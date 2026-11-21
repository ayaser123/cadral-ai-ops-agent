"""One command runs the whole eval suite and prints a score:   python -m evals.run_evals [--offline] [--verbose]

Each case runs through the REAL pipeline (router -> agent -> tools -> DB) on a fresh seeded database, then is scored on
routing, tool use, DB side-effects, citations, 'I don't know' behaviour and answer content. Cost/latency/tokens are tracked."""
import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--offline", action="store_true", help="force the deterministic offline engine (no LLM calls)")
ap.add_argument("--verbose", "-v", action="store_true")
ap.add_argument("--min-score", type=float, default=0.0, help="exit 1 if overall pass-rate is below this (CI gate)")
args = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

tmp = tempfile.mkdtemp(prefix="cadral_eval_")
os.environ["DB_PATH"] = str(Path(tmp) / "eval.db")
if args.offline:
    os.environ["GROQ_API_KEY"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db, llm, pipeline, rag, seed  # noqa: E402
from evals.cases import CASES  # noqa: E402


def run_case(cid, email, text, exp):
    seed.seed(reset=True)
    for t in ("conversations", "messages", "replies", "traces", "escalations", "feedback", "inbox", "outbox", "llm_calls"):
        db.x(f"DELETE FROM {t}")
    turns = text if isinstance(text, list) else [text]
    t0 = time.time()
    conv = None
    for i, tx in enumerate(turns):
        conv, iid = pipeline.ingest("web", email, tx, chat_ref=f"eval-{cid}")
        pipeline.process_inbox(iid)
    lat = (time.time() - t0) * 1000 / len(turns)
    rep = db.q("SELECT * FROM replies WHERE conversation_id=? ORDER BY id DESC LIMIT 1", (conv,), one=True)
    traces = [dict(t, detail=json.loads(t["detail"])) for t in db.q("SELECT * FROM traces WHERE conversation_id=? ORDER BY id", (conv,))]
    cites = json.loads(rep["citations"] or "[]") if rep else []
    reply = (rep["draft"] if rep else "") or ""
    checks = {}
    agent = rep["agent"] if rep else None
    checks["route"] = agent == exp["agent"]
    if "tool" in exp:
        checks["tool"] = any(t["step"] == "tool_call" and t["detail"]["tool"] == exp["tool"] for t in traces)
    if "doc_any" in exp:
        checks["citation"] = any(any(d in c["doc"] for d in exp["doc_any"]) for c in cites)
    if "contains" in exp:
        checks["content"] = all(any(a.strip().lower() in reply.lower() for a in s.split("|")) for s in exp["contains"])
    if exp.get("idk"):
        checks["idk"] = "don't have reliable information" in reply and not cites
    elif exp["agent"] == "support" and "doc_any" in exp:
        checks["grounded"] = bool(cites) and "don't have reliable information" not in reply
    if "reply_status" in exp:
        checks["status"] = rep["status"] == exp["reply_status"]
    if "order_status" in exp:
        checks["order_db"] = db.q("SELECT status FROM orders WHERE id=?", (exp["order_status"][0],), one=True)["status"] == exp["order_status"][1]
    if "refund_status" in exp:
        oid, st = exp["refund_status"]
        r = db.q("SELECT status FROM refunds WHERE order_id=? ORDER BY id DESC LIMIT 1", (oid,), one=True)
        checks["refund_db"] = (r["status"] if r else None) == st
    if exp.get("escalated"):
        e = db.q("SELECT severity FROM escalations WHERE conversation_id=?", (conv,), one=True)
        checks["escalation_row"] = e is not None
        sl = db.q("SELECT 1 FROM outbox WHERE channel='slack' AND body LIKE '%Escalation%'", one=True)
        checks["slack_alert"] = sl is not None
        if "severity" in exp:
            checks["severity"] = bool(e) and e["severity"] == exp["severity"]
    return {"id": cid, "text": turns[-1], "pass": all(checks.values()), "checks": checks, "agent": agent, "expected_agent": exp["agent"],
            "latency_ms": round(lat, 1), "cost_usd": rep["cost_usd"] if rep else 0, "tokens": rep["tokens"] if rep else 0, "reply": reply}


def main():
    db.init()
    engine = "groq:" + os.environ.get("GROQ_MODEL_MAIN", "openai/gpt-oss-120b") if llm.available() else "offline-heuristic"
    print(f"Running {len(CASES)} eval cases | engine={engine} | kb={rag.get_index().n_docs} docs / {len(rag.get_index().chunks)} chunks\n")
    results = [run_case(*c) for c in CASES]
    for r in results:
        if args.verbose or not r["pass"]:
            bad = [k for k, v in r["checks"].items() if not v]
            print(f"{'PASS' if r['pass'] else 'FAIL'} {r['id']} [{r['agent']}] {r['text'][:60]!r} {('failed: ' + ','.join(bad)) if bad else ''}")
            if not r["pass"]:
                print(f"       reply: {r['reply'][:200]}")
    by = defaultdict(list)
    for r in results:
        by[r["id"][0]].append(r)
    names = {"S": "support (RAG)", "I": "I-don't-know", "O": "orders (tools)", "M": "multi-turn memory", "E": "escalation"}
    n = len(results)
    passed = sum(r["pass"] for r in results)
    routing = sum(r["checks"]["route"] for r in results) / n
    lat = [r["latency_ms"] for r in results]
    print("\n" + "=" * 62)
    for k, rs in by.items():
        print(f"  {names[k]:<22} {sum(r['pass'] for r in rs):>3}/{len(rs):<3} {100*sum(r['pass'] for r in rs)/len(rs):5.1f}%")
    print("-" * 62)
    print(f"  Routing accuracy        {100*routing:5.1f}%")
    print(f"  Hallucination guard     {sum(r['pass'] for r in by['I'])}/{len(by['I'])} out-of-scope questions answered 'I don't know'")
    print(f"  Latency  p50 / p95      {statistics.median(lat):.0f} ms / {sorted(lat)[int(0.95*(n-1))]:.0f} ms")
    print(f"  Tokens / cost (list)    {sum(r['tokens'] for r in results)} tok / ${sum(r['cost_usd'] for r in results):.4f}")
    score = passed / n
    print(f"\n  OVERALL SCORE: {passed}/{n} = {100*score:.1f}%   (engine: {engine})")
    print("=" * 62)
    out = Path(__file__).parent / ("results_offline.json" if engine == "offline-heuristic" else "results_llm.json")
    out.write_text(json.dumps({"engine": engine, "score": score, "routing_accuracy": routing, "passed": passed, "total": n,
                               "latency_p50_ms": statistics.median(lat), "results": results}, indent=1), encoding="utf-8")
    if score < args.min_score:
        sys.exit(1)


if __name__ == "__main__":
    main()
