"""Analytics (dashboard) + daily management report (email)."""
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import config, db, notify


def analytics(days=7):
    since = time.time() - days * 86400
    reps = db.q("SELECT * FROM replies WHERE created_at>=?", (since,))
    total = len(reps)
    auto_ok = [r for r in reps if r["auto"] and r["status"] == "sent"]
    reviewed = [r for r in reps if r["status"] in ("approved", "edited", "rejected")]
    good = [r for r in reps if r["status"] in ("sent", "approved")]
    llm = db.q("SELECT COUNT(*) n, COALESCE(SUM(cost_usd),0) c, COALESCE(SUM(prompt_tokens+completion_tokens),0) t, "
               "COALESCE(AVG(latency_ms),0) l FROM llm_calls WHERE created_at>=?", (since,), one=True)
    convs = db.q("SELECT COUNT(DISTINCT conversation_id) n FROM replies WHERE created_at>=?", (since,), one=True)["n"]
    by_agent = Counter(r["agent"] for r in reps)
    by_day = {}
    for r in reps:
        k = datetime.fromtimestamp(r["created_at"], timezone.utc).strftime("%Y-%m-%d")
        d = by_day.setdefault(k, {"total": 0, "good": 0})
        d["total"] += 1
        d["good"] += r["status"] in ("sent", "approved")
    refunds = db.q("SELECT COUNT(*) n, COALESCE(SUM(amount),0) s FROM refunds WHERE status='issued' AND created_at>=?", (since,), one=True)
    return {
        "days": days, "replies": total, "conversations": convs,
        "auto_resolution_pct": round(100 * len(auto_ok) / total, 1) if total else 0,
        "accuracy_pct": round(100 * len(good) / total, 1) if total else 0,
        "avg_response_ms": round(sum(r["latency_ms"] or 0 for r in reps) / total, 0) if total else 0,
        "cost_per_conversation_usd": round(llm["c"] / convs, 5) if convs else 0,
        "total_cost_usd": round(llm["c"], 4), "total_tokens": llm["t"], "llm_calls": llm["n"], "avg_llm_latency_ms": round(llm["l"], 0),
        "by_agent": dict(by_agent),
        "accuracy_over_time": [{"day": k, "accuracy_pct": round(100 * v["good"] / v["total"], 1), "n": v["total"]} for k, v in sorted(by_day.items())],
        "pending_review": len([r for r in reps if r["status"] == "pending"]),
        "reviewed": len(reviewed), "edited": len([r for r in reps if r["status"] == "edited"]),
        "rejected": len([r for r in reps if r["status"] == "rejected"]),
        "refunds_issued": refunds["n"], "refunds_gbp": round(refunds["s"], 2),
        "escalations": db.q("SELECT COUNT(*) n FROM escalations WHERE created_at>=?", (since,), one=True)["n"],
        "feedback_items": db.q("SELECT COUNT(*) n FROM feedback", one=True)["n"],
    }


def top_issues(days=1):
    since = time.time() - days * 86400
    rows = db.q("SELECT detail FROM traces WHERE step='route' AND created_at>=?", (since,))
    import json
    c = Counter()
    for r in rows:
        try:
            c[json.loads(r["detail"]).get("intent", "unknown")] += 1
        except Exception:
            pass
    return c.most_common(5)


def build_report():
    a = analytics(1)
    issues = top_issues(1)
    lines = [f"Northwick & Co - Daily AI Operations Report ({datetime.now(timezone.utc):%Y-%m-%d})", "=" * 56,
             f"Conversations handled : {a['conversations']}   (replies: {a['replies']})",
             f"Auto-resolved         : {a['auto_resolution_pct']}%",
             f"AI accuracy           : {a['accuracy_pct']}%  (edited: {a['edited']}, rejected: {a['rejected']}, awaiting review: {a['pending_review']})",
             f"Avg response time     : {a['avg_response_ms']/1000:.2f}s",
             f"Refunds issued        : {a['refunds_issued']} totalling GBP {a['refunds_gbp']}",
             f"Escalations           : {a['escalations']}",
             f"LLM cost (list price) : ${a['total_cost_usd']}  | per conversation ${a['cost_per_conversation_usd']}",
             "", "Top issues (by routed intent):"] + [f"  - {k}: {v}" for k, v in issues] + \
            ["", "Messages by agent: " + ", ".join(f"{k}={v}" for k, v in a["by_agent"].items())]
    return "\n".join(lines)


def send_daily_report():
    body = build_report()
    d = config.ROOT / "data" / "reports"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"report-{datetime.now(timezone.utc):%Y-%m-%d}.txt").write_text(body, encoding="utf-8")
    notify.queue("email", config.MANAGER_EMAIL or "manager@northwick.example", "Daily AI operations report", body)
    db.kv_set("last_report_date", f"{datetime.now(timezone.utc):%Y-%m-%d}")
    return body


if __name__ == "__main__":
    db.init()
    print(send_daily_report())
