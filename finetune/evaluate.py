"""Compare routers on the held-out test set: accuracy, macro-F1, latency and cost.

  python finetune/evaluate.py --heuristic                     # rules baseline (runs anywhere, no GPU)
  python finetune/evaluate.py --hf Qwen/Qwen2.5-1.5B-Instruct                          # base model (GPU)
  python finetune/evaluate.py --hf Qwen/Qwen2.5-1.5B-Instruct --lora finetune/out/router-lora   # fine-tuned (GPU)
  GROQ_API_KEY=... python finetune/evaluate.py --groq openai/gpt-oss-120b          # large API model
Results are appended to finetune/results.json and rendered by finetune/report.py"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ap = argparse.ArgumentParser()
ap.add_argument("--heuristic", action="store_true")
ap.add_argument("--hf")
ap.add_argument("--lora")
ap.add_argument("--groq")
ap.add_argument("--split", default="test")
ap.add_argument("--gpu-usd-per-hour", type=float, default=0.0, help="0 on free Colab; e.g. 0.35 for a rented T4 to get real cost")
args = ap.parse_args()

LABELS = ["support", "orders", "escalation"]
rows = [json.loads(l) for l in open(Path(__file__).parent / "data" / f"{args.split}.jsonl", encoding="utf-8")]
tok_in = lambda s: max(1, len(s) // 4)


def macro_f1(y, p):
    f = []
    for c in LABELS:
        tp = sum(a == c and b == c for a, b in zip(y, p))
        fp = sum(a != c and b == c for a, b in zip(y, p))
        fn = sum(a == c and b != c for a, b in zip(y, p))
        pr, rc = tp / (tp + fp or 1), tp / (tp + fn or 1)
        f.append(2 * pr * rc / ((pr + rc) or 1))
    return sum(f) / len(f)


def norm(s):
    s = s.strip().lower()
    for c in LABELS:
        if s.startswith(c) or c in s[:20]:
            return c
    return "invalid"


preds, lats, cost, name = [], [], 0.0, ""
sysmsg = rows[0]["messages"][0]["content"]
if args.heuristic:
    from app.agents import heuristic_route
    name = "rules-baseline (heuristic router)"
    for r in rows:
        t0 = time.perf_counter()
        preds.append(heuristic_route(r["messages"][1]["content"])["agent"])
        lats.append((time.perf_counter() - t0) * 1000)
elif args.hf:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.hf)
    model = AutoModelForCausalLM.from_pretrained(args.hf, torch_dtype=torch.float16, device_map="auto")
    if args.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora)
    model.eval()
    name = ("fine-tuned " if args.lora else "base ") + args.hf.split("/")[-1]
    for r in rows:
        prompt = tok.apply_chat_template(r["messages"][:2], tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=4, do_sample=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lats.append((time.perf_counter() - t0) * 1000)
        preds.append(norm(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)))
    cost = sum(lats) / 3.6e6 * args.gpu_usd_per_hour
elif args.groq:
    from app import config, llm
    name = "API " + args.groq
    pi, po = config.PRICING.get(args.groq, (0.59, 0.79))
    for r in rows:
        t0 = time.perf_counter()
        res = llm.chat(r["messages"][:2], model=args.groq, purpose="ft-eval", temperature=0)
        lats.append((time.perf_counter() - t0) * 1000)
        preds.append(norm(res["content"]))
        u = res["usage"]
        cost += (u.get("prompt_tokens", 0) * pi + u.get("completion_tokens", 0) * po) / 1e6
        time.sleep(2.2)  # stay inside free-tier rate limits
else:
    sys.exit("choose one of --heuristic / --hf / --groq")

y = [r["label"] for r in rows]
acc = sum(a == b for a, b in zip(y, preds)) / len(y)
res = {"model": name, "split": args.split, "n": len(rows), "accuracy": round(acc, 4), "macro_f1": round(macro_f1(y, preds), 4),
       "invalid_outputs": preds.count("invalid"), "latency_ms_mean": round(sum(lats) / len(lats), 1),
       "latency_ms_p95": round(sorted(lats)[int(0.95 * (len(lats) - 1))], 1),
       "cost_usd_per_1k_requests": round(cost / len(rows) * 1000, 4), "confusions": dict(Counter(f"{a}->{b}" for a, b in zip(y, preds) if a != b))}
print(json.dumps(res, indent=1))
f = Path(__file__).parent / "results.json"
allr = json.loads(f.read_text()) if f.exists() else []
allr = [x for x in allr if x["model"] != name] + [res]
f.write_text(json.dumps(allr, indent=1))
