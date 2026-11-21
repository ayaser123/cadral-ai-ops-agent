# Write-up: Autonomous AI Operations Agent

## 1. Architecture
Messages from web chat, email (IMAP or webhook) and Telegram are written to a **durable inbox** before anything else. A worker loads **shared memory** (this conversation + the customer's history), the **router** picks a specialist, and the specialist produces a draft reply with a full **trace**. Replies that are confident and low-risk are auto-sent; everything else waits on the dashboard for approve / edit / reject. Human decisions flow into a **feedback store** that improves future replies. Outbound messages and alerts go through a retrying **outbox**. See the diagram in `README.md`.

Agents run on one of two engines behind the same interface: **Groq function-calling (gpt-oss-120b)**, or a **deterministic offline engine**. The offline engine exists so the system (and the evals) run with zero keys, and as the graceful-degradation path when the LLM API is down: *the customer still gets a correct answer for the common cases and nothing is lost*.

## 2. Trade-offs
| Decision | Why | Cost |
|---|---|---|
| SQLite + in-process workers | Zero-ops, free-tier friendly, transactional outbox/inbox | Single node; see scaling |
| Hashed n-gram vectors instead of a neural embedding model | Offline, deterministic, no model download, robust to typos | Weaker semantic recall than e5/bge; swap `rag.embed` for a neural encoder (one function) |
| Feature reranker instead of a cross-encoder | No GPU/latency cost; coverage feature doubles as the "I don't know" signal | Can't resolve genuinely ambiguous lexical ties (see eval S06) |
| Policy in tools, not in prompts | The model cannot be talked into a bad refund: ownership, window, double-refund and GBP 100 limit are enforced in code | Policy changes need a deploy |
| Safety override on the router | Legal/safety/abuse signals always escalate, whatever any model says | Some false-positive escalations (cheap) rather than missed ones (expensive) |
| Auto-send gate (confidence + agent type) | Escalations, refunds over limit and "I don't know" always get a human | Lowers auto-resolution % deliberately |

## 3. Evaluation results (65 cases, `python -m evals.run_evals`)
Engine: offline deterministic (no LLM calls), fresh DB per case.

| Category | Pass |
|---|---|
| Support / RAG (citation + content) | 24/25 (96%) |
| "I don't know" (out-of-scope) | 8/8 (100%) |
| Orders (tool calls + DB side effects) | 21/21 (100%) |
| Multi-turn memory | 1/1 |
| Escalation (route + record + Slack + severity) | 10/10 (100%) |
| **Overall** | **64/65 = 98.5%** · routing 100% · p50 ~0.65 s · $0 |

Honest caveats: (a) the offline rules and the test set were developed together, so this is a **development-set** number; the held-out router test below is the fair generalisation figure. (b) The one miss (S06, *"Is delivery free over a certain amount?"*) retrieves the damaged-items policy because "free" and "delivery" co-occur in one of its sentences, which is a genuine lexical-ambiguity failure that a cross-encoder or LLM reranker fixes. (c) **LLM engine (Groq `openai/gpt-oss-120b`, router `gpt-oss-20b`)**: a first full run scored **57/65 = 87.7%** (support 24/25, orders 15/21, escalation 10/10, memory 1/1; ~$0.011 list-price for 65 cases, ~76k tokens). Most misses were typography that broke exact-match checks (non-breaking hyphens, curly quotes, stripped bold) plus two real defects the eval exposed, both now fixed: (i) the model said *"I'll process that for you now"* / *"I've updated the address"* **without calling the tool** (fix: an action-claim guard that re-prompts until a write tool actually succeeds, plus a reply normaliser), and (ii) pending-approval refunds were being auto-sent (an `auto_ok` bug). The LLM router also escalated a harmless out-of-scope question (*"what is the wifi password"*); escalation now needs a hard risk signal in the text. A follow-up full run was cut off by Groq's free-tier **200k tokens/day** cap (I added a client-side token-per-minute pacer, `LLM_TPM_BUDGET`), so I am not quoting a post-fix LLM-engine score; re-run `python -m evals.run_evals` after the daily quota resets (it writes `evals/results_llm.json`). Note also that the free tier makes the LLM engine slow under load (8k tokens/min), which is why the offline engine doubles as a fallback.

## 4. Fine-tuning results
Task: 3-way router classification. Model: Qwen2.5-1.5B-Instruct + LoRA (r=16), 1,200 train examples (template family A), tested on 180 **held-out** messages in different phrasing (family B).

| Router | Accuracy | Macro-F1 | Latency | Cost / 1k requests |
|---|---|---|---|---|
| Rules baseline (measured) | **92.8%** | 0.927 | <1 ms | $0 |
| Base Qwen2.5-1.5B zero-shot | *run notebook* | | | |
| **Fine-tuned Qwen2.5-1.5B + LoRA** | *run notebook* | | | |
| gpt-oss-120b API (Groq) | *run notebook* | | | |

The training machine for this submission had no GPU, so base/fine-tuned/API rows are produced by `finetune/colab_finetune.ipynb` (free T4, ~10 min total) and written to `finetune/results.json`. I have not run it, so I am not quoting numbers I haven't measured. The baseline row is real: its errors are mostly escalations phrased without keywords ("honestly sick of this...") and casual order requests ("what happened to my package"), which is exactly where a fine-tuned model should gain.

**When to use which**
- **Fine-tuned small model**: narrow, high-volume, latency/cost-sensitive decisions with stable labels (routing, tone, sentiment). Milliseconds, runs on a CPU/small GPU, no per-token fee, data stays in-house. Needs labelled data and retraining when categories change. *Used here for the router.*
- **RAG**: when answers depend on facts that change or must be auditable (policies, prices, stock). Update a document, not a model; citations come free; "I don't know" is possible. *Used for support answers.* Never fine-tune facts in.
- **Large API model**: open-ended reasoning, multi-step tool use, ambiguous or novel inputs, low volume, or when speed-to-ship beats unit cost. *Used for orders reasoning and answer synthesis.*

## 5. Self-improving loop
Every approve / edit / reject is stored with the customer message, AI draft, human version and an optional note. At inference time: (1) the 2 most similar past corrections are added to the prompt (or reused verbatim offline when similarity >= 0.8: demonstrated in the dashboard: a rejected IDK on "price match with other retailers" is answered correctly for the next customer); (2) reviewer notes become standing **guidelines** injected into every agent prompt; (3) `POST /api/feedback/export` writes chat-format JSONL that `train_lora.py --extra` mixes into the next fine-tune. Accuracy-over-time on the dashboard is the loop's scoreboard.

## 6. What breaks at scale, and the fix
| Breaks | Why | Fix |
|---|---|---|
| SQLite write contention, single node | One writer; workers in-process | Postgres + `SELECT ... FOR UPDATE SKIP LOCKED` queues (or Redis/SQS); stateless API replicas |
| SSE event bus is in-memory | Events don't cross replicas | Redis pub/sub |
| RAG index rebuilt in RAM, O(N) scan | Fine for 79 chunks, not 100k | pgvector/Qdrant ANN + Postgres FTS; incremental ingestion; neural embeddings + cross-encoder rerank |
| LLM rate limits (free tier) | 429 under burst | Queue + token-bucket, model fallback chain, response caching for FAQ |
| Feedback retrieval embeds on every call | O(feedback rows) per request | Persist vectors, ANN index; dedupe and decay stale corrections |
| One bad correction can poison answers | Few-shot from raw human edits | Approval workflow for guidelines, A/B on eval set before promoting, per-agent scoping |
| Cost visibility is per call, not per tenant | | Budgets per conversation, alerting on cost/p95 drift |
| Prompt-injection via customer text or retrieved docs | Tools take model-produced args | Already mitigated by server-side identity injection and policy checks; add output filters and a human gate for any new write-tool |
| Auth is a single shared secret with HMAC tokens | | OIDC/SSO, per-user revocation, audit log of every tool call |

## 7. Deployment and status
Dockerised (`Dockerfile`, `docker-compose.yml`), secrets only via environment, `render.yaml` for the free tier. **Not yet done by me and needing your accounts/keys**: pushing to GitHub, the live deployment URL, Slack webhook / SMTP / Telegram credentials, the Colab fine-tune run, the Groq eval run, and the 3-4 minute demo video.
