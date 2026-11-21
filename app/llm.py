"""LLM gateway: Groq (OpenAI-compatible, free tier) with retries + exponential backoff, per-call token/cost/latency logging.
If no key is configured (or Groq is down after retries) callers get LLMUnavailable and use the offline heuristic engine."""
import json
import threading
import time

import httpx

from . import config, db

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class LLMUnavailable(Exception):
    pass


def available():
    return bool(config.GROQ_API_KEY)


def cost(model, pt, ct):
    pi, po = config.PRICING.get(model, (0.0, 0.0))
    return (pt * pi + ct * po) / 1_000_000


def log_call(conv_id, purpose, model, pt, ct, latency_ms, ok=True):
    db.x("INSERT INTO llm_calls(conversation_id,purpose,model,prompt_tokens,completion_tokens,latency_ms,cost_usd,ok,created_at) "
         "VALUES(?,?,?,?,?,?,?,?,?)", (conv_id, purpose, model, pt, ct, latency_ms, cost(model, pt, ct), int(ok), time.time()))


def chat(messages, tools=None, model=None, purpose="", conv_id=None, json_mode=False, temperature=0.2, retries=3,
         base_url=None, api_key=None):
    """Returns {'content', 'tool_calls': [{'id','name','args'}], 'usage'}. Raises LLMUnavailable."""
    key = api_key or config.GROQ_API_KEY
    if not key and not base_url:
        raise LLMUnavailable("no API key")
    model = model or config.GROQ_MODEL_MAIN
    body = {"model": model, "messages": messages, "temperature": temperature}
    if model.startswith("openai/gpt-oss"):
        body["reasoning_effort"] = "low"
    if tools:
        body["tool_choice"] = "auto"
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    _throttle(est_tokens(json.dumps(messages)) + (est_tokens(json.dumps(tools)) if tools else 0) + 500)
    last = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            r = httpx.post(base_url or GROQ_URL, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=40)
            if r.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError(f"{r.status_code}", request=r.request, response=r)
            r.raise_for_status()
            data = r.json()
            msg = data["choices"][0]["message"]
            u = data.get("usage", {})
            lat = (time.time() - t0) * 1000
            log_call(conv_id, purpose, model, u.get("prompt_tokens", 0), u.get("completion_tokens", 0), lat)
            calls = []
            for tc in msg.get("tool_calls") or []:
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                calls.append({"id": tc["id"], "name": tc["function"]["name"], "args": args})
            return {"content": msg.get("content") or "", "tool_calls": calls, "usage": u, "raw": msg}
        except (httpx.HTTPError, KeyError, ValueError) as e:
            last = e
            log_call(conv_id, purpose, model, 0, 0, (time.time() - t0) * 1000, ok=False)
            retry_after = 0
            if isinstance(e, httpx.HTTPStatusError) and e.response is not None:
                retry_after = float(e.response.headers.get("retry-after", 0) or 0)
            time.sleep(min(8, max(retry_after, 1.5 ** attempt)))
    raise LLMUnavailable(f"LLM failed after {retries} attempts: {last}")


_window = []  # (timestamp, tokens) of recent calls
_wlock = threading.Lock()
TPM_BUDGET = int(config.env("LLM_TPM_BUDGET", "6500"))  # Groq free tier ~8k tokens/min: pace ourselves instead of eating 429s


def _throttle(est):
    while True:
        with _wlock:
            now = time.time()
            _window[:] = [w for w in _window if now - w[0] < 60]
            used = sum(t for _, t in _window)
            if used + est <= TPM_BUDGET or not _window:
                _window.append((now, est))
                return
            wait = 60 - (now - _window[0][0]) + 0.2
        time.sleep(max(0.5, wait))


def est_tokens(text):
    return max(1, len(text) // 4)
