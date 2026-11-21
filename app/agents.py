"""Router + Support / Orders / Escalation agents.
Each agent runs in one of two engines with identical interfaces and traces:
  * LLM engine (Groq function calling) when GROQ_API_KEY is set
  * offline heuristic engine (deterministic) otherwise or if the LLM is down - so nothing is ever lost."""
import json
import re

from . import config, feedback, llm, memory, notify, rag, tools, db

SYSTEM_BASE = ("You are the customer-operations assistant for Northwick & Co, a UK online homeware retailer. "
               "Be warm, concise, British English, no more than 120 words. Never invent policies, prices, order facts or links.")

# ---------------------------------------------------------------- escalation signals
ANGER = ["furious", "disgusting", "appalling", "unacceptable", "ridiculous", "worst", "scam", "rip-off", "rip off", "useless",
         "pathetic", "fed up", "never again", "shocking", "outrageous", "awful", "terrible", "disgraceful", "incompetent", "livid",
         "sick of", "joke", "fuming", "hate"]
LEGAL = ["solicitor", "lawyer", "legal action", "sue ", "court", "trading standards", "ombudsman", "chargeback", "fraud",
         "police", "report you", "small claims", "gdpr complaint", "discriminat", "watchdog", "ico "]
SAFETY = ["injur", "burn", "caught fire", "on fire", "hospital", "allergic", "electric shock", "smoke", "hurt "]
HUMAN = ["speak to a manager", "speak to a human", "real person", "talk to a person", "speak to someone", "manager", "human agent",
         "complaint"]
SWEAR = ["fuck", "shit", "bastard", "crap", "bloody useless", "idiots"]
ORDER_WORDS = ["where is my order", "where's my order", "track", "tracking", "my order", "refund", "cancel", "change my address",
               "change the address", "update my address", "delivery address", "has not arrived", "hasn't arrived", "not arrived",
               "status of", "parcel", "dispatched", "return my", "money back", "send it back", "wrong item", "arrived damaged",
               "arrived broken", "faulty", "delayed", "late delivery", "order status"]


def escalation_signals(text):
    t = " " + text.lower() + " "
    sig = []
    for label, words in (("anger", ANGER), ("legal", LEGAL), ("safety", SAFETY), ("human_request", HUMAN), ("profanity", SWEAR)):
        hit = [w for w in words if w in t]
        if hit:
            sig.append((label, hit))
    letters = [c for c in text if c.isalpha()]
    if len(letters) > 15 and sum(c.isupper() for c in letters) / len(letters) > 0.6:
        sig.append(("shouting", ["CAPS"]))
    if text.count("!") >= 3:
        sig.append(("exclamation", ["!!!"]))
    return sig


def heuristic_route(text, ctx=None):
    t = text.lower()
    sig = escalation_signals(text)
    labels = {s[0] for s in sig}
    if labels & {"legal", "safety", "profanity", "human_request", "shouting"} or ("anger" in labels):
        return {"agent": "escalation", "intent": "escalation", "confidence": 0.9, "reason": "signals: " + ", ".join(sorted(labels))}
    if "exclamation" in labels and re.search(r"order|refund|parcel|deliver", t):
        return {"agent": "escalation", "intent": "escalation", "confidence": 0.7, "reason": "heated tone + order issue"}
    has_id = bool(memory.find_order_ids(text))
    follow = bool(ctx and ctx.get("last_order_id") and re.search(r"\b(it|that|this one|the same)\b", t)
                  and re.search(r"refund|cancel|status|where|track|address|money", t))
    if has_id or follow or any(w in t for w in ORDER_WORDS):
        # policy QUESTIONS about refunds/returns (no order context) go to support
        if not has_id and not follow and re.search(r"\b(what is|what's|what are|how (long|much|do|does)|do you|can i|are there|is there|policy)\b", t) \
                and not re.search(r"\bmy (order|parcel|refund|package)\b", t):
            return {"agent": "support", "intent": "policy_question", "confidence": 0.7, "reason": "general policy question"}
        return {"agent": "orders", "intent": "order_action", "confidence": 0.85, "reason": "order id / order-specific request"}
    return {"agent": "support", "intent": "faq", "confidence": 0.75, "reason": "general question -> knowledge base"}


class Tracer:
    def __init__(self, conv_id, publish):
        self.conv_id, self.steps, self._pub = conv_id, [], publish

    def add(self, step, agent, detail):
        self.steps.append({"step": step, "agent": agent, "detail": detail})
        db.x("INSERT INTO traces(conversation_id,step,agent,detail,created_at) VALUES(?,?,?,?,strftime('%s','now'))",
             (self.conv_id, step, agent, json.dumps(detail, default=str)[:4000]))
        self._pub("trace", conversation_id=self.conv_id, step=step, agent=agent, detail=detail)


# ---------------------------------------------------------------- router
def route(text, ctx, tr):
    h = heuristic_route(text, ctx)
    out = dict(h, engine="heuristic")
    engine_note = "heuristic"
    try:
        if config.FT_BASE_URL:
            r = llm.chat([{"role": "system", "content": "Classify the message into support, orders or escalation. Reply JSON {\"agent\":...}."},
                          {"role": "user", "content": text}], model=config.FT_MODEL, purpose="router-ft", conv_id=tr.conv_id,
                         json_mode=False, base_url=config.FT_BASE_URL.rstrip("/") + "/v1/chat/completions", api_key="none")
            agent = json.loads(re.search(r"\{.*\}", r["content"], re.S).group(0))["agent"]
            engine_note = "fine-tuned " + config.FT_MODEL
        elif llm.available():
            r = llm.chat([{"role": "system", "content": "You are a router for an e-commerce support system. Classify the customer's message.\n"
                           "agent=orders: concerns a specific order (status, tracking, refund, cancel, address change).\n"
                           "agent=support: general product/policy/delivery/FAQ question.\n"
                           "agent=escalation: angry, abusive, legal threat, safety issue, or asks for a human/manager.\n"
                           "Reply ONLY JSON: {\"agent\":\"support|orders|escalation\",\"intent\":\"...\",\"confidence\":0-1,\"reason\":\"...\"}\n"
                           f"Context:\n{memory.render(ctx)}"},
                          {"role": "user", "content": text}], model=config.GROQ_MODEL_FAST, purpose="router", conv_id=tr.conv_id, json_mode=True)
            j = json.loads(r["content"])
            agent = j["agent"]
            out.update(intent=j.get("intent", ""), confidence=float(j.get("confidence", 0.8)), reason=j.get("reason", ""))
            engine_note = "llm " + config.GROQ_MODEL_FAST
        else:
            agent = h["agent"]
        if agent == "escalation" and not escalation_signals(text):
            agent = h["agent"]  # LLM alone may not escalate without a hard signal unless very confident
            out["reason"] += " (LLM escalation downgraded: no hard risk signal in text)"
        if agent in ("support", "orders", "escalation"):
            out["agent"] = agent
        # Safety net: hard rules always win for high-risk content, whatever the model says.
        if h["agent"] == "escalation" and h["confidence"] >= 0.9:
            out.update(agent="escalation", reason=h["reason"] + " (safety override)")
    except (llm.LLMUnavailable, ValueError, KeyError, AttributeError) as e:
        engine_note = f"heuristic (llm unavailable: {str(e)[:80]})"
    out["engine"] = engine_note
    tr.add("route", "router", out)
    return out


# ---------------------------------------------------------------- support agent (RAG)
IDK = ("I'm sorry, I don't have reliable information on that in our documentation, so I don't want to guess. "
       "I've passed your question to a member of the team who will get back to you.")


def support_agent(text, ctx, tr):
    hits, ok = rag.retrieve(text)
    tr.add("retrieve", "support", {"query": text, "answerable": ok, "hits": [
        {"id": h["chunk"].id, "score": h["score"], "bm25": h["bm25"], "vec": h["vec"]} for h in hits]})
    learned = feedback.prompt_block(text)
    if learned:
        tr.add("feedback_context", "support", {"injected": learned[:600]})
    # exact-reuse of a human-corrected answer to a near-identical past question (offline learning path)
    ex = [e for e in feedback.similar(text, k=1, min_sim=0.8) if e["action"] == "edited" and e["human_reply"]]
    if ex and not llm.available():
        tr.add("learned_reply", "support", {"from_feedback_id": ex[0]["id"], "similarity": ex[0]["sim"]})
        return {"reply": ex[0]["human_reply"], "citations": [], "confidence": 0.9, "auto_ok": True, "learned": True}
    if not ok:
        return {"reply": IDK, "citations": [], "confidence": 0.2, "auto_ok": False, "idk": True}
    cites = [rag.citation(h) for h in hits[:3]]
    if llm.available():
        try:
            ctx_txt = "\n\n".join(f"[{i+1}] ({c['doc']}) {hits[i]['chunk'].text}" for i, c in enumerate(cites))
            sys = (SYSTEM_BASE + "\nAnswer ONLY from the numbered sources below and cite them like [1]. If the sources do not contain the answer, "
                   "reply exactly: I don't know.\n" + (("\n" + learned) if learned else ""))
            r = llm.chat([{"role": "system", "content": sys}, {"role": "user", "content": f"Sources:\n{ctx_txt}\n\nCustomer: {text}"}],
                         purpose="support", conv_id=tr.conv_id)
            reply = r["content"].strip()
            if reply.lower().startswith("i don't know") or "i don't know" in reply.lower()[:40]:
                return {"reply": IDK, "citations": [], "confidence": 0.2, "auto_ok": False, "idk": True}
            if not re.search(r"\[\d\]", reply):
                reply += " " + " ".join(f"[{i+1}]" for i in range(min(2, len(cites))))
            tr.add("answer", "support", {"engine": "llm", "sources": [c["id"] for c in cites]})
            return {"reply": reply, "citations": cites, "confidence": min(0.95, 0.5 + hits[0]["score"] / 2), "auto_ok": True}
        except llm.LLMUnavailable:
            tr.add("llm_fallback", "support", {"reason": "LLM unavailable, using extractive answer"})
    picked = rag.best_sentences(text, hits)
    reply = " ".join(f"{s.rstrip('.')}. [{i}]" for i, s in picked) if picked else IDK
    reply = "Thanks for getting in touch. " + reply + " Is there anything else I can help with?"
    tr.add("answer", "support", {"engine": "extractive", "sources": [c["id"] for c in cites]})
    return {"reply": reply, "citations": cites, "confidence": min(0.95, 0.5 + hits[0]["score"] / 2), "auto_ok": True}


# ---------------------------------------------------------------- orders agent (tools)
def _run_tool(tr, name, args, email, conv_id, why):
    res = tools.call(name, args, email, conv_id)
    tr.add("tool_call", "orders", {"tool": name, "args": args, "why": why, "result": res})
    return res


def _fmt_status(r):
    items = ", ".join(r["items"])
    if r["status"] == "shipped":
        late = " It is running later than estimated - I'm sorry about that; if it hasn't arrived within 3 working days of the estimate, reply and we'll open a carrier investigation." if r["is_late"] else ""
        return (f"Order {r['order_id']} ({items}) has shipped with {r['carrier']}, tracking {r['tracking']}, estimated delivery {r['eta']}.{late}")
    if r["status"] == "processing":
        return f"Order {r['order_id']} ({items}) is being prepared in our warehouse and should be dispatched within 1 working day."
    if r["status"] == "delivered":
        return f"Order {r['order_id']} ({items}) was delivered on {r['delivered_at']}."
    return f"Order {r['order_id']} is currently {r['status']}."


def _offline_orders(text, ctx, tr, email, conv_id):
    t = text.lower()
    ids = memory.find_order_ids(text) or ([ctx["last_order_id"]] if ctx.get("last_order_id") else [])
    want_refund = bool(re.search(r"refund|money back|return|send it back|faulty|damaged|broken|wrong item", t))
    want_cancel = bool(re.search(r"cancel", t))
    want_addr = bool(re.search(r"address", t))
    if not ids:
        res = _run_tool(tr, "list_customer_orders", {}, email, conv_id, "customer gave no order number; list their orders")
        rows = res.get("orders", [])
        if not rows:
            return {"reply": "I couldn't find any orders for this email address. Could you share your order number (it looks like ORD-1234) or the email you ordered with?",
                    "confidence": 0.5, "auto_ok": False}
        lst = "; ".join(f"{o['id']} ({o['status']}, GBP {o['total']:.2f})" for o in rows[:5])
        return {"reply": f"Happy to help. Which order do you mean? Your recent orders: {lst}.", "confidence": 0.7, "auto_ok": True}
    oid = ids[0]
    if want_cancel:
        r = _run_tool(tr, "update_order", {"order_id": oid, "action": "cancel"}, email, conv_id, "customer asked to cancel")
        if r["ok"]:
            return {"reply": f"Done - order {oid} has been cancelled and you won't be charged.", "confidence": 0.95, "auto_ok": True}
        return {"reply": f"I couldn't cancel {oid}: {r['error'].replace('_', ' ')}. If it has shipped you can return it for a refund once it arrives.", "confidence": 0.8, "auto_ok": True}
    if want_addr:
        m = re.search(r"(?:to|address(?: is)?[:\-]?)\s+(\d+[^.?!]*?[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2})", text)
        if not m:
            return {"reply": f"I can update the address on {oid} while it's still processing. What is the full new delivery address, including postcode?", "confidence": 0.7, "auto_ok": True}
        r = _run_tool(tr, "update_order", {"order_id": oid, "action": "change_address", "new_address": m.group(1)}, email, conv_id, "address change")
        if r["ok"]:
            return {"reply": f"Your delivery address on {oid} is now {m.group(1)}.", "confidence": 0.9, "auto_ok": True}
        return {"reply": f"I couldn't change the address on {oid}: {r['error'].replace('_', ' ')}.", "confidence": 0.8, "auto_ok": True}
    st = _run_tool(tr, "get_order_status", {"order_id": oid}, email, conv_id, "check order before answering")
    if not st["ok"]:
        return {"reply": f"I couldn't find order {oid} under this email address, so I can't share details. Please double-check the number or contact us from the email you ordered with.",
                "confidence": 0.6, "auto_ok": True}
    if want_refund:
        reason = "faulty/damaged item" if re.search(r"faulty|damaged|broken", t) else "customer return"
        r = _run_tool(tr, "issue_refund", {"order_id": oid, "reason": reason}, email, conv_id, "customer requested refund/return")
        if r["ok"] and r["status"] == "issued":
            return {"reply": f"I've refunded GBP {r['amount_gbp']:.2f} for order {oid} to your original payment method; it should arrive within 5-7 working days.", "confidence": 0.95, "auto_ok": True}
        if r["ok"]:
            return {"reply": f"I've submitted a GBP {r['amount_gbp']:.2f} refund for {oid}. Because it's over GBP {config.AUTO_REFUND_LIMIT:.0f}, a manager needs to approve it - you'll hear from us within 1 working day.", "confidence": 0.8, "auto_ok": False}
        return {"reply": f"I'm sorry, I can't refund {oid} right now: {r['error'].replace('_', ' ')}. {'Our returns window is 30 days from delivery.' if 'window' in r['error'] else ''}".strip(), "confidence": 0.8, "auto_ok": True}
    return {"reply": _fmt_status(st), "confidence": 0.95, "auto_ok": True}


def _needed_tool(text, ctx):
    """Which write-tool a clear customer request requires (only when an order is identifiable)."""
    t = text.lower()
    if not (memory.find_order_ids(text) or ctx.get("last_order_id")):
        return None
    if re.search(r"refund|money back", t):
        return "issue_refund"
    if re.search(r"cancel", t):
        return "update_order"
    return None


def orders_agent(text, ctx, tr, email, conv_id):
    learned = feedback.prompt_block(text)
    if llm.available():
        try:
            sys = (SYSTEM_BASE + "\nYou are the Orders agent. ALWAYS use tools for order facts and actions - never state order data from memory. "
                   "Check the order (get_order_status) before refunding. Refund only delivered orders. If a tool returns an error, explain it kindly. "
                   "If a refund is pending_approval, say a manager will approve it. Never say an address was changed, an order cancelled or a refund made unless the tool returned ok=true; if the new address is missing, ask for it.\n" + memory.render(ctx) + (("\n" + learned) if learned else ""))
            msgs = [{"role": "system", "content": sys}] + [
                {"role": "user" if m["role"] == "customer" else "assistant", "content": m["content"]} for m in ctx["history"][-6:]]
            if not msgs or msgs[-1]["role"] != "user" or msgs[-1]["content"] != text:
                msgs.append({"role": "user", "content": text})
            used, called, nudged = [], [], False
            for _ in range(6):
                r = llm.chat(msgs, tools=tools.SCHEMAS, purpose="orders", conv_id=conv_id)
                if not r["tool_calls"]:
                    wrote = any(n in ("issue_refund", "update_order") and u.get("ok") for n, u in zip(called, used))
                    claim = re.search(r"(I(?:'ve| have|'ll have)|has been|have been|is now|are now) (?:\w+ ){0,2}(updated|changed|cancell?ed|refunded|issued|processed)", r["content"])
                    if claim and not wrote and not nudged:
                        nudged = True  # claimed a state change that no successful tool call performed
                        tr.add("guard", "orders", {"reason": "reply claims a completed action but no write tool succeeded; re-prompting"})
                        msgs.append({"role": "assistant", "content": r["content"]})
                        msgs.append({"role": "user", "content": "(system) No write tool has succeeded, so nothing has actually been changed. Either call the correct tool now, or - if information such as the new address is missing - ask the customer for it. Rewrite your reply truthfully."})
                        continue
                    need = _needed_tool(text, ctx)
                    if need and not any(tc_name == need for tc_name in called) and not nudged:
                        nudged = True  # model claimed/offered an action without executing it: force the real function call
                        tr.add("guard", "orders", {"reason": f"reply implied '{need}' but the tool was not called; re-prompting"})
                        msgs.append({"role": "assistant", "content": r["content"]})
                        msgs.append({"role": "user", "content": f"(system) You have not actually called {need}. Call the tool now; do not claim an action you have not executed."})
                        continue
                    pending = any(u.get("status") == "pending_approval" for u in used)
                    return {"reply": r["content"].strip(), "confidence": 0.85, "auto_ok": not pending}
                msgs.append(r["raw"])
                for tc in r["tool_calls"]:
                    res = _run_tool(tr, tc["name"], tc["args"], email, conv_id, "chosen by LLM function calling")
                    used.append(res)
                    called.append(tc["name"])
                    msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(res, default=str)})
            return {"reply": "I'm passing this to a colleague to complete.", "confidence": 0.3, "auto_ok": False}
        except llm.LLMUnavailable:
            tr.add("llm_fallback", "orders", {"reason": "LLM unavailable, using deterministic planner"})
    out = _offline_orders(text, ctx, tr, email, conv_id)
    ex = feedback.similar(text, k=1, min_sim=0.8)
    if ex and ex[0]["action"] == "edited" and ex[0]["human_reply"] and ex[0]["agent"] == "orders" and not re.search(r"ORD-\d", ex[0]["human_reply"]):
        tr.add("learned_reply", "orders", {"from_feedback_id": ex[0]["id"], "similarity": ex[0]["sim"]})
        out["reply"] = ex[0]["human_reply"]
    return out


# ---------------------------------------------------------------- escalation agent
def escalation_agent(text, ctx, tr, conv_id):
    sig = escalation_signals(text)
    labels = [s[0] for s in sig]
    severity = "critical" if {"legal", "safety"} & set(labels) else ("high" if len(labels) >= 2 or "profanity" in labels else "medium")
    c = ctx["customer"]
    summary = (f"{c.get('name') or c['email']} ({c.get('tier')}) - {severity.upper()}. Signals: {', '.join(labels) or 'n/a'}. "
               f"Message: \"{text[:300]}\". History: {ctx['prior_escalations']} prior escalations, {len(ctx['past_conversations'])} past chats. "
               f"Recent orders: {', '.join(o['id'] + ' ' + o['status'] for o in ctx['recent_orders'][:3]) or 'none'}.")
    if llm.available():
        try:
            r = llm.chat([{"role": "system", "content": "Summarise this escalated customer case for a human agent in 3 short bullet points: issue, customer sentiment, recommended next step."},
                          {"role": "user", "content": memory.render(ctx) + "\nConversation:\n" + "\n".join(f"{m['role']}: {m['content']}" for m in ctx['history'][-6:]) + f"\ncustomer: {text}"}],
                         model=config.GROQ_MODEL_FAST, purpose="escalation-summary", conv_id=conv_id)
            summary += "\nAI summary: " + r["content"].strip()
        except llm.LLMUnavailable:
            pass
    eid = db.x("INSERT INTO escalations(conversation_id,reason,summary,severity,created_at) VALUES(?,?,?,?,strftime('%s','now'))",
               (conv_id, ", ".join(labels), summary, severity))
    db.x("UPDATE conversations SET status='escalated' WHERE id=?", (conv_id,))
    tr.add("escalate", "escalation", {"escalation_id": eid, "severity": severity, "signals": labels, "summary": summary})
    notify.slack(f"Escalation #{eid} (conversation {conv_id}, {severity}): {summary[:400]}", level="escalation")
    reply = ("I'm really sorry about this experience, and I completely understand your frustration. I've passed your case, with all the details, "
             "to a senior member of our team who will contact you personally - you won't need to repeat yourself. "
             + ("We're treating this as urgent. " if severity == "critical" else "") + "Thank you for your patience.")
    return {"reply": reply, "confidence": 0.9, "auto_ok": False, "escalated": True, "severity": severity}
