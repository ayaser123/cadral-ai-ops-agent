"""End-to-end message pipeline: inbox (durable) -> memory -> router -> specialist -> reply (auto-send or human review) -> outbox."""
import json
import re
import time
import traceback

from . import agents, db, events, feedback, llm, memory, notify

MAX_ATTEMPTS = 5
AUTO_CONF = 0.6


# ------------------------------------------------------------------ ingest (durable first, then process)
def get_conversation(channel, email, name=None, chat_ref=None):
    if chat_ref:
        c = db.q("SELECT * FROM conversations WHERE channel=? AND chat_ref=? ORDER BY id DESC LIMIT 1", (channel, chat_ref), one=True)
    else:
        c = db.q("SELECT * FROM conversations WHERE channel=? AND customer_email=? AND updated_at>? ORDER BY id DESC LIMIT 1",
                 (channel, email, time.time() - 86400), one=True)
    if c:
        return c["id"]
    return db.x("INSERT INTO conversations(channel,customer_email,customer_name,chat_ref,status,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                (channel, email, name, chat_ref, "open", time.time(), time.time()))


def ingest(channel, email, text, name=None, chat_ref=None):
    """Persist the message BEFORE any processing, so it survives crashes / API failures."""
    email = (email or "").strip().lower()
    conv = get_conversation(channel, email, name, chat_ref)
    iid = db.x("INSERT INTO inbox(channel,payload,conversation_id,created_at) VALUES(?,?,?,?)",
               (channel, json.dumps({"email": email, "text": text, "name": name}), conv, time.time()))
    return conv, iid


def process_inbox(iid):
    row = db.q("SELECT * FROM inbox WHERE id=?", (iid,), one=True)
    if not row or row["status"] in ("done", "dead"):
        return
    p = json.loads(row["payload"])
    db.x("UPDATE inbox SET status='processing' WHERE id=?", (iid,))
    try:
        handle(row["conversation_id"], row["channel"], p["email"], p["text"])
        db.x("UPDATE inbox SET status='done', last_error=NULL WHERE id=?", (iid,))
    except Exception as e:  # nothing is lost: retry with exponential backoff, alert Slack when we give up
        att = row["attempts"] + 1
        dead = att >= MAX_ATTEMPTS
        db.x("UPDATE inbox SET status=?, attempts=?, last_error=?, next_attempt=? WHERE id=?",
             ("dead" if dead else "retry", att, f"{type(e).__name__}: {e}"[:400], time.time() + min(600, 5 * 2 ** att), iid))
        if att in (1, MAX_ATTEMPTS):
            notify.slack(f"Pipeline error on inbox #{iid} (attempt {att}/{MAX_ATTEMPTS}{', GIVING UP - needs human' if dead else ''}): {e}\n{traceback.format_exc()[-300:]}", level="error")
        if dead:
            db.x("UPDATE conversations SET status='escalated' WHERE id=?", (row["conversation_id"],))


def retry_due():
    for r in db.q("SELECT id FROM inbox WHERE status IN ('pending','retry') AND next_attempt<=? ORDER BY id", (time.time(),)):
        process_inbox(r["id"])


# ------------------------------------------------------------------ core
def handle(conv_id, channel, email, text):
    t0 = time.time()
    mid = db.x("INSERT INTO messages(conversation_id,role,content,created_at) VALUES(?,?,?,?)", (conv_id, "customer", text, t0))
    db.x("UPDATE conversations SET updated_at=? WHERE id=?", (t0, conv_id))
    events.publish("message", conversation_id=conv_id, role="customer", content=text, channel=channel)
    ctx = memory.build_context(conv_id, email)
    tr = agents.Tracer(conv_id, events.publish)
    tr.add("memory", "shared", {"customer": ctx["customer"].get("name"), "tier": ctx["customer"].get("tier"), "past_conversations": len(ctx["past_conversations"]),
                                "prior_escalations": ctx["prior_escalations"], "last_order_id": ctx["last_order_id"]})
    r = agents.route(text, ctx, tr)
    if r["agent"] == "escalation":
        out = agents.escalation_agent(text, ctx, tr, conv_id)
    elif r["agent"] == "orders":
        out = agents.orders_agent(text, ctx, tr, email, conv_id)
    else:
        out = agents.support_agent(text, ctx, tr)
    out["reply"] = clean(out["reply"])
    agent = r["agent"]
    latency = (time.time() - t0) * 1000
    agg = db.q("SELECT COALESCE(SUM(cost_usd),0) c, COALESCE(SUM(prompt_tokens+completion_tokens),0) t FROM llm_calls "
               "WHERE conversation_id=? AND created_at>=?", (conv_id, t0), one=True)
    auto = bool(out.get("auto_ok")) and out.get("confidence", 0) >= AUTO_CONF and agent != "escalation"
    status = "sent" if auto else "pending"
    rid = db.x("INSERT INTO replies(conversation_id,message_id,agent,draft,final,citations,status,confidence,auto,created_at,latency_ms,cost_usd,tokens) "
               "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
               (conv_id, mid, agent, out["reply"], out["reply"] if auto else None, json.dumps(out.get("citations", [])), status,
                out.get("confidence", 0), int(auto), time.time(), latency, agg["c"], agg["t"]))
    db.x("UPDATE traces SET reply_id=? WHERE conversation_id=? AND reply_id IS NULL", (rid, conv_id))
    stream(conv_id, rid, out["reply"], agent)
    if auto:
        deliver(rid)
    else:
        events.publish("reply_pending", conversation_id=conv_id, reply_id=rid, agent=agent)
    return rid


def clean(text):
    """Normalise LLM typography so replies are plain, channel-safe text (email/Telegram/web)."""
    t = text.replace("‑", "-").replace("‐", "-").replace("–", "-").replace("—", " - ")
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = t.replace(" ", " ").replace(" ", " ")
    t = re.sub(r"\u3010(?:\d+)\u3011", lambda m: "[" + m.group(0)[1:-1] + "]", t)
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)
    t = re.sub(r"(\d) %", r"\1%", t)
    t = re.sub(r"(\d) (am|pm)\b", r"\1\2", t)
    t = re.sub(r"(\[\d\])(\s*\[\d\])+", lambda m: m.group(1), t)  # collapse repeated citation markers
    return t.strip()


def stream(conv_id, rid, text, agent):
    """Streams the draft to the dashboard token-by-token over SSE."""
    words = text.split(" ")
    for i in range(0, len(words), 3):
        events.publish("stream", conversation_id=conv_id, reply_id=rid, agent=agent, delta=" ".join(words[i:i + 3]) + " ")
        time.sleep(0.03)
    events.publish("stream_end", conversation_id=conv_id, reply_id=rid)


def deliver(rid):
    """Send the final reply to the customer on their original channel (through the retrying outbox)."""
    rep = db.q("SELECT * FROM replies WHERE id=?", (rid,), one=True)
    conv = db.q("SELECT * FROM conversations WHERE id=?", (rep["conversation_id"],), one=True)
    text = rep["final"] or rep["draft"]
    db.x("INSERT INTO messages(conversation_id,role,content,agent,created_at) VALUES(?,?,?,?,?)", (conv["id"], "assistant", text, rep["agent"], time.time()))
    if conv["channel"] == "email":
        notify.queue("email", conv["customer_email"], "Re: your Northwick & Co enquiry", text + "\n\nKind regards,\nNorthwick & Co Customer Care")
    elif conv["channel"] == "telegram":
        notify.queue("telegram", conv["chat_ref"], "", text)
    db.x("UPDATE conversations SET updated_at=?, status=CASE WHEN status='escalated' THEN status ELSE 'resolved' END WHERE id=?", (time.time(), conv["id"]))
    events.publish("message", conversation_id=conv["id"], role="assistant", content=text, agent=rep["agent"], reply_id=rid)


# ------------------------------------------------------------------ human review
def review(rid, action, reviewer, text=None, note=None):
    rep = db.q("SELECT * FROM replies WHERE id=?", (rid,), one=True)
    if not rep:
        raise ValueError("reply not found")
    already_sent = rep["status"] in ("sent", "approved", "edited")
    now = time.time()
    if action == "approve":
        db.x("UPDATE replies SET status=?, final=COALESCE(final,draft), reviewer=?, resolved_at=? WHERE id=?",
             ("sent" if already_sent else "approved", reviewer, now, rid))
        if not already_sent:
            deliver(rid)
        feedback.record(rep, "approved", None, note, reviewer)
    elif action == "edit":
        if not text:
            raise ValueError("edited text required")
        db.x("UPDATE replies SET status='edited', final=?, reviewer=?, resolved_at=? WHERE id=?", (text, reviewer, now, rid))
        feedback.record(rep, "edited", text, note, reviewer)
        deliver(rid)
    elif action == "reject":
        db.x("UPDATE replies SET status='rejected', reviewer=?, resolved_at=? WHERE id=?", (reviewer, now, rid))
        feedback.record(rep, "rejected", None, note, reviewer)
        db.x("UPDATE conversations SET status='needs_human' WHERE id=?", (rep["conversation_id"],))
    else:
        raise ValueError("bad action")
    events.publish("reply_reviewed", reply_id=rid, conversation_id=rep["conversation_id"], action=action)
    return db.q("SELECT * FROM replies WHERE id=?", (rid,), one=True)
