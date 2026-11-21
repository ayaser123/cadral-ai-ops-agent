"""Self-improving loop. Human edits/rejections are stored as feedback and feed back into the system three ways:
 1. Few-shot corrections: the most similar past corrections are injected into the agent prompt (or reused directly offline).
 2. Prompt updates: reviewer notes become standing 'guidelines' appended to every agent's system prompt.
 3. Re-training dataset: export_dataset() writes JSONL (router + reply-tone) used by finetune/."""
import json

import numpy as np

from . import db, rag


def record(reply, action, human_reply=None, note=None, reviewer=None):
    msg = db.q("SELECT content FROM messages WHERE id=?", (reply["message_id"],), one=True)
    db.x("INSERT INTO feedback(reply_id,conversation_id,user_message,agent,ai_reply,human_reply,action,note,reviewer,created_at) "
         "VALUES(?,?,?,?,?,?,?,?,?,strftime('%s','now'))",
         (reply["id"], reply["conversation_id"], msg["content"] if msg else "", reply["agent"], reply["draft"],
          human_reply, action, note, reviewer))


def similar(user_message, k=2, min_sim=0.45):
    rows = db.q("SELECT * FROM feedback WHERE action IN ('edited','rejected') ORDER BY id DESC LIMIT 300")
    if not rows:
        return []
    qv = rag.embed(user_message)
    sims = [(float(rag.embed(r["user_message"]) @ qv), r) for r in rows]
    sims = sorted([s for s in sims if s[0] >= min_sim], key=lambda s: -s[0])[:k]
    return [{"sim": round(s, 3), **r} for s, r in sims]


def guidelines(limit=5):
    rows = db.q("SELECT note FROM feedback WHERE note IS NOT NULL AND note<>'' ORDER BY id DESC LIMIT ?", (limit,))
    return [r["note"] for r in rows]


def prompt_block(user_message):
    parts = []
    g = guidelines()
    if g:
        parts.append("Reviewer guidelines learned from past corrections:\n" + "\n".join(f"- {x}" for x in g))
    for ex in similar(user_message):
        if ex["action"] == "edited":
            parts.append(f"A human corrected a similar reply.\nCustomer: {ex['user_message']}\nBad reply: {ex['ai_reply']}\nCorrected reply: {ex['human_reply']}")
        else:
            parts.append(f"A human REJECTED this reply to a similar message (do not repeat it):\nCustomer: {ex['user_message']}\nRejected: {ex['ai_reply']}")
    return "\n\n".join(parts)


def export_dataset(path):
    """Chat-format JSONL for fine-tuning: edited replies become gold examples."""
    rows = db.q("SELECT * FROM feedback WHERE action='edited' AND human_reply IS NOT NULL")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"messages": [
                {"role": "system", "content": "You are Northwick & Co's customer support assistant. Be warm, concise and accurate."},
                {"role": "user", "content": r["user_message"]},
                {"role": "assistant", "content": r["human_reply"]}]}) + "\n")
    return len(rows)
