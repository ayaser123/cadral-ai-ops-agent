"""Shared memory for all agents: short-term (this conversation) + long-term (customer history across conversations)."""
import re

from . import db

ORDER_RE = re.compile(r"\bORD-?\s?(\d{3,6})\b", re.I)


def find_order_ids(text):
    return [f"ORD-{m}" for m in ORDER_RE.findall(text or "")]


def build_context(conversation_id, email):
    cust = db.q("SELECT * FROM customers WHERE email=?", (email,), one=True) or {"email": email, "name": None, "tier": "unknown"}
    history = db.q("SELECT role,content,agent FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT 10", (conversation_id,))[::-1]
    past = db.q("SELECT c.id,c.created_at,(SELECT content FROM messages m WHERE m.conversation_id=c.id AND role='customer' "
                "ORDER BY m.id LIMIT 1) AS first_msg FROM conversations c WHERE customer_email=? AND c.id<>? ORDER BY c.id DESC LIMIT 5",
                (email, conversation_id))
    esc = db.q("SELECT COUNT(*) n FROM escalations e JOIN conversations c ON c.id=e.conversation_id WHERE c.customer_email=?", (email,), one=True)["n"]
    refunds = db.q("SELECT COUNT(*) n, COALESCE(SUM(amount),0) s FROM refunds r JOIN orders o ON o.id=r.order_id WHERE o.customer_email=? "
                   "AND r.status='issued'", (email,), one=True)
    orders = db.q("SELECT id,status,total FROM orders WHERE customer_email=? ORDER BY placed_at DESC LIMIT 5", (email,))
    # last order id mentioned anywhere in this conversation (pronoun / follow-up resolution: "refund it")
    mentioned = None
    for m in reversed(history):
        ids = find_order_ids(m["content"])
        if ids:
            mentioned = ids[-1]
            break
    return {"customer": cust, "history": history, "past_conversations": past, "prior_escalations": esc,
            "refund_count": refunds["n"], "refund_total": refunds["s"], "recent_orders": orders, "last_order_id": mentioned}


def render(ctx):
    c = ctx["customer"]
    lines = [f"Customer: {c.get('name') or 'unknown'} <{c['email']}> tier={c.get('tier')}",
             f"History: {len(ctx['past_conversations'])} earlier conversations, {ctx['prior_escalations']} escalations, "
             f"{ctx['refund_count']} refunds (GBP {ctx['refund_total']:.2f})"]
    if ctx["recent_orders"]:
        lines.append("Recent orders: " + "; ".join(f"{o['id']} ({o['status']}, GBP {o['total']})" for o in ctx["recent_orders"]))
    if ctx["last_order_id"]:
        lines.append(f"Order last discussed in this chat: {ctx['last_order_id']}")
    return "\n".join(lines)
