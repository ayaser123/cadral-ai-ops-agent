"""Real function tools used by the Orders agent (executed against the database; also exposed as OpenAI tool schemas)."""
import json
import time
from datetime import datetime

from . import config, db

RETURN_WINDOW_DAYS = 30


def _order(order_id):
    o = db.q("SELECT * FROM orders WHERE id=?", (order_id.upper(),), one=True)
    if o:
        o["items"] = json.loads(o["items"])
    return o


def _owns(o, email):
    return o and email and o["customer_email"].lower() == email.lower()


def get_order_status(order_id, customer_email):
    o = _order(order_id)
    if not o:
        return {"ok": False, "error": "order_not_found"}
    if not _owns(o, customer_email):
        return {"ok": False, "error": "order_not_linked_to_this_customer"}
    late = bool(o["eta"] and o["status"] == "shipped" and o["eta"] < datetime.now().strftime("%Y-%m-%d"))
    return {"ok": True, "order_id": o["id"], "status": o["status"], "items": o["items"], "total_gbp": o["total"],
            "carrier": o["carrier"], "tracking": o["tracking"], "eta": o["eta"], "delivered_at": o["delivered_at"],
            "refunded_gbp": o["refunded"], "is_late": late}


def list_customer_orders(customer_email):
    rows = db.q("SELECT id,status,total,placed_at FROM orders WHERE customer_email=? ORDER BY placed_at DESC", (customer_email,))
    return {"ok": True, "orders": rows}


def issue_refund(order_id, reason, customer_email, amount=None, conversation_id=None):
    """Policy-enforced refund. <= AUTO_REFUND_LIMIT is issued immediately; above that it is queued for human approval."""
    o = _order(order_id)
    if not o:
        return {"ok": False, "error": "order_not_found"}
    if not _owns(o, customer_email):
        return {"ok": False, "error": "order_not_linked_to_this_customer"}
    if o["status"] != "delivered":
        return {"ok": False, "error": f"order_not_delivered (status={o['status']}); cancel instead if still processing"}
    days = (datetime.now() - datetime.strptime(o["delivered_at"], "%Y-%m-%d")).days
    if days > RETURN_WINDOW_DAYS and "fault" not in (reason or "").lower() and "damag" not in (reason or "").lower():
        return {"ok": False, "error": f"outside_{RETURN_WINDOW_DAYS}_day_return_window ({days} days since delivery)"}
    remaining = round(o["total"] - o["refunded"], 2)
    if remaining <= 0:
        return {"ok": False, "error": "already_fully_refunded"}
    amount = round(min(float(amount or remaining), remaining), 2)
    if amount > config.AUTO_REFUND_LIMIT:
        rid = db.x("INSERT INTO refunds(order_id,amount,reason,status,conversation_id,created_at) VALUES(?,?,?,?,?,?)",
                   (o["id"], amount, reason, "pending_approval", conversation_id, time.time()))
        return {"ok": True, "refund_id": rid, "status": "pending_approval", "amount_gbp": amount,
                "note": f"Above the GBP {config.AUTO_REFUND_LIMIT:.0f} auto-limit; a human manager must approve."}
    rid = db.x("INSERT INTO refunds(order_id,amount,reason,status,conversation_id,created_at) VALUES(?,?,?,?,?,?)",
               (o["id"], amount, reason, "issued", conversation_id, time.time()))
    db.x("UPDATE orders SET refunded=refunded+? WHERE id=?", (amount, o["id"]))
    return {"ok": True, "refund_id": rid, "status": "issued", "amount_gbp": amount, "arrives": "5-7 working days"}


def update_order(order_id, action, customer_email, new_address=None):
    """action: 'cancel' | 'change_address'. Only allowed while status == processing."""
    o = _order(order_id)
    if not o:
        return {"ok": False, "error": "order_not_found"}
    if not _owns(o, customer_email):
        return {"ok": False, "error": "order_not_linked_to_this_customer"}
    if o["status"] != "processing":
        return {"ok": False, "error": f"cannot_{action}: order is already {o['status']}"}
    if action == "cancel":
        db.x("UPDATE orders SET status='cancelled' WHERE id=?", (o["id"],))
        return {"ok": True, "order_id": o["id"], "new_status": "cancelled"}
    if action == "change_address" and new_address:
        db.x("UPDATE orders SET address=? WHERE id=?", (new_address, o["id"]))
        return {"ok": True, "order_id": o["id"], "new_address": new_address}
    return {"ok": False, "error": "unsupported_action_or_missing_address"}


def approve_refund(refund_id):
    r = db.q("SELECT * FROM refunds WHERE id=?", (refund_id,), one=True)
    if not r or r["status"] != "pending_approval":
        return {"ok": False, "error": "not_pending"}
    db.x("UPDATE refunds SET status='issued' WHERE id=?", (refund_id,))
    db.x("UPDATE orders SET refunded=refunded+? WHERE id=?", (r["amount"], r["order_id"]))
    return {"ok": True}


REGISTRY = {"get_order_status": get_order_status, "list_customer_orders": list_customer_orders,
            "issue_refund": issue_refund, "update_order": update_order}

SCHEMAS = [
    {"type": "function", "function": {"name": "get_order_status", "description": "Look up an order's status, tracking and ETA.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}},
    {"type": "function", "function": {"name": "list_customer_orders", "description": "List the customer's recent orders when they do not give an order id.",
     "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "issue_refund", "description": "Refund a delivered order (policy enforced server-side).",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "amount": {"type": "number", "description": "omit for full refund"},
                    "reason": {"type": "string"}}, "required": ["order_id", "reason"]}}},
    {"type": "function", "function": {"name": "update_order", "description": "Cancel an order or change its delivery address (only while processing).",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "action": {"type": "string", "enum": ["cancel", "change_address"]},
                    "new_address": {"type": "string"}}, "required": ["order_id", "action"]}}},
]


def call(name, args, customer_email, conversation_id=None):
    """Execute a tool; customer identity and conversation id are injected server-side (never trusted from the model)."""
    fn = REGISTRY.get(name)
    if not fn:
        return {"ok": False, "error": f"unknown_tool {name}"}
    args = dict(args)
    args["customer_email"] = customer_email
    if name == "issue_refund":
        args["conversation_id"] = conversation_id
    try:
        return fn(**args)
    except TypeError as e:
        return {"ok": False, "error": f"bad_arguments: {e}"}
