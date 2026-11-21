"""Outbound: Slack alerts, email (SMTP) and Telegram, all through a persistent outbox with retry + backoff,
so nothing is lost if an API fails. Without credentials, messages are marked 'logged' (visible in the dashboard/outbox)."""
import json
import smtplib
import time
from email.message import EmailMessage

import httpx

from . import config, db

MAX_ATTEMPTS = 6


def slack(text, level="info"):
    """Fire-and-forget Slack alert; failures are stored in outbox for retry."""
    icon = {"error": ":rotating_light:", "escalation": ":warning:", "info": ":information_source:"}.get(level, "")
    queue("slack", "", f"{level}", f"{icon} {text}")


def queue(channel, to_addr, subject, body):
    oid = db.x("INSERT INTO outbox(channel,to_addr,subject,body,created_at) VALUES(?,?,?,?,?)",
               (channel, to_addr, subject, body, time.time()))
    try:
        flush_one(oid)
    except Exception:
        pass
    return oid


def _deliver(row):
    ch = row["channel"]
    if ch == "slack":
        if not config.SLACK_WEBHOOK_URL:
            return "logged"
        r = httpx.post(config.SLACK_WEBHOOK_URL, json={"text": row["body"]}, timeout=10)
        r.raise_for_status()
        return "sent"
    if ch == "email":
        if not config.SMTP_HOST:
            return "logged"
        m = EmailMessage()
        m["From"], m["To"], m["Subject"] = config.SMTP_FROM, row["to_addr"], row["subject"]
        m.set_content(row["body"])
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=15) as s:
            s.starttls()
            if config.SMTP_USER:
                s.login(config.SMTP_USER, config.SMTP_PASSWORD)
            s.send_message(m)
        return "sent"
    if ch == "telegram":
        if not config.TELEGRAM_BOT_TOKEN:
            return "logged"
        r = httpx.post(f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage",
                       json={"chat_id": row["to_addr"], "text": row["body"]}, timeout=10)
        r.raise_for_status()
        return "sent"
    return "logged"  # web chat replies are delivered via the live event stream


def flush_one(oid):
    row = db.q("SELECT * FROM outbox WHERE id=?", (oid,), one=True)
    if not row or row["status"] in ("sent", "logged", "dead"):
        return
    try:
        st = _deliver(row)
        db.x("UPDATE outbox SET status=?, attempts=attempts+1, last_error=NULL WHERE id=?", (st, oid))
    except Exception as e:
        att = row["attempts"] + 1
        status = "dead" if att >= MAX_ATTEMPTS else "pending"
        db.x("UPDATE outbox SET status=?, attempts=?, last_error=?, next_attempt=? WHERE id=?",
             (status, att, str(e)[:300], time.time() + min(900, 5 * 2 ** att), oid))


def flush_due():
    for r in db.q("SELECT id FROM outbox WHERE status='pending' AND next_attempt<=?", (time.time(),)):
        flush_one(r["id"])
