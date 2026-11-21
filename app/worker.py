"""Background workers (daemon threads): retry queue, outbox flusher, daily report scheduler, email (IMAP) + Telegram intake."""
import email as emaillib
import imaplib
import threading
import time
from datetime import datetime, timezone
from email.utils import parseaddr

import httpx

from . import config, db, notify, pipeline, reports


def _loop(fn, every, name):
    def run():
        while True:
            try:
                fn()
            except Exception as e:
                try:
                    notify.slack(f"Worker '{name}' error: {e}", level="error")
                except Exception:
                    pass
            time.sleep(every)
    threading.Thread(target=run, name=name, daemon=True).start()


def daily_report_tick():
    now = datetime.now(timezone.utc)
    if now.hour >= config.REPORT_HOUR_UTC and db.kv_get("last_report_date") != f"{now:%Y-%m-%d}":
        reports.send_daily_report()


def imap_poll():
    if not config.IMAP_HOST:
        return
    with imaplib.IMAP4_SSL(config.IMAP_HOST) as m:
        m.login(config.IMAP_USER, config.IMAP_PASSWORD)
        m.select("INBOX")
        _, data = m.search(None, "UNSEEN")
        for num in data[0].split():
            _, msg = m.fetch(num, "(RFC822)")
            em = emaillib.message_from_bytes(msg[0][1])
            name, addr = parseaddr(em["From"])
            body = ""
            for part in em.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode(errors="ignore")
                    break
            text = f"{em['Subject'] or ''}\n{body}".strip()[:4000]
            _, iid = pipeline.ingest("email", addr, text, name=name or None)
            pipeline.process_inbox(iid)


def telegram_poll():
    if not config.TELEGRAM_BOT_TOKEN:
        return
    off = db.kv_get("tg_offset", 0)
    r = httpx.get(f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/getUpdates",
                  params={"offset": off, "timeout": 20}, timeout=30).json()
    for u in r.get("result", []):
        db.kv_set("tg_offset", u["update_id"] + 1)
        m = u.get("message")
        if not m or "text" not in m:
            continue
        chat = str(m["chat"]["id"])
        # Telegram users are matched to customers via /email you@example.com once, else a pseudo identity
        txt = m["text"]
        if txt.startswith("/email "):
            db.kv_set(f"tg_email_{chat}", txt.split(" ", 1)[1].strip().lower())
            notify.queue("telegram", chat, "", "Thanks - your email is linked. How can I help?")
            continue
        email = db.kv_get(f"tg_email_{chat}", f"tg{chat}@telegram.local")
        _, iid = pipeline.ingest("telegram", email, txt, name=m["from"].get("first_name"), chat_ref=chat)
        pipeline.process_inbox(iid)


def start():
    _loop(pipeline.retry_due, 10, "retry-inbox")
    _loop(notify.flush_due, 15, "flush-outbox")
    _loop(daily_report_tick, 60, "daily-report")
    if config.IMAP_HOST:
        _loop(imap_poll, 30, "imap")
    if config.TELEGRAM_BOT_TOKEN:
        _loop(telegram_poll, 1, "telegram")
