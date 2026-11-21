"""SQLite layer. One connection per call (thread-safe, WAL)."""
import json
import sqlite3
import time
from contextlib import contextmanager

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers(email TEXT PRIMARY KEY, name TEXT, tier TEXT DEFAULT 'standard', created_at REAL);
CREATE TABLE IF NOT EXISTS orders(
  id TEXT PRIMARY KEY, customer_email TEXT, status TEXT, total REAL, items TEXT,
  placed_at TEXT, delivered_at TEXT, carrier TEXT, tracking TEXT, eta TEXT, address TEXT, refunded REAL DEFAULT 0);
CREATE TABLE IF NOT EXISTS refunds(id INTEGER PRIMARY KEY AUTOINCREMENT, order_id TEXT, amount REAL, reason TEXT,
  status TEXT, conversation_id INTEGER, created_at REAL);
CREATE TABLE IF NOT EXISTS conversations(id INTEGER PRIMARY KEY AUTOINCREMENT, channel TEXT, customer_email TEXT,
  customer_name TEXT, chat_ref TEXT, status TEXT DEFAULT 'open', created_at REAL, updated_at REAL);
CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id INTEGER, role TEXT,
  content TEXT, agent TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS replies(id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id INTEGER, message_id INTEGER,
  agent TEXT, draft TEXT, final TEXT, citations TEXT, status TEXT, confidence REAL, auto INTEGER DEFAULT 0,
  reviewer TEXT, created_at REAL, resolved_at REAL, latency_ms REAL, cost_usd REAL, tokens INTEGER);
CREATE TABLE IF NOT EXISTS traces(id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id INTEGER, reply_id INTEGER,
  step TEXT, agent TEXT, detail TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS llm_calls(id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id INTEGER, purpose TEXT,
  model TEXT, prompt_tokens INTEGER, completion_tokens INTEGER, latency_ms REAL, cost_usd REAL, ok INTEGER, created_at REAL);
CREATE TABLE IF NOT EXISTS feedback(id INTEGER PRIMARY KEY AUTOINCREMENT, reply_id INTEGER, conversation_id INTEGER,
  user_message TEXT, agent TEXT, ai_reply TEXT, human_reply TEXT, action TEXT, note TEXT, reviewer TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS escalations(id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id INTEGER, reason TEXT,
  summary TEXT, severity TEXT, status TEXT DEFAULT 'open', created_at REAL);
CREATE TABLE IF NOT EXISTS inbox(id INTEGER PRIMARY KEY AUTOINCREMENT, channel TEXT, payload TEXT, status TEXT DEFAULT 'pending',
  attempts INTEGER DEFAULT 0, next_attempt REAL DEFAULT 0, last_error TEXT, conversation_id INTEGER, created_at REAL);
CREATE TABLE IF NOT EXISTS outbox(id INTEGER PRIMARY KEY AUTOINCREMENT, channel TEXT, to_addr TEXT, subject TEXT, body TEXT,
  status TEXT DEFAULT 'pending', attempts INTEGER DEFAULT 0, next_attempt REAL DEFAULT 0, last_error TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS users(email TEXT PRIMARY KEY, pw_hash TEXT, role TEXT);
CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT);
CREATE INDEX IF NOT EXISTS ix_msg_conv ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS ix_trace_conv ON traces(conversation_id);
CREATE INDEX IF NOT EXISTS ix_reply_conv ON replies(conversation_id);
"""


def connect():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(config.DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    return con


@contextmanager
def tx():
    con = connect()
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init():
    with tx() as c:
        c.executescript(SCHEMA)


def q(sql, args=(), one=False):
    with tx() as c:
        rows = [dict(r) for r in c.execute(sql, args).fetchall()]
    return (rows[0] if rows else None) if one else rows


def x(sql, args=()):
    """Execute a write; returns lastrowid."""
    with tx() as c:
        return c.execute(sql, args).lastrowid


def now():
    return time.time()


def kv_get(k, default=None):
    r = q("SELECT v FROM kv WHERE k=?", (k,), one=True)
    return json.loads(r["v"]) if r else default


def kv_set(k, v):
    x("INSERT INTO kv(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, json.dumps(v)))
