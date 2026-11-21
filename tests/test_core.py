import os
import sys
import tempfile
from pathlib import Path

os.environ["DB_PATH"] = str(Path(tempfile.mkdtemp()) / "t.db")
os.environ["GROQ_API_KEY"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from app import agents, db, feedback, pipeline, rag, seed, tools  # noqa: E402


@pytest.fixture(autouse=True)
def fresh():
    seed.seed(reset=True)
    for t in ("conversations", "messages", "replies", "traces", "escalations", "feedback", "inbox", "outbox"):
        db.x(f"DELETE FROM {t}")


def ask(email, text):
    conv, iid = pipeline.ingest("web", email, text, chat_ref=email)
    pipeline.process_inbox(iid)
    return db.q("SELECT * FROM replies WHERE conversation_id=? ORDER BY id DESC LIMIT 1", (conv,), one=True)


def test_kb_has_20_plus_documents():
    assert rag.get_index().n_docs >= 20


def test_hybrid_search_cites_and_idk():
    hits, ok = rag.retrieve("How long do I have to return something?")
    assert ok and hits[0]["chunk"].doc == "returns-policy"
    _, ok = rag.retrieve("What is the capital of France?")
    assert not ok


def test_refund_policy_enforced_server_side():
    assert tools.issue_refund("ORD-1004", "faulty", "bob@example.com")["status"] == "issued"
    assert tools.issue_refund("ORD-1004", "faulty", "bob@example.com")["error"] == "already_fully_refunded"
    assert tools.issue_refund("ORD-1001", "x", "bob@example.com")["error"] == "order_not_linked_to_this_customer"
    assert tools.issue_refund("ORD-1007", "x", "dave@example.com")["status"] == "pending_approval"
    assert "window" in tools.issue_refund("ORD-1005", "changed mind", "carol@example.com")["error"]


def test_orders_agent_uses_real_tool_and_changes_db():
    r = ask("emma@example.com", "Cancel ORD-1009")
    assert r["agent"] == "orders"
    assert db.q("SELECT status FROM orders WHERE id='ORD-1009'", one=True)["status"] == "cancelled"
    assert any(t["step"] == "tool_call" for t in db.q("SELECT step FROM traces"))


def test_escalation_creates_case_and_slack_alert():
    r = ask("carol@example.com", "I'm calling my solicitor, this is a scam")
    assert r["agent"] == "escalation" and r["status"] == "pending"
    assert db.q("SELECT severity FROM escalations", one=True)["severity"] == "critical"
    assert db.q("SELECT 1 FROM outbox WHERE channel='slack'", one=True)


def test_feedback_loop_reuses_human_correction():
    r = ask("alice@example.com", "Do you price match with other retailers?")
    pipeline.review(r["id"], "edit", "agent@cadral.test", "We don't price match, but our sales run in Jan, Jul and Black Friday.", "mention sale dates")
    r2 = ask("bob@example.com", "Do you price match with other retailers?")
    assert "Black Friday" in r2["draft"]
    assert feedback.guidelines() == ["mention sale dates"]


def test_nothing_lost_when_pipeline_fails(monkeypatch):
    conv, iid = pipeline.ingest("web", "alice@example.com", "Where is my order ORD-1001?", chat_ref="x")
    monkeypatch.setattr(agents, "orders_agent", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("API down")))
    pipeline.process_inbox(iid)
    row = db.q("SELECT * FROM inbox WHERE id=?", (iid,), one=True)
    assert row["status"] == "retry" and row["attempts"] == 1 and "API down" in row["last_error"]
    monkeypatch.undo()
    db.x("UPDATE inbox SET next_attempt=0")
    pipeline.retry_due()
    assert db.q("SELECT status FROM inbox WHERE id=?", (iid,), one=True)["status"] == "done"
