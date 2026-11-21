"""FastAPI app: channels, dashboard API, SSE live stream, auth (admin / agent roles)."""
import asyncio
import base64
import hashlib
import hmac
import json
import queue
import threading
import time
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import config, db, events, feedback, llm, notify, pipeline, rag, reports, seed, tools, worker

app = FastAPI(title="Northwick Autonomous AI Operations Agent")
STATIC = Path(__file__).parent / "static"


@app.on_event("startup")
def startup():
    seed.seed()
    rag.get_index()
    worker.start()
    # recover anything that was mid-flight when the process died
    db.x("UPDATE inbox SET status='retry' WHERE status='processing'")


# ---------------------------------------------------------------- auth (HMAC-signed bearer tokens, PBKDF2 passwords)
def make_token(email, role, ttl=12 * 3600):
    payload = base64.urlsafe_b64encode(json.dumps({"e": email, "r": role, "x": int(time.time() + ttl)}).encode()).decode()
    sig = hmac.new(config.SECRET_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def parse_token(tok):
    try:
        payload, sig = tok.rsplit(".", 1)
        if not hmac.compare_digest(sig, hmac.new(config.SECRET_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()):
            return None
        d = json.loads(base64.urlsafe_b64decode(payload))
        return d if d["x"] > time.time() else None
    except Exception:
        return None


def user(authorization: str = Header(None), token: str = Query(None)):
    tok = (authorization or "").replace("Bearer ", "") or token
    d = parse_token(tok or "")
    if not d:
        raise HTTPException(401, "Not authenticated")
    return {"email": d["e"], "role": d["r"]}


def admin(u=Depends(user)):
    if u["role"] != "admin":
        raise HTTPException(403, "Admin only")
    return u


class Login(BaseModel):
    email: str
    password: str


@app.post("/api/login")
def login(b: Login):
    u = db.q("SELECT * FROM users WHERE email=?", (b.email.lower(),), one=True)
    if not u or not seed.check_pw(b.password, u["pw_hash"]):
        raise HTTPException(401, "Invalid credentials")
    return {"token": make_token(u["email"], u["role"]), "email": u["email"], "role": u["role"]}


# ---------------------------------------------------------------- channels
class ChatIn(BaseModel):
    email: str
    text: str
    session: str
    name: str | None = None


def _bg(iid):
    threading.Thread(target=pipeline.process_inbox, args=(iid,), daemon=True).start()


@app.post("/api/chat")
def chat(b: ChatIn):
    """Channel 1: web chat widget."""
    if not b.text.strip() or len(b.text) > 4000:
        raise HTTPException(400, "Message must be 1-4000 characters")
    conv, iid = pipeline.ingest("web", b.email, b.text.strip(), b.name, chat_ref=b.session)
    _bg(iid)
    return {"conversation_id": conv, "inbox_id": iid}


@app.get("/api/chat/{conv}/messages")
def chat_messages(conv: int, session: str):
    c = db.q("SELECT * FROM conversations WHERE id=? AND chat_ref=?", (conv, session), one=True)
    if not c:
        raise HTTPException(404)
    msgs = db.q("SELECT role,content,agent,created_at FROM messages WHERE conversation_id=? ORDER BY id", (conv,))
    pend = db.q("SELECT COUNT(*) n FROM replies WHERE conversation_id=? AND status='pending'", (conv,), one=True)["n"]
    return {"messages": msgs, "awaiting_human": pend > 0, "status": c["status"]}


class EmailIn(BaseModel):
    from_addr: str
    subject: str = ""
    body: str


@app.post("/api/channels/email")
def email_in(b: EmailIn, x_webhook_secret: str = Header(None)):
    """Channel 2: inbound email (webhook for Mailgun/SendGrid-style forwarding; IMAP polling is also supported via worker)."""
    if not hmac.compare_digest(x_webhook_secret or "", config.SECRET_KEY):
        raise HTTPException(401, "bad webhook secret")
    conv, iid = pipeline.ingest("email", b.from_addr, f"{b.subject}\n{b.body}".strip())
    _bg(iid)
    return {"conversation_id": conv, "inbox_id": iid}


# ---------------------------------------------------------------- dashboard API
@app.get("/api/me")
def me(u=Depends(user)):
    return u


@app.get("/api/conversations")
def conversations(status: str | None = None, u=Depends(user)):
    sql = ("SELECT c.*, (SELECT content FROM messages m WHERE m.conversation_id=c.id ORDER BY id DESC LIMIT 1) AS last_msg,"
           "(SELECT COUNT(*) FROM replies r WHERE r.conversation_id=c.id AND r.status='pending') AS pending FROM conversations c ")
    rows = db.q(sql + ("WHERE c.status=? " if status else "") + "ORDER BY c.updated_at DESC LIMIT 100", (status,) if status else ())
    return rows


@app.get("/api/conversations/{cid}")
def conversation(cid: int, u=Depends(user)):
    c = db.q("SELECT * FROM conversations WHERE id=?", (cid,), one=True)
    if not c:
        raise HTTPException(404)
    traces = db.q("SELECT * FROM traces WHERE conversation_id=? ORDER BY id", (cid,))
    for t in traces:
        t["detail"] = json.loads(t["detail"])
    replies = db.q("SELECT * FROM replies WHERE conversation_id=? ORDER BY id", (cid,))
    for r in replies:
        r["citations"] = json.loads(r["citations"] or "[]")
    return {"conversation": c, "messages": db.q("SELECT * FROM messages WHERE conversation_id=? ORDER BY id", (cid,)),
            "replies": replies, "traces": traces,
            "escalation": db.q("SELECT * FROM escalations WHERE conversation_id=? ORDER BY id DESC", (cid,), one=True),
            "memory": __import__("app.memory", fromlist=["x"]).build_context(cid, c["customer_email"])}


class Review(BaseModel):
    action: str
    text: str | None = None
    note: str | None = None


@app.post("/api/replies/{rid}/review")
def review(rid: int, b: Review, u=Depends(user)):
    try:
        return pipeline.review(rid, b.action, u["email"], b.text, b.note)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/analytics")
def analytics(days: int = 7, u=Depends(user)):
    return reports.analytics(days)


@app.get("/api/report/preview")
def report_preview(u=Depends(admin)):
    return {"report": reports.build_report()}


@app.post("/api/report/send")
def report_send(u=Depends(admin)):
    return {"report": reports.send_daily_report()}


@app.get("/api/refunds")
def refunds(u=Depends(user)):
    return db.q("SELECT * FROM refunds ORDER BY id DESC LIMIT 100")


@app.post("/api/refunds/{rid}/approve")
def refund_approve(rid: int, u=Depends(admin)):
    r = tools.approve_refund(rid)
    if not r["ok"]:
        raise HTTPException(400, r["error"])
    return r


@app.get("/api/feedback")
def feedback_list(u=Depends(admin)):
    return db.q("SELECT * FROM feedback ORDER BY id DESC LIMIT 200")


@app.post("/api/feedback/export")
def feedback_export(u=Depends(admin)):
    p = config.ROOT / "data" / "feedback_dataset.jsonl"
    return {"path": str(p), "examples": feedback.export_dataset(p)}


@app.get("/api/system")
def system(u=Depends(admin)):
    return {"llm": "groq:" + config.GROQ_MODEL_MAIN if llm.available() else "offline-heuristic",
            "fine_tuned_router": config.FT_MODEL if config.FT_BASE_URL else None, "kb_docs": rag.get_index().n_docs,
            "kb_chunks": len(rag.get_index().chunks), "slack": bool(config.SLACK_WEBHOOK_URL), "smtp": bool(config.SMTP_HOST),
            "telegram": bool(config.TELEGRAM_BOT_TOKEN), "imap": bool(config.IMAP_HOST),
            "inbox": db.q("SELECT status, COUNT(*) n FROM inbox GROUP BY status"),
            "outbox": db.q("SELECT status, COUNT(*) n FROM outbox GROUP BY status")}


@app.get("/api/outbox")
def outbox(u=Depends(admin)):
    return db.q("SELECT * FROM outbox ORDER BY id DESC LIMIT 100")


@app.get("/api/escalations")
def escalations(u=Depends(user)):
    return db.q("SELECT * FROM escalations ORDER BY id DESC LIMIT 100")


@app.get("/api/events")
async def sse(request: Request, u=Depends(user)):
    q = events.subscribe()

    async def gen():
        try:
            yield "retry: 2000\n\n"
            idle = 0
            while not await request.is_disconnected():
                try:
                    ev = q.get_nowait()
                    idle = 0
                    yield f"data: {json.dumps(ev, default=str)}\n\n"
                except queue.Empty:
                    idle += 1
                    if idle % 100 == 0:
                        yield ": ping\n\n"
                    await asyncio.sleep(0.05)
        finally:
            events.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/health")
def health():
    return {"ok": True, "llm": llm.available(), "kb_docs": rag.get_index().n_docs}


@app.exception_handler(Exception)
async def err(request, exc):
    try:
        notify.slack(f"Unhandled API error on {request.url.path}: {exc}", level="error")
    except Exception:
        pass
    return JSONResponse({"error": "internal error"}, status_code=500)


@app.get("/")
def dashboard():
    return FileResponse(STATIC / "index.html")


@app.get("/chat")
def widget():
    return FileResponse(STATIC / "chat.html")
