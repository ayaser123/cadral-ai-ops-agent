"""Tiny in-process pub/sub used for Server-Sent Events (live dashboard + streaming replies)."""
import queue
import threading

_subs = set()
_lock = threading.Lock()


def subscribe():
    q = queue.Queue(maxsize=1000)
    with _lock:
        _subs.add(q)
    return q


def unsubscribe(q):
    with _lock:
        _subs.discard(q)


def publish(kind, **data):
    ev = {"type": kind, **data}
    with _lock:
        subs = list(_subs)
    for q in subs:
        try:
            q.put_nowait(ev)
        except queue.Full:
            pass
