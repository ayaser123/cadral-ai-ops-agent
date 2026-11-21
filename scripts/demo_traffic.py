"""Populate a running instance with realistic traffic for demos: python scripts/demo_traffic.py [base_url]"""
import sys
import time

import httpx

B = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000") + "/api"
MSGS = [
    ("alice@example.com", "Where is my order ORD-1001?"), ("bob@example.com", "How long do I have to return something?"),
    ("dave@example.com", "I'd like a refund for ORD-1007, the armchair is the wrong colour"),
    ("carol@example.com", "THIS IS UNACCEPTABLE. I'm calling my solicitor!!!"), ("emma@example.com", "Cancel ORD-1009 please"),
    ("grace@example.com", "Do you accept Klarna?"), ("henry@example.com", "What is the capital of France?"),
    ("frank@example.com", "Has ORD-1011 arrived yet? It was due days ago"), ("alice@example.com", "How much is express delivery?"),
]
for i, (e, t) in enumerate(MSGS):
    httpx.post(B + "/chat", json={"email": e, "text": t, "session": f"demo{i}"})
    time.sleep(1.5)
print("sent", len(MSGS), "messages")
