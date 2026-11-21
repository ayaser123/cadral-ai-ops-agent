"""Deterministic seed data (customers + orders relative to today) and default users."""
import hashlib
import hmac
import json
import os
import time
from datetime import datetime, timedelta

from . import config, db


def hash_pw(pw, salt=None):
    salt = salt or os.urandom(8).hex()
    h = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 120_000).hex()
    return f"{salt}${h}"


def check_pw(pw, stored):
    salt, _ = stored.split("$")
    return hmac.compare_digest(hash_pw(pw, salt), stored)


def d(days_ago):
    return (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")


CUSTOMERS = [
    ("alice@example.com", "Alice Johnson", "gold"), ("bob@example.com", "Bob Smith", "standard"),
    ("carol@example.com", "Carol Davies", "standard"), ("dave@example.com", "Dave Patel", "silver"),
    ("emma@example.com", "Emma Wilson", "standard"), ("frank@example.com", "Frank Okafor", "gold"),
    ("grace@example.com", "Grace Taylor", "standard"), ("henry@example.com", "Henry Brown", "standard"),
]


def orders():
    # id, email, status, total, items, placed, delivered, carrier, tracking, eta, address, refunded
    return [
        ("ORD-1001", "alice@example.com", "shipped", 64.90, ["Cotton Throw Blanket", "Scented Candle Cedar x2"], d(3), None, "Royal Mail", "RM123456789GB", d(-2), "12 Park Lane, Leeds LS1 4AB", 0),
        ("ORD-1002", "alice@example.com", "delivered", 39.99, ["Ceramic Vase Tall"], d(9), d(5), "DPD", "DPD998877", d(5), "12 Park Lane, Leeds LS1 4AB", 0),
        ("ORD-1003", "bob@example.com", "processing", 120.00, ["Wool Rug 160x230"], d(0), None, None, None, d(-4), "5 High Street, Bristol BS1 2AA", 0),
        ("ORD-1004", "bob@example.com", "delivered", 89.50, ["Air Fryer 5L"], d(16), d(12), "Evri", "EV55512", d(12), "5 High Street, Bristol BS1 2AA", 0),
        ("ORD-1005", "carol@example.com", "delivered", 55.00, ["Blackout Curtains Pair"], d(50), d(45), "Royal Mail", "RM555000111GB", d(45), "88 Mill Road, Cambridge CB1 3AA", 0),
        ("ORD-1006", "carol@example.com", "shipped", 22.50, ["Picture Frame Trio"], d(2), None, "DPD", "DPD112233", d(-1), "88 Mill Road, Cambridge CB1 3AA", 0),
        ("ORD-1007", "dave@example.com", "delivered", 249.00, ["Velvet Armchair"], d(7), d(3), "DPD", "DPD777666", d(3), "3 Castle Street, Edinburgh EH2 3AA", 0),
        ("ORD-1008", "dave@example.com", "cancelled", 35.00, ["Linen Cushion Cover x2"], d(10), None, None, None, None, "3 Castle Street, Edinburgh EH2 3AA", 0),
        ("ORD-1009", "emma@example.com", "processing", 48.00, ["Egyptian Cotton Towel Set"], d(0), None, None, None, d(-4), "21 Oak Avenue, Cardiff CF10 1AA", 0),
        ("ORD-1010", "emma@example.com", "delivered", 75.00, ["Cast Iron Casserole"], d(25), d(20), "Evri", "EV90010", d(20), "21 Oak Avenue, Cardiff CF10 1AA", 0),
        ("ORD-1011", "frank@example.com", "shipped", 310.00, ["Noise Cancelling Headphones", "Smart LED Desk Lamp x2"], d(12), None, "Royal Mail", "RM404040404GB", d(4), "9 Quay Side, Newcastle NE1 3AA", 0),
        ("ORD-1012", "frank@example.com", "delivered", 18.99, ["Glass Storage Jars Set"], d(14), d(8), "Evri", "EV12012", d(8), "9 Quay Side, Newcastle NE1 3AA", 18.99),
        ("ORD-1013", "grace@example.com", "delivered", 129.00, ["Noise Cancelling Headphones"], d(5), d(2), "DPD", "DPD131313", d(2), "40 Bridge Road, Glasgow G1 1AA", 0),
        ("ORD-1014", "grace@example.com", "processing", 29.99, ["Humidifier Ultrasonic"], d(0), None, None, None, d(-4), "40 Bridge Road, Glasgow G1 1AA", 0),
        ("ORD-1015", "henry@example.com", "delivered", 64.00, ["Knife Block Set"], d(33), d(28), "Royal Mail", "RM151515GB", d(28), "7 Station Way, York YO1 6AA", 0),
        ("ORD-1016", "henry@example.com", "shipped", 15.00, ["Scented Candle Cedar"], d(1), None, "Evri", "EV16016", d(-3), "7 Station Way, York YO1 6AA", 0),
    ]


def seed(reset=False):
    db.init()
    if reset:
        for t in ("orders", "customers", "refunds"):
            db.x(f"DELETE FROM {t}")
    if not db.q("SELECT 1 FROM customers LIMIT 1"):
        for e, n, t in CUSTOMERS:
            db.x("INSERT INTO customers VALUES(?,?,?,?)", (e, n, t, time.time()))
        for o in orders():
            db.x("INSERT INTO orders VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                 (o[0], o[1], o[2], o[3], json.dumps(o[4]), o[5], o[6], o[7], o[8], o[9], o[10], o[11]))
    for email, pw, role in ((config.ADMIN_EMAIL, config.ADMIN_PASSWORD, "admin"), (config.AGENT_EMAIL, config.AGENT_PASSWORD, "agent")):
        if not db.q("SELECT 1 FROM users WHERE email=?", (email,)):
            db.x("INSERT INTO users VALUES(?,?,?)", (email, hash_pw(pw), role))


if __name__ == "__main__":
    seed(reset=True)
    print("seeded")
