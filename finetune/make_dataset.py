"""Builds the router-classification dataset (support | orders | escalation) in chat-JSONL format.

* train / val : template family A  (synthetic, diverse slot-filling)
* test        : template family B  (different phrasing, typos, casing, mixed intents) -> measures generalisation, not memorisation
Run:  python finetune/make_dataset.py        ->  finetune/data/{train,val,test}.jsonl"""
import json
import random
from pathlib import Path

OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)
random.seed(42)
SYSTEM = ("You route customer messages for an online retailer. Reply with exactly one word: "
          "support (general questions/policies/products), orders (about a specific order: status, refund, cancel, address) "
          "or escalation (angry, abusive, legal/safety threat, wants a human).")

ORD = lambda: f"ORD-{random.randint(1000, 1999)}"
PROD = ["kettle", "armchair", "rug", "headphones", "lamp", "candle", "towel set", "air fryer", "mug set", "blanket", "mirror", "casserole dish"]

A = {  # ------------- family A (train/val)
    "support": [
        "What is your returns policy?", "How long does delivery take?", "Do you deliver to {c}?", "How much is express shipping?",
        "Can I pay with {p}?", "Do you sell a {x}?", "How much is the {x}?", "What are your opening hours?", "How do I wash the {x}?",
        "Is the {x} covered by a guarantee?", "How do reward points work?", "Do you have a student discount?", "Are gift cards refundable?",
        "What is the warranty on electronics?", "Do you offer free delivery?", "Is the {x} in stock?", "How do I delete my data?",
        "Where is your head office?", "Do you have a physical shop?", "Can I use two promo codes?", "What colours does the {x} come in?",
        "How can I contact customer care?", "What is your policy on damaged items?", "Is your packaging recyclable?",
        "Hello, I have a quick question about your products", "Do you price match?", "When is the next sale?",
        "What's the difference between standard and express delivery?", "Is the {x} dishwasher safe?", "Tell me about the {x}",
    ],
    "orders": [
        "Where is my order {o}?", "Can you check the status of {o}?", "I'd like a refund for {o}", "Please cancel {o}",
        "My {x} from {o} arrived damaged, I want my money back", "Change the delivery address on {o} to 4 High Street, Leeds LS1 4AB",
        "Track my parcel {o}", "Has {o} been dispatched yet?", "I want to return the {x} I ordered ({o})", "Where's my parcel? Order {o}",
        "Refund {o} please, the {x} is faulty", "Can I change the address for order {o}?", "I haven't received {o} yet",
        "What's happening with my order?", "I need to cancel my last order", "Can I get a refund on my recent order?",
        "{o} - when will it arrive?", "Update the address on {o}", "The {x} in {o} is broken, can I return it?", "Is {o} on its way?",
        "I'd like to send back my order {o}", "Delivery of {o} is late, where is it?", "Please look up order {o} for me",
    ],
    "escalation": [
        "This is absolutely unacceptable, I want to speak to a manager", "I'm going to call my solicitor if this isn't fixed",
        "Your service is disgusting and I'm furious", "I will report you to Trading Standards", "THIS IS A SCAM!!! GIVE ME MY MONEY",
        "I'm doing a chargeback with my bank", "The {x} caught fire and I was injured", "You are useless, I want a human NOW",
        "I've emailed five times and nobody has replied, this is ridiculous", "I'm taking legal action against your company",
        "Worst company ever, I'll never shop here again and I'll tell everyone", "I demand to speak to someone senior immediately",
        "This is fraud, I'm contacting the police", "I want to make a formal complaint about how I've been treated",
        "I'm so fed up with your incompetent staff", "My child had an allergic reaction to the {x}, this is dangerous",
        "Terrible, terrible service. I'm leaving a one-star review everywhere", "Get me a real person, this bot is pathetic",
        "{o} STILL hasn't arrived and I am livid", "Outrageous! I'm contacting the ombudsman",
    ],
}
B = {  # ------------- family B (held-out test): different wording, typos, casual, lowercase
    "support": [
        "hi, do u guys ship to {c}", "whats the cost of next day delivery", "any chance the {x} comes in other sizes?", "is klarna accepted",
        "how long have i got to send stuff back", "do you do gift vouchers", "need to know if the {x} is safe in the microwave",
        "are there any discount codes at the moment?", "when do my loyalty points run out", "what hours is the phone line open",
        "can you tell me about the guarantee on the {x}", "im thinking of buying the {x} - what is it made of?", "do you have shops in london",
        "is there a student offer", "how do i ask you to remove my personal info", "what payment options are there",
        "does the {x} need assembly?", "any plans for a black friday sale?", "what's your sustainability policy", "can i buy a gift card for my mum",
    ],
    "orders": [
        "hey my parcel {o} still hasnt turned up", "cld you cancel the order i just placed ({o})", "money back pls for {o}, item not as described",
        "i ordered a {x} last week and it hasnt shipped - order number {o}", "need to move my delivery to my work address, order {o}",
        "{o}?? any update", "the {x} i got in {o} is cracked. refund?", "can u tell me if my order has left the warehouse", "wrong address on my order, can you fix it",
        "returning order {o}, how do i get my refund", "what happened to my package", "i want to cancel what i bought this morning",
        "tracking number for {o} please", "my {x} was delivered broken (order {o})", "update me on {o}",
    ],
    "escalation": [
        "honestly sick of this. get me your manager", "my lawyer will be in touch about this", "you've ruined my {x} and nearly burned my house down",
        "this company is a joke, i want compensation and a person to talk to", "i'm reporting this to the ombudsman and trading standards",
        "NOBODY ANSWERS MY EMAILS WTF", "dispute with my bank started, you people are thieves", "absolutely appalling. escalate this now",
        "i am beyond angry. {o} is a month late and no one cares", "this is the last time i order from you idiots", "i've been scammed, calling the police",
        "speak to a human. now.", "your colleague was rude and I want a formal complaint logged", "my baby got hurt by the {x}",
    ],
}
COUNTRIES = ["Germany", "France", "Ireland", "the USA", "Spain", "Australia"]
PAY = ["PayPal", "Klarna", "Apple Pay", "Amex"]


def fill(t):
    return t.format(o=ORD(), x=random.choice(PROD), c=random.choice(COUNTRIES), p=random.choice(PAY))


def build(fam, n_per_class):
    rows = []
    for label, tmpl in fam.items():
        for _ in range(n_per_class):
            rows.append((fill(random.choice(tmpl)), label))
    random.shuffle(rows)
    return rows


def write(name, rows):
    with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for text, label in rows:
            f.write(json.dumps({"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text},
                                             {"role": "assistant", "content": label}], "label": label}) + "\n")


if __name__ == "__main__":
    train, val = build(A, 400), build(A, 60)
    seen = {t for t, _ in train}
    val = [r for r in val if r[0] not in seen]
    test = build(B, 60)
    write("train", train)
    write("val", val)
    write("test", test)
    print({"train": len(train), "val": len(val), "test": len(test)})
