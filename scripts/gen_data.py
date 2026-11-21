"""Generate the knowledge base (24 docs: markdown, HTML pages, PDFs, products CSV). Run once: python scripts/gen_data.py"""
import csv
import random
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "data" / "docs"
(D / "pdf").mkdir(parents=True, exist_ok=True)
(D / "web").mkdir(parents=True, exist_ok=True)

MD = {
"returns-policy": """# Returns Policy
Northwick & Co accepts returns within **30 days of delivery**. Items must be unused, in their original packaging with tags attached.
## How to return
1. Log in to your account and choose "Start a return", or reply to your order email.
2. Print the prepaid Royal Mail returns label (UK returns are free).
3. Drop the parcel at any Royal Mail Post Office or postbox for small parcels.
## Exceptions
Items marked **Final Sale**, personalised items, opened underwear and gift cards cannot be returned unless faulty.
## International returns
Customers outside the UK pay return postage. Refunds exclude original shipping costs.""",
"refund-policy": """# Refund Policy
Refunds are issued to the original payment method within **5-7 working days** after we receive and inspect the returned item.
## Refund amounts
- Full item price is refunded for eligible returns.
- Standard delivery charges are refunded only if the item arrived faulty or damaged or we sent the wrong item.
- Partial refunds may apply if an item is returned used or damaged by the customer.
## Missing refunds
If you have not received a refund after 10 working days, contact support with your order number. Banks can take up to 5 days to show the credit.
## Large refunds
Refunds above GBP 100 require manager approval before they are processed.""",
"delivery-info": """# Delivery Information
| Service | Cost | Time |
|---|---|---|
| Standard UK | GBP 3.95 (free over GBP 50) | 3-5 working days |
| Express UK | GBP 6.95 | Next working day if ordered before 2pm |
| EU | GBP 9.95 | 7-10 working days |
| Rest of World | GBP 14.95 | 10-14 working days |
We deliver Monday to Saturday. Orders are dispatched within 1 working day. You receive a tracking link by email once your order ships via Royal Mail, DPD or Evri.
## Delays
If your order has not arrived within 3 working days after the estimated date, contact us and we will open an investigation with the carrier.""",
"order-changes-cancellations": """# Changing or Cancelling an Order
You can cancel free of charge while the order status is **Processing**. Once an order has shipped it cannot be cancelled; please use our returns process after delivery.
## Changing the delivery address
The delivery address can be changed only before dispatch (status Processing). Contact support with your order number and the new address.
## Within one hour
Orders can be edited in your account for 1 hour after being placed.""",
"warranty-guarantee": """# Warranty and Guarantee
All electronics carry a **24 month guarantee**. Homeware, furniture and textiles carry a **12 month guarantee** against manufacturing faults.
Normal wear and tear, accidental damage and misuse are not covered. Your statutory rights under the Consumer Rights Act 2015 are not affected.
To claim, contact support with your order number and photos of the fault.""",
"payment-methods": """# Payment Methods
We accept Visa, Mastercard, American Express, PayPal, Apple Pay, Google Pay and Klarna (pay in 3 interest-free instalments). We do not accept cash, cheques or bank transfers.
Your card is charged when the order is dispatched. All payments are processed securely via Stripe; we never store full card details.""",
"loyalty-rewards": """# Northwick Rewards
Earn **1 point for every GBP 1** spent. **100 points = GBP 5** reward voucher. Points expire **12 months** after they are earned.
Points are not awarded on gift cards or postage. Refunded orders have their points removed.
Members also get early access to sales and a free birthday gift.""",
"gift-cards": """# Gift Cards
Gift cards are valid for **24 months** from purchase, are delivered by email, and can be used on any product. Gift cards cannot be exchanged for cash and are non-refundable. Lost cards can be reissued with proof of purchase.""",
"promo-codes-discounts": """# Promo Codes and Discounts
- **WELCOME10**: 10% off your first order over GBP 30.
- **Student discount**: 15% off with UNiDAYS verification.
- Only one promo code can be used per order. Codes cannot be applied after an order is placed.
- Discount codes are not refundable as cash if an order is cancelled.""",
"damaged-faulty-items": """# Damaged or Faulty Items
Report damaged or faulty goods within **14 days** of delivery with photos of the item and packaging. We will offer a free replacement or a full refund including delivery costs, at your choice. We arrange free collection of faulty items.""",
"privacy-gdpr": """# Privacy and Data Protection (UK GDPR)
We keep order records for 6 years for tax purposes. You may request a copy of your data or ask for deletion at privacy@northwick.example; we respond within **30 days**. We never sell personal data. Marketing emails can be switched off any time using the unsubscribe link.""",
"contact-hours": """# Contact Us
Customer care is open Monday to Friday 9am-5:30pm and Saturday 10am-2pm (UK time). Phone: 0800 123 4567. Email: help@northwick.example. Live chat is available during opening hours; our AI assistant answers 24/7 and hands complex cases to a human colleague. We are an online-only retailer and have no physical shops.""",
}

WEB = {
"faq": ("Frequently Asked Questions", [
 ("Do you ship internationally?", "Yes. We ship to the EU (7-10 working days, GBP 9.95) and the rest of the world (10-14 working days, GBP 14.95)."),
 ("Can I track my parcel?", "Yes, a tracking link is emailed when your order ships. Royal Mail, DPD or Evri tracking can take 24 hours to update."),
 ("Is my payment secure?", "Yes. Payments are processed by Stripe and we never store full card details."),
 ("Can I change my order?", "Orders can be edited for 1 hour in your account, and cancelled or re-addressed while the status is Processing.")]),
"about-us": ("About Northwick & Co", [
 ("Who are we?", "Northwick & Co is a family-run UK homeware and lifestyle retailer founded in 2014 in Leeds."),
 ("Where are you based?", "Our warehouse and head office are in Leeds, West Yorkshire. We are online-only."),
 ("Company numbers", "Registered in England and Wales, company number 09876543.")]),
"size-and-care-guide": ("Product Care Guide", [
 ("Cotton throws and cushions", "Machine wash at 30 degrees, tumble dry low. Do not bleach."),
 ("Ceramic mugs and bowls", "Dishwasher and microwave safe. Avoid sudden temperature changes."),
 ("Wooden items", "Wipe with a damp cloth. Never soak. Re-oil oak items every 6 months."),
 ("Electronics", "Keep away from water and use the supplied charger only.")]),
"sustainability": ("Sustainability", [
 ("Packaging", "All parcels use 100% recyclable cardboard and paper tape. We have been plastic-free in packaging since 2022."),
 ("Carbon", "Standard deliveries are carbon-offset via Gold Standard projects."),
 ("Suppliers", "We audit every supplier annually under our Ethical Trading Code.")]),
"sale-and-clearance": ("Sale and Clearance", [
 ("When are the sales?", "Our main sales run in January and July, plus Black Friday week in November."),
 ("Are sale items returnable?", "Standard sale items follow the 30-day returns policy. Items marked Final Sale cannot be returned unless faulty."),
 ("Price adjustments", "We do not offer price matching or retrospective price adjustments after a sale starts.")]),
}


def html(title, qa):
    body = "".join(f"<h2>{q}</h2><p>{a}</p>" for q, a in qa)
    return f"<!doctype html><html><head><title>{title}</title></head><body><h1>{title}</h1>{body}</body></html>"


def mini_pdf(path, title, paras):
    """Dependency-free single-font PDF writer (text only)."""
    lines = [title, ""]
    for p in paras:
        cur = ""
        for w in p.split():
            if len(cur) + len(w) > 85:
                lines.append(cur)
                cur = w
            else:
                cur = (cur + " " + w).strip()
        lines += [cur, ""]
    pages = [lines[i:i + 46] for i in range(0, len(lines), 46)]
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for pg in pages:
        esc = [l.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for l in pg]
        txt = "BT /F1 11 Tf 50 790 Td 15 TL " + " ".join("(" + l + ") Tj T*" for l in esc) + " ET"
        objs.append(f"<< /Length {len(txt)} >>\nstream\n{txt}\nendstream")
        cid = len(objs)
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents {cid} 0 R /Resources << /Font << /F1 3 0 R >> >> >>")
        kids.append(f"{len(objs)} 0 R")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offs = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode("latin-1", "replace")
    xref = len(out)
    out += f"xref\n0 {len(objs)+1}\n0000000000 65535 f \n".encode()
    for o in offs:
        out += f"{o:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    Path(path).write_bytes(out)


PDF = {
"terms-and-conditions": ("Terms and Conditions of Sale", [
 "1. Contract. A contract is formed when we dispatch your order and take payment. We may refuse any order, for example if a product is mispriced.",
 "2. Prices. All prices include UK VAT at 20%. Delivery charges are shown at checkout.",
 "3. Liability. Nothing limits our liability for death or personal injury caused by negligence, or for fraud.",
 "4. Governing law. These terms are governed by the laws of England and Wales.",
 "5. Disputes. If you are unhappy with our service please contact customer care first. Unresolved disputes may be referred to an approved ADR provider."]),
"extended-warranty-terms": ("Extended Warranty Terms", [
 "Northwick Care is an optional 36 month extended warranty for electronics costing GBP 100 or more, purchasable within 30 days of delivery for 12% of the item price.",
 "Northwick Care covers electrical and mechanical breakdown. It does not cover cosmetic damage, loss or theft.",
 "Claims must be made by emailing claims@northwick.example. We aim to repair or replace within 10 working days."]),
"returns-procedure-guide": ("Internal Returns Procedure", [
 "Agents should verify the order belongs to the customer before discussing any order details. Check the delivered date; returns are accepted up to 30 days after delivery.",
 "Refunds up to GBP 100 may be issued by the automated assistant. Refunds above GBP 100 must be approved by a human manager.",
 "Escalate immediately any message that mentions legal action, solicitors, Trading Standards, chargebacks, injury, fraud or discrimination."]),
}

PRODUCTS = [
 ("Cotton Throw Blanket", "Textiles", 34.99, "Soft woven cotton throw, 130x180cm", "cotton", 12),
 ("Linen Cushion Cover", "Textiles", 14.50, "45x45cm natural linen cover with zip", "linen", 12),
 ("Stoneware Mug Set of 4", "Kitchen", 28.00, "Four 350ml speckled stoneware mugs", "stoneware", 12),
 ("Oak Serving Board", "Kitchen", 24.99, "Solid oak board with leather strap", "oak", 12),
 ("Smart LED Desk Lamp", "Electronics", 59.99, "Dimmable lamp with wireless charging pad", "aluminium", 24),
 ("Bluetooth Speaker Mini", "Electronics", 45.00, "Portable waterproof speaker, 12h battery", "plastic", 24),
 ("Noise Cancelling Headphones", "Electronics", 129.00, "Over-ear headphones, 30h battery", "plastic", 24),
 ("Ceramic Vase Tall", "Decor", 39.00, "Handmade matte ceramic vase 35cm", "ceramic", 12),
 ("Scented Candle Cedar", "Decor", 16.00, "Soy wax candle, 50h burn time", "soy wax", 12),
 ("Wall Mirror Round", "Decor", 79.00, "60cm round mirror with brass frame", "glass", 12),
 ("Bamboo Bath Mat", "Bathroom", 21.99, "Slatted bamboo bath mat", "bamboo", 12),
 ("Egyptian Cotton Towel Set", "Bathroom", 42.00, "Set of 4 600gsm towels", "cotton", 12),
 ("Velvet Armchair", "Furniture", 249.00, "Mid-century velvet armchair, walnut legs", "velvet", 12),
 ("Floating Shelf Pair", "Furniture", 32.00, "Two 60cm oak floating shelves", "oak", 12),
 ("Rattan Storage Basket", "Storage", 27.50, "Large woven rattan basket with handles", "rattan", 12),
 ("Glass Storage Jars Set", "Storage", 19.99, "Set of 3 airtight glass jars", "glass", 12),
 ("Electric Kettle 1.7L", "Electronics", 38.00, "Stainless steel rapid boil kettle", "steel", 24),
 ("Air Fryer 5L", "Electronics", 89.00, "Digital air fryer with 8 presets", "plastic", 24),
 ("Wool Rug 160x230", "Textiles", 149.00, "Hand-tufted wool rug", "wool", 12),
 ("Blackout Curtains Pair", "Textiles", 52.00, "Thermal blackout curtains 140x230cm", "polyester", 12),
 ("Cast Iron Casserole", "Kitchen", 65.00, "24cm enamelled cast iron casserole", "cast iron", 12),
 ("Knife Block Set", "Kitchen", 74.00, "6 piece stainless steel knife set", "steel", 12),
 ("Picture Frame Trio", "Decor", 22.00, "Three oak frames A4", "oak", 12),
 ("Final Sale Tote Bag", "Accessories", 9.99, "FINAL SALE canvas tote bag", "canvas", 0),
 ("Gift Card 50", "Gift Cards", 50.00, "Emailed gift card, valid 24 months", "digital", 0),
 ("Weighted Blanket", "Textiles", 69.00, "6kg cotton weighted blanket", "cotton", 12),
 ("Humidifier Ultrasonic", "Electronics", 33.00, "2.5L quiet humidifier", "plastic", 24),
 ("Bedside Table", "Furniture", 95.00, "Two-drawer bedside table in oak veneer", "oak", 12),
]


def main():
    for k, v in MD.items():
        (D / f"{k}.md").write_text(v, encoding="utf-8")
    for k, (t, qa) in WEB.items():
        (D / "web" / f"{k}.html").write_text(html(t, qa), encoding="utf-8")
    for k, (t, ps) in PDF.items():
        mini_pdf(D / "pdf" / f"{k}.pdf", t, ps)
    random.seed(7)
    with open(D / "products.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sku", "name", "category", "price_gbp", "stock", "description", "material", "guarantee_months"])
        for i, (n, c, p, d, m, g) in enumerate(PRODUCTS, 1):
            w.writerow([f"NW-{1000+i}", n, c, p, random.choice([0, 3, 12, 40, 85]), d, m, g])
    print("docs:", len(MD) + len(WEB) + len(PDF) + 1)


if __name__ == "__main__":
    main()
