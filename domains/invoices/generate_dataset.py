"""Generates dataset.jsonl and split.json for the invoices domain.

Produces 35 synthetic invoices as plain text, each paired with a hand-designed
ground-truth JSON object (vendor, invoice_number, date, line_items, subtotal,
tax, total, currency). 30 records are built procedurally (varied vendors,
item catalogs, layouts, currencies) from the same values used to render their
text, so ground truth is correct by construction. 5 records are hand-authored
"hard cases" that a naive parser is likely to get wrong:

    inv_005  multi-page invoice (items split across two "pages")
    inv_012  missing tax line entirely
    inv_019  two currencies mentioned in one document
    inv_026  printed line items don't sum to the printed subtotal/total
    inv_033  credit memo with negative amounts

Not run automatically by tests; documents/reproduces how dataset.jsonl and
split.json were produced. Re-running overwrites both files with the same
content, since generation is seeded (SEED = 42).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

SEED = 42
DOMAIN_DIR = Path(__file__).parent

VENDORS = [
    "Acme Supplies Inc.", "Bluewave Consulting LLC", "Northwind Traders", "Contoso Hardware Co.",
    "Globex Office Solutions", "Initech Software Systems", "Umbrella Logistics Ltd.", "Stark Industrial Parts",
    "Wayne Business Services", "Wonka Catering Co.", "Hooli Cloud Systems", "Pied Piper Data Services",
    "Aperture Lab Equipment", "Cyberdyne Robotics Supply", "Soylent Foods Wholesale", "Massive Dynamic R&D",
    "Gringotts Financial Print", "Oscorp Chemical Supply", "Vandelay Import Export", "Prestige Worldwide Media",
    "Dunder Mifflin Paper Co.", "Sterling Cooper Ad Services", "Monarch Solutions Group", "Blue Sun Freight",
    "Tyrell Biotech Supply", "Nakatomi Construction Co.", "Weyland Yutani Mining Corp.", "Genco Pura Olive Oil",
    "Abstergo Consulting", "Zorg Industries", "Rekall Travel Services", "Buy N Large Retail Supply",
    "Frobozz Magic Software", "Duff Beverage Distributors", "Krusty Krab Food Supply",
]

ITEM_CATALOG = [
    ("Widget A", 5, 25), ("Widget B", 10, 40), ("Office Chair", 60, 150), ("Desk Lamp", 15, 45),
    ("Consulting Hours", 100, 250), ("Travel Expenses", 50, 400), ("Software License", 200, 900),
    ("Cloud Storage (per TB)", 20, 80), ("Maintenance Fee", 30, 120), ("Printer Toner", 25, 90),
    ("Shipping Pallet", 15, 60), ("Steel Bracket", 3, 18), ("Catering Tray", 40, 120),
    ("Training Session", 150, 500), ("Replacement Part", 8, 55), ("Custom Fabrication", 100, 600),
    ("Annual Subscription", 300, 1200), ("Bulk Paper (case)", 20, 70), ("Security Audit", 500, 2000),
    ("Freight Handling", 40, 200),
]

CURRENCIES = [
    ("USD", "$", "USD"),
    ("EUR", "€", "EUR"),
    ("GBP", "£", "GBP"),
]

TAX_RATES = [0.0, 0.05, 0.07, 0.08, 0.0825, 0.10, 0.20]


def _money(x: float) -> float:
    return round(x + 0.0, 2)


def _make_items(rng: random.Random, n: int) -> list[dict[str, Any]]:
    items = []
    picks = rng.sample(ITEM_CATALOG, k=n)
    for desc, lo, hi in picks:
        qty = rng.randint(1, 12)
        unit_price = _money(rng.uniform(lo, hi))
        amount = _money(qty * unit_price)
        items.append({"description": desc, "qty": qty, "unit_price": unit_price, "amount": amount})
    return items


def _date_str(rng: random.Random) -> str:
    year = rng.choice([2023, 2024])
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    return f"{year:04d}-{month:02d}-{day:02d}"


def _fmt_date(iso_date: str, style: str) -> str:
    year, month, day = iso_date.split("-")
    if style == "iso":
        return iso_date
    if style == "us_slash":
        return f"{month}/{day}/{year}"
    if style == "dot":
        return f"{day}.{month}.{year}"
    return iso_date


def render_style_a(v: dict[str, Any]) -> str:
    lines = ["INVOICE", f"Vendor: {v['vendor']}", "123 Business Rd, Springfield"]
    lines += ["", f"Invoice #: {v['invoice_number']}", f"Date: {v['date_display']}", f"Currency: {v['currency']}", ""]
    lines.append(f"{'Description':<28}{'Qty':>5}{'Unit Price':>14}{'Amount':>12}")
    for it in v["items"]:
        lines.append(f"{it['description']:<28}{it['qty']:>5}{it['unit_price']:>14.2f}{it['amount']:>12.2f}")
    lines.append("")
    lines.append(f"{'Subtotal:':<47}{v['subtotal']:>12.2f}")
    if v["tax"] is not None:
        lines.append(f"{'Tax (' + v['tax_rate_label'] + '):':<47}{v['tax']:>12.2f}")
    lines.append(f"{'Total:':<47}{v['total']:>12.2f}")
    return "\n".join(lines)


def render_style_b(v: dict[str, Any]) -> str:
    lines = [v["vendor"], f"Ref No: {v['invoice_number']}", f"Issued: {v['date_display']}", "Bill To: Customer Account", ""]
    lines.append(f"{'Item':<28}{'Quantity':>10}{'Rate':>10}{'Line Total':>14}")
    for it in v["items"]:
        lines.append(f"{it['description']:<28}{it['qty']:>10}{it['unit_price']:>10.2f}{it['amount']:>14.2f}")
    lines.append("")
    lines.append(f"Sub Total: {v['subtotal']:.2f}")
    if v["tax"] is not None:
        lines.append(f"VAT: {v['tax']:.2f}")
    lines.append(f"Amount Due: {v['total']:.2f}")
    lines.append(f"Currency: {v['currency']}")
    return "\n".join(lines)


def render_style_c(v: dict[str, Any]) -> str:
    lines = ["=== INVOICE ===", f"Company: {v['vendor']}", f"No.: {v['invoice_number']}", f"Date: {v['date_display']}", ""]
    lines.append("| Item                  | Qty | Price   | Total   |")
    lines.append("|-----------------------|-----|---------|---------|")
    for it in v["items"]:
        lines.append(f"| {it['description']:<21} | {it['qty']:<3} | {it['unit_price']:<7.2f} | {it['amount']:<7.2f} |")
    lines.append("")
    sym = v["symbol"]
    lines.append(f"Subtotal: {sym}{v['subtotal']:.2f}")
    if v["tax"] is not None:
        lines.append(f"Sales Tax: {sym}{v['tax']:.2f}")
    lines.append(f"TOTAL: {sym}{v['total']:.2f}")
    return "\n".join(lines)


def render_style_d(v: dict[str, Any]) -> str:
    lines = [f"{v['vendor']} - Statement of Charges", f"Invoice Number: {v['invoice_number']}", f"Date of Issue: {v['date_display']}", f"Currency: {v['currency']}", ""]
    for it in v["items"]:
        lines.append(f"  - {it['description']}: {it['qty']} x {it['unit_price']:.2f} = {it['amount']:.2f}")
    lines.append("")
    lines.append(f"Subtotal ....................... {v['subtotal']:.2f}")
    if v["tax"] is not None:
        lines.append(f"Tax ({v['tax_rate_label']}) ..................... {v['tax']:.2f}")
    lines.append(f"Total Due ...................... {v['total']:.2f}")
    return "\n".join(lines)


STYLES = [render_style_a, render_style_b, render_style_c, render_style_d]
DATE_STYLES = ["iso", "us_slash", "dot"]


def build_normal_record(rng: random.Random, task_id: str, vendor: str) -> dict[str, Any]:
    n_items = rng.randint(1, 4)
    items = _make_items(rng, n_items)
    subtotal = _money(sum(it["amount"] for it in items))
    tax_rate = rng.choice(TAX_RATES)
    tax = _money(subtotal * tax_rate) if tax_rate > 0 else None
    total = _money(subtotal + (tax or 0.0))
    code, symbol, label = rng.choice(CURRENCIES)
    iso_date = _date_str(rng)
    date_style = rng.choice(DATE_STYLES)
    invoice_number = f"{vendor[:2].upper()}-{rng.randint(1000, 9999)}"

    rate_digits = f"{tax_rate * 100:.2f}".rstrip("0").rstrip(".")
    v = {
        "vendor": vendor,
        "invoice_number": invoice_number,
        "date_display": _fmt_date(iso_date, date_style),
        "currency": code,
        "symbol": symbol,
        "items": items,
        "subtotal": subtotal,
        "tax": tax,
        "tax_rate_label": f"{rate_digits}%" if tax_rate else "0%",
        "total": total,
    }
    render = rng.choice(STYLES)
    text = render(v)

    expected = {
        "vendor": vendor,
        "invoice_number": invoice_number,
        "date": iso_date,
        "line_items": [dict(it) for it in items],
        "subtotal": subtotal,
        "tax": tax if tax is not None else 0.0,
        "total": total,
        "currency": code,
    }
    return {"task_id": task_id, "input": {"invoice_text": text}, "expected": expected}


def build_multipage_record() -> dict[str, Any]:
    vendor = "Vandelay Import Export"
    invoice_number = "VE-70051"
    items_p1 = [
        {"description": "Custom Fabrication", "qty": 2, "unit_price": 310.00, "amount": 620.00},
        {"description": "Steel Bracket", "qty": 40, "unit_price": 6.25, "amount": 250.00},
        {"description": "Freight Handling", "qty": 1, "unit_price": 150.00, "amount": 150.00},
    ]
    items_p2 = [
        {"description": "Maintenance Fee", "qty": 1, "unit_price": 95.00, "amount": 95.00},
        {"description": "Replacement Part", "qty": 3, "unit_price": 18.00, "amount": 54.00},
    ]
    all_items = items_p1 + items_p2
    subtotal = _money(sum(it["amount"] for it in all_items))
    tax = _money(subtotal * 0.07)
    total = _money(subtotal + tax)

    text = "\n".join([
        "INVOICE",
        f"Vendor: {vendor}",
        f"Invoice #: {invoice_number}",
        "Date: 2024-02-10",
        "Currency: USD",
        "Page 1 of 2",
        "",
        f"{'Description':<28}{'Qty':>5}{'Unit Price':>14}{'Amount':>12}",
        *(f"{it['description']:<28}{it['qty']:>5}{it['unit_price']:>14.2f}{it['amount']:>12.2f}" for it in items_p1),
        "",
        "--- continued on next page ---",
        "",
        "Page 2 of 2",
        f"{'Description':<28}{'Qty':>5}{'Unit Price':>14}{'Amount':>12}",
        *(f"{it['description']:<28}{it['qty']:>5}{it['unit_price']:>14.2f}{it['amount']:>12.2f}" for it in items_p2),
        "",
        f"{'Subtotal:':<47}{subtotal:>12.2f}",
        f"{'Tax (7%):':<47}{tax:>12.2f}",
        f"{'Total:':<47}{total:>12.2f}",
    ])

    expected = {
        "vendor": vendor,
        "invoice_number": invoice_number,
        "date": "2024-02-10",
        "line_items": all_items,
        "subtotal": subtotal,
        "tax": tax,
        "total": total,
        "currency": "USD",
    }
    return {"task_id": "inv_005", "input": {"invoice_text": text}, "expected": expected}


def build_missing_tax_record() -> dict[str, Any]:
    vendor = "Hooli Cloud Systems"
    invoice_number = "HO-4471"
    items = [
        {"description": "Cloud Storage (per TB)", "qty": 5, "unit_price": 40.00, "amount": 200.00},
        {"description": "Annual Subscription", "qty": 1, "unit_price": 600.00, "amount": 600.00},
    ]
    subtotal = _money(sum(it["amount"] for it in items))
    total = subtotal

    text = "\n".join([
        "Hooli Cloud Systems",
        f"Ref No: {invoice_number}",
        "Issued: 06/18/2024",
        "Bill To: Customer Account",
        "",
        f"{'Item':<28}{'Quantity':>10}{'Rate':>10}{'Line Total':>14}",
        *(f"{it['description']:<28}{it['qty']:>10}{it['unit_price']:>10.2f}{it['amount']:>14.2f}" for it in items),
        "",
        f"Sub Total: {subtotal:.2f}",
        f"Amount Due: {total:.2f}",
        "Currency: USD",
    ])

    expected = {
        "vendor": vendor,
        "invoice_number": invoice_number,
        "date": "2024-06-18",
        "line_items": items,
        "subtotal": subtotal,
        "tax": 0.0,
        "total": total,
        "currency": "USD",
    }
    return {"task_id": "inv_012", "input": {"invoice_text": text}, "expected": expected}


def build_two_currency_record() -> dict[str, Any]:
    vendor = "Monarch Solutions Group"
    invoice_number = "MO-9982"
    items = [
        {"description": "Consulting Hours", "qty": 8, "unit_price": 150.00, "amount": 1200.00},
        {"description": "Travel Expenses", "qty": 1, "unit_price": 40.00, "amount": 40.00},
    ]
    subtotal = _money(sum(it["amount"] for it in items))
    tax = _money(subtotal * 0.20)
    total = _money(subtotal + tax)
    usd_equivalent = _money(total * 1.27)

    text = "\n".join([
        "=== INVOICE ===",
        f"Company: {vendor}",
        f"No.: {invoice_number}",
        "Date: 2024/04/11",
        "",
        "| Item                  | Qty | Price   | Total   |",
        "|-----------------------|-----|---------|---------|",
        *(f"| {it['description']:<21} | {it['qty']:<3} | {it['unit_price']:<7.2f} | {it['amount']:<7.2f} |" for it in items),
        "",
        f"Subtotal: £{subtotal:.2f}",
        f"Sales Tax (20%): £{tax:.2f}",
        f"TOTAL: £{total:.2f}",
        "",
        f"(For reference only, approximate USD equivalent: ${usd_equivalent:.2f} at 1 GBP = 1.27 USD)",
    ])

    expected = {
        "vendor": vendor,
        "invoice_number": invoice_number,
        "date": "2024-04-11",
        "line_items": items,
        "subtotal": subtotal,
        "tax": tax,
        "total": total,
        "currency": "GBP",
    }
    return {"task_id": "inv_019", "input": {"invoice_text": text}, "expected": expected}


def build_bad_arithmetic_record() -> dict[str, Any]:
    vendor = "Dunder Mifflin Paper Co."
    invoice_number = "DM-3301"
    items = [
        {"description": "Bulk Paper (case)", "qty": 3, "unit_price": 12.50, "amount": 37.50},
        {"description": "Printer Toner", "qty": 2, "unit_price": 20.00, "amount": 40.00},
    ]
    real_sum = _money(sum(it["amount"] for it in items))  # 77.50
    stated_subtotal = 85.00  # typo in the source document, does not match real_sum
    stated_tax = _money(stated_subtotal * 0.08)
    stated_total = _money(stated_subtotal + stated_tax)

    text = "\n".join([
        "INVOICE",
        f"Vendor: {vendor}",
        "123 Business Rd, Springfield",
        "",
        f"Invoice #: {invoice_number}",
        "Date: 2024-08-02",
        "Currency: USD",
        "",
        f"{'Description':<28}{'Qty':>5}{'Unit Price':>14}{'Amount':>12}",
        *(f"{it['description']:<28}{it['qty']:>5}{it['unit_price']:>14.2f}{it['amount']:>12.2f}" for it in items),
        "",
        f"{'Subtotal:':<47}{stated_subtotal:>12.2f}",
        f"{'Tax (8.00%):':<47}{stated_tax:>12.2f}",
        f"{'Total:':<47}{stated_total:>12.2f}",
    ])

    expected = {
        "vendor": vendor,
        "invoice_number": invoice_number,
        "date": "2024-08-02",
        "line_items": items,
        "subtotal": stated_subtotal,
        "tax": stated_tax,
        "total": stated_total,
        "currency": "USD",
    }
    return {"task_id": "inv_026", "input": {"invoice_text": text}, "expected": expected}


def build_credit_memo_record() -> dict[str, Any]:
    vendor = "Krusty Krab Food Supply"
    invoice_number = "KK-1187-CM"
    items = [
        {"description": "Catering Tray", "qty": -2, "unit_price": 45.00, "amount": -90.00},
        {"description": "Freight Handling", "qty": -1, "unit_price": 60.00, "amount": -60.00},
    ]
    subtotal = _money(sum(it["amount"] for it in items))
    tax = _money(subtotal * 0.08)
    total = _money(subtotal + tax)

    text = "\n".join([
        "CREDIT MEMO",
        f"Vendor: {vendor}",
        f"Invoice #: {invoice_number}",
        "Date: 2024-09-27",
        "Currency: USD",
        "",
        f"{'Description':<28}{'Qty':>5}{'Unit Price':>14}{'Amount':>12}",
        *(f"{it['description']:<28}{it['qty']:>5}{it['unit_price']:>14.2f}{it['amount']:>12.2f}" for it in items),
        "",
        f"{'Subtotal:':<47}{subtotal:>12.2f}",
        f"{'Tax (8.00%):':<47}{tax:>12.2f}",
        f"{'Total Credit:':<47}{total:>12.2f}",
    ])

    expected = {
        "vendor": vendor,
        "invoice_number": invoice_number,
        "date": "2024-09-27",
        "line_items": items,
        "subtotal": subtotal,
        "tax": tax,
        "total": total,
        "currency": "USD",
    }
    return {"task_id": "inv_033", "input": {"invoice_text": text}, "expected": expected}


HARD_CASE_BUILDERS = {
    "inv_005": build_multipage_record,
    "inv_012": build_missing_tax_record,
    "inv_019": build_two_currency_record,
    "inv_026": build_bad_arithmetic_record,
    "inv_033": build_credit_memo_record,
}


def main() -> None:
    rng = random.Random(SEED)
    normal_vendors = [v for v in VENDORS if v not in {
        "Vandelay Import Export", "Hooli Cloud Systems", "Monarch Solutions Group",
        "Dunder Mifflin Paper Co.", "Krusty Krab Food Supply",
    }]
    rng.shuffle(normal_vendors)

    records: dict[str, dict[str, Any]] = {}
    normal_vendor_iter = iter(normal_vendors)
    for i in range(1, 36):
        task_id = f"inv_{i:03d}"
        if task_id in HARD_CASE_BUILDERS:
            records[task_id] = HARD_CASE_BUILDERS[task_id]()
        else:
            vendor = next(normal_vendor_iter)
            records[task_id] = build_normal_record(rng, task_id, vendor)

    task_ids = sorted(records)
    split_rng = random.Random(SEED)
    shuffled = task_ids[:]
    split_rng.shuffle(shuffled)
    train, holdout = sorted(shuffled[:23]), sorted(shuffled[23:])
    split_of = {tid: "train" for tid in train}
    split_of.update({tid: "holdout" for tid in holdout})

    with (DOMAIN_DIR / "dataset.jsonl").open("w") as f:
        for task_id in task_ids:
            row = dict(records[task_id])
            row["split"] = split_of[task_id]
            f.write(json.dumps(row) + "\n")

    with (DOMAIN_DIR / "split.json").open("w") as f:
        json.dump({"train": train, "holdout": holdout}, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
