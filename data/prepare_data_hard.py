import csv, json, pathlib, sys

SRC = pathlib.Path(sys.argv[1])
OUT = pathlib.Path("data_hard")
OUT.mkdir(exist_ok=True)
HOLDOUT = ["Wingo", "Init7", "Figma", "Infomaniak", "Tamoil", "AMAG",
           "Booking.com", "BLS", "Denner", "Ikea Business", "Canva", "Plakat Global"]

rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
chart = {}
for r in rows:
    chart[r["account"]] = r["account_name"]
CHART = "\n".join(f"{k} {v}" for k, v in sorted(chart.items()))
SYSTEM = "You are a Swiss SME bookkeeper. Reply with ONLY the 4-digit account number from the chart.\n\nChart:\n" + CHART


def held_out(r):
    return any(r["text"].startswith(v + " ") for v in HOLDOUT)


def example(r):
    user = f"Invoice line: {r['text']} | CHF {r['amount_chf']} | VAT {r['vat_rate']}%"
    return {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": user},
                         {"role": "assistant", "content": r["account"]}]}


seen = [r for r in rows if not held_out(r)]
hard = [r for r in rows if held_out(r)]
train = [r for r in seen if r["split"] == "train"]
valid = [r for r in seen if r["split"] != "train"]
for name, part in [("train.jsonl", train), ("valid.jsonl", valid), ("test.jsonl", hard)]:
    with open(OUT / name, "w", encoding="utf-8") as f:
        for r in part:
            f.write(json.dumps(example(r), ensure_ascii=False) + "\n")
    print(name, len(part))
