import json, re, sys
from mlx_lm import load, generate

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
test = [json.loads(l) for l in open("data_hard/test.jsonl", encoding="utf-8")]
adapters = sys.argv[1:] or ["adapters"]


def run(adapter=None):
    model, tok = load(BASE, adapter_path=adapter)
    hits, miss = 0, []
    for ex in test:
        prompt = tok.apply_chat_template(ex["messages"][:-1], add_generation_prompt=True, tokenize=False)
        out = generate(model, tok, prompt=prompt, max_tokens=8)
        m = re.search(r"\d{4}", out)
        got = m.group(0) if m else "none"
        want = ex["messages"][-1]["content"]
        if got == want:
            hits += 1
        else:
            miss.append((ex["messages"][1]["content"], want, got))
    return hits / len(test), miss


rows = [("base", *run(None))]
for a in adapters:
    rows.append((a, *run(a)))

print("| Model | Test accuracy |\n|---|---|")
for name, acc, _ in rows:
    print(f"| {name} | {acc:.1%} |")
for name, _, miss in rows:
    print(f"\n{name}: first misses")
    for text, want, got in miss[:5]:
        print(f"  {text} | expected {want} | got {got}")