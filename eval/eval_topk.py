"""Top-1 and top-3 accuracy by scoring every account in the chart.

Each candidate account is scored by the summed log-probability of its tokens
after the prompt, the same ranking the demo uses. This gives a full ranking per
invoice line, so top-k accuracy is well defined.

Usage:
    python eval/eval_topk.py [--data data_hard/test.jsonl] [--models ID ...]
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "ChadSaglam/swiss-bookkeeping-qwen-lora"]


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def chart_accounts(system_prompt):
    lines = system_prompt.split("Chart:\n", 1)[1].splitlines()
    return [line.split(" ", 1)[0] for line in lines if line[:4].isdigit()]


@torch.no_grad()
def rank_accounts(model, tok, messages, accounts):
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt_ids = tok(prompt, return_tensors="pt").input_ids
    n = prompt_ids.shape[1]
    scores = []
    for acc in accounts:
        ans_ids = tok(acc, add_special_tokens=False, return_tensors="pt").input_ids
        ids = torch.cat([prompt_ids, ans_ids], dim=1)
        logp = model(ids).logits.log_softmax(-1)
        scores.append(sum(logp[0, n - 1 + i, ans_ids[0, i]].item() for i in range(ans_ids.shape[1])))
    order = sorted(range(len(accounts)), key=lambda i: scores[i], reverse=True)
    return [accounts[i] for i in order]


def evaluate(model_id, rows):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
    top1 = top3 = 0
    for row in rows:
        msgs = row["messages"]
        accounts = chart_accounts(msgs[0]["content"])
        ranked = rank_accounts(model, tok, msgs[:-1], accounts)
        gold = msgs[-1]["content"].strip()
        top1 += ranked[0] == gold
        top3 += gold in ranked[:3]
    del model
    return top1, top3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_hard/test.jsonl")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = ap.parse_args()
    rows = load_rows(args.data)
    n = len(rows)
    print(f"| Model | Top-1 | Top-3 |  (n={n})\n|---|---:|---:|")
    for model_id in args.models:
        t1, t3 = evaluate(model_id, rows)
        print(f"| {model_id} | {t1}/{n} ({t1 / n:.1%}) | {t3}/{n} ({t3 / n:.1%}) |", flush=True)


if __name__ == "__main__":
    main()
