---
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B-Instruct
datasets: [ChadSaglam/swiss-sme-bookkeeping-dataset]
library_name: transformers
pipeline_tag: text-generation
tags: [accounting, switzerland, lora, mlx, synthetic-data]
---

# Swiss Bookkeeping Account Suggester

Code and training scripts: https://github.com/ChadSaglam/swiss-bookkeeping-ai

LoRA fine-tune of Qwen2.5-0.5B-Instruct. Given a Swiss SME invoice line, it replies with one 4-digit account number from a six-account test chart.

> Educational model trained on **synthetic** data. Not accounting or tax advice.

## Results
| Model | Seen-vendor test (56 rows) | Held-out-vendor test (74 rows) |
|---|---|---|
| Base Qwen2.5-0.5B-Instruct | 23.2% | 16.2% |
| + LoRA adapter (unfused) | 96.4% (full-data run) | 94.6% |
| + LoRA fused (released weights) | n/a | **93.2%** |

The released weights are the held-out-vendor run: 12 vendors never appear in training. Invoice phrases still overlap with training data, so real-world accuracy will be lower.

## Prompt format (must match training)
System prompt:

    You are a Swiss SME bookkeeper. Reply with ONLY the 4-digit account number from the chart.

    Chart:
    5830 Travel expenses (staff)
    6200 Vehicle expenses
    6500 Office supplies and administration
    6510 Telephone and internet
    6570 IT and software
    6600 Advertising

User message: `Invoice line: <text> | CHF <amount> | VAT <rate>%`

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "ChadSaglam/swiss-bookkeeping-qwen-lora"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo)

CHART = """5830 Travel expenses (staff)
6200 Vehicle expenses
6500 Office supplies and administration
6510 Telephone and internet
6570 IT and software
6600 Advertising"""
SYSTEM = ("You are a Swiss SME bookkeeper. Reply with ONLY the 4-digit account number "
          "from the chart.\n\nChart:\n" + CHART)

msgs = [{"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Invoice line: Swisscom Mobile Abo März | CHF 89.50 | VAT 8.1%"}]
prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
inputs = tok(prompt, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=6, do_sample=False)
n = inputs["input_ids"].size(-1)
print(tok.decode(out.squeeze()[n:], skip_special_tokens=True))
```

## Training
- Data: [swiss-sme-bookkeeping-dataset](https://huggingface.co/datasets/ChadSaglam/swiss-sme-bookkeeping-dataset), 205 train / 93 validation rows after removing 12 vendors for the hard test
- Method: LoRA (rank 8, 16 layers), learning rate 1e-5, batch size 4, 600 iterations
- Framework: mlx-lm on Apple Silicon, then fused into the base weights

## Limits
- Six accounts from a simplified test chart, not the official KMU-Kontenrahmen
- Template-generated, clean invoice text; real invoices are messier
- Unknown vendors with generic wording are the main error source
- No calibrated confidence score
