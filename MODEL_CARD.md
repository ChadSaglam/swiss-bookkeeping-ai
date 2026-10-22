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

## Out-of-phrase check

On 36 hand-written invoice lines with vendors and wording that the training-data generator does not use, the 8-bit MLX variant scored 17/36 (47.2%, 95% interval 32-63%), against 93.2% on the held-out-vendor test. The set is hard by design: the author wrote it with the model's weak spots in mind. The labels were drafted by an AI assistant; the author then reviewed all 36 lines, corrected one (Post Einschreiben, 6500 to 6510, after seeing results) and marked one as debatable (Putztücher Handseife Teeküche). Reported scores use the original labels; the correction does not change the 0.5B score. Treat it as a lower bound for messy real invoices. Data: ood_test.csv in https://huggingface.co/datasets/ChadSaglam/swiss-sme-bookkeeping-dataset

For reference, prompted qwen2.5:7b (Ollama, no fine-tuning, chart of accounts in the prompt, temperature 0) scored 27/36 (75.0%, 95% interval 59-86%) on the same 36 lines using the text only. With the CHF amount appended it scored 30/36 (83.3%); the amounts in this set are independent of the account, so that gap is within noise. Both settings were fixed before the run and the prompt was not changed afterwards. The set was written against the 0.5B model's weak spots, so read this as a rough reference, not a ranking. With the corrected label the 7B text-only score is 28/36 (77.8%); without the two disputed lines it is 26/34 (76.5%). On the same lines the 7B was right where the 0.5B was wrong 17 times, the reverse 7 times (exact McNemar p about 0.06). The two models also differ in size and in whether they see the chart of accounts.

## Robustness: not safe against prompt steering

Words in the invoice text act as content, not as text to ignore. In a local test
(bookkeeping-mcp-agent), 5 of 56 test rows (not used for training) were wrong without changes.
After appending short suffixes such as "(advertising)", "- Werbung" or "pour publicité",
152 of 352 cases were wrong (43%) and 85 of them moved to 6600 Advertising. The 352 cases come
from about 44 base lines and are not independent. Treat every output as a draft for a human to
review, and do not book automatically.
Details: https://github.com/ChadSaglam/bookkeeping-mcp-agent/blob/main/REPORT.md

## Limits
- Six accounts from a simplified test chart, not the official KMU-Kontenrahmen
- Template-generated, clean invoice text; real invoices are messier
- Unknown vendors with generic wording are the main error source
- No calibrated confidence score
