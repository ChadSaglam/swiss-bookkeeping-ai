---
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B-Instruct
datasets: [ChadSaglam/swiss-sme-bookkeeping-dataset]
library_name: transformers
pipeline_tag: text-generation
tags: [accounting, switzerland, lora, mlx, synthetic-data]
---

# Swiss Bookkeeping Account Suggester

A LoRA fine-tune of Qwen2.5-0.5B-Instruct that maps a Swiss SME invoice line to one of six account codes. The released weights are fused for direct use with Transformers.

> Educational model trained on synthetic data. Every suggestion needs human review. This is not accounting or tax advice.

## Resources

- [Gradio demo source](https://github.com/ChadSaglam/swiss-bookkeeping-ai/tree/main/space)
- [Dataset and validation scripts](https://huggingface.co/datasets/ChadSaglam/swiss-sme-bookkeeping-dataset)
- [Training and evaluation code](https://github.com/ChadSaglam/swiss-bookkeeping-ai)
- [8-bit MLX variant](https://huggingface.co/ChadSaglam/swiss-bookkeeping-qwen-lora-mlx-8bit)
- [Project collection](https://huggingface.co/collections/ChadSaglam/swiss-sme-bookkeeping-ai-6ad8c507647f44afb5f2263b)

## Results

| Model | Seen-vendor test (56 rows) | Held-out-vendor test (74 rows) |
|---|---:|---:|
| Base Qwen2.5-0.5B-Instruct | 23.2% | 16.2% |
| + LoRA adapter, unfused | 96.4% (full-data run) | 94.6% |
| + LoRA fused, released weights | n/a | **93.2%** |

The released weights come from the held-out-vendor run. Twelve vendors were excluded from training, but invoice phrases can still overlap. These results do not estimate accuracy on real bookkeeping data.

## Demo

The [Gradio demo source](https://github.com/ChadSaglam/swiss-bookkeeping-ai/tree/main/space) returns the three highest-scoring accounts for an invoice line, CHF amount and VAT rate. The scores are relative model scores, not calibrated probabilities.

Run it locally from the repository root:

```bash
cd space
pip install -r requirements.txt
python app.py
```

## Prompt format

The prompt must match training.

System prompt:

```text
You are a Swiss SME bookkeeper. Reply with ONLY the 4-digit account number from the chart.

Chart:
5830 Travel expenses (staff)
6200 Vehicle expenses
6500 Office supplies and administration
6510 Telephone and internet
6570 IT and software
6600 Advertising
```

User message:

```text
Invoice line: <text> | CHF <amount> | VAT <rate>%
```

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
SYSTEM = (
    "You are a Swiss SME bookkeeper. Reply with ONLY the 4-digit account number "
    "from the chart.\n\nChart:\n" + CHART
)

messages = [
    {"role": "system", "content": SYSTEM},
    {
        "role": "user",
        "content": "Invoice line: Swisscom Mobile Abo März | CHF 89.50 | VAT 8.1%",
    },
]
prompt = tok.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=False
)
inputs = tok(prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=6, do_sample=False)
prompt_length = inputs["input_ids"].size(-1)
print(tok.decode(output.squeeze()[prompt_length:], skip_special_tokens=True))
```

## Training

- Data: [Swiss SME Bookkeeping dataset](https://huggingface.co/datasets/ChadSaglam/swiss-sme-bookkeeping-dataset), with 205 train and 93 validation rows after removing 12 vendors for the held-out test.
- Method: LoRA, rank 8 across 16 layers.
- Settings: learning rate 1e-5, batch size 4, 600 iterations.
- Framework: `mlx-lm` on Apple Silicon, followed by fusion into the base weights.

## Out-of-phrase evaluation

The separate OOD set contains 36 hand-written lines with unseen vendors and wording. It was designed around known weaknesses of the 0.5B model, so it is a stress test rather than a representative benchmark.

- The released 8-bit MLX variant scored **17/36 (47.2%; 95% interval 32–63%)**, compared with 93.2% on the held-out-vendor test.
- A prompted `qwen2.5:7b` reference scored **27/36 (75.0%; 95% interval 59–86%)** using text only and **30/36 (83.3%)** when the CHF amount was appended.
- The 7B model used Ollama, temperature 0 and the account chart in its prompt. It was not fine-tuned. These settings make it a rough reference, not a like-for-like ranking.

The OOD labels were initially drafted with AI assistance and reviewed by the author. One label was corrected from 6500 to 6510 after the first evaluation; this does not change the 0.5B score. The corrected 7B text-only score is 28/36 (77.8%). One additional label remains debatable. The dataset card and `label_review.txt` document the review.

## Robustness

Invoice text is treated as model input, not as untrusted content to ignore. In a local prompt-steering test, 5 of 56 unchanged test rows were classified incorrectly. After short suffixes such as `(advertising)`, `- Werbung` or `pour publicité` were appended, 152 of 352 cases were wrong (43%), including 85 predictions that moved to 6600 Advertising.

The 352 cases came from about 44 base lines and are not independent. Do not book transactions automatically. [Read the robustness report](https://github.com/ChadSaglam/bookkeeping-mcp-agent/blob/main/REPORT.md).

## Intended use and limitations

Suitable for education, reproducible demos and experiments with small text models.

- Six accounts from a simplified test chart, not the official KMU-Kontenrahmen.
- Synthetic, template-generated text that is cleaner than real invoice text.
- Unknown vendors and generic wording remain important error sources.
- No calibrated confidence score and no independent accountant review.
- Not suitable for production bookkeeping, tax decisions or unattended posting.
