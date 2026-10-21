# Swiss Bookkeeping AI

A small LoRA fine-tune that suggests an account for a Swiss SME invoice line.
Everything is synthetic and runs on a laptop (Apple Silicon, MLX).

- Model: https://huggingface.co/ChadSaglam/swiss-bookkeeping-qwen-lora
- 8-bit MLX model: https://huggingface.co/ChadSaglam/swiss-bookkeeping-qwen-lora-mlx-8bit
- Dataset: https://huggingface.co/datasets/ChadSaglam/swiss-sme-bookkeeping-dataset
- Collection: https://huggingface.co/collections/ChadSaglam/swiss-sme-bookkeeping-ai-6ad8c507647f44afb5f2263b

## Result

| Model | Seen-vendor test (56 rows) | Held-out-vendor test (74 rows) |
|---|---|---|
| Base Qwen2.5-0.5B-Instruct | 23.2% | 16.2% |
| + LoRA adapter (unfused) | 96.4% (full-data run) | 94.6% |
| + LoRA fused (released weights) | n/a | **93.2%** |

The released weights come from the held-out-vendor run. Invoice phrases still overlap
with training data, so real-world accuracy will be lower.

## Variants

Held-out-vendor test, 74 rows (about +/-6 points of uncertainty). Details and code:
https://github.com/ChadSaglam/efficiency-lab

| Variant | Accuracy | Peak memory |
|---|---|---|
| fp16 | 93.2% | 1.13 GB |
| 8-bit | 93.2% | 0.71 GB |
| 4-bit | 71.6% | 0.44 GB |

Latency (54-71 ms per line) varied by about 20 ms between runs, so no speed ranking is claimed.

## Robustness: not safe against prompt steering

Words in the invoice text act as content, not as text to ignore. In a local test
(bookkeeping-mcp-agent), 5 of 56 test rows (not used for training) were wrong without changes.
After appending short suffixes such as "(advertising)", "- Werbung" or "pour publicité",
152 of 352 cases were wrong (43%) and 85 of them moved to 6600 Advertising. The 352 cases come
from about 44 base lines and are not independent. Treat every output as a draft for a human to
review, and do not book automatically.
Report: https://github.com/ChadSaglam/bookkeeping-mcp-agent/blob/main/REPORT.md

## Reproduce

    bash train/train.sh

Steps: download the dataset, hold out 12 vendors, train LoRA (rank 8, 600 iterations),
fuse the adapter, evaluate base vs tuned.

## Layout

- data/prepare_data.py: dataset CSV to chat JSONL (random split)
- data/prepare_data_hard.py: same, with 12 vendors held out for the test set
- train/train.sh: end-to-end run
- eval/eval_model.py: accuracy of the base model and adapters
- space/: Gradio demo (needs a Hugging Face Space with Gradio support)
- MODEL_CARD.md: the model card published on the Hub

## Limits

Synthetic data, six accounts from a simplified test chart (not the official KMU-Kontenrahmen),
clean template text. Not accounting or tax advice.
