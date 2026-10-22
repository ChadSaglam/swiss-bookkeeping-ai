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
