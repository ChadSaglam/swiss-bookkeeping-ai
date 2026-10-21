# Swiss Bookkeeping AI

A small LoRA fine-tune that suggests an account for a Swiss SME invoice line.
Everything is synthetic and runs on a laptop (Apple Silicon, MLX).

- Model: https://huggingface.co/ChadSaglam/swiss-bookkeeping-qwen-lora
- Dataset: https://huggingface.co/datasets/ChadSaglam/swiss-sme-bookkeeping-dataset
- Collection: https://huggingface.co/collections/ChadSaglam/swiss-sme-bookkeeping-ai-6ad8c507647f44afb5f2263b

## Result

| Model | Seen-vendor test (56 rows) | Held-out-vendor test (74 rows) |
|---|---|---|
| Base Qwen2.5-0.5B-Instruct | 23.2% | 16.2% |
| + LoRA | 96.4% (full-data run) | 94.6% |

The released weights come from the held-out-vendor run. Invoice phrases still overlap
with training data, so real-world accuracy will be lower.

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
