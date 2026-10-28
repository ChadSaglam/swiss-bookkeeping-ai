# Swiss Bookkeeping AI

Training and evaluation code for a small account-suggestion model built with Qwen2.5-0.5B-Instruct and LoRA. The project runs on Apple Silicon with MLX and includes a Gradio demo.

## Resources

- [Interactive demo](https://huggingface.co/spaces/ChadSaglam/swiss-bookkeeping-ai)
- [Fused model](https://huggingface.co/ChadSaglam/swiss-bookkeeping-qwen-lora)
- [8-bit MLX model](https://huggingface.co/ChadSaglam/swiss-bookkeeping-qwen-lora-mlx-8bit)
- [Dataset and validation scripts](https://huggingface.co/datasets/ChadSaglam/swiss-sme-bookkeeping-dataset)
- [Project collection](https://huggingface.co/collections/ChadSaglam/swiss-sme-bookkeeping-ai-6ad8c507647f44afb5f2263b)

All training data is synthetic. The six-account chart is a simplified test chart, not an official KMU-Kontenrahmen excerpt. Outputs are drafts for human review, not accounting or tax advice.

## Results

| Model | Seen-vendor test (56 rows) | Held-out-vendor test (74 rows) |
|---|---:|---:|
| Base Qwen2.5-0.5B-Instruct | 23.2% | 16.2% |
| + LoRA adapter, unfused | 96.4% (full-data run) | 94.6% |
| + LoRA fused, released weights | n/a | **93.2%** |

The released weights come from the held-out-vendor run. Twelve vendors were excluded from training, but invoice phrases can still overlap. Real-world accuracy will be lower.

### Efficiency variants

Measured on the same 74-row held-out-vendor test:

| Variant | Accuracy | Peak memory |
|---|---:|---:|
| fp16 | 93.2% | 1.13 GB |
| 8-bit | 93.2% | 0.71 GB |
| 4-bit | 71.6% | 0.44 GB |

Latency varied by about 20 ms between runs, so no speed ranking is claimed. See [efficiency-lab](https://github.com/ChadSaglam/efficiency-lab) for details.

## Out-of-phrase and robustness checks

The 8-bit MLX variant scored 17/36 (47.2%; 95% interval 32–63%) on the hand-written OOD stress test. The set uses unseen vendors and wording and was designed around the model’s weak spots. Its labels were reviewed by the author; one correction and one debatable example are documented in the dataset repository.

A separate prompt-steering test found 152 errors in 352 modified cases (43%). Appending advertising-related suffixes moved 85 predictions to 6600 Advertising. These cases came from about 44 base lines and are not independent. [Read the robustness report](https://github.com/ChadSaglam/bookkeeping-mcp-agent/blob/main/REPORT.md).

The [model card](https://huggingface.co/ChadSaglam/swiss-bookkeeping-qwen-lora) contains the full evaluation notes, prompt format and usage example.

## Repository layout

- `data/prepare_data.py`: convert the dataset CSV to chat JSONL using the random split.
- `data/prepare_data_hard.py`: create the split with 12 vendors held out for testing.
- `train/train.sh`: run the end-to-end training workflow.
- `eval/eval_model.py`: evaluate the base model and adapters.
- `space/`: Gradio demo source.
- `MODEL_CARD.md`: model card published on Hugging Face.

## Reproduce

```bash
bash train/train.sh
```

The workflow downloads the dataset, holds out 12 vendors, trains a rank-8 LoRA adapter for 600 iterations, fuses the adapter and evaluates the base and tuned models.

## Limitations

This project covers six accounts and clean, template-generated invoice text. Unknown vendors and generic wording remain important error sources. Model scores are not calibrated, and the OOD labels have not been independently reviewed by an accountant. Do not use the model for unattended posting, production bookkeeping or tax decisions.
