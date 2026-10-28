---
title: Swiss Bookkeeping Account Suggester
emoji: 📒
sdk: gradio
app_file: app.py
pinned: true
license: apache-2.0
models: [ChadSaglam/swiss-bookkeeping-qwen-lora]
datasets: [ChadSaglam/swiss-sme-bookkeeping-dataset]
short_description: Suggests a Swiss SME account for an invoice line
---

# Swiss Bookkeeping Account Suggester

Enter an invoice line, CHF amount and VAT rate to see the three highest-scoring accounts.

- [Model card](https://huggingface.co/ChadSaglam/swiss-bookkeeping-qwen-lora)
- [Dataset](https://huggingface.co/datasets/ChadSaglam/swiss-sme-bookkeeping-dataset)
- [Source code](https://github.com/ChadSaglam/swiss-bookkeeping-ai)

This educational demo uses synthetic training data and a simplified six-account chart. Scores are not calibrated probabilities. Review every suggestion; this is not accounting or tax advice.
