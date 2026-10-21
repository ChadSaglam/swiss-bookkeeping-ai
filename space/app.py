import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "ChadSaglam/swiss-bookkeeping-qwen-lora"
CHART = {"6510": "Telephone and internet", "6570": "IT and software", "6200": "Vehicle expenses",
         "5830": "Travel expenses (staff)", "6500": "Office supplies and administration", "6600": "Advertising"}
SYSTEM = ("You are a Swiss SME bookkeeper. Reply with ONLY the 4-digit account number from the chart.\n\nChart:\n"
          + "\n".join(f"{k} {v}" for k, v in sorted(CHART.items())))

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).eval()


def score(prompt_ids, answer):
    ans_ids = tok(answer, add_special_tokens=False, return_tensors="pt").input_ids
    ids = torch.cat([prompt_ids, ans_ids], dim=1)
    with torch.no_grad():
        logp = model(ids).logits.log_softmax(-1)
    n = prompt_ids.shape[1]
    return sum(logp[0, n - 1 + i, ans_ids[0, i]].item() for i in range(ans_ids.shape[1]))


def suggest(text, amount, vat):
    if not text or not text.strip():
        raise gr.Error("Please type an invoice line.")
    user = f"Invoice line: {text} | CHF {amount} | VAT {vat}%"
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    pids = tok(prompt, return_tensors="pt").input_ids
    scores = torch.tensor([score(pids, a) for a in CHART]).softmax(0)
    return {f"{a} {CHART[a]}": float(s) for a, s in zip(CHART, scores)}


demo = gr.Interface(
    suggest,
    [gr.Textbox(label="Invoice line", placeholder="Swisscom Mobile Abo März"),
     gr.Number(label="Amount (CHF)", value=89.5), gr.Dropdown([8.1, 3.8, 2.6, 0], value=8.1, label="VAT %")],
    gr.Label(num_top_classes=3, label="Suggested accounts"),
    title="📒 Swiss Bookkeeping Account Suggester",
    description="Synthetic data demo. Not accounting or tax advice.",
    flagging_mode="never",
)
demo.launch()
