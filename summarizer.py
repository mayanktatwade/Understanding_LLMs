import torch
from transformers import (AutoModelForCausalLM,AutoTokenizer)

model_path = "/content/financial_llm2"  # or your custom save path

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
model.to("cuda" if torch.cuda.is_available() else "cpu")


# Load text document
with open("/content/TechNova_Financial_Report_2024.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

# Split into manageable chunks (tweak size if needed)
chunk_size = 1000  # characters
chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]

all_summaries = []

for idx, chunk in enumerate(chunks):
    print(f"🧩 Summarizing chunk {idx+1}/{len(chunks)}...")

    prompt = f"""
You are a helpful assistant. Summarize the following financial or business-related text clearly and concisely.

### Text:
{chunk}

### Summary:
""".strip()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    summary = response.split("### Summary:")[-1].strip()
    all_summaries.append(summary)

# Combine all summaries
final_summary = "\n\n".join(all_summaries)

# Print final formatted summary
print("\n📘 Final Summary:\n")
print(textwrap.fill(final_summary, width=100))

