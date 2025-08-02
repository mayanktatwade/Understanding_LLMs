import textwrap
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
from transformers import (AutoModelForCausalLM,AutoTokenizer)



model_path = "/content/financial_llm2"  # or your custom save path
doc_path = "/content/TechNova_Financial_Report_2024.txt" # Enter you text document path

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
model.to("cuda" if torch.cuda.is_available() else "cpu")


# Load and split the document
with open(doc_path, "r", encoding="utf-8") as f:
    full_text = f.read()

chunk_size = 300  # characters (tune this)
chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

# Embed each chunk
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = embedder.encode(chunks)

# Store in FAISS index
index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
index.add(np.array(chunk_embeddings))

import pickle

# Save chunks to file
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

# Save FAISS index
faiss.write_index(index, "faiss_index.index")
print("✅ Saved FAISS index and chunks.")



print("💬 Ask anything about uploaded .txt (type 'exit' to stop):")

while True:
    question = input("\nYou: ")
    if question.lower() in ['exit', 'quit']:
        break

    # Encode the user question
    question_embedding = embedder.encode([question])

    # Retrieve top 5 similar chunks using FAISS
    k = 5
    _, indices = index.search(np.array(question_embedding), k=k)
    unique_indices = np.unique(indices[0])

    # Prepare context by combining the top retrieved chunks
    retrieved_chunks = [(i, chunks[i]) for i in unique_indices if i < len(chunks)]

    retrieved_context = "\n---\n".join([f"[Chunk {i}]\n{txt}" for i, txt in retrieved_chunks])

    # Construct the prompt
    prompt = f"""
    You are a helpful financial assistant.

    ### Context:
    {retrieved_context}

    ### Question:
    {question}

    ### Answer:
    """.strip()

    # Debug: print token count to ensure it's within model limits
    prompt_tokens = len(tokenizer.encode(prompt))
    # print(f"🔢 [Debug] Prompt token count: {prompt_tokens}")

    # Generate the answer from your LLM
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,           # Increased for longer answers
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id  # Helps limit runaway text
    )

    # Decode and format the output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.split("### Answer:")[-1].strip()

    print("\n📘 Bot:\n" + textwrap.fill(answer, width=100))

