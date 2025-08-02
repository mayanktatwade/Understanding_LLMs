# FINE-TUNING LLMs for Financial Analysis and Portfolio Strategy
The link to my report - https://drive.google.com/file/d/1OyuC2Pn8QGH6qE5e8AEhfu5FzmgotlgS/view?usp=sharing

# 💼 Financial LLM Assistant

A modular AI-driven assistant designed for financial document analysis, chatbot interaction, summarization, and investment portfolio evaluation. This project leverages Hugging Face Transformers, PEFT LoRA fine-tuning, FAISS for retrieval, and SentenceTransformers for embeddings.

## 📁 Project Structure

- **training.py**
Fine-tunes an LLM (e.g., `google/gemma-2b`) on financial Q\&A data using PEFT and LoRA for efficiency.
- **chatbot.py**
A retrieval-augmented chatbot that answers user queries from a financial report using FAISS and chunk embeddings.
- **summarizer.py**
Summarizes large financial reports into concise highlights using chunked generation.
- **portfolio_analysis.py**
Analyzes an investment portfolio using real-time stock data and provides personalized financial advice.


## 🚀 Setup Instructions

1. **Install dependencies:**

```bash
pip install -U transformers datasets accelerate peft trl bitsandbytes faiss-cpu sentence-transformers yfinance
```

2. **Authenticate with Hugging Face (for model access/upload):**

```bash
huggingface-cli login
```

3. **Upload required files:**
    - `Financial-QA-10k.csv` (for training)
    - `TechNova_Financial_Report_2024.txt` (for chatbot/summarizer)

## 🧠 Model Training (Optional)

Run the training script to fine-tune the model:

```bash
python training.py
```

*Make sure to toggle `do_training = True` if needed.*

## 💬 Chatbot Usage

Run the chatbot to ask questions about a financial report:

```bash
python chatbot.py
```

**Example Interaction**

```
💬 Ask anything about uploaded .txt (type 'exit' to stop):

You: What were TechNova’s total revenue and net income in FY 2024?
📘 Bot:
Total revenue = $89.2 billion     
Net income = $19.

You: Which product segment generated the highest revenue for TechNova?
📘 Bot:
[Chunk 4]     Consumer Devices      
### Explanation:     
[Chunk 5]     The "Consumer Devices" segment generated the highest revenue for TechNova, with a total revenue of $45.6 billion in FY 2024, representing 50.5% of the company's total revenue...

You: What are TechNova’s sustainability goals for the future?
📘 Bot:
- Reducing carbon footprint via renewables  
- Designing sustainable, low-waste products  
- Increasing recycled material usage  
- Eco-friendly packaging & customer recycling  
- Investing in sustainability-focused R&D
```


## 📝 Summarizer Output

The summarizer breaks down long reports into concise summaries.
**Example conclusion:**
> TechNova Inc. saw significant growth in revenue and profitability in FY 2024...
> Net income increased by 12.5%, with major growth in cloud and consumer electronics...
> The company achieved 82% renewable energy use, repurchased shares, and projects FY 2025 revenue of \$96–98B.

## 📊 Portfolio Analysis

Run the analysis script:

```bash
python portfolio_analysis.py
```

**Example Output:**

```
📊 Portfolio Summary:
- AAPL: 15 shares @ $135 (Current: $208.57) – ROI: 54.5%
- MSFT: 12 shares @ $250 (Current: $534.53) – ROI: 113.81%
- UNH: 6 shares @ $480 (Current: $252.5) – ROI: -47.4%
...

📘 LLM Advice:
1. The portfolio has an average ROI of **40.75%**
2. Diversify away from underperformers (e.g., UNH)
3. Add more defensive or healthcare stocks
4. Regularly review risk profile and rebalance
```

## 📌 Notes

- The model uses 4-bit quantization and LoRA to make training efficient on consumer GPUs.
- For production, integrate with Gradio or Streamlit for UI, or expose as an API via FastAPI.
- FAISS + Sentence Transformers provides fast, meaningful retrieval over long documents.

Let me know if you want this README as a downloadable file or included inside your ZIP archive!


