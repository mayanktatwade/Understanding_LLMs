import yfinance as yf
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
from transformers import (AutoModelForCausalLM,AutoTokenizer)

model_path = "/content/financial_llm2"  # or your custom save path

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
model.to("cuda" if torch.cuda.is_available() else "cpu")



# Define portfolio
portfolio1 = [
    {"symbol": "AAPL", "quantity": 15, "buy_price": 135},
    {"symbol": "JNJ", "quantity": 20, "buy_price": 160},
    {"symbol": "XOM", "quantity": 18, "buy_price": 85},
    {"symbol": "MSFT", "quantity": 12, "buy_price": 250},
    {"symbol": "JPM", "quantity": 10, "buy_price": 130},
    {"symbol": "KO",  "quantity": 25, "buy_price": 55},
    {"symbol": "UNH", "quantity": 6,  "buy_price": 480},
    {"symbol": "V",   "quantity": 10, "buy_price": 190},
    {"symbol": "NVDA","quantity": 8,  "buy_price": 240},
    {"symbol": "VEA", "quantity": 30, "buy_price": 45},
    {"symbol": "BND", "quantity": 40, "buy_price": 75},
    {"symbol": "SPY", "quantity": 15, "buy_price": 400}
]

portfolio2 = [
    {"symbol": "INTC", "quantity": 25, "buy_price": 1000},  # Struggling chipmaker
    {"symbol": "T",    "quantity": 50, "buy_price": 1000},  # Declining telecom with high debt
    {"symbol": "WFC",  "quantity": 18, "buy_price": 1000},  # Banking stock with regulatory issues
    {"symbol": "PYPL", "quantity": 12, "buy_price": 180}, # Fallen fintech giant
    {"symbol": "DIS",  "quantity": 15, "buy_price": 900},  # Media stock with streaming losses
    {"symbol": "BABA", "quantity": 10, "buy_price": 1020}, # Chinese stock with political risk
    {"symbol": "ARKK", "quantity": 20, "buy_price": 600},  # Volatile, underperforming ETF
    {"symbol": "F",    "quantity": 30, "buy_price": 120},  # Legacy automaker with EV struggles
    {"symbol": "META", "quantity": 8,  "buy_price": 3000}, # Tech stock with uncertain growth
    {"symbol": "GE",   "quantity": 20, "buy_price": 605},  # Industrial conglomerate in decline
    {"symbol": "SLV",  "quantity": 40, "buy_price": 202},  # Silver ETF with high volatility
    {"symbol": "NFLX", "quantity": 5,  "buy_price": 3500}  # High-competition streaming stock
]

# CHOOSE WHICH PORTFOLIO TO CHECK FOR
portfolio = portfolio1

# Fetch real-time prices and compute ROI/value
total_value = 0
for asset in portfolio:
    ticker = yf.Ticker(asset["symbol"])
    info = ticker.info
    asset["current_price"] = info.get("currentPrice", 0)
    asset["sector"] = info.get("sector", "Unknown")
    asset["roi"] = round(((asset["current_price"] - asset["buy_price"]) / asset["buy_price"]) * 100, 2)
    asset["value"] = round(asset["quantity"] * asset["current_price"], 2)
    total_value += asset["value"]

# Filter out invalid or zero-value assets
valid_assets = [a for a in portfolio if a["current_price"] > 0]

# Calculating ROI
average_roi = sum([a["roi"] for a in valid_assets]) / len(valid_assets)

# Build portfolio summary string
summary_lines = []
for asset in valid_assets:
    summary_lines.append(
        f"- {asset['symbol']}: {asset['quantity']} shares @ ${asset['buy_price']} "
        f"(Current: ${asset['current_price']}) – Sector: {asset['sector']} – ROI: {asset['roi']}% – Value: ${asset['value']}"
    )
portfolio_summary = "\n".join(summary_lines)

# Add overall framing to encourage balanced tone
overall_comment = (
    f"The portfolio has a total value of approximately ${round(total_value, 2)} "
    f"and includes a diverse set of holdings with notable gains in several sectors."
)

# Final prompt for your LLM
prompt = f"""
You are a professional financial advisor.

The client's average portfolio ROI is {average_roi:.2f}%.

Analyze and advise based on:
1. Diversification and sector exposure
2. High- and low-performing assets
3. Risk profile considering the ROI
4. Suggestions for rebalancing, replacements, or new additions
5. Long-term strategy guidance

### Portfolio:
{portfolio_summary}

### Advice:
(End your advice with a final summary or conclusion.)
""".strip()

# Generate advice using your model
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=True,
    top_k=30,
    top_p=0.85,
    temperature=0.5,
    repetition_penalty=1.15,
    eos_token_id=tokenizer.eos_token_id
)

# Decode and show output
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
answer = response.split("### Advice:")[-1].strip()


print("\n📊 Portfolio Summary:\n" + portfolio_summary)
print("\n📘 LLM Advice:\n" + answer)

