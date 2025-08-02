# !pip install -U datasets accelerate peft trl bitsandbytes transformers faiss-cpu
# Also perform Hugging face login with: huggingface-cli login --token YOUR_HF_TOKEN

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model
)
from trl import SFTTrainer
from sklearn.model_selection import train_test_split
from datasets import Dataset
import pandas as pd

# Config
MODEL_NAME = "google/gemma-2b"  # Lightweight compared to 7B+
DATA_PATH = "Financial-QA-10k.csv"  # Make sure this file is uploaded to Colab
OUTPUT_DIR = "./financial_llm2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def training_function():
  # Load and prepare dataset
  def load_dataset(file_path):
    df = pd.read_csv(file_path)
    df = df[:500]
    df['text'] = df.apply(lambda x: f"### Instruction: {x['question']}\n\n### Response: {x['answer']}", axis=1)
    train_df, eval_df = train_test_split(df, test_size=0.1, random_state=42)
    return Dataset.from_pandas(train_df), Dataset.from_pandas(eval_df)

  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  tokenizer.pad_token = tokenizer.eos_token

  model = AutoModelForCausalLM.from_pretrained(
      MODEL_NAME,
      load_in_4bit=True,
      device_map="auto",
      torch_dtype=torch.float16
  )

  model = prepare_model_for_kbit_training(model)

  lora_config = LoraConfig(
      r=8,
      lora_alpha=32,
      target_modules=["q_proj", "v_proj"],
      lora_dropout=0.05,
      bias="none",
      task_type="CAUSAL_LM"
  )

  model = get_peft_model(model, lora_config)
  model.print_trainable_parameters()

  def tokenize_function(examples):
      tokens = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
      tokens["labels"] = tokens["input_ids"].copy()
      return tokens

  train_dataset, eval_dataset = load_dataset(DATA_PATH)
  tokenized_train = train_dataset.map(tokenize_function, batched=True)
  tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

  from trl import SFTConfig

  training_args = TrainingArguments(
          output_dir=OUTPUT_DIR,
          per_device_train_batch_size=1,
          per_device_eval_batch_size=1,
          gradient_accumulation_steps=4,
          num_train_epochs=3,
          logging_dir=f"{OUTPUT_DIR}/logs",
          logging_steps=10,
          save_steps=500,
          eval_strategy="steps",  # Still might not be supported in old version,
          eval_steps=500,
          learning_rate=2e-5,
          fp16=True,
          warmup_steps=100,
          report_to="none",
          save_safetensors=False
      )

  trainer = SFTTrainer(
      model=model,
      args=training_args,
      train_dataset=tokenized_train,
      eval_dataset=tokenized_eval,
      processing_class=tokenizer
  )

  trainer.train()
  model.save_pretrained(OUTPUT_DIR)
  tokenizer.save_pretrained(OUTPUT_DIR)
  print("✅ Training complete! Model saved to:", OUTPUT_DIR)


do_training = False; #Set True if need to train, I have already trained
if do_training == True:
  training_function()

