#!/usr/bin/env python3
"""
Evaluate MT0-LoRA model on detoxification task.
"""

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from peft import PeftModel
from detoxify import Detoxify
import evaluate

# -------------------------
# CONFIGURATION
# -------------------------

BASE_MODEL = "bigscience/mt0-large"
LORA_PATH = "./mt0-detox-lora"  # Path to your saved LoRA adapters
CACHE_DIR = "./model_cache"
EVAL_DATA_PATH = "./results/merged_data/eval_ru.xlsx"  # Update this path
OUTPUT_CSV = "./mt0_lora_evaluation_results.csv"
PREDICTIONS_CSV = "./mt0_lora_predictions.csv"

# CSV columns
INPUT_COL = "toxic_sentence"
TARGET_COL = "non_toxic_sentence"
LANG = "ru"  # Language for BERTScore

# -------------------------
# DEVICE SETUP
# -------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"✅ Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("⚠️ Using CPU")

# -------------------------
# LOAD MT0-LORA MODEL
# -------------------------

print("Loading base model...")
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float32,
)

print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model = model.to(device)
model.eval()

mt0_tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)

print("✅ MT0-LoRA Model loaded!\n")

# -------------------------
# LOAD METRICS (GLOBAL)
# -------------------------

print("Loading evaluation models...")

# Toxicity
print("  Loading Detoxify (multilingual)...")
detox_model = Detoxify('multilingual')

# Semantic Similarity
print("  Loading BERTScore...")
bertscore_metric = evaluate.load("bertscore")

# Fluency
print("  Loading GPT-2...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
lm_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
lm_model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"✅ Evaluation models loaded! Using device: {device}\n")

# -------------------------
# METRIC FUNCTIONS
# -------------------------

def compute_toxicity(predictions: list) -> float:
    """Compute average toxicity score."""
    scores = []
    for txt in tqdm(predictions, desc="    Toxicity", leave=False):
        if not txt or not txt.strip():
            scores.append(0.0)
            continue
        score = detox_model.predict(txt)['toxicity']
        scores.append(score)
    return float(np.mean(scores))

def compute_similarity(predictions: list, references: list, lang: str) -> float:
    """Compute average BERTScore F1."""
    # Handle empty strings
    preds_clean = [p if p and p.strip() else "." for p in predictions]
    refs_clean = [r if r and r.strip() else "." for r in references]
    
    # Use appropriate language for BERTScore
    bert_lang = lang if lang in ["en", "de", "fr", "es", "ru", "zh"] else "en"
    
    bert_results = bertscore_metric.compute(
        predictions=preds_clean,
        references=refs_clean,
        lang=bert_lang
    )
    return float(np.mean(bert_results["f1"]))

def compute_fluency(texts: list) -> float:
    """Compute average fluency (negative log-likelihood)."""
    scores = []
    with torch.no_grad():
        for txt in tqdm(texts, desc="    Fluency", leave=False):
            if not txt or not txt.strip():
                scores.append(float('-inf'))
                continue
            enc = tokenizer(txt, return_tensors="pt", truncation=True, padding=True).to(device)
            outputs = lm_model(**enc, labels=enc["input_ids"])
            loss = outputs.loss.item()
            scores.append(-loss)
    return float(np.mean(scores))

# -------------------------
# INFERENCE FUNCTION
# -------------------------

def detoxify(text: str, max_new_tokens: int = 128) -> str:
    prompt = f"Detoxify this text: {text}"
    inputs = mt0_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
    
    return mt0_tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------------------------
# MAIN EVALUATION
# -------------------------

def main():
    # Load evaluation data
    print("Loading evaluation data...")
    df = pd.read_excel(EVAL_DATA_PATH)
    df = df.dropna(subset=[INPUT_COL, TARGET_COL])
    print(f"Loaded {len(df)} samples\n")
    
    # Get texts
    original_texts = df[INPUT_COL].tolist()
    references = df[TARGET_COL].tolist()
    
    # Generate predictions
    print("Generating detoxified texts...")
    predictions = []
    for text in tqdm(original_texts, desc="  Running MT0-LoRA"):
        predictions.append(detoxify(text))
    
    # Add predictions to dataframe
    df['prediction'] = predictions
    
    # Compute metrics
    print("\nComputing metrics...")
    
    print("  Computing toxicity (original)...")
    toxicity_orig = compute_toxicity(original_texts)
    
    print("  Computing toxicity (predicted)...")
    toxicity_pred = compute_toxicity(predictions)
    
    print("  Computing similarity...")
    similarity = compute_similarity(predictions, references, LANG)
    
    print("  Computing fluency...")
    fluency = compute_fluency(predictions)
    
    # Create results
    results = {
        "model": "MT0-LoRA",
        "language": LANG,
        "samples": len(df),
        "toxicity_original": toxicity_orig,
        "toxicity_predicted": toxicity_pred,
        "toxicity_reduction": toxicity_orig - toxicity_pred,
        "similarity": similarity,
        "fluency": fluency
    }
    
    # Save results
    print("\nSaving results...")
    results_df = pd.DataFrame([results])
    results_df.to_csv(OUTPUT_CSV, index=False)
    df.to_csv(PREDICTIONS_CSV, index=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Model: MT0-LoRA")
    print(f"Language: {LANG}")
    print(f"Samples: {len(df)}")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'Value':<15}")
    print("-" * 40)
    print(f"{'Toxicity (Original)':<25} {toxicity_orig:<15.4f}")
    print(f"{'Toxicity (Predicted)':<25} {toxicity_pred:<15.4f}")
    print(f"{'Toxicity Reduction':<25} {toxicity_orig - toxicity_pred:<15.4f}")
    print(f"{'Similarity (BERTScore)':<25} {similarity:<15.4f}")
    print(f"{'Fluency':<25} {fluency:<15.4f}")
    print(f"{'='*60}")
    print(f"\nResults saved to: {OUTPUT_CSV}")
    print(f"Predictions saved to: {PREDICTIONS_CSV}")

if __name__ == "__main__":
    main()