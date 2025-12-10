#!/usr/bin/env python3
"""
Evaluate baseline models (Duplicate and Delete) on eval datasets.
Usage: python -m src.evaluate_baseline
"""

import csv
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import evaluate

# -------------------------
# CONFIGURATION
# -------------------------

LANGUAGES = ["ar", "de", "uk", "zh"]
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
EVAL_DATA_DIR = RESULTS_DIR / "merged_data"
OUTPUT_CSV = RESULTS_DIR / "baseline_evaluation_results.csv"

# Column names in the Excel files
INPUT_COL = "toxic_sentence"
TARGET_COL = "non_toxic_sentence"

print(f"Project root: {PROJECT_ROOT}")
print(f"Eval data dir: {EVAL_DATA_DIR}")
print(f"Results dir: {RESULTS_DIR}")

# -------------------------
# LOAD STOPWORDS
# -------------------------

def load_stopwords(language: str) -> set:
    """Load multilingual toxic lexicon stopwords for a language."""
    try:
        lexicon_dataset = load_dataset("textdetox/multilingual_toxic_lexicon", split=language)
        return set([t.lower() for t in lexicon_dataset["text"]])
    except Exception as e:
        print(f"  [WARN] Could not load stopwords for {language}: {e}")
        return set()

# -------------------------
# BASELINES
# -------------------------

class DuplicateBaseline:
    """Returns text as-is"""
    name = "Duplicate"
    
    def detoxify(self, text, lang=None):
        return text

class DeleteBaseline:
    """Deletes words that match toxic stopwords"""
    name = "Delete"
    
    def __init__(self):
        self.stopwords_cache = {}
    
    def get_stopwords(self, lang: str) -> set:
        if lang not in self.stopwords_cache:
            self.stopwords_cache[lang] = load_stopwords(lang)
        return self.stopwords_cache[lang]
    
    def detoxify(self, text, lang=None):
        if lang is None or not isinstance(text, str):
            return text if isinstance(text, str) else ""
        
        stopwords = self.get_stopwords(lang)
        if not stopwords:
            return text
            
        words = text.split()
        safe_words = []
        for w in words:
            w_lower = w.lower()
            # Check if word or cleaned version is in stopwords
            cleaned = ''.join(c for c in w_lower if c.isalnum())
            if w_lower not in stopwords and cleaned not in stopwords:
                safe_words.append(w)
        return " ".join(safe_words) if safe_words else ""

# -------------------------
# DEVICE SETUP
# -------------------------

def get_device():
    """Get the best available device."""
    if torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            print("✅ Using Apple Silicon GPU (MPS)")
            return torch.device("mps")
    if torch.cuda.is_available():
        print(f"✅ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("⚠️ Using CPU")
    return torch.device("cpu")

device = get_device()

# -------------------------
# LOAD METRICS (GLOBAL)
# -------------------------

print("\nLoading evaluation models...")

# Toxicity - XLM-R Large (supports Amharic, Russian, etc.)
print("  Loading XLM-R toxicity classifier...")
tox_tokenizer = AutoTokenizer.from_pretrained("textdetox/xlmr-large-toxicity-classifier")
tox_model = AutoModelForSequenceClassification.from_pretrained(
    "textdetox/xlmr-large-toxicity-classifier",
    torch_dtype=torch.float32
).to(device)
tox_model.eval()

# Semantic Similarity
print("  Loading BERTScore...")
bertscore_metric = evaluate.load("bertscore")

# Fluency
print("  Loading GPT-2...")
lm_device = "cpu" if device.type == "mps" else device  # GPT-2 can have issues on MPS
tokenizer = AutoTokenizer.from_pretrained("gpt2")
lm_model = AutoModelForCausalLM.from_pretrained("gpt2").to(lm_device)
lm_model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"✅ Models loaded!")
print(f"   XLM-R Toxicity: {device}")
print(f"   GPT-2 Fluency: {lm_device}")

# -------------------------
# METRIC FUNCTIONS
# -------------------------

def compute_toxicity(predictions: list) -> float:
    """Compute average toxicity score using XLM-R Large (supports Amharic/Russian)."""
    scores = []
    BATCH_SIZE_EVAL = 16

    for i in tqdm(range(0, len(predictions), BATCH_SIZE_EVAL), desc="    Toxicity", leave=False):
        batch_texts = predictions[i:i + BATCH_SIZE_EVAL]

        # Handle empty/nan strings
        clean_batch = [str(t) if t and str(t).strip() else "." for t in batch_texts]

        inputs = tox_tokenizer(
            clean_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = tox_model(**inputs)
            # Apply softmax to get probabilities
            probs = torch.softmax(outputs.logits, dim=-1)
            # The second column (index 1) is the "Toxic" probability
            batch_scores = probs[:, 1].cpu().tolist()
            scores.extend(batch_scores)
        
        # Clear cache periodically
        if device.type == "mps":
            torch.mps.empty_cache()

    return float(np.mean(scores)) if scores else 0.0

def compute_similarity(predictions: list, references: list, lang: str) -> float:
    """Compute average BERTScore F1."""
    # Handle empty strings
    preds_clean = [str(p) if p and str(p).strip() else "." for p in predictions]
    refs_clean = [str(r) if r and str(r).strip() else "." for r in references]
    
    # Use appropriate language model for BERTScore
    bert_lang = lang if lang in ["en", "de", "fr", "es", "ru", "zh"] else "en"
    
    bert_results = bertscore_metric.compute(
        predictions=preds_clean,
        references=refs_clean,
        lang=bert_lang,
        device="cpu" if device.type == "mps" else str(device)
    )
    return float(np.mean(bert_results["f1"]))

def compute_fluency(texts: list) -> float:
    """Compute average fluency (negative log-likelihood)."""
    scores = []
    with torch.no_grad():
        for txt in tqdm(texts, desc="    Fluency", leave=False):
            if not txt or not str(txt).strip():
                scores.append(float('-inf'))
                continue
            try:
                enc = tokenizer(str(txt), return_tensors="pt", truncation=True, 
                               max_length=512, padding=True).to(lm_device)
                outputs = lm_model(**enc, labels=enc["input_ids"])
                loss = outputs.loss.item()
                scores.append(-loss)
            except Exception:
                scores.append(float('-inf'))
    
    # Filter out -inf values for mean calculation
    valid_scores = [s for s in scores if s != float('-inf')]
    return float(np.mean(valid_scores)) if valid_scores else float('-inf')

# -------------------------
# LOAD DATA
# -------------------------

def load_eval_data(lang: str) -> tuple:
    """Load eval data from Excel file for a specific language."""
    eval_file = EVAL_DATA_DIR / f"eval_{lang}.xlsx"
    
    if not eval_file.exists():
        print(f"  [WARN] File not found: {eval_file}")
        return [], []
    
    try:
        df = pd.read_excel(eval_file)
        
        # Check if required columns exist
        if INPUT_COL not in df.columns or TARGET_COL not in df.columns:
            print(f"  [WARN] Required columns not found in {eval_file}")
            print(f"         Available columns: {list(df.columns)}")
            return [], []
        
        # Drop rows with NaN values
        df = df.dropna(subset=[INPUT_COL, TARGET_COL])
        
        original_texts = df[INPUT_COL].tolist()
        references = df[TARGET_COL].tolist()
        
        return original_texts, references
        
    except Exception as e:
        print(f"  [ERROR] Failed to load {eval_file}: {e}")
        return [], []

# -------------------------
# EVALUATION
# -------------------------

def evaluate_baseline(name: str, predictions: list, references: list, lang: str) -> dict:
    """Evaluate a baseline model."""
    print(f"  Evaluating {name}...")
    
    toxicity = compute_toxicity(predictions)
    similarity = compute_similarity(predictions, references, lang)
    fluency = compute_fluency(predictions)
    
    return {
        "toxicity": toxicity,
        "similarity": similarity,
        "fluency": fluency
    }

def run_evaluation_for_language(lang: str, baselines: list) -> dict:
    """Run all baselines for a single language."""
    print(f"\n{'='*60}")
    print(f"Processing: {lang.upper()}")
    print(f"{'='*60}")
    
    original_texts, references = load_eval_data(lang)
    
    if not original_texts:
        print(f"  [SKIP] No data for {lang}")
        return {}
    
    print(f"  Loaded {len(original_texts)} eval samples")
    
    # Show sample
    print(f"  Sample toxic: {str(original_texts[0])[:60]}...")
    print(f"  Sample clean: {str(references[0])[:60]}...")
    
    results = {}
    
    for baseline in baselines:
        # Generate predictions
        predictions = []
        for text in tqdm(original_texts, desc=f"  Running {baseline.name}", leave=False):
            predictions.append(baseline.detoxify(text, lang))
        
        # Evaluate
        metrics = evaluate_baseline(baseline.name, predictions, references, lang)
        results[baseline.name] = metrics
        
        print(f"    {baseline.name}: Tox={metrics['toxicity']:.4f}, "
              f"SIM={metrics['similarity']:.4f}, Flu={metrics['fluency']:.4f}")
    
    return results

# -------------------------
# MAIN
# -------------------------

def main():
    # Initialize baselines
    baselines = [DuplicateBaseline(), DeleteBaseline()]
    
    # Store all results
    all_results = []
    
    # Check if eval data directory exists
    if not EVAL_DATA_DIR.exists():
        print(f"[ERROR] Eval data directory not found: {EVAL_DATA_DIR}")
        print("Please ensure the directory exists with eval_<lang>.xlsx files")
        return
    
    # List available files
    print(f"\nLooking for eval files in: {EVAL_DATA_DIR}")
    available_files = list(EVAL_DATA_DIR.glob("eval_*.xlsx"))
    print(f"Found files: {[f.name for f in available_files]}")
    
    # Process each language
    for lang in LANGUAGES:
        lang_results = run_evaluation_for_language(lang, baselines)
        
        for baseline_name, metrics in lang_results.items():
            all_results.append({
                "language": lang,
                "model": baseline_name,
                "toxicity": metrics["toxicity"],
                "similarity": metrics["similarity"],
                "fluency": metrics["fluency"]
            })
    
    if not all_results:
        print("\n[ERROR] No results to save. Check your data files.")
        return
    
    # Save to CSV
    print(f"\n{'='*60}")
    print("Saving results to CSV...")
    print(f"{'='*60}")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["language", "model", "toxicity", "similarity", "fluency"])
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"Results saved to: {OUTPUT_CSV}")
    
    # Print summary table
    print(f"\n{'='*60}")
    print("FINAL SUMMARY - BASELINE EVALUATION ON EVAL SET")
    print(f"{'='*60}")
    print(f"{'Lang':<6} {'Model':<12} {'Toxicity↓':<12} {'SIM↑':<12} {'Fluency↑':<12}")
    print("-" * 60)
    
    for row in all_results:
        print(f"{row['language']:<6} {row['model']:<12} {row['toxicity']:<12.4f} "
              f"{row['similarity']:<12.4f} {row['fluency']:<12.4f}")
    
    # Print averages per baseline
    print("-" * 60)
    for baseline in baselines:
        baseline_rows = [r for r in all_results if r['model'] == baseline.name]
        if baseline_rows:
            avg_tox = np.mean([r['toxicity'] for r in baseline_rows])
            avg_sim = np.mean([r['similarity'] for r in baseline_rows])
            avg_flu = np.mean([r['fluency'] for r in baseline_rows])
            print(f"{'AVG':<6} {baseline.name:<12} {avg_tox:<12.4f} {avg_sim:<12.4f} {avg_flu:<12.4f}")

if __name__ == "__main__":
    main()