#!/usr/bin/env python3

import json
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from detoxify import Detoxify
import evaluate

# -------------------------
# CONFIGURATION
# -------------------------

INPUT_FILE = "results/es_baseline_output.jsonl"  # Path to your JSONL file
LANG = "en"

# -------------------------
# LOAD STOPWORDS
# -------------------------

def load_stopwords(language="en"):
    """Load multilingual toxic lexicon stopwords"""
    lexicon_dataset = load_dataset("textdetox/multilingual_toxic_lexicon", split=language)
    return set([t.lower() for t in lexicon_dataset["text"]])

stopwords = load_stopwords(LANG)

# -------------------------
# BASELINES
# -------------------------

class DuplicateBaseline:
    """Returns text as-is"""
    name = "Duplicate (Identity)"
    def detoxify(self, text, lang=None):
        return text

class DeleteBaseline:
    """Deletes words that match toxic stopwords"""
    name = "Lexicon-based Deletion"
    def __init__(self, stopwords):
        self.stopwords = stopwords
    def detoxify(self, text, lang=None):
        words = text.split()
        safe_words = []
        for w in words:
            if w.lower() not in self.stopwords:
                cleaned = re.sub(r'[^\w\s]', '', w.lower())
                if cleaned not in self.stopwords:
                    safe_words.append(w)
        return " ".join(safe_words) if safe_words else ""

baselines = [DuplicateBaseline(), DeleteBaseline(stopwords)]

# -------------------------
# LOAD DATA
# -------------------------

original_texts = []
references = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]
    for item in data:
        original_texts.append(item["text"])
        references.append(item.get("reference", ""))

# -------------------------
# LOAD METRICS
# -------------------------

# Toxicity
detox_model = Detoxify('original')  # predicts toxicity

def compute_toxicity(predictions):
    scores = []
    for txt in predictions:
        score = detox_model.predict(txt)['toxicity']
        scores.append(score)
    return np.mean(scores)

# Semantic Similarity
bertscore_metric = evaluate.load("bertscore")

# Fluency
device = "cuda" if torch.cuda.is_available() else "cpu"
lm_model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
lm_model = AutoModelForCausalLM.from_pretrained(lm_model_name).to(device)
lm_model.eval()

# Fix for padding issue
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def compute_fluency(texts):
    """Compute average negative log-likelihood (perplexity proxy)"""
    scores = []
    with torch.no_grad():
        for txt in texts:
            enc = tokenizer(txt, return_tensors="pt", truncation=True, padding=True).to(device)
            outputs = lm_model(**enc, labels=enc["input_ids"])
            loss = outputs.loss.item()
            scores.append(-loss)  # higher is better (less perplexity)
    return np.mean(scores)

# -------------------------
# EVALUATION FUNCTION
# -------------------------

def evaluate_baseline(name, predictions, references, inputs):
    print(f"\n{'-'*50}\nEvaluating: {name}\n{'-'*50}")

    # STA (toxicity)
    avg_toxicity = compute_toxicity(predictions)
    print(f"Style Transfer Accuracy (toxicity): {avg_toxicity:.4f} (Lower is better)")

    # SIM (BERTScore vs Reference)
    bert_ref = bertscore_metric.compute(predictions=predictions, references=references, lang="en")
    avg_f1_ref = np.mean(bert_ref["f1"])
    print(f"Semantic Similarity (SIM) vs Reference: {avg_f1_ref:.4f} (Higher is better)")

    # Fluency
    avg_fluency = compute_fluency(predictions)
    print(f"Fluency (LM-based): {avg_fluency:.4f} (Higher is better)")

    return {
        "toxicity": avg_toxicity,
        "sim": avg_f1_ref,
        "fluency": avg_fluency
    }

# -------------------------
# RUN BASELINES
# -------------------------

model_outputs = {model.name: [] for model in baselines}

for text in tqdm(original_texts, desc="Running baselines"):
    for model in baselines:
        model_outputs[model.name].append(model.detoxify(text, LANG))

# -------------------------
# EVALUATE
# -------------------------

all_results = {}
for name, outputs in model_outputs.items():
    results = evaluate_baseline(name, outputs, references, original_texts)
    all_results[name] = results

print("\nAll results:", all_results)