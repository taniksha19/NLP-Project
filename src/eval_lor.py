# ============================================================
# CELL 1: Install dependencies
# ============================================================
# pip install transformers sentence-transformers torch tqdm datasets detoxify evaluate bert_score openpyxl

# ============================================================
# CELL 2: Imports
# ============================================================
import json
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    XLMRobertaTokenizer,
    XLMRobertaForMaskedLM,
)
from sentence_transformers import SentenceTransformer, util
import evaluate

# ============================================================
# CELL 3: Device Setup (MPS-optimized)
# ============================================================
def get_device():
    """Get the best available device with MPS support."""
    if torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            print("✅ Using Apple Silicon GPU (MPS)")
            return torch.device("mps")
        else:
            print("⚠️ MPS available but not built, using CPU")
            return torch.device("cpu")
    elif torch.cuda.is_available():
        print(f"✅ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("⚠️ No GPU detected, using CPU")
        return torch.device("cpu")

device = get_device()

# MPS-specific settings
if device.type == "mps":
    import os
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ============================================================
# CELL 4: Configuration
# ============================================================
CONFIG = {
    "results_dir": "../results/merged_data",
    "toxic_words_path": "./results/toxic_words_by_language.json",
    "output_dir": "./results/detox_output_improved",
    "languages": ["ar", "de", "uk", "zh"],
    "max_samples": None,
}

Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)

# ============================================================
# CELL 5: Helper - Load Excel Eval Data
# ============================================================
def load_eval_excel(lang: str) -> pd.DataFrame:
    """Load evaluation data from Excel file for a language."""
    excel_path = Path(CONFIG["results_dir"]) / f"eval_{lang}.xlsx"

    if not excel_path.exists():
        print(f"  [WARN] File not found: {excel_path}")
        return None

    df = pd.read_excel(excel_path)
    print(f"  Loaded {len(df)} rows from {excel_path.name}")
    print(f"  Columns: {df.columns.tolist()}")
    return df

def detect_columns(df: pd.DataFrame) -> tuple:
    """Auto-detect toxic and reference columns."""
    toxic_col, ref_col = None, None

    for col in df.columns:
        col_lower = col.lower()
        if toxic_col is None and 'toxic' in col_lower and 'non' not in col_lower:
            toxic_col = col
        elif ref_col is None and ('non_toxic' in col_lower or 'neutral' in col_lower or 'reference' in col_lower):
            ref_col = col

    if toxic_col is None:
        for candidate in ['toxic_sentence', 'toxic', 'text', 'input', 'source']:
            if candidate in df.columns:
                toxic_col = candidate
                break

    if ref_col is None:
        for candidate in ['non_toxic_sentence', 'neutral_sentence', 'reference', 'target']:
            if candidate in df.columns:
                ref_col = candidate
                break

    return toxic_col, ref_col

# ============================================================
# CELL 6: Load Toxic Words (Custom + HuggingFace Lexicon)
# ============================================================
print("Loading toxic words...")

toxic_words_path = Path(CONFIG["toxic_words_path"])
if toxic_words_path.exists():
    with open(toxic_words_path, "r", encoding="utf-8") as f:
        toxic_dict = json.load(f)
    toxic_words_by_lang = {
        lang: set(w.lower() for w in words)
        for lang, words in toxic_dict.items()
    }
    print(f"  Loaded custom toxic words from {toxic_words_path}")
else:
    toxic_words_by_lang = {}
    print(f"  [WARN] Custom toxic words not found at {toxic_words_path}")

print("  Loading textdetox/multilingual_toxic_lexicon from HuggingFace...")
try:
    toxic_lexicon_dataset = load_dataset("textdetox/multilingual_toxic_lexicon")
    for lang in toxic_lexicon_dataset.keys():
        hf_words = set(w.lower() for w in toxic_lexicon_dataset[lang]["text"])
        if lang in toxic_words_by_lang:
            toxic_words_by_lang[lang] = toxic_words_by_lang[lang] | hf_words
        else:
            toxic_words_by_lang[lang] = hf_words
    print("  ✅ Merged HuggingFace toxic lexicon")
except Exception as e:
    print(f"  [WARN] Could not load HF lexicon: {e}")

print("\nToxic words loaded:")
for lang, words in toxic_words_by_lang.items():
    print(f"  {lang}: {len(words)} words")

# ============================================================
# CELL 7: Masking Function
# ============================================================
def mask_toxic_words(text: str, lang: str, mask_token: str = "<mask>"):
    """Mask toxic words in text based on language."""
    if lang not in toxic_words_by_lang:
        return text, 0

    toxic_set = toxic_words_by_lang[lang]
    tokens = text.split()
    masked_tokens = []
    num_masks = 0

    for tok in tokens:
        clean = re.sub(r"[^\w']", "", tok.lower())
        if clean in toxic_set:
            masked_tokens.append(mask_token)
            num_masks += 1
        else:
            masked_tokens.append(tok)

    return " ".join(masked_tokens), num_masks

test_text = "You are an idiot and a fool"
masked, n = mask_toxic_words(test_text, "en")
print(f"\nTest masking:")
print(f"  Original: {test_text}")
print(f"  Masked: {masked} ({n} masks)")

# ============================================================
# CELL 8: Load Models (MPS-compatible)
# ============================================================
print("\nLoading models...")

print("  Loading XLM-RoBERTa...")
xlm_tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
xlm_model = XLMRobertaForMaskedLM.from_pretrained(
    "xlm-roberta-base",
    torch_dtype=torch.float32
).to(device)
xlm_model.eval()

print("  Loading LaBSE for semantic similarity...")
sim_model = SentenceTransformer("sentence-transformers/LaBSE")
if device.type == "mps":
    sim_model = sim_model.to("cpu")
    sim_device = "cpu"
else:
    sim_device = device

print("  Loading XLM-R toxicity classifier...")
from transformers import AutoModelForSequenceClassification
tox_tokenizer = AutoTokenizer.from_pretrained("textdetox/xlmr-large-toxicity-classifier")
tox_model = AutoModelForSequenceClassification.from_pretrained(
    "textdetox/xlmr-large-toxicity-classifier",
    torch_dtype=torch.float32
).to(device)
tox_model.eval()

print(f"✅ All models loaded!")
print(f"   XLM-RoBERTa (mask-fill): {device}")
print(f"   XLM-R (toxicity): {device}")
print(f"   LaBSE: {sim_device if device.type == 'mps' else device}")

# ============================================================
# CELL 9: Improved Mask Filling Function (MPS-compatible)
# ============================================================
def fill_masks_improved(
    masked_text: str,
    original_text: str,
    lang: str = "en",
    top_k: int = 30,
    max_candidates: int = 5,
    toxicity_threshold: float = 0.5,
) -> str:
    """Improved mask filling with MPS compatibility."""
    mask_token = xlm_tokenizer.mask_token
    result_text = masked_text.replace("<mask>", mask_token)

    toxic_set = toxic_words_by_lang.get(lang, set())
    
    original_embedding = sim_model.encode(original_text, convert_to_tensor=True)
    if device.type == "mps":
        original_embedding = original_embedding.cpu()

    iteration = 0
    max_iterations = masked_text.count("<mask>") + 5

    while mask_token in result_text and iteration < max_iterations:
        iteration += 1

        inputs = xlm_tokenizer(result_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = xlm_model(**inputs)

        mask_idx = (inputs["input_ids"] == xlm_tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        if len(mask_idx) == 0:
            break

        first_mask_idx = mask_idx[0].item()
        logits = outputs.logits[0, first_mask_idx]
        
        if device.type == "mps":
            logits = logits.cpu()
        
        top_k_values, top_k_indices = torch.topk(logits, top_k)

        candidates = []
        for idx in top_k_indices:
            token = xlm_tokenizer.decode([idx.item()]).strip()
            token_lower = token.lower()

            if not token or len(token) < 2:
                continue
            if not any(c.isalnum() for c in token):
                continue
            if token_lower in toxic_set:
                continue

            candidate_text = result_text.replace(mask_token, token, 1)

            try:
                # Use XLM-R toxicity classifier
                tox_inputs = tox_tokenizer(
                    candidate_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(device)
                with torch.no_grad():
                    tox_outputs = tox_model(**tox_inputs)
                    tox_probs = torch.softmax(tox_outputs.logits, dim=-1)
                    sentence_tox = tox_probs[0, 1].item()  # Toxic probability
                if sentence_tox > toxicity_threshold:
                    continue
            except:
                pass

            candidates.append({'token': token, 'text': candidate_text})
            if len(candidates) >= max_candidates:
                break

        if not candidates:
            for idx in top_k_indices:
                token = xlm_tokenizer.decode([idx.item()]).strip()
                if token and len(token) >= 2 and token.lower() not in toxic_set:
                    candidates = [{'token': token, 'text': result_text.replace(mask_token, token, 1)}]
                    break

        if not candidates:
            token = xlm_tokenizer.decode([top_k_indices[0].item()]).strip()
            result_text = result_text.replace(mask_token, token, 1)
            continue

        best_candidate = None
        best_similarity = -1

        for cand in candidates:
            cand_embedding = sim_model.encode(cand['text'], convert_to_tensor=True)
            if device.type == "mps":
                cand_embedding = cand_embedding.cpu()
            similarity = float(util.cos_sim(original_embedding, cand_embedding).item())
            if similarity > best_similarity:
                best_similarity = similarity
                best_candidate = cand

        result_text = best_candidate['text']

    return result_text

print(f"\nTesting improved fill:")
print(f"  Filled: {fill_masks_improved(masked, test_text, 'en')}")

# ============================================================
# CELL 10: Load Evaluation Models (MPS-compatible)
# ============================================================
print("\nLoading evaluation models...")

print("  Loading BERTScore...")
bertscore_metric = evaluate.load("bertscore")

print("  Loading GPT-2 for fluency...")
lm_tokenizer = AutoTokenizer.from_pretrained("gpt2")

lm_device = "cpu" if device.type == "mps" else device
lm_model = AutoModelForCausalLM.from_pretrained("gpt2").to(lm_device)
lm_model.eval()

if lm_tokenizer.pad_token is None:
    lm_tokenizer.pad_token = lm_tokenizer.eos_token

print(f"✅ Evaluation models loaded!")
print(f"   GPT-2 Fluency: {lm_device}")

# ============================================================
# CELL 11: Evaluation Functions (MPS-compatible)
# ============================================================
def compute_toxicity(predictions: list) -> tuple:
    """Compute toxicity scores using XLM-R Large (supports Amharic/Russian)."""
    scores = []
    BATCH_SIZE_EVAL = 16

    for i in tqdm(range(0, len(predictions), BATCH_SIZE_EVAL), desc="  Computing toxicity", leave=False):
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

    return float(np.mean(scores)), scores

def compute_similarity(predictions: list, references: list, lang: str = "en") -> tuple:
    """Compute BERTScore F1 between predictions and references."""
    preds_clean = [str(p) if p and str(p).strip() else "." for p in predictions]
    refs_clean = [str(r) if r and str(r).strip() else "." for r in references]

    bert_lang = lang if lang in ["am", "ar", "en", "de", "uk", "es", "ru", "zh"] else "en"

    print(f"  Computing BERTScore (lang={bert_lang})...")
    bert_results = bertscore_metric.compute(
        predictions=preds_clean,
        references=refs_clean,
        lang=bert_lang,
        device="cpu" if device.type == "mps" else str(device)
    )
    f1_scores = bert_results["f1"]
    return np.mean(f1_scores), f1_scores

def compute_fluency(texts: list) -> tuple:
    """Compute fluency as negative log-likelihood."""
    scores = []
    with torch.no_grad():
        for txt in tqdm(texts, desc="  Computing fluency", leave=False):
            if not txt or not str(txt).strip():
                scores.append(float('-inf'))
                continue
            try:
                enc = lm_tokenizer(str(txt), return_tensors="pt", truncation=True, max_length=512, padding=True)
                enc = {k: v.to(lm_device) for k, v in enc.items()}
                outputs = lm_model(**enc, labels=enc["input_ids"])
                loss = outputs.loss.item()
                scores.append(-loss)
            except Exception as e:
                scores.append(float('-inf'))
    
    valid_scores = [s for s in scores if s != float('-inf')]
    return np.mean(valid_scores) if valid_scores else float('-inf'), scores

# ============================================================
# Helper function for safe fluency formatting
# ============================================================
def format_fluency(value):
    """Safely format fluency value."""
    if value is None or value == float('-inf'):
        return "N/A"
    return f"{value:.4f}"

# ============================================================
# CELL 12: Main Pipeline for One Language
# ============================================================
def run_detox_for_language(lang: str, max_samples=None):
    """Run improved detox pipeline for one language using Excel data."""
    print(f"\n{'='*60}")
    print(f"Processing: {lang}")
    print(f"{'='*60}")

    df = load_eval_excel(lang)
    if df is None or len(df) == 0:
        print(f"  [SKIP] No data found for {lang}")
        return None

    toxic_col, ref_col = detect_columns(df)

    if toxic_col is None:
        print(f"  [ERROR] Could not find toxic text column in {df.columns.tolist()}")
        return None

    print(f"  Using toxic column: '{toxic_col}'")
    print(f"  Using reference column: '{ref_col}'" if ref_col else "  No reference column found")

    if max_samples and len(df) > max_samples:
        df = df.head(max_samples)
        print(f"  Limited to {max_samples} samples")

    original_texts, reference_texts, detoxed_texts = [], [], []
    masked_texts, num_masks_list = [], []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  Detoxifying {lang}"):
        toxic_text = str(row.get(toxic_col, "")) if pd.notna(row.get(toxic_col)) else ""
        ref_text = str(row.get(ref_col, "")) if ref_col and pd.notna(row.get(ref_col)) else ""

        if not toxic_text.strip():
            continue

        masked_text, num_masks = mask_toxic_words(toxic_text, lang)

        if num_masks > 0:
            detoxed_text = fill_masks_improved(masked_text, toxic_text, lang=lang)
        else:
            detoxed_text = toxic_text

        original_texts.append(toxic_text)
        reference_texts.append(ref_text if ref_text else toxic_text)
        masked_texts.append(masked_text)
        detoxed_texts.append(detoxed_text)
        num_masks_list.append(num_masks)
        
        if device.type == "mps" and idx % 50 == 0:
            torch.mps.empty_cache()

    n = len(original_texts)
    if n == 0:
        print(f"  [WARN] No valid samples for {lang}")
        return None

    print(f"\n  Evaluating {n} samples...")

    print("  Computing toxicity (original)...")
    tox_orig_mean, tox_orig_scores = compute_toxicity(original_texts)

    print("  Computing toxicity (detoxed)...")
    tox_detox_mean, tox_detox_scores = compute_toxicity(detoxed_texts)

    sim_mean, sim_scores = compute_similarity(detoxed_texts, reference_texts, lang=lang)
    flu_mean, flu_scores = compute_fluency(detoxed_texts)

    samples = []
    for i in range(n):
        flu_val = flu_scores[i] if flu_scores[i] != float('-inf') else None
        samples.append({
            "original": original_texts[i],
            "reference": reference_texts[i],
            "masked": masked_texts[i],
            "detoxed": detoxed_texts[i],
            "num_masks": num_masks_list[i],
            "toxicity_orig": float(tox_orig_scores[i]),
            "toxicity": float(tox_detox_scores[i]),
            "similarity": float(sim_scores[i]),
            "fluency": float(flu_val) if flu_val is not None else None,
        })

    # Handle fluency mean
    flu_mean_val = flu_mean if flu_mean != float('-inf') else None

    metrics = {
        "toxicity_orig": float(tox_orig_mean),
        "toxicity": float(tox_detox_mean),
        "toxicity_reduction": float(tox_orig_mean - tox_detox_mean),
        "similarity": float(sim_mean),
        "fluency": float(flu_mean_val) if flu_mean_val is not None else None,
        "total_samples": n,
        "samples_masked": sum(1 for m in num_masks_list if m > 0),
    }

    print(f"\n  [{lang}] Results (n={n}):")
    print(f"    Toxicity (orig):    {metrics['toxicity_orig']:.4f}")
    print(f"    Toxicity (detox):   {metrics['toxicity']:.4f} (↓ Lower is better)")
    print(f"    Toxicity Reduction: {metrics['toxicity_reduction']:.4f}")
    print(f"    Similarity (SIM):   {metrics['similarity']:.4f} (↑ Higher is better)")
    print(f"    Fluency:            {format_fluency(metrics['fluency'])} (↑ Higher is better)")
    print(f"    Samples masked:     {metrics['samples_masked']}/{n}")

    return {"samples": samples, "metrics": metrics}

# ============================================================
# CELL 13: Check Available Eval Files
# ============================================================
print("\nChecking for evaluation files...")
results_dir = Path(CONFIG["results_dir"])

print(results_dir)
available_langs = []
for lang in CONFIG["languages"]:
    eval_file = results_dir / f"eval_{lang}.xlsx"
    if eval_file.exists():
        available_langs.append(lang)
        print(f"  ✅ Found: eval_{lang}.xlsx")
    else:
        print(f"  ❌ Missing: eval_{lang}.xlsx")

print(f"\nAvailable languages: {available_langs}")

# ============================================================
# CELL 14: Run Pipeline for All Available Languages
# ============================================================
all_results = {}
summary = {}

for lang in available_langs:
    if lang not in toxic_words_by_lang:
        print(f"\n[WARN] No toxic words for {lang}, skipping...")
        continue

    result = run_detox_for_language(lang, max_samples=CONFIG["max_samples"])

    if result:
        all_results[lang] = result
        summary[lang] = result["metrics"]

        out_file = Path(CONFIG["output_dir"]) / f"detox_results_{lang}.json"
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"  Saved: {out_file}")

        csv_file = Path(CONFIG["output_dir"]) / f"detox_predictions_{lang}.csv"
        pd.DataFrame(result["samples"]).to_csv(csv_file, index=False)
        print(f"  Saved: {csv_file}")

        summary_file = Path(CONFIG["output_dir"]) / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
    
    if device.type == "mps":
        torch.mps.empty_cache()

# ============================================================
# CELL 15: Final Summary
# ============================================================
print("\n" + "="*70)
print("FINAL SUMMARY - IMPROVED MASK FILLING (MPS)")
print("="*70)

if summary:
    print(f"{'Lang':<6} {'Tox_Orig':<10} {'Tox_Det':<10} {'Tox_Red↓':<10} {'SIM↑':<10} {'Fluency↑':<10} {'Masked':<10}")
    print("-" * 70)
    for lang, m in summary.items():
        flu_str = format_fluency(m['fluency'])
        print(f"{lang:<6} {m['toxicity_orig']:<10.4f} {m['toxicity']:<10.4f} {m['toxicity_reduction']:<10.4f} {m['similarity']:<10.4f} {flu_str:<10} {m['samples_masked']}/{m['total_samples']}")

    summary_csv = Path(CONFIG["output_dir"]) / "evaluation_summary.csv"
    summary_rows = []
    for lang, m in summary.items():
        summary_rows.append({
            "language": lang,
            "model": "Unsupervised (Mask+Fill)",
            "toxicity_orig": m["toxicity_orig"],
            "toxicity": m["toxicity"],
            "toxicity_reduction": m["toxicity_reduction"],
            "similarity": m["similarity"],
            "fluency": m["fluency"],
            "total_samples": m["total_samples"],
            "samples_masked": m["samples_masked"],
        })
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"\nSummary saved to: {summary_csv}")
else:
    print("No results to summarize. Check that eval files exist.")

# ============================================================
# CELL 16: View Sample Results
# ============================================================
if all_results:
    first_lang = list(all_results.keys())[0]
    print(f"\n{'='*60}")
    print(f"Sample detoxifications ({first_lang}):")
    print("="*60)

    for i, sample in enumerate(all_results[first_lang]["samples"][:5]):
        print(f"\n--- Example {i+1} ---")
        print(f"🔴 Original:  {sample['original']}")
        print(f"🟡 Masked:    {sample['masked']}")
        print(f"🟢 Detoxed:   {sample['detoxed']}")
        if sample['reference'] != sample['original']:
            print(f"🔵 Reference: {sample['reference']}")
        tox_delta = sample['toxicity_orig'] - sample['toxicity']
        print(f"   Toxicity:  {sample['toxicity_orig']:.3f} → {sample['toxicity']:.3f} (Δ = {tox_delta:.3f})")
        flu_str = format_fluency(sample['fluency'])
        print(f"   SIM: {sample['similarity']:.3f} | Fluency: {flu_str}")