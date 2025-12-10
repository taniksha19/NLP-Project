from datasets import load_dataset
from collections import Counter
import numpy as np
import json
import os

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
LANGUAGES = ['de', 'uk', 'ru', 'en', 'es', 'am', 'zh', 'ar', 'hi']
TOP_N_WORDS = 100  # Number of toxic words to extract per language

# Get the script's directory and construct path to results folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PARENT_DIR, 'results')

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(RESULTS_DIR, 'toxic_words_by_language.json')

print(f"Script directory: {SCRIPT_DIR}")
print(f"Results directory: {RESULTS_DIR}")
print(f"Output file: {OUTPUT_FILE}\n")

# ---------------------------------------------------------
# LOG ODDS RATIO FUNCTION
# ---------------------------------------------------------
def compute_lor(toxic_texts, nontoxic_texts, alpha=1.0):
    toxic_words = Counter()
    nontoxic_words = Counter()

    # Count words in each class
    for text in toxic_texts:
        if text:  # Handle None/empty texts
            toxic_words.update(text.lower().split())

    for text in nontoxic_texts:
        if text:  # Handle None/empty texts
            nontoxic_words.update(text.lower().split())

    # Totals
    total_toxic = sum(toxic_words.values())
    total_non = sum(nontoxic_words.values())

    if total_toxic == 0 or total_non == 0:
        return {}

    vocab = set(toxic_words.keys()) | set(nontoxic_words.keys())
    V = len(vocab)

    lor_scores = {}

    for w in vocab:
        pt = (toxic_words[w] + alpha) / (total_toxic + alpha * V)
        pn = (nontoxic_words[w] + alpha) / (total_non + alpha * V)
        lor_scores[w] = np.log(pt / pn)

    return lor_scores

# ---------------------------------------------------------
# MAIN PROCESSING
# ---------------------------------------------------------
print("="*60)
print("LOADING MULTILINGUAL TOXICITY DATASET")
print("="*60)

ds = load_dataset("textdetox/multilingual_toxicity_dataset")
print(f"Available splits: {list(ds.keys())}\n")

# Dictionary to store results
toxic_words_dict = {}

# Process each language
for lang in LANGUAGES:
    print(f"\n{'='*60}")
    print(f"Processing: {lang.upper()}")
    print(f"{'='*60}")
    
    if lang not in ds:
        print(f"❌ Language '{lang}' not found in dataset. Skipping.")
        toxic_words_dict[lang] = []
        continue
    
    try:
        # Load language data
        lang_data = ds[lang]
        
        # Extract texts and labels
        texts = lang_data['text']
        labels = lang_data['toxic']
        
        # Filter toxic vs non-toxic
        toxic_texts = [t for t, y in zip(texts, labels) if y == 1]
        nontoxic_texts = [t for t, y in zip(texts, labels) if y == 0]
        
        print(f"  Toxic samples: {len(toxic_texts)}")
        print(f"  Non-toxic samples: {len(nontoxic_texts)}")
        
        if len(toxic_texts) == 0 or len(nontoxic_texts) == 0:
            print(f"  ⚠️ Insufficient data for LOR computation")
            toxic_words_dict[lang] = []
            continue
        
        # Compute LOR scores
        lor_scores = compute_lor(toxic_texts, nontoxic_texts)
        
        if not lor_scores:
            print(f"  ⚠️ No LOR scores computed")
            toxic_words_dict[lang] = []
            continue
        
        # Get top toxic words
        top_toxic = sorted(lor_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_N_WORDS]
        toxic_words_list = [word for word, score in top_toxic]
        
        # Store in dictionary
        toxic_words_dict[lang] = toxic_words_list
        
        print(f"  ✅ Extracted {len(toxic_words_list)} toxic words")
        print(f"  Top 10: {toxic_words_list[:10]}")
        
    except Exception as e:
        print(f"  ❌ Error processing {lang}: {e}")
        toxic_words_dict[lang] = []

# ---------------------------------------------------------
# SAVE TO JSON
# ---------------------------------------------------------
print(f"\n{'='*60}")
print("SAVING RESULTS")
print(f"{'='*60}")

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(toxic_words_dict, f, ensure_ascii=False, indent=2)

print(f"✅ Saved to: {OUTPUT_FILE}")

# Display summary
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for lang, words in toxic_words_dict.items():
    print(f"{lang:5s}: {len(words):3d} words")

# Optional: Display sample from JSON
print(f"\n{'='*60}")
print("SAMPLE OUTPUT")
print(f"{'='*60}")
for lang in LANGUAGES[:3]:  # Show first 3 languages
    if toxic_words_dict.get(lang):
        print(f"\n{lang.upper()}: {toxic_words_dict[lang][:10]}")

print(f"\n✅ Complete! Check {OUTPUT_FILE}")