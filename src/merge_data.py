import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import gc
import os

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_NAME = "s-nlp/mt0-xl-detox-orpo"
BATCH_SIZE = 4  # Reduced for MPS stability
OUTPUT_DIR = "../results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

ALL_LANGUAGES = ['hi']
PARADETOX_LANGS = ['hi']

# ==========================================
# DEVICE SELECTION (MPS PRIORITY)
# ==========================================
def get_device():
    """Select best available device: MPS > CUDA > CPU"""
    if torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            print("✓ Using MPS (Apple Silicon)")
            return "mps"
        else:
            print("⚠ MPS available but not built, falling back to CPU")
    if torch.cuda.is_available():
        print(f"✓ Using CUDA: {torch.cuda.get_device_name(0)}")
        return "cuda"
    print("⚠ Using CPU - this will be slow")
    return "cpu"

# ==========================================
# PART 1: LOAD & MERGE DATASETS FOR ONE LANGUAGE
# ==========================================
def prepare_dataset_for_language(lang):
    print(f"\n{'='*60}")
    print(f"Processing Language: {lang.upper()}")
    print(f"{'='*60}")
    
    all_dfs = []
    
    if lang in PARADETOX_LANGS:
        try:
            print(f"Loading paradetox data for {lang}...")
            ds_lang = load_dataset("textdetox/multilingual_paradetox", split=lang)
            df_lang = ds_lang.to_pandas()
            df_lang['lang'] = lang
            
            rename_map = {}
            if "toxic_sentence" in df_lang.columns:
                rename_map["toxic_sentence"] = "toxic_sentence"
            if "neutral_sentence" in df_lang.columns:
                rename_map["neutral_sentence"] = "non_toxic_sentence"
            elif "non_toxic_sentence" not in df_lang.columns and "neutral_sentence" not in df_lang.columns:
                for col in df_lang.columns:
                    if 'neutral' in col.lower() or 'detox' in col.lower():
                        rename_map[col] = "non_toxic_sentence"
                        break
            
            if rename_map:
                df_lang = df_lang.rename(columns=rename_map)
            
            if 'toxic_sentence' in df_lang.columns and 'non_toxic_sentence' in df_lang.columns:
                df_lang = df_lang[['toxic_sentence', 'non_toxic_sentence', 'lang']]
                all_dfs.append(df_lang)
                print(f"  ✓ Loaded {len(df_lang)} paradetox rows")
            else:
                print(f"  ⚠ Skipping paradetox: missing required columns")
                
        except Exception as e:
            print(f"  ⚠ Could not load paradetox: {e}")
    else:
        print(f"  ℹ No paradetox data available for {lang}")
    
    try:
        print(f"Loading toxicity data for {lang}...")
        ds_tox_lang = load_dataset("textdetox/multilingual_toxicity_dataset", split=lang)
        df_tox_lang = ds_tox_lang.to_pandas()
        df_tox_lang['lang'] = lang
        
        if 'is_toxic' in df_tox_lang.columns:
            df_tox_lang = df_tox_lang[df_tox_lang['is_toxic'] == 1].copy()
        elif 'toxic' in df_tox_lang.columns:
            df_tox_lang = df_tox_lang[df_tox_lang['toxic'] == 1].copy()
        
        text_col_found = False
        for col in ['text', 'comment_text', 'sentence', 'content', 'toxic_comment']:
            if col in df_tox_lang.columns:
                df_tox_lang = df_tox_lang.rename(columns={col: "toxic_sentence"})
                text_col_found = True
                break
        
        if not text_col_found:
            print(f"  ⚠ No text column found in toxicity data")
        else:
            df_tox_lang['non_toxic_sentence'] = None
            df_tox_lang = df_tox_lang[['toxic_sentence', 'non_toxic_sentence', 'lang']]
            all_dfs.append(df_tox_lang)
            print(f"  ✓ Loaded {len(df_tox_lang)} toxicity rows")
        
    except Exception as e:
        print(f"  ⚠ Could not load toxicity data: {e}")
    
    if not all_dfs:
        print(f"  ❌ No data loaded for {lang}")
        return None
    
    merged_df = pd.concat(all_dfs, ignore_index=True)
    print(f"  Rows before deduplication: {len(merged_df)}")
    merged_df = merged_df.drop_duplicates(subset=['toxic_sentence'])
    print(f"  ✓ Total rows after dedup: {len(merged_df)}")
    return merged_df

# ==========================================
# PART 2: GENERATE MISSING LABELS (MPS/GPU/CPU)
# ==========================================
def generate_labels(df, lang, tokenizer, model, device):
    print(f"\n--- Generating Missing Labels for {lang.upper()} ---")
    
    mask = df['non_toxic_sentence'].isna() | (df['non_toxic_sentence'] == "")
    
    if mask.sum() == 0:
        print("  ✓ No missing labels found. Dataset is complete!")
        return df

    print(f"  Rows needing generation: {mask.sum()}")

    lang_prompts = {
        'en': 'Detoxify: ', 'ru': 'Детоксифицируй: ', 'uk': 'Детоксифікуй: ',
        'de': 'Detoxifizieren: ', 'es': 'Desintoxicar: ', 'fr': 'Détoxifier: ',
        'ar': 'إزالة السموم: ', 'hi': 'विषाक्तता हटाएँ: ', 'zh': '排毒： ',
        'am': 'Detoxify: ', 'it': 'Disintossicare: ', 'he': 'לנקות רעלים: ',
        'hin': 'विषाक्तता हटाएँ: ', 'tt': 'Detoksifikatsiya: ', 'ja': '解毒する： ',
        'default': 'Detoxify: '
    }

    df_process = df[mask].copy()
    toxic_texts = df_process['toxic_sentence'].tolist()
    langs = df_process['lang'].fillna('default').tolist()
    
    generated_results = []
    print(f"  Starting generation for {len(toxic_texts)} texts...")
    
    for i in tqdm(range(0, len(toxic_texts), BATCH_SIZE), desc=f"  {lang}"):
        batch_texts = toxic_texts[i : i + BATCH_SIZE]
        batch_langs = langs[i : i + BATCH_SIZE]
        
        input_strings = []
        for t, l in zip(batch_texts, batch_langs):
            prompt = lang_prompts.get(l, lang_prompts['default'])
            input_strings.append(prompt + str(t))
        
        try:
            inputs = tokenizer(
                input_strings, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=128
            ).to(device)
            
            with torch.no_grad():
                # MPS doesn't support all operations, use float32 for stability
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=128, 
                    num_beams=2,
                    do_sample=False  # Deterministic for MPS stability
                )
            
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated_results.extend(decoded)
            
        except Exception as e:
            print(f"\n  ⚠ Batch failed: {e}. Filling with empty strings.")
            generated_results.extend([""] * len(batch_texts))
        
        # Memory management for MPS
        if device == "mps" and i % 20 == 0:
            torch.mps.empty_cache()
        elif device == "cuda" and i % 50 == 0:
            torch.cuda.empty_cache()

    df.loc[mask, 'non_toxic_sentence'] = generated_results
    print(f"  ✓ Generated {len(generated_results)} detoxified versions")
    return df

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("MULTILINGUAL DETOXIFICATION DATASET GENERATION")
    print("="*60)
    print(f"Processing {len(ALL_LANGUAGES)} languages: {', '.join(ALL_LANGUAGES)}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    print("\n" + "="*60)
    print("LOADING MODEL")
    print("="*60)
    
    device = get_device()
    tokenizer = None
    model = None
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        if device == "mps":
            # MPS: Load in float32 for stability (float16 has issues on MPS)
            print("Loading model for MPS (Apple Silicon)...")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32,  # MPS works better with float32
                low_cpu_mem_usage=True
            )
            model = model.to(device)
            print("✓ Model loaded successfully on MPS\n")
            
        elif device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    MODEL_NAME,
                    device_map="auto",
                    quantization_config=quantization_config
                )
                print("✓ Model loaded with 8-bit quantization\n")
            except Exception as quant_error:
                print(f"⚠ Quantization failed: {quant_error}")
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    MODEL_NAME,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
                print("✓ Model loaded in FP16\n")
        else:
            print("Loading model for CPU...")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32
            )
            model = model.to(device)
            print("✓ Model loaded on CPU\n")
            
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        print("\nTroubleshooting for MPS:")
        print("1. Ensure PyTorch >= 2.0: pip install -U torch")
        print("2. Check macOS >= 12.3")
        print("3. Try reducing BATCH_SIZE to 2")
        print("4. Install accelerate: pip install -U accelerate")
        import sys
        sys.exit(1)
    
    if model is None or tokenizer is None:
        print("❌ Critical error: Model or tokenizer is None")
        import sys
        sys.exit(1)
    
    # Process each language
    success_count = 0
    failed_langs = []
    
    for lang in ALL_LANGUAGES:
        try:
            lang_df = prepare_dataset_for_language(lang)
            
            if lang_df is None or len(lang_df) == 0:
                print(f"  ⚠ Skipping {lang}: No data available")
                failed_langs.append(lang)
                continue
            
            final_df = generate_labels(lang_df, lang, tokenizer, model, device)
            
            output_file = os.path.join(OUTPUT_DIR, f"detox_dataset_{lang}.csv")
            final_df.to_csv(output_file, index=False)
            
            print(f"\n  ✅ Saved: {output_file}")
            print(f"     Total rows: {len(final_df)}")
            print(f"     Complete entries: {(~final_df['non_toxic_sentence'].isna()).sum()}")
            
            success_count += 1
            
            del lang_df, final_df
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\n  ❌ Failed to process {lang}: {e}")
            failed_langs.append(lang)
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"✅ Successfully processed: {success_count}/{len(ALL_LANGUAGES)} languages")
    if failed_langs:
        print(f"❌ Failed languages: {', '.join(failed_langs)}")
    print(f"\nAll files saved in: {OUTPUT_DIR}/")