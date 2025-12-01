#!/usr/bin/env python3
"""
Finetuning script for multilingual text detoxification model.
Trains on the merged detoxification datasets.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset as HFDataset
import json
from pathlib import Path
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
BASE_MODEL = "s-nlp/mt0-xl-detox-orpo"  # Or start from a smaller model like "google/mt5-base"
DATA_DIR = "data"
OUTPUT_DIR = "models/finetuned_detox"
MAX_LENGTH = 128
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5
SAVE_STEPS = 500
EVAL_STEPS = 500
LOGGING_STEPS = 100

# Languages to train on
LANGUAGES = ['en', 'ru', 'uk', 'de', 'es', 'am', 'zh', 'ar', 'hi']

# Language-specific prompts (optional, can help model understand task)
LANG_PROMPTS = {
    'en': 'Detoxify: ',
    'ru': 'Детоксифицируй: ',
    'uk': 'Детоксифікуй: ',
    'de': 'Detoxifizieren: ',
    'es': 'Desintoxicar: ',
    'am': 'Detoxify: ',
    'zh': '排毒： ',
    'ar': 'إزالة السموم: ',
    'hi': 'विषाक्तता हटाएँ: ',
    'default': 'Detoxify: '
}

# ==========================================
# DATASET CLASS
# ==========================================
class DetoxificationDataset(Dataset):
    """Dataset for detoxification task."""
    
    def __init__(self, tokenizer, data, max_length=128, lang_prompts=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lang_prompts = lang_prompts or {}
        
        # Prepare data
        self.inputs = []
        self.targets = []
        
        for _, row in data.iterrows():
            toxic = str(row['toxic_sentence']).strip()
            non_toxic = str(row['non_toxic_sentence']).strip()
            lang = str(row.get('lang', 'default')).strip()
            
            # Skip empty or invalid rows
            if not toxic or not non_toxic or toxic == 'nan' or non_toxic == 'nan':
                continue
            
            # Add language-specific prompt (optional)
            prompt = self.lang_prompts.get(lang, self.lang_prompts.get('default', ''))
            input_text = prompt + toxic
            
            self.inputs.append(input_text)
            self.targets.append(non_toxic)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        source = self.inputs[idx]
        target = self.targets[idx]
        
        # Tokenize inputs
        model_inputs = self.tokenizer(
            source,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                target,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        
        # Replace padding token id's of the labels by -100 so it's ignored by the loss function
        labels = labels['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        model_inputs['labels'] = labels
        
        return {
            'input_ids': model_inputs['input_ids'].squeeze(),
            'attention_mask': model_inputs['attention_mask'].squeeze(),
            'labels': labels
        }

# ==========================================
# LOAD AND PREPARE DATA
# ==========================================
def load_all_datasets(data_dir, languages):
    """Load and combine datasets from all languages."""
    all_data = []
    
    print(f"\n{'='*60}")
    print("LOADING DATASETS")
    print(f"{'='*60}")
    
    for lang in languages:
        csv_path = os.path.join(data_dir, f"detox_dataset_{lang}.csv")
        
        if not os.path.exists(csv_path):
            print(f"⚠️  File not found: {csv_path}, skipping {lang}")
            continue
        
        try:
            df = pd.read_csv(csv_path)
            
            # Filter out rows with missing data
            df = df.dropna(subset=['toxic_sentence', 'non_toxic_sentence'])
            df = df[df['toxic_sentence'].astype(str).str.strip() != '']
            df = df[df['non_toxic_sentence'].astype(str).str.strip() != '']
            
            # Ensure lang column exists
            if 'lang' not in df.columns:
                df['lang'] = lang
            
            all_data.append(df)
            print(f" {lang}: {len(df)} samples")
            
        except Exception as e:
            print(f" Error loading {lang}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No data loaded! Check data directory and file names.")
    
    # Combine all datasets
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Shuffle
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n Total samples: {len(combined_df)}")
    
    # Split into train/val (90/10)
    split_idx = int(len(combined_df) * 0.9)
    train_df = combined_df[:split_idx]
    val_df = combined_df[split_idx:]
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    return train_df, val_df

# ==========================================
# MAIN TRAINING FUNCTION
# ==========================================
def main():
    print("\n" + "="*60)
    print("MULTILINGUAL TEXT DETOXIFICATION - FINETUNING")
    print("="*60)
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print(" WARNING: Training on CPU will be very slow!")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load tokenizer and model
    print(f"\n{'='*60}")
    print("LOADING MODEL AND TOKENIZER")
    print(f"{'='*60}")
    print(f"Base model: {BASE_MODEL}")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    if device == "cuda":
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                BASE_MODEL,
                device_map="auto",
                quantization_config=quantization_config
            )
            print(" Model loaded with 8-bit quantization")
        except:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                BASE_MODEL,
                device_map="auto",
                torch_dtype=torch.float16
            )
            print(" Model loaded in FP16")
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
        model = model.to(device)
        print(" Model loaded on CPU")
    
    # Load datasets
    train_df, val_df = load_all_datasets(DATA_DIR, LANGUAGES)
    
    # Create datasets
    print(f"\n{'='*60}")
    print("PREPARING DATASETS")
    print(f"{'='*60}")
    
    train_dataset = DetoxificationDataset(
        tokenizer, train_df, max_length=MAX_LENGTH, lang_prompts=LANG_PROMPTS
    )
    val_dataset = DetoxificationDataset(
        tokenizer, val_df, max_length=MAX_LENGTH, lang_prompts=LANG_PROMPTS
    )
    
    print(f"✅ Train dataset: {len(train_dataset)} samples")
    print(f"✅ Val dataset: {len(val_dataset)} samples")
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=500,
        logging_steps=LOGGING_STEPS,
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        predict_with_generate=True,
        fp16=device == "cuda",
        report_to="none",  # Set to "tensorboard" if you want TensorBoard logs
    )
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    print(f"\n{'='*60}")
    print("STARTING TRAINING")
    print(f"{'='*60}")
    
    trainer.train()
    
    # Save final model
    print(f"\n{'='*60}")
    print("SAVING MODEL")
    print(f"{'='*60}")
    
    final_model_path = os.path.join(OUTPUT_DIR, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print(f"✅ Model saved to: {final_model_path}")
    print("\n🎉 Training complete!")

if __name__ == "__main__":
    main()

