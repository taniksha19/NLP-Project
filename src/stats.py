#!/usr/bin/env python3
"""
Statistics computation script for textdetox/multilingual_paradetox dataset.
Computes dataset size, length, toxicity, and similarity statistics.
"""

import random
from typing import Dict, List

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def simple_tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization."""
    return text.split()


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def compute_dataset_size(dataset: Dict) -> Dict:
    """Compute dataset size statistics."""
    stats = {
        "overall": {"total_samples": 0},
        "per_language": {}
    }
    
    for split_name, split_data in dataset.items():
        num_samples = len(split_data)
        stats["per_language"][split_name] = {"samples": num_samples}
        stats["overall"]["total_samples"] += num_samples
    
    return stats


def compute_length_stats(dataset: Dict) -> Dict:
    """Compute length statistics for toxic and neutral sentences."""
    stats = {
        "overall": {
            "toxic": {"avg_tokens": [], "median_tokens": [], "avg_chars": []},
            "neutral": {"avg_tokens": [], "median_tokens": [], "avg_chars": []},
            "length_diff": []
        },
        "per_language": {}
    }
    
    for split_name, split_data in dataset.items():
        toxic_tokens = []
        neutral_tokens = []
        toxic_chars = []
        neutral_chars = []
        length_diffs = []
        
        for example in split_data:
            toxic = example.get("toxic_sentence", "")
            neutral = example.get("neutral_sentence", "")
            
            toxic_tok = simple_tokenize(toxic)
            neutral_tok = simple_tokenize(neutral)
            
            toxic_tokens.append(len(toxic_tok))
            neutral_tokens.append(len(neutral_tok))
            toxic_chars.append(len(toxic))
            neutral_chars.append(len(neutral))
            length_diffs.append(len(neutral_tok) - len(toxic_tok))
        
        lang_stats = {
            "toxic": {
                "avg_tokens": np.mean(toxic_tokens),
                "median_tokens": np.median(toxic_tokens),
                "avg_chars": np.mean(toxic_chars)
            },
            "neutral": {
                "avg_tokens": np.mean(neutral_tokens),
                "median_tokens": np.median(neutral_tokens),
                "avg_chars": np.mean(neutral_chars)
            },
            "avg_length_diff": np.mean(length_diffs)
        }
        
        stats["per_language"][split_name] = lang_stats
        
        # Aggregate for overall
        stats["overall"]["toxic"]["avg_tokens"].extend(toxic_tokens)
        stats["overall"]["neutral"]["avg_tokens"].extend(neutral_tokens)
        stats["overall"]["toxic"]["avg_chars"].extend(toxic_chars)
        stats["overall"]["neutral"]["avg_chars"].extend(neutral_chars)
        stats["overall"]["length_diff"].extend(length_diffs)
    
    # Compute overall statistics
    stats["overall"]["toxic"]["avg_tokens"] = np.mean(stats["overall"]["toxic"]["avg_tokens"])
    stats["overall"]["toxic"]["median_tokens"] = np.median(stats["overall"]["toxic"]["avg_tokens"])
    stats["overall"]["toxic"]["avg_chars"] = np.mean(stats["overall"]["toxic"]["avg_chars"])
    
    stats["overall"]["neutral"]["avg_tokens"] = np.mean(stats["overall"]["neutral"]["avg_tokens"])
    stats["overall"]["neutral"]["median_tokens"] = np.median(stats["overall"]["neutral"]["avg_tokens"])
    stats["overall"]["neutral"]["avg_chars"] = np.mean(stats["overall"]["neutral"]["avg_chars"])
    
    stats["overall"]["avg_length_diff"] = np.mean(stats["overall"]["length_diff"])
    
    return stats


def load_toxicity_model():
    """Load the toxicity classification model."""
    print("Loading toxicity model: unitary/toxic-bert...")
    model_name = "unitary/toxic-bert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Toxicity model loaded on {device}")
    return model, tokenizer, device


def compute_toxicity_score(text: str, model, tokenizer, device) -> float:
    """Compute toxicity score for a single text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        # Return the maximum toxicity score across all labels
        toxicity_score = float(probs.max().item())
    
    return toxicity_score


def compute_toxicity_stats(dataset: Dict, model, tokenizer, device) -> Dict:
    """Compute toxicity statistics."""
    print("Computing toxicity statistics...")
    stats = {
        "overall": {
            "toxic_avg": [],
            "neutral_avg": [],
            "reduction": []
        },
        "per_language": {}
    }
    
    for split_name, split_data in dataset.items():
        print(f"  Processing {split_name}...")
        toxic_scores = []
        neutral_scores = []
        reductions = []
        
        for example in tqdm(split_data, desc=f"    {split_name}", leave=False):
            toxic = example.get("toxic_sentence", "")
            neutral = example.get("neutral_sentence", "")
            
            toxic_score = compute_toxicity_score(toxic, model, tokenizer, device)
            neutral_score = compute_toxicity_score(neutral, model, tokenizer, device)
            reduction = toxic_score - neutral_score
            
            toxic_scores.append(toxic_score)
            neutral_scores.append(neutral_score)
            reductions.append(reduction)
        
        lang_stats = {
            "toxic_avg": np.mean(toxic_scores),
            "neutral_avg": np.mean(neutral_scores),
            "reduction_avg": np.mean(reductions)
        }
        
        stats["per_language"][split_name] = lang_stats
        stats["overall"]["toxic_avg"].extend(toxic_scores)
        stats["overall"]["neutral_avg"].extend(neutral_scores)
        stats["overall"]["reduction"].extend(reductions)
    
    # Compute overall statistics
    stats["overall"]["toxic_avg"] = np.mean(stats["overall"]["toxic_avg"])
    stats["overall"]["neutral_avg"] = np.mean(stats["overall"]["neutral_avg"])
    stats["overall"]["reduction_avg"] = np.mean(stats["overall"]["reduction"])
    
    return stats


def load_embedding_model():
    """Load the multilingual embedding model."""
    print("Loading embedding model: paraphrase-multilingual-MiniLM-L12-v2...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("Embedding model loaded")
    return model


def compute_similarity_stats(dataset: Dict, embedding_model) -> Dict:
    """Compute similarity statistics."""
    print("Computing similarity statistics...")
    stats = {
        "overall": {
            "cosine_sim": [],
            "edit_distance": []
        },
        "per_language": {}
    }
    
    for split_name, split_data in dataset.items():
        print(f"  Processing {split_name}...")
        cosine_sims = []
        edit_dists = []
        
        # Batch process embeddings for efficiency
        toxic_texts = [ex.get("toxic_sentence", "") for ex in split_data]
        neutral_texts = [ex.get("neutral_sentence", "") for ex in split_data]
        
        # Get embeddings
        toxic_embeddings = embedding_model.encode(toxic_texts, show_progress_bar=False)
        neutral_embeddings = embedding_model.encode(neutral_texts, show_progress_bar=False)
        
        # Compute cosine similarities
        for i in tqdm(range(len(split_data)), desc=f"    {split_name}", leave=False):
            toxic_emb = toxic_embeddings[i:i+1]
            neutral_emb = neutral_embeddings[i:i+1]
            cosine_sim = cosine_similarity(toxic_emb, neutral_emb)[0][0]
            cosine_sims.append(cosine_sim)
            
            # Compute edit distance
            toxic = toxic_texts[i]
            neutral = neutral_texts[i]
            edit_dist = levenshtein_distance(toxic, neutral)
            edit_dists.append(edit_dist)
        
        lang_stats = {
            "cosine_sim_avg": np.mean(cosine_sims),
            "edit_distance_avg": np.mean(edit_dists)
        }
        
        stats["per_language"][split_name] = lang_stats
        stats["overall"]["cosine_sim"].extend(cosine_sims)
        stats["overall"]["edit_distance"].extend(edit_dists)
    
    # Compute overall statistics
    stats["overall"]["cosine_sim_avg"] = np.mean(stats["overall"]["cosine_sim"])
    stats["overall"]["edit_distance_avg"] = np.mean(stats["overall"]["edit_distance"])
    
    return stats


def print_statistics(size_stats, length_stats, toxicity_stats, similarity_stats):
    """Print all statistics in a structured format."""
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    
    # Dataset Size
    print("\n1. DATASET SIZE")
    print("-" * 80)
    print(f"{'Language':<15} {'Samples':<15}")
    print("-" * 80)
    for lang, stats in sorted(size_stats["per_language"].items()):
        print(f"{lang:<15} {stats['samples']:<15}")
    print("-" * 80)
    print(f"{'OVERALL':<15} {size_stats['overall']['total_samples']:<15}")
    
    # Length Statistics
    print("\n2. LENGTH STATISTICS")
    print("-" * 80)
    print(f"{'Language':<15} {'Toxic Avg Tokens':<20} {'Neutral Avg Tokens':<20} {'Avg Length Diff':<20}")
    print("-" * 80)
    for lang in sorted(length_stats["per_language"].keys()):
        stats = length_stats["per_language"][lang]
        print(f"{lang:<15} {stats['toxic']['avg_tokens']:<20.2f} {stats['neutral']['avg_tokens']:<20.2f} {stats['avg_length_diff']:<20.2f}")
    print("-" * 80)
    overall = length_stats["overall"]
    print(f"{'OVERALL':<15} {overall['toxic']['avg_tokens']:<20.2f} {overall['neutral']['avg_tokens']:<20.2f} {overall['avg_length_diff']:<20.2f}")
    
    # Toxicity Statistics
    print("\n3. TOXICITY STATISTICS")
    print("-" * 80)
    print(f"{'Language':<15} {'Toxic Avg':<15} {'Neutral Avg':<15} {'Reduction Avg':<15}")
    print("-" * 80)
    for lang in sorted(toxicity_stats["per_language"].keys()):
        stats = toxicity_stats["per_language"][lang]
        print(f"{lang:<15} {stats['toxic_avg']:<15.4f} {stats['neutral_avg']:<15.4f} {stats['reduction_avg']:<15.4f}")
    print("-" * 80)
    overall = toxicity_stats["overall"]
    print(f"{'OVERALL':<15} {overall['toxic_avg']:<15.4f} {overall['neutral_avg']:<15.4f} {overall['reduction_avg']:<15.4f}")
    
    # Similarity Statistics
    print("\n4. SIMILARITY STATISTICS")
    print("-" * 80)
    print(f"{'Language':<15} {'Cosine Sim Avg':<20} {'Edit Distance Avg':<20}")
    print("-" * 80)
    for lang in sorted(similarity_stats["per_language"].keys()):
        stats = similarity_stats["per_language"][lang]
        print(f"{lang:<15} {stats['cosine_sim_avg']:<20.4f} {stats['edit_distance_avg']:<20.2f}")
    print("-" * 80)
    overall = similarity_stats["overall"]
    print(f"{'OVERALL':<15} {overall['cosine_sim_avg']:<20.4f} {overall['edit_distance_avg']:<20.2f}")


def main():
    """Main function to compute and display all statistics."""
    set_seed(42)
    
    print("Loading dataset: textdetox/multilingual_paradetox...")
    dataset = load_dataset("textdetox/multilingual_paradetox")
    print(f"Dataset loaded. Available splits: {list(dataset.keys())}\n")
    
    # Compute statistics
    print("Computing dataset size statistics...")
    size_stats = compute_dataset_size(dataset)
    
    print("Computing length statistics...")
    length_stats = compute_length_stats(dataset)
    
    # Load models
    toxicity_model, toxicity_tokenizer, toxicity_device = load_toxicity_model()
    embedding_model = load_embedding_model()
    
    # Compute toxicity and similarity stats
    toxicity_stats = compute_toxicity_stats(dataset, toxicity_model, toxicity_tokenizer, toxicity_device)
    similarity_stats = compute_similarity_stats(dataset, embedding_model)
    
    # Print all statistics
    print_statistics(size_stats, length_stats, toxicity_stats, similarity_stats)
    
    print("\n" + "="*80)
    print("Statistics computation complete!")
    print("="*80)


if __name__ == "__main__":
    main()

