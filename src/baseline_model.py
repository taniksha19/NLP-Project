#!/usr/bin/env python3

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import List, Set, Optional
from datasets import load_dataset
from tqdm import tqdm

SPACES = re.compile(r"\s+")


def detoxify(
    text: str,
    stopwords: List[str],
    remove_all_terms: bool = False,
    remove_no_terms: bool = False,
) -> str:

    if remove_no_terms:
        return text
    if remove_all_terms:
        return ""

    tokens = [
        token
        for token in SPACES.split(text)
        if not stopwords or token.lower().strip() not in stopwords
    ]
    return " ".join(tokens)


def load_stopwords(language: Optional[str] = None) -> Set[str]:
    """
    Load stopwords for a specific language or all languages.
    
    Args:
        language: Language code (e.g., 'en', 'ru', 'es'). If None, loads all.
    
    Returns:
        Set of stopwords
    """
    if language is not None:
        logging.info(f"Loading stopwords for {language}")
        stopwords = load_dataset("textdetox/multilingual_toxic_lexicon")[language]["text"]
        return set(stopwords)
    else:
        logging.info("No specific language for stopwords provided. Loading all stopwords")
        stopwords_dataset = load_dataset("textdetox/multilingual_toxic_lexicon")
        words = []
        for lang in stopwords_dataset.keys():
            words.extend(stopwords_dataset[lang]["text"])
        return set(words)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="The trivial baseline for the PAN 2024 text detoxification task "
        "that removes all/none/or specified stopwords from given text for "
        "detoxification."
    )

    parser.add_argument(
        "--input",
        required=False,
        type=str,
        help="The input file path, expected a jsonl file. Required if --dataset is not provided.",
    )
    parser.add_argument(
        "--dataset",
        required=False,
        type=str,
        default=None,
        help="Hugging Face dataset name (e.g., 'textdetox/multilingual_paradetox'). "
        "If provided, will process the dataset instead of JSONL file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="The output file path or directory. For JSONL input: file path. "
        "For dataset: directory path where results will be saved.",
    )
    parser.add_argument(
        "--language",
        required=False,
        type=str,
        default=None,
        help="Specify language. Should be one of "
        "['am', 'es', 'ru', 'uk', 'en', 'zh', 'ar', 'hi', 'de']. "
        "Without specification will load all stopwords.",
        choices=["am", "es", "ru", "uk", "en", "zh", "ar", "hi", "de"],
    )
    parser.add_argument(
        "--remove-all-terms",
        required=False,
        default=False,
        type=bool,
        help="Generate the empty string.",
    )
    parser.add_argument(
        "--remove-no-terms",
        required=False,
        default=False,
        type=bool,
        help="Output the text without modification.",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Specify logging level (default: INFO).",
    )

    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    # Validate arguments
    if not args.dataset and not args.input:
        parser.error("Either --input or --dataset must be provided")

    # Load stopwords
    stopwords = load_stopwords(args.language)
    logging.info(f"Loaded {len(stopwords)} stopwords")

    # Process dataset from Hugging Face
    if args.dataset:
        logging.info(f"Loading dataset: {args.dataset}")
        dataset = load_dataset(args.dataset)
        
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get available splits (languages)
        available_splits = list(dataset.keys())
        logging.info(f"Available language splits: {available_splits}")
        
        # Process each language split
        for split_name in available_splits:
            logging.info(f"\nProcessing split: {split_name}")
            split_data = dataset[split_name]
            
            # Determine language for stopwords (use split name if it's a language code)
            lang_for_stopwords = split_name if split_name in [
                "am", "es", "ru", "uk", "en", "zh", "ar", "hi", "de"
            ] else args.language
            
            # Load language-specific stopwords if available
            if lang_for_stopwords:
                split_stopwords = load_stopwords(lang_for_stopwords)
            else:
                split_stopwords = stopwords
            
            # Process each example
            output_file = output_dir / f"{split_name}_baseline_output.jsonl"
            with open(output_file, "w", encoding="utf-8") as f:
                for idx, example in enumerate(tqdm(split_data, desc=f"Processing {split_name}")):
                    # Get toxic text (try different field names)
                    toxic_text = example.get("toxic_sentence") or example.get("text", "")
                    
                    # Detoxify the text
                    detoxified_text = detoxify(
                        toxic_text,
                        split_stopwords,
                        remove_all_terms=args.remove_all_terms,
                        remove_no_terms=args.remove_no_terms,
                    )
                    
                    # Create output entry
                    result = {
                        "id": f"{split_name}-{idx:04d}",
                        "text": detoxified_text,
                    }
                    
                    # Add original text and reference if available
                    if "toxic_sentence" in example:
                        result["original_text"] = example["toxic_sentence"]
                    if "neutral_sentence" in example:
                        result["reference"] = example["neutral_sentence"]
                    
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
            logging.info(f"Processed {len(split_data)} examples for {split_name}")
            logging.info(f"Results saved to {output_file}")
        
        logging.info(f"\nAll done! Results saved to {output_dir}/")
    
    # Process JSONL file
    else:
        logging.info("Started processing texts from JSONL file.")
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(args.input, "rb") as input_file, \
             open(output_path, "w", encoding="UTF-8") as output_file:
            for line in tqdm(input_file, desc="Processing input file"):
                instance = json.loads(line)
                instance["text"] = detoxify(
                    instance["text"], stopwords, args.remove_all_terms, args.remove_no_terms
                )
                
                output_file.write(json.dumps(instance, ensure_ascii=False))
                output_file.write("\n")
        
        logging.info(f"All done. Outputs are written to {args.output}")


if __name__ == "__main__":
    main()