# Multilingual Text Detoxification - Baseline System

This project implements a baseline system for multilingual text detoxification for the PAN 2024 text detoxification task. The baseline removes toxic stopwords from input text using a multilingual toxic lexicon.

## Project Structure

```
NLP-Project/
├── data/                # Input datasets (optional, for JSONL files)
├── results/             # Output results
│   └── dataset_results/ # Results from Hugging Face dataset processing
│       ├── en_baseline_output.jsonl
│       ├── ru_baseline_output.jsonl
│       ├── uk_baseline_output.jsonl
│       └── ... (other language outputs)
├── src/
│   └── baseline_model.py  # Baseline detoxification script
├── requirements.txt
└── README.md
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

The baseline model supports two input modes: JSONL files or Hugging Face datasets.

### Processing Hugging Face Datasets (Recommended)

Run the baseline on the multilingual_paradetox dataset:

```bash
python src/baseline_model.py --dataset textdetox/multilingual_paradetox --output results/dataset_results
```

This will:
- Automatically process all language splits (en, ru, uk, es, am, zh, ar, hi, de)
- Use language-specific stopwords for each split
- Save separate output files for each language (e.g., `en_baseline_output.jsonl`, `ru_baseline_output.jsonl`)

### Processing JSONL Files

Run the baseline on a local JSONL file:

```bash
python src/baseline_model.py --input data/input.jsonl --output results/output.jsonl --language en
```

### Command Line Arguments

- `--input` (optional): Input JSONL file path. Required if `--dataset` is not provided. Each line should be a JSON object with at least a `"text"` field.
- `--dataset` (optional): Hugging Face dataset name (e.g., `textdetox/multilingual_paradetox`). If provided, will process the dataset instead of JSONL file.
- `--output` (required): Output file path (for JSONL) or directory path (for datasets) where detoxified results will be written.
- `--language` (optional): Language code for stopwords. Options: `am`, `es`, `ru`, `uk`, `en`, `zh`, `ar`, `hi`, `de`. If not specified, all stopwords from all languages will be loaded. For datasets, language-specific stopwords are used automatically per split.
- `--remove-all-terms` (optional, default: `False`): If `True`, generates empty strings for all texts.
- `--remove-no-terms` (optional, default: `False`): If `True`, outputs text without any modification.
- `--log-level` (optional, default: `INFO`): Logging level. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.

### Examples

```bash
# Process multilingual_paradetox dataset (all languages)
python src/baseline_model.py --dataset textdetox/multilingual_paradetox --output results/dataset_results

# Process JSONL file with English stopwords
python src/baseline_model.py --input data/input.jsonl --output results/output.jsonl --language en

# Process JSONL file without specifying language (uses all stopwords)
python src/baseline_model.py --input data/input.jsonl --output results/output.jsonl
```

## Input Format

### JSONL File Format

For JSONL files, each line should be a JSON object with at least a `"text"` field:

```json
{"id": "en-sample00", "text": "this is some toxic text."}
{"id": "en-sample01", "text": "another example sentence."}
```

### Hugging Face Dataset Format

The `textdetox/multilingual_paradetox` dataset contains:
- `toxic_sentence`: Original toxic text
- `neutral_sentence`: Reference detoxified text (gold standard)
- Multiple language splits: `en`, `ru`, `uk`, `es`, `am`, `zh`, `ar`, `hi`, `de`

## Output Format

### JSONL File Output

For JSONL input, the output maintains the same structure with the `"text"` field containing the detoxified version:

```json
{"id": "en-sample00", "text": "this is some text."}
{"id": "en-sample01", "text": "another example sentence."}
```

### Dataset Output

For dataset processing, each language split generates a separate JSONL file with the following structure:

```json
{"id": "en-0000", "text": "detoxified text", "original_text": "toxic sentence", "reference": "gold standard neutral sentence"}
{"id": "en-0001", "text": "detoxified text", "original_text": "toxic sentence", "reference": "gold standard neutral sentence"}
```

Output files are saved as `{language}_baseline_output.jsonl` in the specified output directory.

## Dataset

This project uses the [multilingual_paradetox](https://huggingface.co/datasets/textdetox/multilingual_paradetox) dataset from Hugging Face, which provides parallel toxic and neutral sentence pairs for multiple languages. The dataset is part of the PAN 2024 Text Detoxification shared task.

**Supported Languages:**
- English (en)
- Russian (ru)
- Ukrainian (uk)
- Spanish (es)
- Amharic (am)
- Chinese (zh)
- Arabic (ar)
- Hindi (hi)
- German (de)

## Model

The baseline uses a simple stopword removal approach:
- Loads toxic stopwords from the `textdetox/multilingual_toxic_lexicon` dataset
- Removes tokens that match stopwords (case-insensitive)
- Uses language-specific stopwords when processing datasets
- Preserves the original text structure and non-toxic words

## Limitations

- The baseline only removes exact word matches from the lexicon
- Punctuation attached to words (e.g., "fuck.") may not be detected
- Misspelled toxic words will not be removed
- This is a trivial baseline intended for comparison purposes

## Build LOR tables for all languages 

python -m src.mask_reg.build_lor \
  --languages en,ru,uk,es,am,zh,ar,hi,de \
  --tokenizer google/mt5-small \
  --mask_fraction 0.05 \
  --out_dir artifacts/lor_mf005

  This will create *.jsonl + *.meta.json for each language under artifacts/lor_mf005/.

## Train the LoRA-finetuned detox model (per language)

python -m src.mask_reg.train_mt5 \
  --lang en \
  --dataset textdetox/multilingual_paradetox \
  --lor_dir artifacts/lor_mf005 \
  --model_name google/mt5-small \
  --out_dir checkpoints/mask-reg-mt5-small-en-lora-mf005 \
  --epochs 5 \
  --train_bs 1 \
  --eval_bs 1 \
  --grad_accum 8 \
  --max_source_len 96 \
  --max_target_len 96 \
  --merge_lora

This trains a LoRA adapter and then writes a merged HF model to:
checkpoints/mask-reg-mt5-small-en-lora-mf005-MERGED/
To train another language, just change --lang (and optionally --out_dir naming).

## Run Inference (Mask)

python -m src.mask_reg.infer_mt5 \
  --lang en \
  --lor_dir artifacts/lor_mf005 \
  --model_dir checkpoints/mask-reg-mt5-small-en-lora-mf005-MERGED \
  --tokenizer_name google/mt5-small \
  --threshold_mult 0.9



## TODO

- Improve tokenization to handle punctuation better
- Add support for fuzzy matching or stemming
- Integrate with PAN evaluation scripts
- Add more sophisticated detoxification methods