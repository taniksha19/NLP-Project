# Multilingual Text Detoxification - Baseline System

This project implements a baseline system for multilingual text detoxification for the PAN 2024 text detoxification task. The baseline removes toxic stopwords from input text using a multilingual toxic lexicon.

## Project Structure

```
NLP-Project/
├── data/                # Input datasets (JSONL format)
│   └── input.jsonl
├── results/             # Output results
│   └── output.jsonl
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

Run the baseline detoxification script on a JSONL input file:

```bash
python src/baseline_model.py --input data/input.jsonl --output results/output.jsonl --language en
```

### Command Line Arguments

- `--input` (required): Input JSONL file path. Each line should be a JSON object with at least a `"text"` field.
- `--output` (required): Output JSONL file path where detoxified results will be written.
- `--language` (optional): Language code for stopwords. Options: `am`, `es`, `ru`, `uk`, `en`, `zh`, `ar`, `hi`, `de`. If not specified, all stopwords from all languages will be loaded.
- `--remove-all-terms` (optional, default: `False`): If `True`, generates empty strings for all texts.
- `--remove-no-terms` (optional, default: `False`): If `True`, outputs text without any modification.
- `--log-level` (optional, default: `INFO`): Logging level. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.

### Example

```bash
# Process English text with English stopwords
python src/baseline_model.py --input data/input.jsonl --output results/output.jsonl --language en

# Process without specifying language (uses all stopwords)
python src/baseline_model.py --input data/input.jsonl --output results/output.jsonl
```

## Input Format

The input file should be in JSONL format, where each line is a JSON object:

```json
{"id": "en-sample00", "text": "this is some toxic text."}
{"id": "en-sample01", "text": "another example sentence."}
```

## Output Format

The output file will be in the same JSONL format, with the `"text"` field containing the detoxified version:

```json
{"id": "en-sample00", "text": "this is some text."}
{"id": "en-sample01", "text": "another example sentence."}
```

## Model

The baseline uses a simple stopword removal approach:
- Loads toxic stopwords from the `textdetox/multilingual_toxic_lexicon` dataset
- Removes tokens that match stopwords (case-insensitive)
- Preserves the original text structure and non-toxic words

## Limitations

- The baseline only removes exact word matches from the lexicon
- Punctuation attached to words (e.g., "fuck.") may not be detected
- Misspelled toxic words will not be removed
- This is a trivial baseline intended for comparison purposes

## TODO

- Improve tokenization to handle punctuation better
- Add support for fuzzy matching or stemming
- Integrate with PAN evaluation scripts
- Add more sophisticated detoxification methods




