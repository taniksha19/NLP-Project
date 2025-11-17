# Multilingual Text Detoxification - Baseline System

This project implements a baseline system for multilingual text detoxification using a pre-trained mT5 model.

## Project Structure

```
multilingual-detox/
├── data/                # downloaded datasets will go here
├── results/
│   └── baseline_results.csv
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset_loader.py
│   ├── baseline_model.py
│   ├── evaluation.py
│   ├── utils.py
│   └── run_baseline.py
├── requirements.txt
└── README.md
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the baseline evaluation:

```bash
python src/run_baseline.py
```

This will:
1. Load datasets for English, Russian, Spanish, and Ukrainian
2. Run detoxification on the first 300 samples per language
3. Compute evaluation metrics (Style Transfer Accuracy, Semantic Similarity, Fluency)
4. Save results to `results/baseline_results.csv`

## Evaluation Metrics

- **Style Transfer Accuracy (STA)**: Percentage of generated sentences that do not contain toxic words
- **Semantic Similarity (SIM)**: Cosine similarity between gold and generated detoxified texts using LaBSE embeddings
- **Fluency (FL)**: Heuristic score based on sentence length and token repetition

## Datasets

The system uses the following HuggingFace datasets:
- English: `s-nlp/paradetox`
- Russian: `s-nlp/ru_paradetox`
- Spanish: `textdetox/es_paradetox`
- Ukrainian: `textdetox/uk_paradetox`

## Model

The baseline uses the pre-trained model: `textdetox/mt5-xl-detox-baseline`

## TODO

- Replace heuristic metrics with PAN evaluation scripts
- Add mT5 fine-tuning pipeline
- Add mBART training experiments
- Add multilingual data augmentation methods



