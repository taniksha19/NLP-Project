# Multilingual Text Detoxification

A comprehensive multilingual text detoxification system supporting 8 languages, implementing both unsupervised baseline methods and supervised fine-tuning approaches for the PAN 2024 Text Detoxification shared task.

## Overview

Text detoxification aims to transform toxic text into neutral alternatives while preserving semantic meaning. This project explores multiple approaches across **Arabic, German, Ukrainian, Chinese, English, Russian, Spanish, and Amharic**.

## Approaches

### 1. Baseline Methods
- **Duplicate**: Returns the original toxic text unchanged (lower bound)
- **Delete**: Removes toxic words identified via a multilingual toxic lexicon

### 2. Unsupervised Mask-and-Fill
Uses Log-Odds Ratio (LOR) to identify toxic tokens and replaces them using masked language models (mBERT/XLM-RoBERTa).

### 3. Supervised Fine-tuning
Fine-tunes **MT0-Large** with LoRA adapters on parallel toxic-neutral sentence pairs for sequence-to-sequence detoxification.

## Results

### Evaluation Metrics
- **Toxicity (↓)**: Lower is better - measures toxicity of detoxified output
- **SIM (↑)**: Higher is better - semantic similarity to original
- **Fluency (↑)**: Higher is better - language model perplexity (negative log-likelihood)
- **Tox_Red (↓)**: Toxicity reduction from original

### Baseline Results

| Lang | Method | Toxicity ↓ | SIM ↑ | Fluency ↑ |
|------|--------|-----------|-------|-----------|
| am | Duplicate | 0.9982 | 0.9825 | -2.80 |
| am | Delete | 0.9982 | 0.9827 | -2.78 |
| ar | Duplicate | 0.9950 | 0.9766 | -2.26 |
| ar | Delete | 0.9950 | 0.9770 | -2.25 |
| de | Duplicate | 0.9830 | 0.9683 | -5.28 |
| de | Delete | 0.9735 | 0.9671 | -5.24 |
| en | Duplicate | 0.9894 | 0.9583 | -5.13 |
| en | Delete | 0.6742 | 0.9563 | -5.49 |
| es | Duplicate | 0.9674 | 0.9389 | -5.45 |
| es | Delete | 0.8885 | 0.9406 | -5.46 |
| ru | Duplicate | 0.8550 | 0.9452 | -2.85 |
| ru | Delete | 0.8550 | 0.9452 | -2.85 |
| uk | Duplicate | 0.8695 | 0.9734 | -3.09 |
| uk | Delete | 0.8695 | 0.9736 | -3.08 |
| zh | Duplicate | 0.9715 | 0.9549 | -3.66 |
| zh | Delete | 0.9715 | 0.9549 | -3.66 |

### Unsupervised Mask-Filling Results (MPS)

| Lang | Tox_Orig | Tox_Det | Tox_Red ↓ | SIM ↑ | Fluency ↑ | Masked |
|------|----------|---------|-----------|-------|-----------|--------|
| en | 0.9894 | 0.3785 | **0.6110** | 0.9484 | -5.27 | 265/302 |
| es | 0.9674 | 0.7708 | **0.1966** | 0.9361 | -5.43 | 305/500 |
| de | 0.9830 | 0.9581 | 0.0248 | 0.9661 | -5.27 | 91/440 |
| ar | 0.9950 | 0.9950 | 0.0000 | 0.9796 | -2.26 | 0/453 |
| uk | 0.8695 | 0.8695 | 0.0000 | 0.9627 | -3.09 | 0/400 |
| zh | 0.9715 | 0.9715 | 0.0000 | 0.9549 | -3.66 | 0/401 |
| ru | 0.8550 | 0.8550 | 0.0000 | 0.9452 | -2.85 | 0/368 |
| am | 0.9982 | 0.9982 | 0.0000 | 0.9825 | -2.80 | 0/501 |

### MT0-Large + LoRA Fine-tuning Results

| Lang | Toxicity ↓ | SIM ↑ | Fluency ↑ | Tox_Orig | Tox_Red |
|------|-----------|-------|-----------|----------|---------|
| **Overall** | **0.8344** | **0.9195** | **-4.55** | 0.9563 | 0.1219 |
| en | **0.6585** | 0.9792 | -5.14 | 0.9894 | **0.3309** |
| es | **0.7030** | 0.9244 | -5.70 | 0.9675 | **0.2644** |
| ar | **0.7989** | 0.8524 | -3.89 | 0.9950 | **0.1961** |
| de | 0.8899 | 0.9419 | -5.43 | 0.9830 | 0.0931 |
| ru | 0.8321 | 0.8593 | -3.92 | 0.8552 | 0.0231 |
| uk | 0.8541 | 0.9101 | -3.97 | 0.8697 | 0.0156 |
| am | 0.9408 | 0.8650 | -4.15 | 0.9982 | 0.0574 |
| zh | 0.9592 | 0.9285 | -4.14 | 0.9714 | 0.0122 |

### Key Findings

1. **English** shows the best detoxification across all methods, with MT0-LoRA achieving 0.33 toxicity reduction
2. **Spanish** and **Arabic** also benefit significantly from supervised fine-tuning
3. **Low-resource languages** (Amharic, Chinese) remain challenging with minimal toxicity reduction
4. The **unsupervised mask-filling** approach works best for English (0.61 reduction) but struggles with other languages due to limited toxic token identification

## Project Structure

```
NLP-Project/
├── src/
│   ├── baseline_model.py    # Baseline detoxification implementations
│   ├── eval_baseline.py     # Evaluate baseline methods (Delete & Duplicate)
│   ├── eval_lor.py          # Evaluate unsupervised LOR mask-filling approach
│   ├── eval_model.py        # Evaluate supervised MT0-LoRA model
│   └── stats.py             # Generate dataset statistics
├── results/
│   └── dataset_results/     # Results for training
├── mt0_lora_evaluation_results.csv
├── mt0_lora_predictions.csv
├── requirements.txt
└── README.md
```

## Evaluation Scripts

| Script | Description |
|--------|-------------|
| `eval_baseline.py` | Evaluates the Delete and Duplicate baseline methods on the test set |
| `eval_lor.py` | Evaluates the unsupervised Log-Odds Ratio mask-filling approach |
| `eval_model.py` | Evaluates the fine-tuned MT0-LoRA model on detoxification |
| `stats.py` | Generates statistics and analysis of the multilingual dataset |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Process HuggingFace Dataset

```bash
python src/baseline_model.py --dataset textdetox/multilingual_paradetox --output results/dataset_results
```

### Process Local JSONL File

```bash
python src/baseline_model.py --input data/input.jsonl --output results/output.jsonl --language en
```

## Dataset

Uses the [multilingual_paradetox](https://huggingface.co/datasets/textdetox/multilingual_paradetox) dataset from HuggingFace, providing parallel toxic-neutral sentence pairs for the PAN 2024 shared task.

**Supported Languages:** English (en), Russian (ru), Ukrainian (uk), Spanish (es), Amharic (am), Chinese (zh), Arabic (ar), German (de)

## References

- PAN 2024 Text Detoxification Shared Task
- [Multilingual ParaDetox Dataset](https://huggingface.co/datasets/textdetox/multilingual_paradetox)
- [MT0 Model](https://huggingface.co/bigscience/mt0-large)

## Contributors

- [Taniksha Datar](https://github.com/taniksha19)
- [Ashwin](https://github.com/Ashwin1102)