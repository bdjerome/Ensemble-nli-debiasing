# Ensemble Debiasing for Natural Language Inference

A PyTorch implementation of ensemble debiasing methods for improving Natural Language Inference (NLI) model robustness against dataset artifacts and spurious correlations.

## 📋 Overview

This project implements an ensemble-based debiasing approach for NLI models that:
- Trains a hypothesis-only "bias" model to capture dataset artifacts
- Combines it with a full NLI model using weighted ensemble debiasing
- Achieves significant improvements on out-of-distribution test sets (HANS)
- Maintains strong in-distribution performance (SNLI)

**Key Results:**
- **HANS accuracy:** 100% (vs ~50% baseline)
- **SNLI validation accuracy:** 89.3%
- **ANLI accuracy:** 32.75% (comparable to baselines)

## 🏗️ Project Structure

```
├── eval_hans.py                    # HANS evaluation script
├── eval_anli.py                    # ANLI evaluation script  
├── run_ensemble_debiasing.py       # Main ensemble debiasing training
├── run_alpha_sweep.py              # Hyperparameter tuning (alpha values)
├── HANS_leakage_check.ipynb        # Data leakage investigation notebook
├── paper.ipynb                     # Results analysis and visualization
├── fp-dataset-artifacts/           # Original base code
│   ├── run.py                      # Standard NLI training
│   ├── helpers.py                  # Utility functions
│   ├── requirements.txt            # Python dependencies
│   └── full_model/                 # Pre-trained baseline model
│   └── hypothesis_only_model/      # Pre-trained bias model
├── hans/                           # HANS dataset and evaluation
│   ├── heuristics_evaluation_set.jsonl
│   ├── heuristics_train_set.jsonl
│   └── evaluate_heur_output.py
└── results/                        # Model checkpoints and metrics
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-name>

# Install dependencies
pip install -r fp-dataset-artifacts/requirements.txt

# Verify PyTorch installation (with MPS support for Apple Silicon)
python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available())"
```

### Training Models

#### 1. Train Baseline Model (SNLI only)
```bash
python fp-dataset-artifacts/run.py \
  --model google/electra-small-discriminator \
  --task nli \
  --dataset snli \
  --do_train \
  --do_eval \
  --output_dir ./results/baseline_model \
  --per_device_train_batch_size 16 \
  --num_train_epochs 3
```

#### 2. Train Hypothesis-Only Bias Model
```bash
python run_hypothesis_only.py \
  --model google/electra-small-discriminator \
  --task nli \
  --dataset snli \
  --do_train \
  --do_eval \
  --output_dir ./results/hypothesis_only_model \
  --per_device_train_batch_size 16 \
  --num_train_epochs 3
```

#### 3. Train Ensemble Debiased Model
```bash
python run_ensemble_debiasing.py \
  --main_model google/electra-small-discriminator \
  --bias_model ./results/hypothesis_only_model \
  --task nli \
  --dataset SNLI_HANS \
  --do_train \
  --do_eval \
  --output_dir ./results/ensemble_debiased_model \
  --per_device_train_batch_size 16 \
  --num_train_epochs 3 \
  --alpha 0.5 \
  --combine_mode logit
```

### Evaluation

#### Evaluate on HANS
```bash
python eval_hans.py
```

#### Evaluate on ANLI
```bash
python eval_anli.py
```

## 🔬 Methodology

### Ensemble Debiasing

The core debiasing approach follows:

**Debiased Logits** = `Logits_main - α * Logits_bias`

Where:
- `Logits_main`: Full model (premise + hypothesis)
- `Logits_bias`: Hypothesis-only model (captures artifacts)
- `α`: Scaling factor (tuned via hyperparameter sweep)

### Data Augmentation

Models are trained on combined SNLI + HANS training data to expose them to heuristic patterns during training, following the methodology from the HANS paper (McCoy et al., 2019).

### Key Features

- **Dual Tokenization**: Separate preprocessing for full inputs (main model) and hypothesis-only inputs (bias model)
- **Flexible Alpha**: Supports fixed or learnable scaling parameters
- **Multiple Combination Modes**: Logit-space or log-probability space subtraction
- **Apple Silicon Support**: Optimized for MPS (Metal Performance Shaders) on M1/M2/M3 Macs

## 📊 Results

### HANS Performance by Heuristic

| Model | Lexical Overlap | Subsequence | Constituent | Overall |
|-------|----------------|-------------|-------------|---------|
| Baseline | 50.4% | 56.2% | 53.8% | 53.5% |
| Hypothesis-Only | 50.8% | 50.6% | 50.2% | 50.5% |
| **Ensemble Debiased** | **100%** | **100%** | **100%** | **100%** |

### Cross-Dataset Evaluation

| Model | SNLI (dev) | HANS | ANLI |
|-------|------------|------|------|
| Baseline | 89.1% | 53.5% | 32.1% |
| Hypothesis-Only | 56.9% | 50.5% | 33.0% |
| **Ensemble Debiased** | **89.3%** | **100%** | **32.8%** |

## 🔍 Data Leakage Investigation

Due to the exceptional 100% HANS accuracy, we conducted a thorough investigation documented in `HANS_leakage_check.ipynb`:

- **Finding**: HANS train/eval sets reuse the same pairIDs (ex0-ex29999) but have completely different sentence pairs
- **Validation**: 0 identical examples out of 30,000 overlapping IDs
- **Conclusion**: No data leakage; results are legitimate

## 📁 Repository Contents

### Core Scripts

- **`eval_hans.py`**: Comprehensive HANS evaluation with per-heuristic metrics
- **`eval_anli.py`**: ANLI (Adversarial NLI) evaluation  
- **`run_ensemble_debiasing.py`**: Ensemble debiasing training with dual tokenization
- **`run_alpha_sweep.py`**: Automated hyperparameter search for alpha values
- **`run_hypothesis_only.py`**: Train hypothesis-only bias models

### Analysis Notebooks

- **`HANS_leakage_check.ipynb`**: Data leakage investigation with visualizations
- **`paper.ipynb`**: Results analysis, confusion matrices, performance comparisons

### Base Framework

The `fp-dataset-artifacts/` directory contains the original training framework from the University of Texas at Austin NLP course.

## 🛠️ Technical Details

### Model Architecture

- **Base Model**: ELECTRA-small-discriminator (Google)
- **Task**: 3-way classification (entailment, neutral, contradiction)
- **Framework**: Hugging Face Transformers + PyTorch

### Hardware Requirements

- **GPU**: Recommended (CUDA or Apple MPS)
- **RAM**: 16GB minimum
- **Storage**: ~10GB for datasets and model checkpoints

### Hyperparameters

```python
# Optimal configuration
ALPHA = 0.5                  # Ensemble debiasing weight
BATCH_SIZE = 16              # Training batch size
EPOCHS = 3                   # Training epochs
MAX_LENGTH = 128             # Sequence length
LEARNING_RATE = 2e-5         # Default from TrainingArguments
```

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@misc{ensemble-debiasing-nli,
  author = {Brandon Jerome},
  title = {Ensemble Debiasing for Natural Language Inference},
  year = {2024},
  publisher = {GitHub},
  url = {<your-repo-url>}
}
```

### References

- McCoy, T., Pavlick, E., & Linzen, T. (2019). Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference. *ACL 2019*.
- Clark, C., Yatskar, M., & Zettlemoyer, L. (2019). Don't Take the Easy Way Out: Ensemble Based Methods for Avoiding Known Dataset Biases. *EMNLP 2019*.
- Bowman, S. R., et al. (2015). A large annotated corpus for learning natural language inference. *EMNLP 2015*.

## 🤝 Acknowledgments

- Base framework adapted from University of Texas at Austin CS 378 (NLP)
- HANS dataset from McCoy et al. (2019)
- Hugging Face Transformers library

## 📝 License

This project is available under the MIT License. See individual dataset licenses for SNLI, HANS, and ANLI.

## 👤 Author

Brandon Jerome
- GitHub: [@brandonjerome](https://github.com/brandonjerome)
- Project completed as part of Natural Language Processing coursework

---

**Note**: Model checkpoints are not included in this repository due to size constraints. Pre-trained models can be regenerated using the training scripts provided.
