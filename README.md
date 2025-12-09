# Abstract

Natural Language Inference (NLI) models 
can achieve high accuracy for in domain 
problems,  but  can  generalize  poorly  to 
adversarial  settings  when  they  learn 
embedded patterns in the training set that 
are  not  displayed  in  a  new  test  set.  We 
investigate  a  simple  debiased  model 
implementation  in  conjunction  with  data 
augmentation to improve generalization to 
challenging  evaluation  datasets.  Our 
technique shows modest improvements on 
template-based  heuristic  challenges  but 
fails  to  yield  meaningful  gains  on  more 
complex reasoning challenges, revealing a 
tension between mitigating known bias and 
generating  robust  reasoning.  We  provide 
empirical evidence and reproducible code 
showing that suppressing artifacts alone is 
insufficient  for  building  next  level  NLI 
systems.

View the full paper at ensemble_debiasing_nli_report.pdf in the repository

## Methodology Overview

This project implements an ensemble-based debiasing approach for NLI models that:
- Trains a hypothesis-only "bias" model to capture dataset artifacts
- Combines it with a full NLI model using weighted ensemble debiasing
- Achieves significant improvements on out-of-distribution test sets (HANS)
- Maintains strong in-distribution performance (SNLI)
- Maintains but does not improve out-of-distribution challenge set (ANLI)

## Project Structure

```
├── eval_hans.py                    # HANS evaluation script
├── eval_anli.py                    # ANLI evaluation script  
├── run_ensemble_debiasing.py       # Main ensemble debiasing training
├── run_alpha_sweep.py              # Hyperparameter tuning (alpha values)
├── HANS_leakage_check.ipynb        # Data leakage investigation notebook
├── paper_visuals.ipynb             # Visualization used in my paper
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

## Quick Start

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
Will have to update file paths in the .py file to point to correct model files
```bash
python eval_hans.py
```

#### Evaluate on ANLI
Will have to update file paths in the .py file to point to correct model files
```bash
python eval_anli.py
```

## Repository Contents

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

### References

- McCoy, T., Pavlick, E., & Linzen, T. (2019). Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference. *ACL 2019*.
- Clark, C., Yatskar, M., & Zettlemoyer, L. (2019). Don't Take the Easy Way Out: Ensemble Based Methods for Avoiding Known Dataset Biases. *EMNLP 2019*.
- Bowman, S. R., et al. (2015). A large annotated corpus for learning natural language inference. *EMNLP 2015*.

## Acknowledgments

- Base framework adapted from University of Texas at Austin CS 378 (NLP)
- HANS dataset from McCoy et al. (2019)
- Hugging Face Transformers library

## License

This project is available under the MIT License. See individual dataset licenses for SNLI, HANS, and ANLI.

## 👤 Author

Brandon Jerome
- GitHub: [@brandonjerome](https://github.com/brandonjerome)
- Project completed as part of Natural Language Processing coursework

---

**Note**: Model checkpoints are not included in this repository due to size constraints. Pre-trained models can be regenerated using the training scripts provided.
