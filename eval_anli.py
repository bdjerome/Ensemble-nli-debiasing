import os
import json
import numpy as np
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# Label mapping for ANLI
LABEL_MAP = {
    0: 0,  # entailment
    1: 1,  # neutral
    2: 2   # contradiction
}

def load_anli_dataset():
    """Load ANLI dataset from HuggingFace"""
    print(f"Loading ANLI dataset from HuggingFace")
    # Load all three rounds and concatenate
    ds = load_dataset("facebook/anli")
    datasets_list = []
    for split in ['test_r1', 'test_r2', 'test_r3']:
        datasets_list.append(ds[split])
    
    # Concatenate all test sets
    combined = concatenate_datasets(datasets_list)
    print(f"Loaded {len(combined)} total test examples from ANLI")
    return combined

def load_model_and_tokenizer(model_path):
    """Load trained model and tokenizer"""
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    #forcing onto gpu if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    return model, tokenizer, device

def evaluate_on_anli(model, tokenizer, dataset, device, batch_size=32, use_hypothesis_only=False):
    """Run inference on ANLI dataset"""
    predictions = []
    gold_labels = []
    
    print("Running inference...")
    
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i+batch_size]
        
        # Tokenize inputs
        if use_hypothesis_only:
            # For hypothesis-only model, only use hypothesis
            inputs = tokenizer(
                batch['hypothesis'],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
        else:
            # For full model, use premise and hypothesis
            inputs = tokenizer(
                batch['premise'],
                batch['hypothesis'],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        
        predictions.extend(preds)
        
        # Store gold labels
        for label in batch['label']:
            gold_labels.append(label)
    
    return predictions, gold_labels

def compute_metrics(predictions, gold_labels):
    """Compute overall and per-heuristic accuracy"""
    predictions = np.array(predictions)
    gold_labels = np.array(gold_labels)
    
    # Overall accuracy
    overall_acc = (predictions == gold_labels).mean()
    
    # Confusion matrix using sklearn
    cm = confusion_matrix(gold_labels, predictions, labels=[0, 1, 2])
    
    # Entailment vs non-entailment accuracy
    entailment_mask = gold_labels == 0
    non_entailment_mask = (gold_labels == 2) | (gold_labels == 1)
    
    entailment_acc = (predictions[entailment_mask] == gold_labels[entailment_mask]).mean()
    non_entailment_acc = (predictions[non_entailment_mask] == gold_labels[non_entailment_mask]).mean()
    
    return {
        "overall_accuracy": overall_acc,
        "entailment_accuracy": entailment_acc,
        "non_entailment_accuracy": non_entailment_acc,
        "confusion_matrix": cm.tolist()  # Convert to list for JSON serialization
    }

def main():
    # Paths
    full_model_path = "results/full_model"
    hypothesis_model_path = "results/hypothesis_only_model"
    debiased_model_path = "results/ensemble_debiased_alpha_0_5_3epochs"


    #New Evaluation
    snli_hans_debiased_model = "results/ensemble_snli_hans_alpha_0_5"

    snli_hans_base_model = "results/snli_hans_base_model"
    
    # Load ANLI dataset
    anli_dataset = load_anli_dataset()
    
    # Evaluate full model
    print("\n" + "="*50)
    print("EVALUATING FULL MODEL ON ANLI")
    print("="*50)
    model, tokenizer, device = load_model_and_tokenizer(full_model_path)
    predictions, gold_labels = evaluate_on_anli(
        model, tokenizer, anli_dataset, device, use_hypothesis_only=False
    )
    
    # Compute and save metrics
    metrics = compute_metrics(predictions, gold_labels)
    print("\nFull Model Results:")
    print(json.dumps(metrics, indent=2))
    
    os.makedirs("./results", exist_ok=True)
    with open("./results/full_model_anli_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    #Evaluate hypothesis-only model
    print("\n" + "="*50)
    print("EVALUATING HYPOTHESIS-ONLY MODEL ON ANLI")
    print("="*50)
    model, tokenizer, device = load_model_and_tokenizer(hypothesis_model_path)
    predictions, gold_labels = evaluate_on_anli(
        model, tokenizer, anli_dataset, device, use_hypothesis_only=True
    )
    
    # Compute and save metrics
    metrics = compute_metrics(predictions, gold_labels)
    print("\nHypothesis-Only Model Results:")
    print(json.dumps(metrics, indent=2))
    
    with open("./results/hypothesis_only_model_anli_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    #Evaluate debiased ensemble model
    print("\n" + "="*50)
    print("EVALUATING DEBIASED MODEL ON ANLI")
    print("="*50)
    model, tokenizer, device = load_model_and_tokenizer(debiased_model_path)
    predictions, gold_labels = evaluate_on_anli(
        model, tokenizer, anli_dataset, device, use_hypothesis_only=False
    )
    
    # Compute and save metrics
    metrics = compute_metrics(predictions, gold_labels)
    print("\nDebiased Model Results:")
    print(json.dumps(metrics, indent=2))
    
    with open("./results/debiased_model_anli_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)


    #Evaluate SNLI+HANS debiased model
    print("\n" + "="*50)
    print("EVALUATING SNLI-HANS DEBIASED MODEL ON ANLI")
    print("="*50)
    model, tokenizer, device = load_model_and_tokenizer(snli_hans_debiased_model)
    predictions, gold_labels = evaluate_on_anli(
        model, tokenizer, anli_dataset, device, use_hypothesis_only=False
    )
    
    # Compute and save metrics
    metrics = compute_metrics(predictions, gold_labels)
    print("\nDebiased Model Results:")
    print(json.dumps(metrics, indent=2))
    
    with open("./results/snli_hans_debiased_anli_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    #Evaluate SNLI+HANS base model
    print("\n" + "="*50)
    print("EVALUATING SNLI-HANS BASE MODEL ON ANLI")
    print("="*50)
    model, tokenizer, device = load_model_and_tokenizer(snli_hans_base_model)
    predictions, gold_labels = evaluate_on_anli(
        model, tokenizer, anli_dataset, device, use_hypothesis_only=False
    )
    
    # Compute and save metrics
    metrics = compute_metrics(predictions, gold_labels)
    print("\nDebiased Model Results:")
    print(json.dumps(metrics, indent=2))
    
    with open("./results/snli_hans_base_anli_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()