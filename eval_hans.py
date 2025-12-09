import os
import json
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# Label mapping for NLI
LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    "non-entailment": 2  # HANS uses "non-entailment" instead of "contradiction"
}

def load_hans_dataset(hans_path):
    """Load HANS dataset from jsonl file"""
    print(f"Loading HANS dataset from {hans_path}")
    dataset = load_dataset('json', data_files=hans_path)
    print(f"Loaded {len(dataset['train'])} examples")
    return dataset['train']

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

def evaluate_on_hans(model, tokenizer, dataset, device, batch_size=32, use_hypothesis_only=False):
    """Run inference on HANS dataset"""
    predictions = []
    gold_labels = []
    heuristics = []
    subtypes = []
    
    print("Running inference...")
    
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i+batch_size]
        
        # Tokenize inputs
        if use_hypothesis_only:
            # For hypothesis-only model, only use hypothesis
            inputs = tokenizer(
                batch['sentence2'],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
        else:
            # For full model, use premise and hypothesis
            inputs = tokenizer(
                batch['sentence1'],
                batch['sentence2'],
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
        
        # Store gold labels and metadata
        for label, heuristic, subtype in zip(batch['gold_label'], batch['heuristic'], batch['subcase']):
            gold_labels.append(LABEL_MAP[label])
            heuristics.append(heuristic)
            subtypes.append(subtype)
    
    return predictions, gold_labels, heuristics, subtypes

def compute_metrics(predictions, gold_labels, heuristics, subtypes):
    """Compute overall and per-heuristic accuracy"""
    predictions = np.array(predictions)
    gold_labels = np.array(gold_labels)
    
    # Overall accuracy
    overall_acc = (predictions == gold_labels).mean()
    
    # Confusion matrix using sklearn
    cm = confusion_matrix(gold_labels, predictions, labels=[0, 1, 2])
    
    # Per-heuristic accuracy
    heuristic_accuracies = {}
    for heuristic in set(heuristics):
        mask = np.array([h == heuristic for h in heuristics])
        heuristic_acc = (predictions[mask] == gold_labels[mask]).mean()
        heuristic_accuracies[heuristic] = heuristic_acc
    
    # Entailment vs non-entailment accuracy
    entailment_mask = gold_labels == 0
    non_entailment_mask = gold_labels == 2
    
    entailment_acc = (predictions[entailment_mask] == gold_labels[entailment_mask]).mean()
    non_entailment_acc = (predictions[non_entailment_mask] == gold_labels[non_entailment_mask]).mean()
    
    return {
        "overall_accuracy": overall_acc,
        "entailment_accuracy": entailment_acc,
        "non_entailment_accuracy": non_entailment_acc,
        "heuristic_accuracies": heuristic_accuracies,
        "confusion_matrix": cm.tolist()  # Convert to list for JSON serialization
    }

#formatting predictions so I can use the official HANS evaluation script
def save_predictions(predictions, dataset, output_file):
    """Save predictions in HANS format for official evaluation"""
    with open(output_file, 'w') as f:
        f.write("pairID,gold_label\n")
        for pred, example in zip(predictions, dataset):
            pred_label = ["entailment", "non-entailment", "non-entailment"][pred]
            f.write(f"{example['pairID']},{pred_label}\n")
    print(f"Predictions saved to {output_file}")

def main():
    # Paths
    hans_path = "./hans/heuristics_evaluation_set.jsonl"
    full_model_path = "results/full_model"
    hypothesis_model_path = "results/hypothesis_only_model"
    debiased_model_path = "results/ensemble_debiased_alpha_0_5_3epochs"


    #New Evaluation
    snli_hans_debiased_model = "results/ensemble_snli_hans_alpha_0_5"

    snli_hans_base_model = "results/snli_hans_base_model"
    
    # Load HANS dataset
    hans_dataset = load_hans_dataset(hans_path)
    
    #Evaluate full model
    print("\n" + "="*50)
    print("EVALUATING FULL MODEL")
    print("="*50)
    model, tokenizer, device = load_model_and_tokenizer(full_model_path)
    predictions, gold_labels, heuristics, subtypes = evaluate_on_hans(
        model, tokenizer, hans_dataset, device, use_hypothesis_only=False
    )
    
    # Compute and save metrics
    metrics = compute_metrics(predictions, gold_labels, heuristics, subtypes)
    print("\nFull Model Results:")
    print(json.dumps(metrics, indent=2))
    
    os.makedirs("./results", exist_ok=True)
    with open("./results/full_model_hans_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    save_predictions(predictions, hans_dataset, "./results/full_model_predictions.txt")
    
    #Evaluate hypothesis-only model
    print("\n" + "="*50)
    print("EVALUATING HYPOTHESIS-ONLY MODEL")
    print("="*50)
    model, tokenizer, device = load_model_and_tokenizer(hypothesis_model_path)
    predictions, gold_labels, heuristics, subtypes = evaluate_on_hans(
        model, tokenizer, hans_dataset, device, use_hypothesis_only=True
    )
    
    # Compute and save metrics
    metrics = compute_metrics(predictions, gold_labels, heuristics, subtypes)
    print("\nHypothesis-Only Model Results:")
    print(json.dumps(metrics, indent=2))
    
    with open("./results/hypothesis_only_model_hans_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    save_predictions(predictions, hans_dataset, "./results/hypothesis_only_predictions.txt")
    
    #Debiased model evaluation
    print("\n" + "="*50)
    print("EVALUATING DEBIASED MODEL")
    print("="*50)
    model, tokenizer, device = load_model_and_tokenizer(debiased_model_path)
    predictions, gold_labels, heuristics, subtypes = evaluate_on_hans(
        model, tokenizer, hans_dataset, device, use_hypothesis_only=False
    )
    
    # Compute and save metrics
    metrics = compute_metrics(predictions, gold_labels, heuristics, subtypes)
    print("\nDebiased Model Results:")
    print(json.dumps(metrics, indent=2))
    
    with open("./results/debiased_model_hans_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    save_predictions(predictions, hans_dataset, "./results/debiased_model_predictions.txt")

    #New Debiased Model Evaluation
    print("\n" + "="*50)
    print("EVALUATING NEW-DEBIASED MODEL")
    print("="*50)
    model, tokenizer, device = load_model_and_tokenizer(snli_hans_debiased_model)
    predictions, gold_labels, heuristics, subtypes = evaluate_on_hans(
        model, tokenizer, hans_dataset, device, use_hypothesis_only=False
    )
    
    # Compute and save metrics
    metrics = compute_metrics(predictions, gold_labels, heuristics, subtypes)
    print("\n New Debiased Model Results:")
    print(json.dumps(metrics, indent=2))
    
    with open("./results/snli_hans_debiased_hans_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    save_predictions(predictions, hans_dataset, "./results/snli_hans_debiased_model_predictions.txt")

    #Base with hans snli concat model evaluation
    print("\n" + "="*50)
    print("EVALUATING NEW-DEBIASED MODEL")
    print("="*50)
    model, tokenizer, device = load_model_and_tokenizer(snli_hans_base_model)
    predictions, gold_labels, heuristics, subtypes = evaluate_on_hans(
        model, tokenizer, hans_dataset, device, use_hypothesis_only=False
    )
    
    # Compute and save metrics
    metrics = compute_metrics(predictions, gold_labels, heuristics, subtypes)
    print("\n New Debiased Model Results:")
    print(json.dumps(metrics, indent=2))
    
    with open("./results/snli_hans_base_hans_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    save_predictions(predictions, hans_dataset, "./results/snli_hans_base_predictions.txt")

if __name__ == "__main__":
    main()