import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    Trainer, TrainingArguments, HfArgumentParser
import evaluate
from helpers import compute_accuracy
import os
import json
import torch

NUM_PREPROCESSING_WORKERS = 2


def prepare_dataset_nli_hypothesis_only(examples, tokenizer, max_seq_length=None):
    """
    Preprocesses NLI dataset by tokenizing ONLY the hypothesis (ignoring premise)
    """
    max_seq_length = tokenizer.model_max_length if max_seq_length is None else max_seq_length

    tokenized_examples = tokenizer(
        examples['hypothesis'],  # Only hypothesis
        truncation=True,
        max_length=max_seq_length,
        padding='max_length'
    )

    tokenized_examples['label'] = examples['label']
    return tokenized_examples


def main():
    argp = HfArgumentParser(TrainingArguments)
    
    argp.add_argument('--model', type=str,
                      default='google/electra-small-discriminator',
                      help="""Base model to fine-tune.""")
    argp.add_argument('--dataset', type=str, default='snli',
                      help="""Dataset to use (default: snli).""")
    argp.add_argument('--max_length', type=int, default=128,
                      help="""Maximum sequence length.""")
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit the number of examples to train on.')
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit the number of examples to evaluate on.')

    training_args, args = argp.parse_args_into_dataclasses()
    
    # Device detection
    print(f"PyTorch version: {torch.__version__}")
    if torch.backends.mps.is_available():
        print("MPS (Apple Silicon GPU) is available and will be used automatically by Trainer")
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    elif torch.cuda.is_available():
        print("CUDA GPU is available and will be used automatically by Trainer")
    else:
        print("No GPU available, using CPU")

    print("\n" + "="*70)
    print("HYPOTHESIS-ONLY TRAINING MODE")
    print("="*70 + "\n")

    # Load dataset
    if args.dataset.endswith('.json') or args.dataset.endswith('.jsonl'):
        dataset = datasets.load_dataset('json', data_files=args.dataset)
        eval_split = 'train'
    else:
        if args.dataset == 'snli':
            dataset = datasets.load_dataset('snli')
        elif args.dataset == 'mnli':
            dataset = datasets.load_dataset('glue', 'mnli')
        else:
            dataset = datasets.load_dataset(args.dataset)
        eval_split = 'validation_matched' if args.dataset == 'mnli' else 'validation'
    
    # Initialize model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, 
        num_labels=3  # entailment, neutral, contradiction
    )
    
    # Make tensor contiguous if needed
    if hasattr(model, 'electra'):
        for param in model.electra.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Prepare dataset preprocessing function (hypothesis-only)
    prepare_train_dataset = prepare_eval_dataset = \
        lambda exs: prepare_dataset_nli_hypothesis_only(exs, tokenizer, args.max_length)

    print("Preprocessing data (hypothesis-only)...")
    
    # Remove SNLI examples with no label
    if args.dataset == 'snli':
        dataset = dataset.filter(lambda ex: ex['label'] != -1)
    
    # Prepare training dataset
    train_dataset = None
    eval_dataset = None
    train_dataset_featurized = None
    eval_dataset_featurized = None
    
    if training_args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        train_dataset_featurized = train_dataset.map(
            prepare_train_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=train_dataset.column_names
        )
        print(f"Training samples: {len(train_dataset)}")
    
    if training_args.do_eval:
        eval_dataset = dataset[eval_split]
        if args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        eval_dataset_featurized = eval_dataset.map(
            prepare_eval_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=eval_dataset.column_names
        )
        print(f"Evaluation samples: {len(eval_dataset)}")

    # Setup metrics
    eval_predictions = None
    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_accuracy(eval_preds)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_featurized,
        eval_dataset=eval_dataset_featurized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions
    )

    print(f"Model device: {next(model.parameters()).device}")
    
    # Train
    if training_args.do_train:
        print("\nStarting hypothesis-only training...")
        trainer.train()
        trainer.save_model()
        print("Model saved!")

    # Evaluate
    if training_args.do_eval:
        print("\nEvaluating hypothesis-only model...")
        results = trainer.evaluate()

        print('\n' + "="*70)
        print('HYPOTHESIS-ONLY EVALUATION RESULTS:')
        print("="*70)
        print(f"Accuracy: {results['eval_accuracy']:.4f} ({results['eval_accuracy']*100:.2f}%)")
        print(f"Loss: {results['eval_loss']:.4f}")
        print("="*70 + "\n")

        # Save results
        os.makedirs(training_args.output_dir, exist_ok=True)

        with open(os.path.join(training_args.output_dir, 'eval_metrics.json'), encoding='utf-8', mode='w') as f:
            json.dump(results, f, indent=2)

        with open(os.path.join(training_args.output_dir, 'eval_predictions.jsonl'), encoding='utf-8', mode='w') as f:
            for i, example in enumerate(eval_dataset):
                example_with_prediction = dict(example)
                example_with_prediction['predicted_scores'] = eval_predictions.predictions[i].tolist()
                example_with_prediction['predicted_label'] = int(eval_predictions.predictions[i].argmax())
                f.write(json.dumps(example_with_prediction))
                f.write('\n')
        
        print(f"Results saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
