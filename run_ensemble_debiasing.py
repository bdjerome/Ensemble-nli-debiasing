import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    Trainer, TrainingArguments, HfArgumentParser
from transformers.modeling_outputs import SequenceClassifierOutput
from helpers import prepare_dataset_nli, compute_accuracy
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# basic logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_PREPROCESSING_WORKERS = 2

#creating ensemble model class compatible with HuggingFace Trainer
class EnsembleDebiasingModel(nn.Module):
    """
    Ensemble debiasing wrapper for HuggingFace Trainer.

    Improvements over the basic subtraction:
      - Accepts explicit bias_* inputs (hypothesis-only) so bias_model receives
        the inputs it was trained on.
      - Supports a tunable alpha scalar (float or learnable).
      - Supports two combination modes: 'logit' (raw logits subtraction) and
        'logprob' (subtract log-probabilities, numerically stable).
      - Optional normalization / temperature on bias logits.
      - Handles no-label forward (returns logits only, Trainer uses loss when labels provided).
    """

    def __init__(
        self,
        main_model,
        bias_model,
        num_labels=3,
        alpha: float = 1.0,
        learn_alpha: bool = False,
        combine_mode: str = "logit",  # "logit" or "logprob"
        normalize_bias: bool = False,
        bias_temperature: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert combine_mode in ("logit", "logprob"), "combine_mode must be 'logit' or 'logprob'"

        self.main_model = main_model
        self.bias_model = bias_model
        self.num_labels = num_labels
        self.combine_mode = combine_mode
        self.normalize_bias = normalize_bias
        self.bias_temperature = bias_temperature
        self.eps = eps

        # Freeze bias model by default
        for param in self.bias_model.parameters():
            param.requires_grad = False
        self.bias_model.eval()

        # alpha can be a learnable parameter or fixed scalar
        if learn_alpha:
            self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        else:
            self.register_buffer("alpha", torch.tensor(float(alpha)))

        # Mirror main_model config for Trainer compatibility
        self.config = main_model.config

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        # optional bias inputs (pass hypothesis-only inputs here)
        bias_input_ids=None,
        bias_attention_mask=None,
        bias_token_type_ids=None,
        labels=None,
        **kwargs
    ):
        """
        If bias_input_* are provided they will be used to call bias_model.
        Otherwise bias_model is called with the same inputs as main_model (not recommended
        if bias_model was trained on hypothesis-only).

        combine_mode:
          - 'logit': debiased_logits = logits_main - alpha * logits_bias
          - 'logprob': use log_softmax to compute log-probs and subtract scaled log-probs;
                       use nll_loss on the resulting log-probs.
        """

        device = next(self.main_model.parameters()).device

        # Main model forward
        main_outputs = self.main_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        logits_main = main_outputs.logits  # [batch, num_labels]

        # Prepare bias inputs (use explicit bias inputs if provided)
        if bias_input_ids is None:
            bias_input_ids = input_ids
            bias_attention_mask = attention_mask
            bias_token_type_ids = token_type_ids

        # Ensure bias inputs on same device
        if bias_input_ids is not None:
            bias_input_ids = bias_input_ids.to(device)
        if bias_attention_mask is not None:
            bias_attention_mask = bias_attention_mask.to(device)
        if bias_token_type_ids is not None:
            bias_token_type_ids = bias_token_type_ids.to(device)

        # Bias model forward (no grad)
        with torch.no_grad():
            bias_outputs = self.bias_model(
                input_ids=bias_input_ids,
                attention_mask=bias_attention_mask,
                token_type_ids=bias_token_type_ids,
                return_dict=True,
            )
            logits_bias = bias_outputs.logits  # [batch, num_labels]

        # Optionally normalize / temperature-scale bias logits to match scales
        if self.normalize_bias:
            # normalization using batch std
            std = logits_bias.detach().std(dim=1, keepdim=True)  # shape [batch, 1]
            logits_bias = logits_bias / (std + self.eps)

        if self.bias_temperature != 1.0:
            logits_bias = logits_bias / float(self.bias_temperature)

        # Combine models
        if self.combine_mode == "logit":
            # combine in logits space: logits_main - alpha * logits_bias
            alpha = self.alpha if isinstance(self.alpha, torch.Tensor) else torch.tensor(self.alpha, device=device)
            debiased_logits = logits_main - alpha * logits_bias
            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(debiased_logits.view(-1, self.num_labels), labels.view(-1))

        else:  # "logprob" combine
            # Use log-softmax then subtract: logp_comb = logp_main - alpha * logp_bias
            # NLL loss accepts log-probs directly
            logp_main = F.log_softmax(logits_main, dim=-1)
            logp_bias = F.log_softmax(logits_bias, dim=-1)
            alpha = self.alpha if isinstance(self.alpha, torch.Tensor) else torch.tensor(self.alpha, device=device)
            logp_comb = logp_main - alpha * logp_bias
            debiased_logits = logp_comb  # keep log-probs in logits slot for downstream eval
            loss = None
            if labels is not None:
                # negative log-likelihood on log-probs
                loss = F.nll_loss(logp_comb, labels.view(-1))

        # Return SequenceClassifierOutput for Trainer compatibility
        return SequenceClassifierOutput(
            loss=loss,
            logits=debiased_logits
        )

# Tokenize both premise and hypothesis for main model and only hypothesis for bias model
def prepare_dataset_nli_dual(examples, main_tokenizer, bias_tokenizer, max_length=128):
    
    # Tokenize full input for main model
    main_examples = main_tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_attention_mask=True,
    )

    # Tokenize hypothesis-only for bias model
    bias_examples = bias_tokenizer(
        examples["hypothesis"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_attention_mask=True,
    )

    # Choose token_type_ids if tokenizers provide them (some models do not)
    out = {
        # main model fields
        "input_ids": main_examples["input_ids"],
        "attention_mask": main_examples["attention_mask"],
    }
    if "token_type_ids" in main_examples:
        out["token_type_ids"] = main_examples["token_type_ids"]

    # bias model fields (names must match EnsembleDebiasingModel forward)
    out["bias_input_ids"] = bias_examples["input_ids"]
    out["bias_attention_mask"] = bias_examples["attention_mask"]
    if "token_type_ids" in bias_examples:
        out["bias_token_type_ids"] = bias_examples["token_type_ids"]

    # keep labels if present
    if "label" in examples:
        out["labels"] = examples["label"]

    return out

#copying code from run.py for main function
def main():
    argp = HfArgumentParser(TrainingArguments)
    
    argp.add_argument('--main_model', type=str,
                      default='google/electra-small-discriminator',
                      help='Base model for the main (debiased) model')
    argp.add_argument('--bias_model', type=str, required=True,
                      help='Path to the trained biased (hypothesis-only) model checkpoint')
    argp.add_argument('--task', type=str, choices=['nli', 'qa'], required=True,
                      help='Task: nli or qa')
    argp.add_argument('--dataset', type=str, default=None,
                      help='Dataset override')
    argp.add_argument('--max_length', type=int, default=128,
                      help='Maximum sequence length')
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit training examples')
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit evaluation examples')
    
    argp.add_argument('--alpha', type=float, default=1.0, help='Alpha scaling for bias logits subtraction')
    argp.add_argument('--combine_mode', type=str, default='logit', choices=['logit', 'logprob'])
    argp.add_argument('--normalize_bias', action='store_true', help='Normalize bias logits before combining')

    training_args, args = argp.parse_args_into_dataclasses()

    training_args.remove_unused_columns = False  # important to keep bias_input_* fields passed to model
    
    # Device setup
    print(f"PyTorch version: {torch.__version__}")
    if torch.backends.mps.is_available():
        print("MPS (Apple Silicon GPU) is available")
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    elif torch.cuda.is_available():
        print("CUDA GPU is available")
    else:
        print("Using CPU")

    # Dataset selection
    if args.dataset and (args.dataset.endswith('.json') or args.dataset.endswith('.jsonl')):
        dataset = datasets.load_dataset('json', data_files=args.dataset)
        eval_split = 'train'
    elif args.dataset == 'SNLI_HANS':
        # Load SNLI + HANS combined dataset
        snli_dataset = datasets.load_dataset('snli')
        hans_dataset = datasets.load_dataset('json', data_files='./hans/heuristics_train_set.jsonl')

        #converting HANS column names to match SNLI dataset
        def convert_hans_to_snli(example):
            if example['gold_label'] == 'entailment':
                example['label'] = 0
            elif example['gold_label'] == 'non-entailment':
                example['label'] = 2
            return example

        hans_dataset = hans_dataset.map(convert_hans_to_snli)
        # Cast label to match SNLI's ClassLabel type
        from datasets import ClassLabel
        hans_dataset = hans_dataset.cast_column('label', 
            ClassLabel(names=['entailment', 'neutral', 'contradiction']))
        hans_dataset = hans_dataset.rename_column('sentence1', 'premise')
        hans_dataset = hans_dataset.rename_column('sentence2', 'hypothesis')


        drop_cols = [c for c in hans_dataset['train'].column_names if c not in ("premise","hypothesis","label","heuristic")]
        if drop_cols:
            hans_dataset['train'] = hans_dataset['train'].remove_columns(drop_cols)

        # Filter SNLI to only include examples with label != -1
        snli_filtered = snli_dataset.filter(lambda ex: ex['label'] != -1)

        #combine SNLI and HANS datasets
        combined_train = datasets.concatenate_datasets([snli_filtered['train'], hans_dataset['train']])
        # Wrap back into DatasetDict for compatibility with rest of code
        dataset = datasets.DatasetDict({'train': combined_train, 'validation': snli_filtered['validation']})
        eval_split = 'validation'
        print(f"Combined SNLI and HANS datasets: {len(dataset['train'])} training examples")
    else:
        default_datasets = {'qa': ('squad',), 'nli': ('snli',)}
        dataset_id = tuple(args.dataset.split(':')) if args.dataset is not None else default_datasets[args.task]
        eval_split = 'validation_matched' if dataset_id == ('glue', 'mnli') else 'validation'
        dataset = datasets.load_dataset(*dataset_id)
    
    # Load models
    print(f"Loading main model from: {args.main_model}")
    main_model = AutoModelForSequenceClassification.from_pretrained(args.main_model, num_labels=3)
    
    print(f"Loading biased model from: {args.bias_model}")
    bias_model = AutoModelForSequenceClassification.from_pretrained(args.bias_model, num_labels=3)
    
    # Create ensemble model
    model = EnsembleDebiasingModel(main_model, bias_model, num_labels=3, alpha=args.alpha,
                                    combine_mode=args.combine_mode, normalize_bias=args.normalize_bias)
    
    # Make tensors contiguous if needed
    for m in [main_model, bias_model]:
        if hasattr(m, 'electra'):
            for param in m.electra.parameters():
                if not param.is_contiguous():
                    param.data = param.data.contiguous()
    
    # Loading tokenizer for both models
    bias_tokenizer = AutoTokenizer.from_pretrained(args.bias_model, use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained(args.main_model, use_fast=True)

    # Prepare dataset
    prepare_train_dataset = lambda exs: prepare_dataset_nli_dual(exs, tokenizer, bias_tokenizer, args.max_length)
    prepare_eval_dataset  = lambda exs: prepare_dataset_nli_dual(exs, tokenizer, bias_tokenizer, args.max_length)

    print("Preprocessing data...")
    #remove examples with label = -1
    if args.dataset is None or args.dataset == 'snli':
        dataset = dataset.filter(lambda ex: ex['label'] != -1)
    
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


    # Sanity checks: ensure bias_input fields exist
    if training_args.do_train and train_dataset_featurized is not None:
        sample = train_dataset_featurized[0]
        logger.info(f"Train featurized columns: {train_dataset_featurized.column_names}")
        assert "bias_input_ids" in train_dataset_featurized.column_names, "bias_input_ids missing from featurized train dataset"
    if training_args.do_eval and eval_dataset_featurized is not None:
        logger.info(f"Eval featurized columns: {eval_dataset_featurized.column_names}")
        assert "bias_input_ids" in eval_dataset_featurized.column_names, "bias_input_ids missing from featurized eval dataset"


    # Compute metrics
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
        trainer.train()
        # Save only the main model (not the bias model)
        main_model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)

    # Evaluate
    if training_args.do_eval:
        results = trainer.evaluate()
        print('Evaluation results:')
        print(results)

        os.makedirs(training_args.output_dir, exist_ok=True)

        with open(os.path.join(training_args.output_dir, 'eval_metrics.json'), encoding='utf-8', mode='w') as f:
            json.dump(results, f)

        with open(os.path.join(training_args.output_dir, 'eval_predictions.jsonl'), encoding='utf-8', mode='w') as f:
            for i, example in enumerate(eval_dataset):
                example_with_prediction = dict(example)
                example_with_prediction['predicted_scores'] = eval_predictions.predictions[i].tolist()
                example_with_prediction['predicted_label'] = int(eval_predictions.predictions[i].argmax())
                f.write(json.dumps(example_with_prediction))
                f.write('\n')


if __name__ == "__main__":
    main()
