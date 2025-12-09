import subprocess
import json
import os
import sys
from pathlib import Path
import argparse


def run_training(alpha, config):
    """Run one training job for a specific alpha value."""
    safe_alpha = str(alpha).replace('.', '_')
    output_dir = config['output_root'] / f"alpha_{safe_alpha}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / "run.log"
    
    cmd = [
        sys.executable,
        "fp-dataset-artifacts/run_ensemble_debiasing.py",
        "--main_model", config['main_model'],
        "--bias_model", config['bias_model'],
        "--task", "nli",
        "--dataset", "snli",
        "--do_train", "--do_eval",
        "--output_dir", str(output_dir),
        "--max_train_samples", str(config['max_train_samples']),
        "--max_eval_samples", str(config['max_eval_samples']),
        "--per_device_train_batch_size", str(config['batch_size']),
        "--per_device_eval_batch_size", str(config['batch_size']),
        "--num_train_epochs", str(config['num_epochs']),
        "--alpha", str(alpha),
        "--combine_mode", config['combine_mode'],
    ]
    
    print(f"\n{'='*60}")
    print(f"Running alpha={alpha} -> {output_dir}")
    print(f"Command: {' '.join(cmd[:10])}...")
    print(f"{'='*60}\n")
    
    with open(log_file, 'w') as f:
        result = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=config['project_root']
        )
    
    if result.returncode != 0:
        print(f"Warning: Training for alpha={alpha} failed (exit code {result.returncode})")
        print(f"   Check log: {log_file}")
        return None
    
    print(f"Training completed for alpha={alpha}")
    return output_dir


def run_hans_evaluation_direct(alpha, output_dir, config):
    """Run HANS evaluation using eval_hans_cli.py on the trained model."""
    hans_log = output_dir / "hans_run.log"
    
    # Use the saved model from training
    trained_model = str(output_dir)
    
    cmd = [
        sys.executable,
        "eval_hans_cli.py",
        "--model_path", trained_model,
        "--hans_path", "hans/heuristics_evaluation_set.jsonl",
        "--output_file", str(output_dir / "hans_predictions.txt"),
        "--metrics_file", str(output_dir / "hans_metrics.json"),
    ]
    
    print(f"   Running HANS evaluation...")
    
    with open(hans_log, 'w') as f:
        result = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=config['project_root']
        )
    
    if result.returncode != 0:
        print(f"   HANS evaluation failed (exit code {result.returncode})")
        print(f"      Check log: {hans_log}")
        return None
    
    hans_metrics_file = output_dir / "hans_metrics.json"
    if hans_metrics_file.exists():
        print(f"   HANS evaluation completed: {hans_metrics_file}")
        return hans_metrics_file
    
    return None


def convert_to_hans_csv(output_dir, label_mapping=None):
    """Convert eval_predictions.jsonl to HANS CSV format."""
    if label_mapping is None:
        label_mapping = {0: 'entailment', 1: 'non-entailment', 2: 'non-entailment'}
    
    predictions_file = output_dir / "eval_predictions.jsonl"
    hans_csv = output_dir / "hans_predictions.csv"
    
    if not predictions_file.exists():
        print(f"   No predictions file found: {predictions_file}")
        return None
    
    try:
        with open(predictions_file, 'r', encoding='utf-8') as inf, \
             open(hans_csv, 'w', encoding='utf-8') as outf:
            
            outf.write('pairID,gold_label\n')
            
            for line in inf:
                try:
                    j = json.loads(line)
                    # Try multiple possible field names for pair ID
                    pair_id = (j.get('pairID') or j.get('pair_id') or 
                              j.get('pairId') or j.get('pair'))
                    pred = int(j.get('predicted_label', j.get('pred', 1)))
                    
                    if pair_id is None:
                        continue
                    
                    label = label_mapping.get(pred, 'non-entailment')
                    outf.write(f'{pair_id},{label}\n')
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    continue
        
        print(f"   Converted to HANS CSV: {hans_csv}")
        return hans_csv
        
    except Exception as e:
        print(f"   Error converting to HANS CSV: {e}")
        return None


def run_hans_evaluation(hans_csv, output_dir, config):
    """Run the official HANS evaluator script."""
    hans_output = output_dir / "hans_eval.txt"
    hans_script = config['project_root'] / "hans" / "evaluate_heur_output.py"
    
    if not hans_script.exists():
        print(f"   HANS evaluator not found: {hans_script}")
        return None
    
    if hans_csv is None or not hans_csv.exists():
        print(f"   No HANS CSV to evaluate}")
        return None
    
    try:
        with open(hans_output, 'w') as f:
            result = subprocess.run(
                [sys.executable, str(hans_script), str(hans_csv)],
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=config['project_root']
            )
        
        if result.returncode == 0:
            print(f"   HANS evaluation: {hans_output}")
            return hans_output
        else:
            print(f"   HANS evaluation failed (exit code {result.returncode})")
            return None
            
    except Exception as e:
        print(f"   Error running HANS evaluation: {e}")
        return None


def extract_snli_accuracy(output_dir):
    """Extract SNLI dev accuracy from eval_metrics.json."""
    metrics_file = output_dir / "eval_metrics.json"
    
    if not metrics_file.exists():
        return None
    
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # Try multiple possible key names
        for key in ('eval_accuracy', 'accuracy', 'eval_acc', 'acc'):
            if key in metrics:
                return metrics[key]
        
        return None
        
    except Exception as e:
        print(f"   Error reading metrics: {e}")
        return None


def parse_hans_results(hans_output_file):
    """Parse HANS evaluator text output and extract key metrics."""
    if hans_output_file is None or not hans_output_file.exists():
        return {}
    
    try:
        with open(hans_output_file, 'r') as f:
            content = f.read()
        
        results = {'overall': None, 'heuristics': {}}
        
        # Parse overall accuracy (first line usually has it)
        lines = content.strip().split('\n')
        for line in lines:
            if 'overall' in line.lower() or line.startswith('Accuracy'):
                # Try to extract number
                parts = line.split()
                for part in parts:
                    try:
                        acc = float(part.strip('%,'))
                        if 0 <= acc <= 100:
                            results['overall'] = acc / 100.0  # normalize to [0,1]
                            break
                    except ValueError:
                        continue
        
        return results
        
    except Exception as e:
        print(f"   Error parsing HANS results: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description='Run alpha sweep for ensemble debiasing')
    parser.add_argument('--alphas', nargs='+', type=float, 
                       default=[0.0, 0.25, 0.5, 1.0, 2.0],
                       help='Alpha values to sweep over')
    parser.add_argument('--main_model', type=str,
                       default='./fp-dataset-artifacts/full_model',
                       help='Path to main (full) model')
    parser.add_argument('--bias_model', type=str,
                       default='./fp-dataset-artifacts/hypothesis_only_model',
                       help='Path to bias (hypothesis-only) model')
    parser.add_argument('--output_root', type=str,
                       default='results/ensemble_alphas',
                       help='Root directory for outputs')
    parser.add_argument('--max_train_samples', type=int, default=None,
                       help='Max training samples (None = use full dataset)')
    parser.add_argument('--max_eval_samples', type=int, default=None,
                       help='Max eval samples (None = use full dataset)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training/eval')
    parser.add_argument('--num_epochs', type=int, default=1,
                       help='Number of epochs to train')
    parser.add_argument('--combine_mode', type=str, default='logit',
                       choices=['logit', 'logprob'],
                       help='Combination mode for ensemble')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip alphas that already have results')
    
    args = parser.parse_args()
    
    # Setup config
    project_root = Path.cwd()
    config = {
        'project_root': project_root,
        'main_model': args.main_model,
        'bias_model': args.bias_model,
        'output_root': Path(args.output_root),
        'max_train_samples': args.max_train_samples,
        'max_eval_samples': args.max_eval_samples,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'combine_mode': args.combine_mode,
    }
    
    config['output_root'].mkdir(parents=True, exist_ok=True)
    
    # Summary files
    snli_summary_file = config['output_root'] / "summary_snli.csv"
    hans_summary_file = config['output_root'] / "summary_hans.csv"
    
    # Initialize summary files
    with open(snli_summary_file, 'w') as f:
        f.write('alpha,snli_dev_accuracy\n')
    
    with open(hans_summary_file, 'w') as f:
        f.write('alpha,hans_overall_accuracy\n')
    
    print(f"\nStarting alpha sweep")
    print(f"   Alphas: {args.alphas}")
    print(f"   Main model: {config['main_model']}")
    print(f"   Bias model: {config['bias_model']}")
    print(f"   Output: {config['output_root']}")
    print(f"   Train samples: {config['max_train_samples'] or 'full dataset'}")
    print(f"   Eval samples: {config['max_eval_samples'] or 'full dataset'}\n")
    
    results = []
    
    for alpha in args.alphas:
        safe_alpha = str(alpha).replace('.', '_')
        output_dir = config['output_root'] / f"alpha_{safe_alpha}"
        
        # Skip if already exists
        if args.skip_existing and (output_dir / "eval_metrics.json").exists():
            print(f"\nSkipping alpha={alpha} (already exists)")
            snli_acc = extract_snli_accuracy(output_dir)
            if snli_acc is not None:
                with open(snli_summary_file, 'a') as f:
                    f.write(f'{alpha},{snli_acc}\n')
            continue
        
        # 1. Train
        output_dir = run_training(alpha, config)
        if output_dir is None:
            results.append({'alpha': alpha, 'status': 'failed'})
            continue
        
        # 2. Extract SNLI accuracy (from training eval)
        snli_acc = extract_snli_accuracy(output_dir)
        
        # 3. Run HANS evaluation on trained model
        hans_metrics_file = run_hans_evaluation_direct(alpha, output_dir, config)
        
        hans_acc = None
        if hans_metrics_file and hans_metrics_file.exists():
            try:
                with open(hans_metrics_file, 'r') as f:
                    hans_metrics = json.load(f)
                    hans_acc = hans_metrics.get('overall_accuracy')
            except:
                pass
        
        # 4. Save to summaries
        if snli_acc is not None:
            with open(snli_summary_file, 'a') as f:
                f.write(f'{alpha},{snli_acc}\n')
            print(f"   SNLI dev accuracy: {snli_acc:.4f}")
        
        if hans_acc is not None:
            with open(hans_summary_file, 'a') as f:
                f.write(f'{alpha},{hans_acc}\n')
            print(f"   HANS overall accuracy: {hans_acc:.4f}")
        
        results.append({
            'alpha': alpha,
            'status': 'success',
            'snli_acc': snli_acc,
            'hans_acc': hans_acc,
            'output_dir': str(output_dir)
        })
    
    # Final summary
    print(f"\n{'='*60}")
    print("Alpha sweep completed!")
    print(f"{'='*60}")
    print(f"\nSummary files:")
    print(f"  - SNLI: {snli_summary_file}")
    print(f"  - HANS: {hans_summary_file}")
    print(f"\nResults:")
    print(f"{'Alpha':<8} {'Status':<10} {'SNLI Acc':<12} {'HANS Acc':<12}")
    print("-" * 50)
    for r in results:
        status = r['status']
        snli = f"{r.get('snli_acc', 0):.4f}" if r.get('snli_acc') else "N/A"
        hans = f"{r.get('hans_acc', 0):.4f}" if r.get('hans_acc') else "N/A"
        print(f"{r['alpha']:<8} {status:<10} {snli:<12} {hans:<12}")
    
    print(f"\nNext steps:")
    print(f"   - Review logs in: {config['output_root']}/alpha_*/run.log")
    print(f"   - HANS detailed results: {config['output_root']}/alpha_*/hans_eval.txt")
    print(f"   - Load summaries to compare alphas and pick best value\n")


if __name__ == "__main__":
    main()
