import subprocess
import json
import os

def run_hans_evaluation(predictions_file, output_file):
    """
    Run the HANS evaluation script and capture output
    
    Args:
        predictions_file: Path to the predictions file
        output_file: Path to save the evaluation results
    """
    
    # Run the evaluation script
    result = subprocess.run(
        ['python', 'hans/evaluate_heur_output.py', predictions_file],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    if result.returncode != 0:
        print(f"Error running evaluation: {result.stderr}")
        return
    
    # Parse the output
    output_lines = result.stdout.strip().split('\n')
    
    results = {
        "heuristic_entailed": {},
        "heuristic_non_entailed": {},
        "subcase": {},
        "template": {}
    }
    
    current_section = None
    
    for line in output_lines:
        line = line.strip()
        if not line:
            continue
            
        if line == "Heuristic entailed results:":
            current_section = "heuristic_entailed"
        elif line == "Heuristic non-entailed results:":
            current_section = "heuristic_non_entailed"
        elif line == "Subcase results:":
            current_section = "subcase"
        elif line == "Template results:":
            current_section = "template"
        elif ":" in line and current_section:
            parts = line.rsplit(": ", 1)
            if len(parts) == 2:
                key, value = parts
                try:
                    results[current_section][key] = float(value)
                except ValueError:
                    pass
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Also save raw output to txt
    txt_output = output_file.replace('.json', '.txt')
    with open(txt_output, 'w') as f:
        f.write(result.stdout)
    
    print(f"Results saved to {output_file} and {txt_output}")
    print("\n" + result.stdout)
    
    return results

if __name__ == "__main__":
    # Evaluate both models
    print("="*60)
    print("EVALUATING FULL MODEL ON HANS")
    print("="*60)
    run_hans_evaluation(
        'results/full_model_predictions.txt',
        'results/full_model_hans_evaluation.json'
    )
    
    print("\n" + "="*60)
    print("EVALUATING HYPOTHESIS-ONLY MODEL ON HANS")
    print("="*60)
    run_hans_evaluation(
        'results/hypothesis_only_predictions.txt',
        'results/hypothesis_only_hans_evaluation.json'
    )
