#!/usr/bin/env python
# 07_evaluate_model.py
# Purpose: Evaluate the fine-tuned model on the test set

import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from rouge_score import rouge_scorer
from datasets import load_from_disk
from transformers import AutoTokenizer
from unsloth import FastModel
import config

def load_test_samples():
    """Load test samples from the dataset splits"""
    dataset_path = os.path.join(config.DATA_DIR, "dataset_splits.json")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset splits not found at {dataset_path}. "
                              f"Please run 01_data_load.py first.")
    
    with open(dataset_path, "r") as f:
        splits = json.load(f)
    
    test_samples = splits["test"]
    print(f"Loaded {len(test_samples)} test samples")
    
    return test_samples

def load_base_model():
    """Load the base model for comparison"""
    base_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "base_model.pt")
    if not os.path.exists(base_checkpoint_path):
        raise FileNotFoundError(f"Base model not found at {base_checkpoint_path}. "
                              f"Please run 03_model_load.py first.")
    
    print(f"Loading base model from {base_checkpoint_path}...")
    checkpoint = torch.load(base_checkpoint_path)
    model = checkpoint["model"]
    tokenizer = checkpoint["tokenizer"]
    cpu_only = checkpoint.get("cpu_only", False)
    
    return model, tokenizer, cpu_only

def load_finetuned_model():
    """Load the fine-tuned model"""
    if not os.path.exists(config.MODEL_OUTPUT_DIR):
        raise FileNotFoundError(f"Fine-tuned model not found at {config.MODEL_OUTPUT_DIR}. "
                              f"Please run 06_train_model.py first.")
    
    print(f"Loading fine-tuned model from {config.MODEL_OUTPUT_DIR}...")
    model = FastModel.from_pretrained(config.MODEL_OUTPUT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_OUTPUT_DIR)
    
    # Determine CPU/GPU mode
    cpu_only = not torch.cuda.is_available()
    
    return model, tokenizer, cpu_only

def evaluate_model(model, tokenizer, test_samples, model_name="model", cpu_only=False, num_samples=None):
    """Evaluate a model on test samples and compute metrics"""
    # Limit the number of samples for evaluation if specified
    if num_samples is not None:
        test_samples = test_samples[:min(num_samples, len(test_samples))]
    
    # Set device
    device = "cuda" if torch.cuda.is_available() and not cpu_only else "cpu"
    model = model.to(device)
    
    # Initialize Rouge scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Initialize result lists
    results = []
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []
    generation_times = []
    
    # Evaluate each sample
    print(f"Evaluating {model_name} on {len(test_samples)} test samples...")
    
    for i, sample in enumerate(test_samples):
        # Create the instruction prompt
        instruction = config.INSTRUCTION
        prompt = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"
        
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate text
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.GENERATION_MAX_NEW_TOKENS,
                temperature=config.GENERATION_TEMPERATURE,
                top_p=config.GENERATION_TOP_P,
                top_k=config.GENERATION_TOP_K,
            )
        generation_time = time.time() - start_time
        
        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = generated_text.split("<start_of_turn>model\n")[-1].strip()
        
        # Get the reference text
        reference_text = sample["caption"]
        
        # Calculate ROUGE scores
        rouge_scores = scorer.score(reference_text, generated_text)
        
        # Store scores
        rouge_1_scores.append(rouge_scores['rouge1'].fmeasure)
        rouge_2_scores.append(rouge_scores['rouge2'].fmeasure)
        rouge_l_scores.append(rouge_scores['rougeL'].fmeasure)
        generation_times.append(generation_time)
        
        # Store the result
        results.append({
            "sample_id": i,
            "reference": reference_text,
            "generated": generated_text,
            "rouge1": rouge_scores['rouge1'].fmeasure,
            "rouge2": rouge_scores['rouge2'].fmeasure,
            "rougeL": rouge_scores['rougeL'].fmeasure,
            "generation_time": generation_time
        })
        
        # Print progress
        if (i + 1) % 5 == 0 or (i + 1) == len(test_samples):
            print(f"Evaluated {i + 1}/{len(test_samples)} samples")
    
    # Calculate average scores
    avg_metrics = {
        "rouge1": sum(rouge_1_scores) / len(rouge_1_scores),
        "rouge2": sum(rouge_2_scores) / len(rouge_2_scores),
        "rougeL": sum(rouge_l_scores) / len(rouge_l_scores),
        "generation_time": sum(generation_times) / len(generation_times)
    }
    
    print(f"\nEvaluation results for {model_name}:")
    print(f"Average ROUGE-1: {avg_metrics['rouge1']:.4f}")
    print(f"Average ROUGE-2: {avg_metrics['rouge2']:.4f}")
    print(f"Average ROUGE-L: {avg_metrics['rougeL']:.4f}")
    print(f"Average generation time: {avg_metrics['generation_time']:.4f} seconds")
    
    return results, avg_metrics

def compare_models(base_results, finetuned_results, base_metrics, finetuned_metrics):
    """Compare the performance of base and fine-tuned models"""
    # Create comparison table
    print("\n--- Model Comparison ---")
    print("                     Base Model       Fine-tuned Model   Improvement")
    print(f"ROUGE-1 score:      {base_metrics['rouge1']:.4f}           {finetuned_metrics['rouge1']:.4f}          {finetuned_metrics['rouge1'] - base_metrics['rouge1']:.4f}")
    print(f"ROUGE-2 score:      {base_metrics['rouge2']:.4f}           {finetuned_metrics['rouge2']:.4f}          {finetuned_metrics['rouge2'] - base_metrics['rouge2']:.4f}")
    print(f"ROUGE-L score:      {base_metrics['rougeL']:.4f}           {finetuned_metrics['rougeL']:.4f}          {finetuned_metrics['rougeL'] - base_metrics['rougeL']:.4f}")
    print(f"Generation time:    {base_metrics['generation_time']:.4f}s         {finetuned_metrics['generation_time']:.4f}s        {base_metrics['generation_time'] - finetuned_metrics['generation_time']:.4f}s")
    
    # Create comparison bar chart
    metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    base_values = [base_metrics['rouge1'], base_metrics['rouge2'], base_metrics['rougeL']]
    finetuned_values = [finetuned_metrics['rouge1'], finetuned_metrics['rouge2'], finetuned_metrics['rougeL']]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, base_values, width, label='Base Model')
    plt.bar(x + width/2, finetuned_values, width, label='Fine-tuned Model')
    
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title('Performance Comparison: Base vs. Fine-tuned Model')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(config.FIGURES_DIR, 'model_comparison.png'))
    print(f"Comparison chart saved to {os.path.join(config.FIGURES_DIR, 'model_comparison.png')}")
    
    # Create side-by-side text comparison examples
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    comparison_file = os.path.join(config.RESULTS_DIR, 'model_comparison_examples.txt')
    
    with open(comparison_file, 'w') as f:
        num_examples = min(5, len(base_results), len(finetuned_results))
        
        for i in range(num_examples):
            f.write(f"=== Example {i+1} ===\n\n")
            f.write(f"REFERENCE:\n{base_results[i]['reference']}\n\n")
            f.write(f"BASE MODEL OUTPUT:\n{base_results[i]['generated']}\n\n")
            f.write(f"FINE-TUNED MODEL OUTPUT:\n{finetuned_results[i]['generated']}\n\n")
            f.write(f"METRICS COMPARISON:\n")
            f.write(f"ROUGE-1: {base_results[i]['rouge1']:.4f} → {finetuned_results[i]['rouge1']:.4f} (Δ: {finetuned_results[i]['rouge1'] - base_results[i]['rouge1']:.4f})\n")
            f.write(f"ROUGE-2: {base_results[i]['rouge2']:.4f} → {finetuned_results[i]['rouge2']:.4f} (Δ: {finetuned_results[i]['rouge2'] - base_results[i]['rouge2']:.4f})\n")
            f.write(f"ROUGE-L: {base_results[i]['rougeL']:.4f} → {finetuned_results[i]['rougeL']:.4f} (Δ: {finetuned_results[i]['rougeL'] - base_results[i]['rougeL']:.4f})\n\n")
            f.write("="*50 + "\n\n")
    
    print(f"Example comparisons saved to {comparison_file}")
    
    # Save detailed results to CSV
    base_df = pd.DataFrame(base_results)
    finetuned_df = pd.DataFrame(finetuned_results)
    
    base_df.to_csv(os.path.join(config.RESULTS_DIR, 'base_model_results.csv'), index=False)
    finetuned_df.to_csv(os.path.join(config.RESULTS_DIR, 'finetuned_model_results.csv'), index=False)
    
    # Save comparison metrics
    comparison_metrics = {
        "base_model": base_metrics,
        "finetuned_model": finetuned_metrics,
        "improvements": {
            "rouge1": finetuned_metrics['rouge1'] - base_metrics['rouge1'],
            "rouge2": finetuned_metrics['rouge2'] - base_metrics['rouge2'],
            "rougeL": finetuned_metrics['rougeL'] - base_metrics['rougeL'],
            "generation_time": base_metrics['generation_time'] - finetuned_metrics['generation_time']
        }
    }
    
    with open(os.path.join(config.RESULTS_DIR, 'comparison_metrics.json'), 'w') as f:
        json.dump(comparison_metrics, f, indent=2)
    
    print(f"Detailed comparison metrics saved to {os.path.join(config.RESULTS_DIR, 'comparison_metrics.json')}")

def main():
    """Evaluate and compare models"""
    import time
    
    # Create output directories
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    
    # Load test samples
    test_samples = load_test_samples()
    
    # Set the number of test samples to evaluate
    num_test_samples = min(config.EVAL_NUM_SAMPLES, len(test_samples))
    
    try:
        # Load and evaluate base model
        base_model, base_tokenizer, base_cpu_only = load_base_model()
        base_results, base_metrics = evaluate_model(base_model, base_tokenizer, test_samples, 
                                               model_name="Base Model", cpu_only=base_cpu_only,
                                               num_samples=num_test_samples)
        
        # Load and evaluate fine-tuned model
        finetuned_model, finetuned_tokenizer, finetuned_cpu_only = load_finetuned_model()
        finetuned_results, finetuned_metrics = evaluate_model(finetuned_model, finetuned_tokenizer, test_samples,
                                                       model_name="Fine-tuned Model", cpu_only=finetuned_cpu_only,
                                                       num_samples=num_test_samples)
        
        # Compare models
        compare_models(base_results, finetuned_results, base_metrics, finetuned_metrics)
        
        print("Model evaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
