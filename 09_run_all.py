#!/usr/bin/env python
# run_all.py
# Purpose: Run all steps of the Gemma 3 X-ray analysis fine-tuning pipeline

import os
import sys
import subprocess
import time
import argparse

def run_step(step_number, step_name, script_name, exit_on_error=True):
    """Run a single step in the process and check for success"""
    print("\n" + "=" * 80)
    print(f"STEP {step_number}: {step_name}")
    print("=" * 80)
    
    start_time = time.time()
    result = subprocess.run([sys.executable, script_name], check=False)
    duration = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n✓ Step {step_number} completed successfully in {duration:.1f} seconds")
        return True
    else:
        print(f"\n✗ Step {step_number} failed with exit code {result.returncode}")
        if exit_on_error:
            print("Exiting pipeline due to error")
            sys.exit(1)
        return False

def main():
    """Run all steps in the fine-tuning pipeline"""
    parser = argparse.ArgumentParser(description="Run the Gemma 3 X-ray fine-tuning pipeline")
    parser.add_argument("--start-step", type=int, default=1, help="Step to start from (1-8)")
    parser.add_argument("--end-step", type=int, default=8, help="Step to end at (1-8)")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip model evaluation step")
    parser.add_argument("--skip-push", action="store_true", help="Skip pushing to Hugging Face Hub")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue pipeline even if a step fails")
    
    args = parser.parse_args()
    
    # Define steps
    steps = [
        (1, "Data Loading", "01_data_load.py"),
        (2, "Data Analysis", "02_data_analysis.py"),
        (3, "Model Loading", "03_model_load.py"),
        (4, "LoRA Configuration", "04_lora_config.py"),
        (5, "Training Preparation", "05_prepare_training.py"),
        (6, "Model Training", "06_train_model.py"),
        (7, "Model Evaluation", "07_evaluate_model.py"),
        (8, "Push to Hugging Face", "08_push_to_hf.py")
    ]
    
    # Filter steps based on arguments
    filtered_steps = [
        step for step in steps 
        if args.start_step <= step[0] <= args.end_step
    ]
    
    if args.skip_evaluation:
        filtered_steps = [step for step in filtered_steps if step[0] != 7]
    
    if args.skip_push:
        filtered_steps = [step for step in filtered_steps if step[0] != 8]
    
    # Print execution plan
    print("Gemma 3 X-ray Fine-tuning Pipeline")
    print("\nExecution Plan:")
    for step_num, step_name, _ in filtered_steps:
        print(f"  {step_num}. {step_name}")
    
    # Confirm execution
    if input("\nPress Enter to continue or Ctrl+C to abort..."):
        pass
    
    # Run steps
    pipeline_start_time = time.time()
    
    for step_num, step_name, script_name in filtered_steps:
        success = run_step(step_num, step_name, script_name, exit_on_error=not args.continue_on_error)
        if not success and not args.continue_on_error:
            break
    
    pipeline_duration = time.time() - pipeline_start_time
    minutes, seconds = divmod(pipeline_duration, 60)
    hours, minutes = divmod(minutes, 60)
    
    print("\n" + "=" * 80)
    print(f"Pipeline completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print("=" * 80)

if __name__ == "__main__":
    main()
