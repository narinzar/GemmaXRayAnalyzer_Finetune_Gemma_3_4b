# run_all.py
# Purpose: Run all steps in sequence to fine-tune and deploy the X-ray analyzer

import os
import sys
import subprocess
import time

def run_step(step_number, step_name, script_name):
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
        return False

def main():
    """Run all steps in sequence"""
    print("Starting the fine-tuning and deployment process for the X-ray analyzer")
    
    # Make sure the examples directory exists
    if not os.path.exists("examples"):
        print("Setting up example images...")
        if not run_step(0, "Setup Examples", "setup_examples.py"):
            return
    
    # Step 1: Load Model
    if not run_step(1, "Load Model", "01_load_model.py"):
        return
    
    # Step 2: Define LoRA Configuration
    if not run_step(2, "Define LoRA Configuration", "02_define_lora_config.py"):
        return
    
    # Step 3: Prepare Dataset
    if not run_step(3, "Prepare Dataset", "03_prepare_dataset.py"):
        return
    
    # Step 4: Define Trainer
    if not run_step(4, "Define Trainer", "04_define_trainer.py"):
        return
    
    # Step 5: Train
    if not run_step(5, "Train Model", "05_train.py"):
        return
    
    # Step 6: Launch Gradio App
    print("\n" + "=" * 80)
    print("STEP 6: Launch Gradio App")
    print("=" * 80)
    print("Running the Gradio app for testing...")
    print("Press Ctrl+C to stop the app")
    
    subprocess.run([sys.executable, "06_gradio_app.py"], check=False)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting...")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
