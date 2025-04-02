#!/usr/bin/env python
# 06_train_model_simplified.py
# Purpose: Train the model using LoRA with simplified configuration based on Unsloth example

# Import unsloth first to avoid warnings and ensure optimizations
import unsloth
from unsloth import FastModel, is_bf16_supported

# Disable Unsloth cache
import os
os.environ["DISABLE_UNSLOTH_CACHE"] = "1"

# Standard imports
import sys
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from trl import SFTTrainer, SFTConfig
from datasets import load_from_disk
import config

def train_model():
    """Train the model using LoRA with simplified configuration"""
    
    # Check if prepare info exists
    prepare_info_path = os.path.join(config.CHECKPOINT_DIR, "prepare_info.json")
    if not os.path.exists(prepare_info_path):
        raise FileNotFoundError(f"Prepare info not found at {prepare_info_path}. "
                              f"Please run 05_prepare_training.py first.")
    
    # Check if model info exists
    model_info_path = os.path.join(config.CHECKPOINT_DIR, "model_info.json")
    if not os.path.exists(model_info_path):
        raise FileNotFoundError(f"Model info not found at {model_info_path}. "
                              f"Please run 03_model_load.py first.")
    
    # Load model info
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)
    
    model_config = model_info["model_config"]
    cpu_only = model_config.get("cpu_only", False)
    
    # Load the model (fresh instance)
    print("Loading model and tokenizer...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_config["model_name"],
        max_seq_length=model_config["max_seq_length"],
        load_in_4bit=model_config["load_in_4bit"],
        dtype=None,
    )
    
    # Apply LoRA config (simplified from Unsloth example)
    model = FastModel.get_peft_model(
        model,
        r=16,                             # LoRA rank
        lora_alpha=16,                    # LoRA alpha parameter
        lora_dropout=0,                   # No dropout for stability
        bias="none",                      # No bias training
        random_state=3407,                # Random seed for reproducibility
    )
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = load_from_disk(os.path.join(config.DATA_DIR, "train_dataset"))
    val_dataset = load_from_disk(os.path.join(config.DATA_DIR, "val_dataset"))
    
    print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Configure SFT trainer exactly as in the Unsloth example
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=config.MAX_STEPS,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=config.OUTPUT_DIR,
            report_to="none",
        ),
    )
    
    # Track GPU memory usage
    if torch.cuda.is_available() and not cpu_only:
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")
    else:
        print("Running on CPU mode")
        start_gpu_memory = 0
        max_memory = 0
    
    # Run training
    print("\n--- Starting Training ---")
    training_start_time = time.time()
    
    try:
        trainer_stats = trainer.train()
        training_success = True
    except Exception as e:
        print(f"Training error: {e}")
        print("Saving partial results...")
        training_success = False
        trainer_stats = None
    
    training_end_time = time.time()
    
    # Calculate training time
    training_time = training_end_time - training_start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Training success: {training_success}")
    
    # Report GPU memory usage if available
    if torch.cuda.is_available() and not cpu_only:
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_training = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory/max_memory*100, 3)
        print(f"Peak reserved memory = {used_memory} GB")
        print(f"Peak reserved memory for training = {used_memory_for_training} GB")
        print(f"Peak memory utilization = {used_percentage:.2f}%")
    
    # Extract training metrics if available
    training_losses = []
    eval_losses = []
    
    if hasattr(trainer, 'state') and hasattr(trainer.state, 'log_history'):
        for entry in trainer.state.log_history:
            if 'loss' in entry and 'eval_loss' not in entry:
                step = entry.get('step', 0)
                loss = entry['loss']
                if isinstance(loss, torch.Tensor):
                    loss = loss.item()
                training_losses.append((step, loss))
            elif 'eval_loss' in entry:
                step = entry.get('step', 0)
                loss = entry['eval_loss']
                if isinstance(loss, torch.Tensor):
                    loss = loss.item()
                eval_losses.append((step, loss))
    
    # Plot training and evaluation losses if available
    if training_losses:
        plt.figure(figsize=(10, 6))
        
        steps, losses = zip(*training_losses)
        plt.plot(steps, losses, label='Training Loss')
        
        if eval_losses:
            steps, losses = zip(*eval_losses)
            plt.plot(steps, losses, label='Validation Loss')
        
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Create figures directory
        os.makedirs(config.FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(config.FIGURES_DIR, 'training_loss.png'))
        print(f"Training loss plot saved to {os.path.join(config.FIGURES_DIR, 'training_loss.png')}")
    
    # Save training metrics
    training_metrics = {
        "training_time_seconds": training_time,
        "training_time_formatted": f"{int(hours)}h {int(minutes)}m {int(seconds)}s",
        "training_success": training_success,
    }
    
    # Convert tuples to lists for JSON serialization
    if training_losses:
        training_metrics["training_losses"] = [[step, float(loss)] for step, loss in training_losses]
        training_metrics["final_train_loss"] = float(training_losses[-1][1])
    
    if eval_losses:
        training_metrics["eval_losses"] = [[step, float(loss)] for step, loss in eval_losses]
        training_metrics["final_eval_loss"] = float(eval_losses[-1][1])
    
    # Create results directory
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "training_metrics.json"), "w") as f:
        json.dump(training_metrics, f, indent=2)
    
    # Save the model
    os.makedirs(config.MODEL_OUTPUT_DIR, exist_ok=True)
    print(f"Saving fine-tuned model to {config.MODEL_OUTPUT_DIR}...")
    
    try:
        trainer.save_model(config.MODEL_OUTPUT_DIR)
        tokenizer.save_pretrained(config.MODEL_OUTPUT_DIR)
        model_saved = True
    except Exception as e:
        print(f"Error saving model: {e}")
        print("Trying alternative saving method...")
        try:
            # Try saving with model's own method
            model.save_pretrained(config.MODEL_OUTPUT_DIR)
            tokenizer.save_pretrained(config.MODEL_OUTPUT_DIR)
            model_saved = True
        except Exception as e2:
            print(f"Error in alternative saving method: {e2}")
            model_saved = False
    
    # Save a file that indicates training status
    with open(os.path.join(config.MODEL_OUTPUT_DIR, "training_status.txt"), "w") as f:
        f.write(f"Training completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training time: {int(hours)}h {int(minutes)}m {int(seconds)}s\n")
        f.write(f"Training success: {training_success}\n")
        f.write(f"Model saved: {model_saved}\n")
        if training_losses:
            f.write(f"Final training loss: {training_losses[-1][1]}\n")
        if eval_losses:
            f.write(f"Final validation loss: {eval_losses[-1][1]}\n")
    
    if training_success and model_saved:
        print("Model training and saving completed successfully!")
    else:
        print("Model training or saving encountered issues.")
        print("Check the logs and training_status.txt for details.")
    
    return model, tokenizer, trainer_stats

def main():
    """Train the model and save results"""
    try:
        model, tokenizer, trainer_stats = train_model()
        print("Training process completed!")
    except Exception as e:
        print(f"Error during training process: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()