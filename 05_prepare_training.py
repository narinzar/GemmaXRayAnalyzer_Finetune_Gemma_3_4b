#!/usr/bin/env python
# 05_prepare_training.py
# Purpose: Prepare datasets for training

import os
import json
import torch
from datasets import Dataset
import config

def format_conversation(caption, instruction=None):
    """Format a conversation using the model's chat template"""
    if instruction is None:
        instruction = config.INSTRUCTION
    
    # Format using Gemma's chat format
    return {
        "text": f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{caption}<end_of_turn>"
    }

def prepare_datasets():
    """Prepare training and validation datasets"""
    # Check if LoRA model checkpoint exists
    lora_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "lora_model.pt")
    if not os.path.exists(lora_checkpoint_path):
        raise FileNotFoundError(f"LoRA model checkpoint not found at {lora_checkpoint_path}. "
                              f"Please run 04_lora_config.py first.")
    
    # Check if dataset splits exist
    dataset_path = os.path.join(config.DATA_DIR, "dataset_splits.json")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset splits not found at {dataset_path}. "
                              f"Please run 01_data_load.py first.")
    
    # Load model and tokenizer
    print(f"Loading model and tokenizer from {lora_checkpoint_path}...")
    checkpoint = torch.load(lora_checkpoint_path)
    model = checkpoint["model"]
    tokenizer = checkpoint["tokenizer"]
    lora_config = checkpoint.get("lora_config", {})
    cpu_only = checkpoint.get("cpu_only", False)
    
    # Load dataset splits
    print(f"Loading dataset splits from {dataset_path}...")
    with open(dataset_path, "r") as f:
        splits = json.load(f)
    
    train_samples = splits["train"]
    val_samples = splits["validation"]
    
    # Format datasets
    print("Formatting datasets for training...")
    train_formatted = [format_conversation(sample["caption"]) for sample in train_samples]
    val_formatted = [format_conversation(sample["caption"]) for sample in val_samples]
    
    # Convert to Dataset objects
    train_dataset = Dataset.from_list(train_formatted)
    val_dataset = Dataset.from_list(val_formatted)
    
    print(f"Prepared {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    
    # Save formatted datasets
    train_dataset.save_to_disk(os.path.join(config.DATA_DIR, "train_dataset"))
    val_dataset.save_to_disk(os.path.join(config.DATA_DIR, "val_dataset"))
    
    print(f"Datasets saved to {config.DATA_DIR}")
    
    # Save a sample for reference
    with open(os.path.join(config.DATA_DIR, "sample.txt"), "w") as f:
        f.write(f"SAMPLE INPUT FORMAT:\n\n{train_formatted[0]['text']}")
    
    print(f"Sample saved to {os.path.join(config.DATA_DIR, 'sample.txt')}")
    
    # Save prepare_dataset info
    prepare_info = {
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "instruction": config.INSTRUCTION,
        "lora_config": lora_config
    }
    
    with open(os.path.join(config.CHECKPOINT_DIR, "prepare_info.json"), "w") as f:
        json.dump(prepare_info, f, indent=2)
    
    # Return model, tokenizer, datasets, and CPU flag
    return model, tokenizer, train_dataset, val_dataset, cpu_only

def main():
    """Prepare datasets for training"""
    # Create necessary directories
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    # Prepare datasets
    model, tokenizer, train_dataset, val_dataset, cpu_only = prepare_datasets()
    
    print("Dataset preparation completed successfully")

if __name__ == "__main__":
    main()
