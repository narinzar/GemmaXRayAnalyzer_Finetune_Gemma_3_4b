#!/usr/bin/env python
# 05_prepare_training.py
# Purpose: Prepare datasets for training

import os
import json
import torch
from datasets import Dataset
from unsloth import FastModel
from transformers import AutoTokenizer
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
    # Check if LoRA configuration exists
    lora_config_path = os.path.join(config.CHECKPOINT_DIR, "lora_config.json")
    model_info_path = os.path.join(config.CHECKPOINT_DIR, "model_info.json")
    lora_tokenizer_dir = os.path.join(config.CHECKPOINT_DIR, "lora_tokenizer")
    
    if not os.path.exists(lora_config_path):
        raise FileNotFoundError(f"LoRA configuration not found at {lora_config_path}. "
                              f"Please run 04_lora_config.py first.")
    
    if not os.path.exists(model_info_path):
        raise FileNotFoundError(f"Model info not found at {model_info_path}. "
                              f"Please run 03_model_load.py first.")
    
    # Check if dataset splits exist
    dataset_path = os.path.join(config.DATA_DIR, "dataset_splits.json")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset splits not found at {dataset_path}. "
                              f"Please run 01_data_load.py first.")
    
    # Load model configurations
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)
    
    with open(lora_config_path, 'r') as f:
        lora_config = json.load(f)
    
    model_config = model_info["model_config"]
    cpu_only = model_config.get("cpu_only", False)
    
    # Load model and tokenizer
    print(f"Loading model and tokenizer...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_config["model_name"],
        max_seq_length=model_config["max_seq_length"],
        load_in_4bit=model_config["load_in_4bit"],
        dtype=None,
        use_gradient_checkpointing=model_config["use_gradient_checkpointing"]
    )
    
    # Apply LoRA configuration
    model = FastModel.get_peft_model(
        model,
        finetune_language_layers=lora_config["finetune_language_layers"],
        finetune_attention_modules=lora_config["finetune_attention_modules"],
        finetune_mlp_modules=lora_config["finetune_mlp_modules"],
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        bias=lora_config["bias"],
        random_state=lora_config["random_state"],
    )
    
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
    
    # Create data directory if it doesn't exist
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
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