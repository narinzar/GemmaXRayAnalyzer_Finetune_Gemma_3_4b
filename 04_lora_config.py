#!/usr/bin/env python
# 04_lora_config.py
# Purpose: Configure LoRA parameters for fine-tuning

import os
import json
import torch
from unsloth import FastModel
import config

def configure_lora(model=None, tokenizer=None, cpu_only=None):
    """Configure LoRA parameters for efficient fine-tuning"""
    # Create checkpoint directory
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # If model wasn't passed, load from configuration
    if model is None:
        model_info_path = os.path.join(config.CHECKPOINT_DIR, "model_info.json")
        tokenizer_dir = os.path.join(config.CHECKPOINT_DIR, "tokenizer")
        
        if not os.path.exists(model_info_path):
            raise FileNotFoundError(f"Model info not found at {model_info_path}. "
                                  f"Please run 03_model_load.py first.")
        
        # Load model configuration
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        
        model_config = model_info["model_config"]
        cpu_only = model_config.get("cpu_only", False)
        
        print(f"Loading model using configuration from {model_info_path}...")
        
        # Load model and tokenizer
        model, tokenizer = FastModel.from_pretrained(
            model_name=model_config["model_name"],
            max_seq_length=model_config["max_seq_length"],
            load_in_4bit=model_config["load_in_4bit"],
            dtype=None,
            use_gradient_checkpointing=model_config["use_gradient_checkpointing"]
        )
    
    print("Configuring LoRA parameters...")
    try:
        # Apply LoRA configuration using Unsloth's helper function
        model = FastModel.get_peft_model(
            model,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=config.LORA_RANK,
            lora_alpha=config.LORA_ALPHA,
            lora_dropout=config.LORA_DROPOUT,
            bias=config.LORA_BIAS,
            random_state=config.RANDOM_SEED,
        )
        print("LoRA configuration applied successfully")
    except Exception as e:
        print(f"Error applying LoRA configuration: {e}")
        raise
    
    # Print model architecture and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")
    
    # Save LoRA configuration for reference
    lora_config_data = {
        "r": config.LORA_RANK,
        "lora_alpha": config.LORA_ALPHA,
        "lora_dropout": config.LORA_DROPOUT,
        "bias": config.LORA_BIAS,
        "random_state": config.RANDOM_SEED,
        "finetune_language_layers": True,
        "finetune_attention_modules": True, 
        "finetune_mlp_modules": True,
        "trainable_params": trainable_params,
        "total_params": total_params
    }
    
    # Save to file
    with open(os.path.join(config.CHECKPOINT_DIR, "lora_config.json"), "w") as f:
        json.dump(lora_config_data, f, indent=2)
    
    # Save tokenizer separately
    tokenizer_dir = os.path.join(config.CHECKPOINT_DIR, "lora_tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)
    
    print(f"LoRA configuration saved to {os.path.join(config.CHECKPOINT_DIR, 'lora_config.json')}")
    print(f"Tokenizer saved to {tokenizer_dir}")
    
    return model, tokenizer, lora_config_data, cpu_only

def main():
    """Configure LoRA and save model"""
    # Configure LoRA
    model, tokenizer, lora_config_data, cpu_only = configure_lora()
    
    print("LoRA configuration completed successfully")

if __name__ == "__main__":
    main()