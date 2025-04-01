#!/usr/bin/env python
# 04_lora_config.py
# Purpose: Configure LoRA parameters for fine-tuning

import os
import torch
import json
from unsloth import FastModel
import config

def configure_lora(model=None, tokenizer=None, cpu_only=None):
    """Configure LoRA parameters for efficient fine-tuning"""
    # Create checkpoint directory
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # If model wasn't passed, load from checkpoint
    if model is None:
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "base_model.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Base model checkpoint not found at {checkpoint_path}. "
                                  f"Please run 03_model_load.py first.")
        
        print(f"Loading model from checkpoint {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        model = checkpoint["model"]
        tokenizer = checkpoint["tokenizer"]
        cpu_only = checkpoint.get("cpu_only", False)
    
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
    lora_config = {
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
        json.dump(lora_config, f, indent=2)
    
    # Save model with LoRA
    torch.save({
        "model": model,
        "tokenizer": tokenizer,
        "lora_config": lora_config,
        "cpu_only": cpu_only
    }, os.path.join(config.CHECKPOINT_DIR, "lora_model.pt"))
    
    print(f"LoRA-configured model saved to {os.path.join(config.CHECKPOINT_DIR, 'lora_model.pt')}")
    
    return model, tokenizer, lora_config, cpu_only

def main():
    """Configure LoRA and save model"""
    # Configure LoRA
    model, tokenizer, lora_config, cpu_only = configure_lora()
    
    print("LoRA configuration completed successfully")

if __name__ == "__main__":
    main()
