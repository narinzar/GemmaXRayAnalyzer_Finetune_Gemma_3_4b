#!/usr/bin/env python
# 03_model_load.py
# Purpose: Load the Gemma 3 model and save its configuration for later use

import os
import json
import torch
from unsloth import FastModel
import config
from dotenv import load_dotenv

def load_model():
    """Load the base Gemma 3 model and save its configuration"""
    # Create checkpoint directory
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # Check for GPU availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        print(f"GPU available: {gpu_name}")
        print(f"CUDA Version: {cuda_version}")
        cpu_only = False
    else:
        print("WARNING: No GPU detected. Processing will be done on CPU.")
        cpu_only = True
    
    # Load the model and tokenizer with Unsloth's FastModelz
    print(f"Loading {config.MODEL_NAME} model...")
    try:
        model, tokenizer = FastModel.from_pretrained(
            model_name=config.MODEL_NAME,
            max_seq_length=config.SEQUENCE_LENGTH,
            load_in_4bit=config.USE_4BIT and not cpu_only,
            dtype=None,  # Automatically determine best dtype
            use_gradient_checkpointing="unsloth"  # Use gradient checkpointing for memory efficiency
        )
        print(f"Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    # Save model config for future reference without trying to pickle the model
    model_config = {
        "model_name": config.MODEL_NAME,
        "max_seq_length": config.SEQUENCE_LENGTH,
        "load_in_4bit": config.USE_4BIT and not cpu_only,
        "use_gradient_checkpointing": "unsloth",
        "cpu_only": cpu_only
    }
    
    # Extract tokenizer info
    tokenizer_info = {
        "vocab_size": len(tokenizer.vocab) if hasattr(tokenizer, 'vocab') else None,
        "tokenizer_name": type(tokenizer).__name__
    }
    
    # Save the model configuration to a JSON file
    model_info = {
        "model_config": model_config,
        "tokenizer_info": tokenizer_info,
        "model_type": type(model).__name__,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_params": sum(p.numel() for p in model.parameters())
    }
    
    # Save the configuration as JSON
    with open(os.path.join(config.CHECKPOINT_DIR, "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=2)
    
    print(f"Model configuration saved to {os.path.join(config.CHECKPOINT_DIR, 'model_info.json')}")
    
    if hasattr(tokenizer, "save_pretrained"):
        # Save the tokenizer files directly
        tokenizer_dir = os.path.join(config.CHECKPOINT_DIR, "tokenizer")
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer.save_pretrained(tokenizer_dir)
        print(f"Tokenizer saved to {tokenizer_dir}")
    
    print(f"Tokenizer vocabulary size: {tokenizer_info['vocab_size']}")
    
    if model_info["trainable_params"] > 0:
        print(f"Model has {model_info['trainable_params']:,} trainable parameters " 
              f"out of {model_info['total_params']:,} total parameters "
              f"({model_info['trainable_params']/model_info['total_params']*100:.2f}%)")
    
    # Return the model and tokenizer for potential further processing
    return model, tokenizer, cpu_only

def main():
    """Load model and save its configuration"""
    # Load environment variables
    load_dotenv()
    
    # Check if HF_TOKEN is set
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("WARNING: HF_TOKEN not found in environment variables.")
        print("You may encounter issues downloading the model.")
        print("Please add your Hugging Face token to the .env file.")
    else:
        # Set HF_TOKEN as an environment variable for the current process
        os.environ["HF_TOKEN"] = hf_token
        print("Hugging Face token loaded from .env file")
    
    # Load model and save its configuration
    model, tokenizer, cpu_only = load_model()
    
    print("Model loading completed successfully")

if __name__ == "__main__":
    main()
