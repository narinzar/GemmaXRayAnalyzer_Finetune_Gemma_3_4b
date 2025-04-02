#!/usr/bin/env python
# 03_model_load_fixed.py
# Purpose: Load the Gemma 3 model and save model info (but not the model itself)

import os
import json
import torch
from unsloth import FastModel
import config
from dotenv import load_dotenv

# Fix for HybridCache float error
os.environ["DISABLE_UNSLOTH_CACHE"] = "1"

def load_model():
    """Load the base Gemma 3 model and save its configuration info"""
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
    
    # Load the model and tokenizer with Unsloth's FastModel
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
    
    # Save model config for future reference (don't try to save the model itself)
    model_config = {
        "model_name": config.MODEL_NAME,
        "max_seq_length": config.SEQUENCE_LENGTH,
        "load_in_4bit": config.USE_4BIT and not cpu_only,
        "use_gradient_checkpointing": "unsloth",
        "cpu_only": cpu_only
    }
    
    # Get model info without saving the actual model
    vocab_size = len(tokenizer.get_vocab()) if hasattr(tokenizer, 'get_vocab') else None
    
    # Save just the model info to a JSON file
    model_info = {
        "model_config": model_config,
        "tokenizer_vocab_size": vocab_size,
        "model_type": type(model).__name__,
        "cpu_only": cpu_only
    }
    
    # Save to JSON file
    with open(os.path.join(config.CHECKPOINT_DIR, "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=2)
    
    print(f"Model info saved to {os.path.join(config.CHECKPOINT_DIR, 'model_info.json')}")
    print(f"Tokenizer vocabulary size: {vocab_size}")
    
    return model, tokenizer, cpu_only

def main():
    """Load model and save its info"""
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
    
    # Load model
    model, tokenizer, cpu_only = load_model()
    
    print("Model loading completed successfully")

if __name__ == "__main__":
    main()