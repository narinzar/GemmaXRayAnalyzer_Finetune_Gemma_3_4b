#!/usr/bin/env python
# 03_model_load.py
# Purpose: Load the Gemma 3 model and save it for later use

import os
import json
import torch
from unsloth import FastModel
import config
from dotenv import load_dotenv

def load_model():
    """Load the base Gemma 3 model and save it for later use"""
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
    
    # Instead of trying to save the model directly, we'll save its configuration
    # and reload it in the evaluation script
    model_config = {
        "model_name": config.MODEL_NAME,
        "max_seq_length": config.SEQUENCE_LENGTH,
        "load_in_4bit": config.USE_4BIT and not cpu_only,
        "use_gradient_checkpointing": "unsloth",
        "cpu_only": cpu_only
    }
    
    # Save model configuration
    with open(os.path.join(config.CHECKPOINT_DIR, "base_model_config.json"), "w") as f:
        json.dump(model_config, f, indent=2)
    
    # Save tokenizer separately
    tokenizer_path = os.path.join(config.CHECKPOINT_DIR, "base_tokenizer")
    os.makedirs(tokenizer_path, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_path)
    
    print(f"Base model configuration saved to {os.path.join(config.CHECKPOINT_DIR, 'base_model_config.json')}")
    print(f"Base tokenizer saved to {tokenizer_path}")
    print(f"Tokenizer vocabulary size: {len(tokenizer.vocab) if hasattr(tokenizer, 'vocab') else None}")
    
    # Now we need to create a marker file that the evaluation script will recognize
    # This file indicates that the base model loading has been completed
    with open(os.path.join(config.CHECKPOINT_DIR, "base_model.marker"), "w") as f:
        f.write("Base model loading completed at: " + time.strftime("%Y-%m-%d %H:%M:%S"))
    
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
    
    print("Model loading and configuration saving completed successfully")

if __name__ == "__main__":
    import time
    main()