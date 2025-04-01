#!/usr/bin/env python
# 06_train_model.py
# Purpose: Train the model using LoRA

import os
import sys
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from trl import SFTTrainer
from datasets import load_from_disk
import config
from unsloth import is_bf16_supported

# Fix for HybridCache float error
os.environ["DISABLE_UNSLOTH_CACHE"] = "1"

def train_model():
    """Train the model using LoRA and save checkpoints"""
    
    # Check if prepare info exists
    prepare_info_path = os.path.join(config.CHECKPOINT_DIR, "prepare_info.json")
    if not os.path.exists(prepare_info_path):
        raise FileNotFoundError(f"Prepare info not found at {prepare_info_path}. "
                              f"Please run 05_prepare_training.py first.")
    
    # Check if LoRA model checkpoint exists
    lora_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "lora_model.pt")
    if not os.path.exists(lora_checkpoint_path):
        raise FileNotFoundError(f"LoRA model checkpoint not found at {lora_checkpoint_path}. "
                              f"Please run 04_lora_config.py first.")
    
    # Load model and tokenizer
    print(f"Loading model and tokenizer from {lora_checkpoint_path}...")
    checkpoint = torch.load(lora_checkpoint_path)
    model = checkpoint["model"]
    tokenizer = checkpoint["tokenizer"]
    cpu_only = checkpoint.get("cpu_only", False)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = load_from_disk(os.path.join(config.DATA_DIR, "train_dataset"))
    val_dataset = load_from_disk(os.path.join(config.DATA_DIR, "val_dataset"))
    
    print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    