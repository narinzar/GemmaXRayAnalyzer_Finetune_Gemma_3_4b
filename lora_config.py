# 02_define_lora_config.py
# Purpose: Configure LoRA (Low-Rank Adaptation) parameters for efficient fine-tuning

import os
import torch
from unsloth import FastModel

# Check if previous step output exists
if not os.path.exists("checkpoints/01_model_loaded.pt"):
    raise FileNotFoundError("Run 01_load_model.py first to create the model checkpoint")

# Load model, tokenizer, and vision processor from previous step
print("Loading model, tokenizer, and vision processor from checkpoint...")
checkpoint = torch.load("checkpoints/01_model_loaded.pt")
model = checkpoint["model"]
tokenizer = checkpoint["tokenizer"]
vision_processor = checkpoint["vision_processor"]

print("Configuring LoRA parameters...")

# Apply LoRA configuration using Unsloth's helper function
# LoRA (Low-Rank Adaptation) enables efficient fine-tuning by only updating
# a small number of parameters rather than the entire model
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=True,     # Fine-tune vision layers to process X-ray features
    finetune_language_layers=True,   # Fine-tune the language layers for medical descriptions
    finetune_attention_modules=True, # Fine-tune attention modules which are crucial for understanding context
    finetune_mlp_modules=True,       # Fine-tune MLP (Multi-Layer Perceptron) modules for improved reasoning
    r=8,                             # LoRA rank - higher for medical tasks as they require more capacity
    lora_alpha=16,                   # LoRA alpha parameter - scaling factor for LoRA updates
    lora_dropout=0.05,               # Dropout rate for LoRA layers - small amount to prevent overfitting
    bias="none",                     # Whether to train bias parameters - "none" means don't train biases
)

# Print model architecture and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")

# Save for next steps
torch.save({
    "model": model,
    "tokenizer": tokenizer,
    "vision_processor": vision_processor
}, "checkpoints/02_lora_configured.pt")

print("LoRA-configured model saved to checkpoints/02_lora_configured.pt")

if __name__ == "__main__":
    print("LoRA configuration completed successfully")
