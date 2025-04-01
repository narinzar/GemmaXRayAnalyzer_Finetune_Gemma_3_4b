# 04_define_trainer.py
# Purpose: Set up the trainer with appropriate configuration for efficient fine-tuning on medical X-rays

import os
import torch
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
from torch.utils.data import DataLoader

# Check if previous step output exists
if not os.path.exists("checkpoints/03_dataset_prepared.pt"):
    raise FileNotFoundError("Run 03_prepare_dataset.py first to prepare the dataset")

# Load model, tokenizer, vision processor, and datasets from previous step
print("Loading model, tokenizer, vision processor, and datasets from checkpoint...")
checkpoint = torch.load("checkpoints/03_dataset_prepared.pt")
model = checkpoint["model"]
tokenizer = checkpoint["tokenizer"]
vision_processor = checkpoint["vision_processor"]
train_dataset = checkpoint["train_dataset"]
val_dataset = checkpoint["val_dataset"]
conditions = checkpoint["conditions"]

print("Configuring trainer...")

# Set up output directory for model checkpoints
output_dir = "./results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Configure the trainer using SFTTrainer from TRL (Transformer Reinforcement Learning) library
# SFTTrainer is designed specifically for Supervised Fine-Tuning
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    dataset_text_field="text",  # Field in the dataset that contains the text to train on
    max_seq_length=512,         # Use a shorter sequence length for efficiency
    args=SFTConfig(
        # Training parameters
        per_device_train_batch_size=1,      # Batch size of 1 for large images and memory constraints
        gradient_accumulation_steps=8,      # Accumulate gradients over multiple steps to increase effective batch size
        max_steps=200,                      # Maximum number of training steps
        learning_rate=1e-4,                 # Learning rate - lower for medical tasks to prevent overfitting
        bf16=True,                          # Use bfloat16 precision for faster training if supported
        logging_steps=10,                   # Log metrics every N steps
        output_dir=output_dir,              # Directory to save model checkpoints
        optim="adamw_8bit",                 # Optimizer to use - adamw_8bit is memory efficient
        weight_decay=0.02,                  # Weight decay for regularization - higher for medical to reduce overfitting
        lr_scheduler_type="cosine",         # Learning rate scheduler type
        warmup_ratio=0.05,                  # Portion of steps for learning rate warmup
        max_grad_norm=0.5,                  # Clip gradients to prevent exploding gradients
        save_strategy="steps",              # When to save checkpoints
        save_steps=50,                      # Save checkpoint every N steps
        evaluation_strategy="steps",        # Evaluate during training
        eval_steps=50,                      # Evaluate every N steps
        report_to="none",                   # Don't report to any tracking system
    ),
)

print("Trainer configured with the following settings:")
print(f"Batch size: {trainer.args.per_device_train_batch_size}")
print(f"Gradient accumulation steps: {trainer.args.gradient_accumulation_steps}")
print(f"Effective batch size: {trainer.args.per_device_train_batch_size * trainer.args.gradient_accumulation_steps}")
print(f"Learning rate: {trainer.args.learning_rate}")
print(f"Max steps: {trainer.args.max_steps}")

# Save for next steps
torch.save({
    "model": model,
    "tokenizer": tokenizer,
    "vision_processor": vision_processor,
    "trainer": trainer,
    "conditions": conditions
}, "checkpoints/04_trainer_defined.pt")

print("Trainer configured and saved to checkpoints/04_trainer_defined.pt")

if __name__ == "__main__":
    print("Trainer configuration completed successfully")
