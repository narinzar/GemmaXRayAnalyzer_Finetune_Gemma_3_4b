# 03_prepare_dataset.py
# Purpose: Load and prepare the medical X-ray dataset for fine-tuning

import os
import torch
import numpy as np
from PIL import Image
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Check if previous step output exists
if not os.path.exists("checkpoints/02_lora_configured.pt"):
    raise FileNotFoundError("Run 02_define_lora_config.py first to create the LoRA-configured model checkpoint")

# Load model, tokenizer, and vision processor from previous step
print("Loading model, tokenizer, and vision processor from checkpoint...")
checkpoint = torch.load("checkpoints/02_lora_configured.pt")
model = checkpoint["model"]
tokenizer = checkpoint["tokenizer"]
vision_processor = checkpoint["vision_processor"]

print("Loading and preparing X-ray dataset...")

# Load NIH ChestX-ray14 dataset from Hugging Face
# This dataset contains 112,120 X-ray images with 14 disease labels
dataset = load_dataset("nih-chest-xrays/nih-chest-xrays", "small", split="train")
print(f"Loaded dataset with {len(dataset)} examples")

# Define the conditions/labels we want to classify
CONDITIONS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", 
    "Mass", "Nodule", "Pneumonia", "Pneumothorax", 
    "Consolidation", "Edema", "Emphysema", "Fibrosis", 
    "Pleural_Thickening", "Hernia", "No Finding"
]

# Create a custom dataset class for the X-ray images
class XrayDataset(Dataset):
    def __init__(self, dataset, vision_processor, tokenizer, split='train'):
        self.dataset = dataset
        self.vision_processor = vision_processor
        self.tokenizer = tokenizer
        
        # Split the dataset
        if split == 'train':
            self.indices = list(range(int(0.8 * len(dataset))))
        else:  # validation
            self.indices = list(range(int(0.8 * len(dataset)), len(dataset)))
            
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get the actual index from our indices list
        actual_idx = self.indices[idx]
        
        # Get image and label
        item = self.dataset[actual_idx]
        image = item['image']
        
        # Extract the labels (1 for present, 0 for absent)
        labels = [item['labels'][condition] for condition in CONDITIONS]
        
        # Find the conditions that are present (value == 1)
        present_conditions = [CONDITIONS[i] for i, value in enumerate(labels) if value == 1]
        
        # If no condition is found, default to "No Finding"
        if not present_conditions:
            present_conditions = ["No Finding"]
        
        # Create a description for the model to learn
        description = f"This is a chest X-ray showing: {', '.join(present_conditions)}"
        
        # Process the image
        processed_image = self.vision_processor(image, return_tensors="pt")
        
        # Format the input for the model
        messages = [
            {
                "role": "user",
                "content": "Analyze this chest X-ray and describe what conditions are present."
            },
            {
                "role": "assistant",
                "content": description
            }
        ]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(messages)
        
        return {
            "image": processed_image,
            "text": text,
            "description": description,
            "labels": labels
        }

# Create train and validation datasets
train_dataset = XrayDataset(dataset, vision_processor, tokenizer, split='train')
val_dataset = XrayDataset(dataset, vision_processor, tokenizer, split='val')

print(f"Created train dataset with {len(train_dataset)} examples")
print(f"Created validation dataset with {len(val_dataset)} examples")

# Print a sample example to verify formatting
sample_idx = 0
sample = train_dataset[sample_idx]
print("\nSample example:")
print("-" * 50)
print(f"Image shape: {sample['image'].shape}")
print(f"Description: {sample['description']}")
print(f"Labels: {sample['labels']}")
print("-" * 50)

# Save for next steps
torch.save({
    "model": model,
    "tokenizer": tokenizer,
    "vision_processor": vision_processor,
    "train_dataset": train_dataset,
    "val_dataset": val_dataset,
    "conditions": CONDITIONS
}, "checkpoints/03_dataset_prepared.pt")

print("Dataset prepared and saved to checkpoints/03_dataset_prepared.pt")

if __name__ == "__main__":
    print("Dataset preparation completed successfully")
