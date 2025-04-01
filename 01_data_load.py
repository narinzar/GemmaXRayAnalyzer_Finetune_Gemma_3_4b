#!/usr/bin/env python
# 01_data_load.py
# Purpose: Load and prepare the X-ray dataset

import os
import json
import config
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def main():
    """Load X-ray dataset and split into train/val/test sets"""
    
    # Create necessary directories
    for directory in [config.DATA_DIR, config.CHECKPOINT_DIR, config.RESULTS_DIR, 
                     config.FIGURES_DIR, config.OUTPUT_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    print(f"Loading {config.DATASET_NAME} dataset...")
    radiology_dataset = load_dataset(config.DATASET_NAME)
    full_dataset = radiology_dataset["train"]
    print(f"Dataset loaded with {len(full_dataset)} examples")
    
    # Extract all data samples
    all_samples = []
    for sample in full_dataset:
        all_samples.append({
            'caption': sample['caption']
        })
    
    # Perform the split based on configuration
    total_val_test_size = config.TEST_SIZE + config.VAL_SIZE
    train_samples, temp_samples = train_test_split(
        all_samples, 
        test_size=total_val_test_size, 
        random_state=config.RANDOM_SEED
    )
    
    # Calculate relative sizes for val and test sets
    relative_val_size = config.VAL_SIZE / total_val_test_size
    
    val_samples, test_samples = train_test_split(
        temp_samples, 
        test_size=(1-relative_val_size), 
        random_state=config.RANDOM_SEED
    )
    
    print(f"Dataset split:")
    print(f"  Training set: {len(train_samples)} samples")
    print(f"  Validation set: {len(val_samples)} samples")
    print(f"  Test set: {len(test_samples)} samples")
    
    # Save splits to disk
    dataset_path = os.path.join(config.DATA_DIR, "dataset_splits.json")
    with open(dataset_path, 'w') as f:
        json.dump({
            'train': train_samples,
            'validation': val_samples,
            'test': test_samples
        }, f)
    
    print(f"Dataset splits saved to {dataset_path}")
    
    # Return the splits for potential further processing
    return train_samples, val_samples, test_samples

if __name__ == "__main__":
    main()
