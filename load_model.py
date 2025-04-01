# 01_load_model.py
# Purpose: Initialize and load the Gemma 3 model using Unsloth for efficient fine-tuning

import os
import torch
from dotenv import load_dotenv
from unsloth import FastModel
from transformers import AutoProcessor, AutoModelForVision2Seq

# Load environment variables from .env file (for Hugging Face token)
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file. Please add your Hugging Face token.")

# Check if GPU is available
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("WARNING: No GPU detected. Fine-tuning will be very slow on CPU.")

# Model configuration
MODEL = "unsloth/gemma-3-4b-it"  # Using the 4B instruction-tuned variant

# Load the text model and tokenizer with Unsloth's FastModel
print("Loading Gemma 3 model and tokenizer...")
model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL,
    max_seq_length=2048,    # Maximum context window size
    dtype=None,             # Automatically determine best dtype
    load_in_4bit=True,      # Use 4-bit quantization to reduce memory usage
    full_finetuning=False   # Use parameter-efficient fine-tuning via LoRA
)

print(f"Model {MODEL} loaded successfully")
print(f"Tokenizer vocabulary size: {len(tokenizer)}")

# Load vision processor for handling X-ray images
print("Loading vision processor...")
# Using the CLIP processor for image preprocessing
vision_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("Vision processor loaded successfully")

# Save for next steps
if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")
    
torch.save({
    "model": model,
    "tokenizer": tokenizer,
    "vision_processor": vision_processor
}, "checkpoints/01_model_loaded.pt")

print("Model, tokenizer, and vision processor saved to checkpoints/01_model_loaded.pt")

if __name__ == "__main__":
    print("Model loaded successfully and ready for LoRA configuration")
