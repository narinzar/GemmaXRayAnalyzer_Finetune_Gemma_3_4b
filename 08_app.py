#!/usr/bin/env python
# 08_app.py
# Purpose: Gradio app for X-ray analysis with Gemma 3

import os
import time
import gradio as gr
import torch
from unsloth import FastModel
from transformers import AutoTokenizer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_USERNAME = os.getenv("HF_USERNAME", "your_username")
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "GemmaXRayAnalyzer_Finetune_Gemma_3_4b")

# Model repository ID
MODEL_ID = f"{HF_USERNAME}/{HF_MODEL_NAME}"

# Demo instruction/prompt
INSTRUCTION = "You are an expert radiologist. Analyze this X-ray image and describe what you see in detail."

# Function to load model and tokenizer
def load_model():
    print(f"Loading model from {MODEL_ID}...")
    try:
        # Try to load from specified model name
        model = FastModel.from_pretrained(MODEL_ID)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        print("Model loaded successfully from Hugging Face Hub")
    except Exception as e:
        print(f"Error loading from {MODEL_ID}: {e}")
        print("Falling back to base Gemma model")
        
        # Fall back to base Gemma model
        model = FastModel.from_pretrained("unsloth/gemma-3-4b-it")
        tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-4b-it")
        print("Base Gemma model loaded successfully as fallback")
    
    return model, tokenizer

# Load model at startup
print("Initializing model...")
model, tokenizer = load_model()

# Function to analyze X-ray image
def analyze_xray(prompt, max_tokens=256, temperature=0.7, top_p=0.9):
    if not prompt:
        return "Please enter a prompt about an X-ray image."
    
    # Format the prompt
    full_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    
    # Tokenize the prompt
    inputs = tokenizer(full_prompt, return_tensors="pt")
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model_device = model.device
    if model_device != device:
        model.to(device)
    
    # Start timer
    start_time = time.time()
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    
    # Compute generation time
    gen_time = time.time() - start_time
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the model's response
    if "<start_of_turn>model\n" in response:
        response = response.split("<start_of_turn>model\n")[-1].strip()
    
    # Add generation time
    response += f"\n\n_Generated in {gen_time:.2f} seconds_"
    
    return response

# Create the Gradio interface
demo = gr.Interface(
    fn=analyze_xray,
    inputs=[
        gr.Textbox(
            label="Prompt",
            placeholder="Analyze this chest X-ray showing...",
            value=INSTRUCTION,
            lines=4
        ),
        gr.Slider(
            minimum=50, maximum=512, value=256, step=1,
            label="Maximum Tokens"
        ),
        gr.Slider(
            minimum=0.1, maximum=1.5, value=0.7, step=0.1,
            label="Temperature"
        ),
        gr.Slider(
            minimum=0.1, maximum=1.0, value=0.9, step=0.1,
            label="Top-p"
        )
    ],
    outputs=gr.Markdown(),
    title="🩻 X-ray Analysis with Gemma 3",
    description="This demo showcases the fine-tuned Gemma 3 model for medical X-ray analysis. Enter your prompt describing the X-ray image or condition you want to analyze.",
    examples=[
        ["Analyze this chest X-ray showing opacity in the lower right lung"],
        ["Describe the findings in this X-ray of a patient with suspected pneumonia"],
        ["What can you tell me about this X-ray showing a possible fracture in the wrist?"],
        ["Generate a detailed report for this abdominal X-ray showing bowel obstruction"],
    ],
    article="""
    ## Disclaimer
    
    This is a demonstration tool and should not be used for actual medical diagnosis. 
    Always consult a qualified healthcare professional for medical advice.
    """
)

# Launch the app
if __name__ == "__main__":
    demo.launch()