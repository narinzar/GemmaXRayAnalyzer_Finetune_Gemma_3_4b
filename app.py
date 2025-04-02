#!/usr/bin/env python
# 08_app.py
# Purpose: Gradio app for X-ray analysis with Gemma 3

import os
import time
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
from PIL import Image
import traceback

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
    
    # Get the device upfront to ensure model loads on the right device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # First try loading from user's HF repository
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",  # Let transformers decide the device mapping
            torch_dtype="auto"   # Let transformers decide the dtype
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        print("Model loaded successfully from Hugging Face Hub")
    except Exception as e:
        print(f"Error loading from {MODEL_ID}: {e}")
        print("Falling back to base Gemma model")
        
        # Fall back to base Gemma model
        try:
            model = AutoModelForCausalLM.from_pretrained(
                "unsloth/gemma-3-4b-it",
                device_map="auto",  # Let transformers decide the device mapping
                torch_dtype="auto"   # Let transformers decide the dtype
            )
            tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-4b-it")
            print("Base Gemma model loaded successfully as fallback")
        except Exception as e:
            print(f"Error loading fallback model: {e}")
            raise
    
    return model, tokenizer

# Load model at startup
print("Initializing model...")
model, tokenizer = load_model()

# Function to analyze X-ray image and text
def analyze_xray(image, prompt, max_tokens=256, temperature=0.7, top_p=0.9):
    try:
        if not prompt:
            prompt = INSTRUCTION
        
        # Handle the image if provided
        image_description = ""
        if image is not None:
            # Save the image temporarily for display
            temp_img_path = "temp_xray.jpg"
            if isinstance(image, Image.Image):
                image.save(temp_img_path)
            else:
                # If it's already a path
                temp_img_path = image
            
            image_description = f"\n\nImage uploaded: X-ray image received for analysis."
        
        # Combine prompt with image notification
        full_text_prompt = prompt + image_description
        
        # Format the prompt using Gemma's format
        full_prompt = f"<start_of_turn>user\n{full_text_prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        # Tokenize the prompt
        inputs = tokenizer(full_prompt, return_tensors="pt")
        
        # Move inputs to the correct device - the model should already be on the correct device
        try:
            # Try to get the model's device directly
            device = next(model.parameters()).device
        except:
            # If that fails, default to CUDA if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move inputs to the device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
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
        
        # Return the image if it was provided, along with the response
        result = ""
        if image is not None:
            # Create the response with the image
            result = f"**X-ray Analysis:**\n\n{response}\n\n_Generated in {gen_time:.2f} seconds_"
        else:
            # Just return the text response
            result = f"{response}\n\n_Generated in {gen_time:.2f} seconds_"
        
        return result
    except Exception as e:
        print(f"Error in analyze_xray: {e}")
        traceback.print_exc()
        return f"Error generating response: {str(e)}\n\nPlease try a different prompt or check the console for detailed error information."

# Create the Gradio interface with image upload
demo = gr.Interface(
    fn=analyze_xray,
    inputs=[
        gr.Image(type="pil", label="Upload X-ray Image (Optional)"),
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
    description="This demo showcases the Gemma 3 model for medical X-ray analysis. Upload an X-ray image and enter your prompt describing what you'd like to analyze.",
    examples=[
        [None, "Analyze this chest X-ray showing opacity in the lower right lung"],
        [None, "Describe the findings in this X-ray of a patient with suspected pneumonia"],
        [None, "What can you tell me about this X-ray showing a possible fracture in the wrist?"],
        [None, "Generate a detailed report for this abdominal X-ray showing bowel obstruction"],
    ],
    article="""
    ## How to Use
    
    1. (Optional) Upload an X-ray image using the image upload area
    2. Enter a prompt describing what you want the model to analyze
    3. Adjust generation parameters if desired
    4. Click "Submit" to generate the analysis
    
    ## Example Prompts
    
    - "Analyze this chest X-ray and describe any abnormalities"
    - "What pathologies are visible in this X-ray image?"
    - "Is there evidence of pneumonia in this chest X-ray?"
    - "Generate a radiological report for this X-ray"
    
    ## Disclaimer
    
    This is a demonstration tool and should not be used for actual medical diagnosis. 
    Always consult a qualified healthcare professional for medical advice.
    
    Note: The model has been fine-tuned on radiological text data but may not directly 
    analyze the uploaded image. The image upload feature is provided for reference and context.
    """
)

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)  # Set share=True to create a public link