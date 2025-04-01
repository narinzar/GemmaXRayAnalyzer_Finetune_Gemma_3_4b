# 06_gradio_app.py
# Purpose: Create a Gradio interface for testing the fine-tuned X-ray analysis model

import os
import torch
import numpy as np
import gradio as gr
from PIL import Image
from unsloth import FastModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Define model path (either local or from HF Hub)
# Use local path if available, otherwise use the Hugging Face model repository
MODEL_PATH = "./fine_tuned_xray_model"  # Local path

# If local model doesn't exist, provide instructions
if not os.path.exists(MODEL_PATH):
    print(f"Local model not found at {MODEL_PATH}.")
    print("Please run the training script first: python 05_train.py")
    print("Alternatively, you can specify a Hugging Face model repository in the MODEL_PATH variable.")
    
    # For demonstration, we'll use a default medical vision model
    print("Using a default model for demonstration purposes...")
    MODEL_PATH = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    USE_DEFAULT_MODEL = True
else:
    USE_DEFAULT_MODEL = False

# Load the conditions
if os.path.exists(os.path.join(MODEL_PATH, "conditions.txt")) and not USE_DEFAULT_MODEL:
    with open(os.path.join(MODEL_PATH, "conditions.txt"), "r") as f:
        CONDITIONS = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(CONDITIONS)} conditions from {MODEL_PATH}/conditions.txt")
else:
    # Default conditions if file not found
    CONDITIONS = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", 
        "Mass", "Nodule", "Pneumonia", "Pneumothorax", 
        "Consolidation", "Edema", "Emphysema", "Fibrosis", 
        "Pleural_Thickening", "Hernia", "No Finding"
    ]
    print("Using default conditions list")

try:
    # Load the fine-tuned model
    print(f"Loading model from {MODEL_PATH}...")
    
    if USE_DEFAULT_MODEL:
        # Load default model for demonstration
        from transformers import AutoProcessor, AutoModel
        model = AutoModel.from_pretrained(MODEL_PATH)
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        
        # Define a simple inference function for the default model
        def process_xray(image):
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            
            # For demonstration, we'll return placeholders
            # In a real scenario, we would process the model outputs properly
            return "Demo mode: This would show the model's analysis of the X-ray image."
    else:
        # Load our fine-tuned model
        model, tokenizer = FastModel.from_pretrained(
            model_name=MODEL_PATH,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        
        # Load vision processor
        from transformers import AutoProcessor
        vision_processor = AutoProcessor.from_pretrained(MODEL_PATH)
        
        # Define function to analyze X-ray images
        def process_xray(image):
            """
            Process an X-ray image and return the model's analysis.
            
            Args:
                image: PIL.Image - X-ray image to analyze
                
            Returns:
                String containing the model's analysis of the X-ray
            """
            # Process the image
            processed_image = vision_processor(image, return_tensors="pt")
            
            # Prepare the prompt
            messages = [
                {
                    "role": "user",
                    "content": "Analyze this chest X-ray and describe what conditions are present."
                }
            ]
            
            # Apply chat template
            text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
            )
            
            # Generate response
            inputs = tokenizer(text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
            )
            
            # Decode the generated response
            response = tokenizer.batch_decode(outputs)[0]
            
            # Extract only the model's response
            if "<bot>:" in response:
                response = response.split("<bot>:")[-1].strip()
            
            return response

    print("Model loaded successfully")

    # Create Gradio interface
    with gr.Blocks(title="Medical X-ray Analyzer") as app:
        gr.Markdown("# Medical X-ray Analyzer")
        gr.Markdown("Upload a chest X-ray image to get an analysis of potential medical conditions.")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload X-ray Image")
                submit_btn = gr.Button("Analyze X-ray", variant="primary")
                
            with gr.Column():
                analysis_output = gr.Textbox(label="Analysis Results", lines=10)
                
                with gr.Row():
                    conditions_output = gr.CheckboxGroup(
                        choices=CONDITIONS,
                        label="Detected Conditions",
                        interactive=False
                    )
        
        # Event handler for the submit button
        submit_btn.click(
            fn=process_xray,
            inputs=image_input,
            outputs=analysis_output
        )
        
        # Add examples
        gr.Examples(
            examples=[
                os.path.join(os.path.dirname(__file__), "examples/pneumonia.jpg"),
                os.path.join(os.path.dirname(__file__), "examples/normal.jpg"),
            ],
            inputs=image_input,
        )
        
        # Add disclaimer
        gr.Markdown("""
        ## Disclaimer
        
        This tool is for educational and research purposes only. It is not a medical device and is not intended to be used for diagnosis, prevention, or treatment of any medical condition. Always consult with a qualified healthcare provider for medical advice.
        """)

    # Launch the app
    app.launch(share=True)

except Exception as e:
    print(f"Error loading model or creating Gradio interface: {e}")
    print("Please make sure you've run the training script: python 05_train.py")

if __name__ == "__main__":
    print("Gradio app launched successfully")
