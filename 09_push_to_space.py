#!/usr/bin/env python
# push_to_space.py
# Purpose: Push the Gradio app to Hugging Face Space

import os
import sys
import subprocess
from huggingface_hub import login, HfApi, create_repo
from dotenv import load_dotenv

def push_to_space():
    """Push the Gradio app to a Hugging Face Space"""
    # Load environment variables
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    hf_username = os.getenv("HF_USERNAME")
    space_name = os.getenv("HF_REPO_NAME", "GemmaXRayAnalyzer_Finetune_Gemma_3_4b")
    
    if not hf_token or not hf_username:
        print("ERROR: HF_TOKEN or HF_USERNAME not found in .env file")
        print("Please set these environment variables and try again")
        return False
    
    # Login to Hugging Face
    login(token=hf_token)
    
    # Space ID
    space_id = f"{hf_username}/{space_name}"
    print(f"Pushing Gradio app to Space: {space_id}")
    
    # Files to include in the Space
    files_to_include = [
        "app.py",  # Gradio app
        "huggingface-space-config.yml",  # Space configuration
        "requirements.txt",  # Dependencies for the Space
    ]
    
    # Check if all required files exist
    for file in files_to_include:
        if not os.path.exists(file):
            print(f"ERROR: Required file '{file}' not found")
            return False
    
    # Create requirements_space.txt if it doesn't exist
    if not os.path.exists("requirements.txt"):
        with open("requirements.txt", "w") as f:
            f.write("gradio>=3.50.2\n")
            f.write("unsloth>=2023.11.0\n")
            f.write("transformers>=4.34.0\n")
            f.write("torch>=2.0.0\n")
            f.write("python-dotenv>=1.0.0\n")
    
    # Create or update the Space
    try:
        api = HfApi()
        
        # Create the Space if it doesn't exist
        try:
            create_repo(space_id, repo_type="space", space_sdk="gradio", exist_ok=True)
        except Exception as e:
            print(f"Note: {e}")
            print("Attempting to update existing Space...")
        
        # Option 1: Using the API (simplest)
        print("Uploading files to Space...")
        for file in files_to_include:
            api.upload_file(
                path_or_fileobj=file,
                path_in_repo=file,
                repo_id=space_id,
                repo_type="space"
            )
        
        print(f"Successfully pushed to Space: https://huggingface.co/spaces/{space_id}")
        print("The Space will now build automatically. This may take a few minutes.")
        return True
        
    except Exception as e:
        print(f"Error pushing to Space: {e}")
        print("Attempting alternative method...")
        
        # Option 2: Using git (if API fails)
        try:
            # Clone the Space repository
            temp_dir = "temp_space"
            if os.path.exists(temp_dir):
                subprocess.run(f"rm -rf {temp_dir}", shell=True)
            
            subprocess.run(f"git clone https://huggingface.co/spaces/{space_id} {temp_dir}", shell=True, check=True)
            
            # Copy files to the cloned repository
            for file in files_to_include:
                subprocess.run(f"cp {file} {temp_dir}/", shell=True, check=True)
            
            # Commit and push changes
            subprocess.run("git add .", cwd=temp_dir, shell=True, check=True)
            subprocess.run("git commit -m 'Update Gradio app'", cwd=temp_dir, shell=True, check=True)
            subprocess.run("git push", cwd=temp_dir, shell=True, check=True)
            
            # Clean up
            subprocess.run(f"rm -rf {temp_dir}", shell=True)
            
            print(f"Successfully pushed to Space: https://huggingface.co/spaces/{space_id}")
            print("The Space will now build automatically. This may take a few minutes.")
            return True
            
        except Exception as e2:
            print(f"Error with git method: {e2}")
            return False

if __name__ == "__main__":
    push_to_space()