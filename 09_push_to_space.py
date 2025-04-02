#!/usr/bin/env python
# 08_push_to_hf_enhanced.py
# Purpose: Enhanced script to push the fine-tuned model to Hugging Face Hub

import os
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path
from huggingface_hub import HfApi, login, create_repo
from dotenv import load_dotenv
import config

def check_model_exists():
    """Check if the fine-tuned model exists and is complete"""
    model_dir = config.MODEL_OUTPUT_DIR
    
    if not os.path.exists(model_dir):
        print(f"❌ Fine-tuned model not found at {model_dir}")
        print("Please run 06_train_model.py first to train the model")
        return False
    
    # Check for key model files
    required_files = ["adapter_config.json", "tokenizer_config.json", "tokenizer.json"]
    for file in required_files:
        if not os.path.exists(os.path.join(model_dir, file)):
            print(f"❌ Required model file {file} not found in {model_dir}")
            print("Model directory exists but appears incomplete")
            return False
    
    # Check for model weights (either .bin or .safetensors)
    found_weights = False
    for ext in [".bin", ".safetensors"]:
        if list(Path(model_dir).glob(f"*{ext}")):
            found_weights = True
            break
    
    if not found_weights:
        print(f"❌ No model weight files found in {model_dir}")
        print("Model directory exists but appears to be missing weight files")
        return False
    
    # Model exists and key files found
    print(f"✅ Model found at {model_dir} with required files")
    
    # Estimate model size
    model_size_bytes = sum(f.stat().st_size for f in Path(model_dir).glob('**/*') if f.is_file())
    model_size_gb = model_size_bytes / (1024 ** 3)
    print(f"📊 Estimated model size: {model_size_gb:.2f} GB")
    
    return True

def get_hf_credentials():
    """Get Hugging Face credentials from environment variables"""
    # Load environment variables
    load_dotenv()
    
    hf_token = os.getenv("HF_TOKEN")
    hf_username = os.getenv("HF_USERNAME")
    hf_model_name = os.getenv("HF_MODEL_NAME", config.HF_MODEL_NAME)
    
    if not hf_token:
        print("❌ HF_TOKEN not found in environment variables")
        print("Please add it to the .env file")
        return None, None, None
    
    if not hf_username:
        print("❌ HF_USERNAME not found in environment variables")
        print("Please add it to the .env file")
        return None, None, None
    
    return hf_token, hf_username, hf_model_name

def prepare_model_for_upload(model_dir, repo_id, readme_content=None):
    """Prepare the model directory for upload by adding README and model card info"""
    # Create a README.md if it doesn't exist or if content is provided
    readme_path = os.path.join(model_dir, "README.md")
    if readme_content or not os.path.exists(readme_path):
        if not readme_content:
            readme_content = f"""# {repo_id.split('/')[-1]}

A fine-tuned version of {config.MODEL_NAME} for medical X-ray analysis.

## Model Description

This model was fine-tuned on the {config.DATASET_NAME} dataset to improve its ability to analyze and describe medical X-ray images.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

# Create a prompt for X-ray analysis
instruction = "You are an expert radiologist. Analyze this X-ray image and describe what you see in detail."
prompt = f"<start_of_turn>user\\n{{instruction}}<end_of_turn>\\n<start_of_turn>model\\n"

# Generate text
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7, top_p=0.9)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Training Details

- Base model: {config.MODEL_NAME}
- Dataset: {config.DATASET_NAME}
- Fine-tuning method: LoRA (Low-Rank Adaptation)
- LoRA rank: {config.LORA_RANK}
- Training steps: {config.MAX_STEPS}
- Batch size: {config.BATCH_SIZE} x {config.GRADIENT_ACCUMULATION_STEPS} = {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS}
- Learning rate: {config.LEARNING_RATE}
"""
        
        with open(readme_path, "w") as f:
            f.write(readme_content)
            print(f"✅ Created README.md in {model_dir}")
    
    # Create a model card metadata file
    model_card = {
        "language": ["en"],
        "license": "mit",
        "tags": ["gemma", "medical", "radiology", "x-ray", "finetune"],
        "datasets": [config.DATASET_NAME],
        "model-index": [
            {
                "name": repo_id.split('/')[-1],
                "results": []
            }
        ]
    }
    
    with open(os.path.join(model_dir, "model_card.json"), "w") as f:
        json.dump(model_card, f, indent=2)
        print(f"✅ Created model_card.json in {model_dir}")
    
    return True

def push_to_hub_api(model_dir, hf_token, repo_id):
    """Push the model to Hugging Face Hub using the Python API"""
    print(f"🚀 Pushing model to Hugging Face Hub as {repo_id}...")
    
    # Login to Hugging Face
    login(token=hf_token)
    
    # Create the API client
    api = HfApi()
    
    # Create the repo if it doesn't exist
    try:
        create_repo(repo_id, token=hf_token, repo_type="model", exist_ok=True)
        print(f"✅ Repository {repo_id} created or already exists")
    except Exception as e:
        print(f"⚠️ Error creating repository: {e}")
        print("Attempting to continue anyway...")
    
    # Push model to Hub
    try:
        print("📤 Uploading model files to Hugging Face Hub...")
        api.upload_folder(
            folder_path=model_dir,
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=["*.pt", "__pycache__", "*.pyc", ".git*", "old_*"],
            commit_message=f"Upload fine-tuned {repo_id.split('/')[-1]} model"
        )
        print(f"✅ Model successfully pushed to Hugging Face Hub: https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"❌ Error pushing model to Hub via API: {e}")
        return False

def push_to_hub_cli(model_dir, hf_token, repo_id):
    """Push the model to Hugging Face Hub using git and the command-line interface"""
    print(f"🚀 Pushing model to Hugging Face Hub as {repo_id} using git...")
    
    # Before proceeding, check if git-lfs is installed
    result = subprocess.run("git lfs --version", shell=True, capture_output=True)
    if result.returncode != 0:
        print("⚠️ Git LFS not detected. Installing Git LFS is recommended for large model files.")
        print("Visit https://git-lfs.github.com/ for installation instructions")
        
        # Ask user if they want to continue without Git LFS
        if input("Continue without Git LFS? (y/n): ").lower() != 'y':
            return False
    else:
        print(f"✅ Git LFS detected: {result.stdout.decode().strip()}")
    
    # Create a temporary directory for the git repository
    temp_dir = f"temp_model_upload_{int(time.time())}"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    try:
        # Initialize git in the temporary directory
        print("🔄 Initializing git repository...")
        subprocess.run("git init", cwd=temp_dir, shell=True, check=True)
        
        # Setup Git LFS if available
        if result.returncode == 0:
            print("🔄 Setting up Git LFS...")
            subprocess.run("git lfs install", cwd=temp_dir, shell=True, check=True)
            subprocess.run('git lfs track "*.bin" "*.safetensors" "*.h5"', cwd=temp_dir, shell=True, check=True)
            subprocess.run("git add .gitattributes", cwd=temp_dir, shell=True, check=True)
        
        # Copy model files to the temporary directory
        print("🔄 Copying model files...")
        for item in os.listdir(model_dir):
            src = os.path.join(model_dir, item)
            dst = os.path.join(temp_dir, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        
        # Configure git
        print("🔄 Configuring git...")
        subprocess.run('git config --local user.email "user@example.com"', cwd=temp_dir, shell=True, check=True)
        subprocess.run('git config --local user.name "Hugging Face User"', cwd=temp_dir, shell=True, check=True)
        
        # Login to Hugging Face
        print("🔄 Logging in to Hugging Face...")
        # Use the token directly without storing it in git config
        login_cmd = f"huggingface-cli login --token {hf_token}"
        subprocess.run(login_cmd, shell=True, capture_output=True, check=True)
        
        # Create the repository if it doesn't exist
        print(f"🔄 Creating repository {repo_id}...")
        create_cmd = f"huggingface-cli repo create {repo_id} --type model --yes"
        subprocess.run(create_cmd, shell=True, check=False)  # Don't check as it might already exist
        
        # Add the remote
        print("🔄 Adding remote...")
        remote_url = f"https://huggingface.co/{repo_id}"
        subprocess.run(f"git remote add origin {remote_url}", cwd=temp_dir, shell=True, check=True)
        
        # Add all files
        print("🔄 Adding files...")
        subprocess.run("git add .", cwd=temp_dir, shell=True, check=True)
        
        # Commit
        print("🔄 Committing changes...")
        subprocess.run(f'git commit -m "Upload fine-tuned model"', cwd=temp_dir, shell=True, check=True)
        
        # Push (with force flag)
        print("🔄 Pushing to Hugging Face Hub...")
        push_cmd = f"git push -f origin main"
        result = subprocess.run(push_cmd, cwd=temp_dir, shell=True)
        
        if result.returncode == 0:
            print(f"✅ Model successfully pushed to Hugging Face Hub: https://huggingface.co/{repo_id}")
            return True
        else:
            print("❌ Git push failed, trying with HTTPS remote URL...")
            # Try alternative push method
            subprocess.run(f"git remote remove origin", cwd=temp_dir, shell=True)
            subprocess.run(f"git remote add origin https://{hf_token}@huggingface.co/{repo_id}", cwd=temp_dir, shell=True, check=True)
            result = subprocess.run(push_cmd, cwd=temp_dir, shell=True)
            
            if result.returncode == 0:
                print(f"✅ Model successfully pushed to Hugging Face Hub: https://huggingface.co/{repo_id}")
                return True
            else:
                print("❌ Git push failed")
                return False
    except Exception as e:
        print(f"❌ Error pushing model to Hub via git: {e}")
        return False
    finally:
        # Clean up temporary directory
        print("🧹 Cleaning up temporary directory...")
        shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    """Push the fine-tuned model to Hugging Face Hub"""
    print("🔍 Preparing to push model to Hugging Face Hub...")
    
    # Parse command-line arguments
    use_cli = "--cli" in sys.argv
    
    # Check if model exists
    if not check_model_exists():
        sys.exit(1)
    
    # Get HF credentials
    hf_token, hf_username, hf_model_name = get_hf_credentials()
    if not hf_token or not hf_username:
        sys.exit(1)
    
    # Construct repository ID
    repo_id = f"{hf_username}/{hf_model_name}"
    print(f"📌 Will push model to {repo_id}")
    
    # Prepare model for upload
    prepare_model_for_upload(config.MODEL_OUTPUT_DIR, repo_id)
    
    # Try pushing with selected method
    if use_cli:
        success = push_to_hub_cli(config.MODEL_OUTPUT_DIR, hf_token, repo_id)
    else:
        success = push_to_hub_api(config.MODEL_OUTPUT_DIR, hf_token, repo_id)
        
        # If API method fails, try CLI method as backup
        if not success and input("API upload failed. Try using git method? (y/n): ").lower() == 'y':
            success = push_to_hub_cli(config.MODEL_OUTPUT_DIR, hf_token, repo_id)
    
    if success:
        # Create a file indicating successful push
        with open(os.path.join(config.RESULTS_DIR, "push_success.txt"), "w") as f:
            f.write(f"Model pushed to Hugging Face Hub at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Repository: https://huggingface.co/{repo_id}\n")
        
        print("\n🎉 Push to Hugging Face Hub completed successfully!")
        print(f"🔗 Your model is now available at: https://huggingface.co/{repo_id}")
    else:
        print("\n❌ Failed to push model to Hugging Face Hub")
        print("🛠️ Try running with --cli flag: python 08_push_to_hf_enhanced.py --cli")
        sys.exit(1)

if __name__ == "__main__":
    main()