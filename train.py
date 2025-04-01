# 05_train.py
# Purpose: Execute the fine-tuning process for medical X-ray analysis

import os
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
from huggingface_hub import login
from dotenv import load_dotenv

# Load environment variables for Hugging Face token
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file. Please add your Hugging Face token.")

# Log in to Hugging Face
login(token=HF_TOKEN)

# Check if previous step output exists
if not os.path.exists("checkpoints/04_trainer_defined.pt"):
    raise FileNotFoundError("Run 04_define_trainer.py first to configure the trainer")

# Load model, tokenizer, vision processor, and trainer from previous step
print("Loading model, tokenizer, vision processor, and trainer from checkpoint...")
checkpoint = torch.load("checkpoints/04_trainer_defined.pt")
model = checkpoint["model"]
tokenizer = checkpoint["tokenizer"]
vision_processor = checkpoint["vision_processor"]
trainer = checkpoint["trainer"]
conditions = checkpoint["conditions"]

print("Starting fine-tuning process for medical X-ray analysis...")
print("This may take a while depending on your GPU and dataset size.")
print("Training information will be displayed below:")

# Start timer to measure training duration
start_time = time.time()

# Run the training process
trainer_stats = trainer.train()

# Calculate training duration
training_duration = time.time() - start_time
hours, remainder = divmod(training_duration, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")

# Extract and display training metrics
training_log = trainer.state.log_history
print("\nTraining Metrics:")

# Separate training and evaluation metrics
train_metrics = []
eval_metrics = []

for entry in training_log:
    if "loss" in entry and "eval_loss" not in entry:
        step = entry.get("step", 0)
        loss = entry["loss"]
        train_metrics.append({"Step": step, "Loss": loss})
        print(f"Step {step}: Training Loss = {loss:.6f}")
    elif "eval_loss" in entry:
        step = entry.get("step", 0)
        eval_loss = entry["eval_loss"]
        eval_metrics.append({"Step": step, "Loss": eval_loss})
        print(f"Step {step}: Evaluation Loss = {eval_loss:.6f}")

# Create visualizations of the training and evaluation loss
if train_metrics:
    df_train = pd.DataFrame(train_metrics)
    plt.figure(figsize=(10, 6))
    plt.plot(df_train["Step"], df_train["Loss"], label="Training Loss")
    
    if eval_metrics:
        df_eval = pd.DataFrame(eval_metrics)
        plt.plot(df_eval["Step"], df_eval["Loss"], label="Evaluation Loss", linestyle="--")
    
    plt.title("Training and Evaluation Loss over Time")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig("training_loss.png")
    print("Training and evaluation loss plot saved to training_loss.png")

# Save the fine-tuned model
output_dir = "./fine_tuned_xray_model"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"Saving fine-tuned model to {output_dir}...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
vision_processor.save_pretrained(output_dir)

# Also save the conditions list for inference
conditions_file = os.path.join(output_dir, "conditions.txt")
with open(conditions_file, "w") as f:
    for condition in conditions:
        f.write(f"{condition}\n")

# Push to Hugging Face Hub
repo_name = f"gemma-3-4b-xray-analyzer-{int(time.time())}"
print(f"Pushing fine-tuned model to Hugging Face Hub as '{repo_name}'...")

try:
    trainer.model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)
    vision_processor.push_to_hub(repo_name)
    
    # Push conditions file to Hub
    with open(conditions_file, "r") as f:
        conditions_content = f.read()
    
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(
        path_or_fileobj=conditions_file,
        path_in_repo="conditions.txt",
        repo_id=repo_name,
        token=HF_TOKEN
    )
    
    print(f"Model successfully pushed to Hugging Face Hub: https://huggingface.co/{repo_name}")
except Exception as e:
    print(f"Error pushing to Hugging Face Hub: {e}")
    print("Continuing with local model...")

if __name__ == "__main__":
    print("Fine-tuning completed successfully")
    print(f"You can now run the Gradio app to test the model: python 06_gradio_app.py")
