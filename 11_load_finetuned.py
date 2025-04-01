#!/usr/bin/env python
# load_finetuned_model.py
# Purpose: Utility script to load and use the fine-tuned model

import os
import torch
import argparse
from unsloth import FastModel
from transformers import AutoTokenizer
from dotenv import load_dotenv

def load_model_from_local(model_path):
    """Load a fine-tuned model from a local directory"""
    print(f"Loading model from local path: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} does not exist")
    
    # Load model and tokenizer
    model = FastModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer

def load_model_from_hub(repo_id):
    """Load a fine-tuned model from Hugging Face Hub"""
    print(f"Loading model from Hugging Face Hub: {repo_id}")
    
    # Load environment variables for HF_TOKEN
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token:
        print("Warning: HF_TOKEN not found in environment variables")
        print("You may encounter issues downloading the model if it's private")
    
    # Load model and tokenizer
    model = FastModel.from_pretrained(repo_id, token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(repo_id, token=hf_token)
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=256, temperature=0.7, top_p=0.9, top_k=50):
    """Generate a response from the model"""
    # Format the prompt with the model's chat template
    formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    
    # Tokenize the prompt
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model = model.to(device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the model's response
    if "<start_of_turn>model\n" in response:
        response = response.split("<start_of_turn>model\n")[-1].strip()
    
    return response

def interactive_session(model, tokenizer):
    """Run an interactive session with the model"""
    print("\n" + "=" * 50)
    print("Interactive X-ray Analysis Session")
    print("Enter your prompt or type 'exit' to quit")
    print("=" * 50 + "\n")
    
    while True:
        # Get user input
        user_input = input("You: ")
        
        # Check if user wants to exit
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Exiting interactive session")
            break
        
        # Generate and print response
        print("\nThinking...\n")
        response = generate_response(model, tokenizer, user_input)
        print(f"Model: {response}\n")

def main():
    """Main function to load and interact with the model"""
    parser = argparse.ArgumentParser(description="Load and use a fine-tuned Gemma model")
    parser.add_argument("--local", type=str, help="Path to local model directory", default="models/gemma_xray_model")
    parser.add_argument("--hub", type=str, help="Hugging Face Hub repository ID (username/repo_name)")
    parser.add_argument("--prompt", type=str, help="Single prompt to generate a response for")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--max_tokens", type=int, default=256, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    
    args = parser.parse_args()
    
    # Determine which model to load
    if args.hub:
        model, tokenizer = load_model_from_hub(args.hub)
    else:
        model, tokenizer = load_model_from_local(args.local)
    
    # Run in specified mode
    if args.interactive:
        interactive_session(model, tokenizer)
    elif args.prompt:
        response = generate_response(
            model, 
            tokenizer, 
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
        print(f"Response: {response}")
    else:
        # Default to interactive mode if no mode specified
        interactive_session(model, tokenizer)

if __name__ == "__main__":
    main()
