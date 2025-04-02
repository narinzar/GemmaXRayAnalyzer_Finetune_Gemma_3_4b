#!/usr/bin/env python
# 10_load_finetuned.py
# Purpose: Load and use the fine-tuned model for inference

import os
import sys
import time
import torch
import argparse
from transformers import AutoTokenizer
from unsloth import FastModel
from dotenv import load_dotenv
import config

def load_model_from_local(model_path=None):
    """
    Load the fine-tuned model from a local directory
    
    Parameters:
    -----------
    model_path : str
        Path to the model directory. If None, uses the path from config.py
        
    Returns:
    --------
    model : FastModel
        The loaded model
    tokenizer : AutoTokenizer
        The loaded tokenizer
    """
    if model_path is None:
        model_path = config.MODEL_OUTPUT_DIR
        
    print(f"Loading model from local path: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} does not exist. "
                              f"Run training first or specify a valid path.")
    
    # Load model and tokenizer
    try:
        model = FastModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"✅ Model loaded successfully from {model_path}")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

def load_model_from_hub(repo_id=None):
    """
    Load the fine-tuned model from Hugging Face Hub
    
    Parameters:
    -----------
    repo_id : str
        Hugging Face repository ID (username/repo_name). If None, constructs from env vars.
        
    Returns:
    --------
    model : FastModel
        The loaded model
    tokenizer : AutoTokenizer
        The loaded tokenizer
    """
    # Load environment variables for credentials
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    hf_username = os.getenv("HF_USERNAME")
    hf_model_name = os.getenv("HF_MODEL_NAME", config.HF_MODEL_NAME)
    
    # Construct repo_id if not provided
    if repo_id is None:
        if not hf_username:
            print("❌ HF_USERNAME not found in environment variables.")
            print("Please provide it in .env file or use --hub argument.")
            sys.exit(1)
        repo_id = f"{hf_username}/{hf_model_name}"
    
    print(f"Loading model from Hugging Face Hub: {repo_id}")
    
    # Load model and tokenizer
    try:
        model = FastModel.from_pretrained(repo_id, token=hf_token)
        tokenizer = AutoTokenizer.from_pretrained(repo_id, token=hf_token)
        print(f"✅ Model loaded successfully from Hugging Face Hub: {repo_id}")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Error loading model from Hub: {e}")
        sys.exit(1)

def generate_response(model, tokenizer, prompt, max_new_tokens=256, 
                     temperature=0.7, top_p=0.9, top_k=50, 
                     system_prompt=None):
    """
    Generate a response from the model
    
    Parameters:
    -----------
    model : FastModel
        The loaded model
    tokenizer : AutoTokenizer
        The loaded tokenizer
    prompt : str
        The prompt to generate a response for
    max_new_tokens : int
        Maximum number of tokens to generate
    temperature : float
        Sampling temperature (higher = more random)
    top_p : float
        Nucleus sampling parameter
    top_k : int
        Top-k sampling parameter
    system_prompt : str
        Optional system prompt to prepend
        
    Returns:
    --------
    response : str
        The generated response
    generation_time : float
        Time taken to generate the response
    """
    # Prepare the prompt
    if system_prompt:
        formatted_prompt = f"<start_of_turn>system\n{system_prompt}<end_of_turn>\n<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    else:
        formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    
    # Tokenize the prompt
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model = model.to(device)
    
    # Start timing
    start_time = time.time()
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=1.1,  # Helps avoid repetition
        )
    
    # Calculate generation time
    generation_time = time.time() - start_time
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract model's response
    if "<start_of_turn>model\n" in response:
        response = response.split("<start_of_turn>model\n")[-1].strip()
    
    return response, generation_time

def interactive_session(model, tokenizer, system_prompt=None, temperature=0.7, max_tokens=256):
    """
    Run an interactive session with the model
    
    Parameters:
    -----------
    model : FastModel
        The loaded model
    tokenizer : AutoTokenizer
        The loaded tokenizer
    system_prompt : str
        Optional system prompt to set context for all interactions
    temperature : float
        Sampling temperature (higher = more random)
    max_tokens : int
        Maximum number of tokens to generate
    """
    print("\n" + "="*50)
    print("🩻 Interactive X-ray Analysis Session")
    print("  Type your prompt or 'exit' to quit")
    print("="*50 + "\n")
    
    if system_prompt:
        print(f"System prompt: {system_prompt}\n")
    
    while True:
        user_input = input("You > ")
        
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Exiting interactive session")
            break
        
        print("\nGenerating...")
        response, gen_time = generate_response(
            model, 
            tokenizer, 
            user_input,
            max_new_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt
        )
        
        print(f"\nModel > {response}")
        print(f"\n[Generated in {gen_time:.2f}s]")
        print("-"*50)

def main():
    """Main function to parse arguments and run the appropriate mode"""
    parser = argparse.ArgumentParser(description="Load and use the fine-tuned Gemma X-ray model")
    
    # Model source
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--local", metavar="PATH", type=str, 
                             help="Path to local model directory")
    source_group.add_argument("--hub", metavar="REPO_ID", type=str,
                             help="Hugging Face Hub repository ID (username/repo_name)")
    
    # Operation mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--interactive", action="store_true", 
                           help="Run in interactive mode")
    mode_group.add_argument("--prompt", type=str, 
                           help="Single prompt to generate a response for")
    
    # Generation parameters
    parser.add_argument("--max-tokens", type=int, default=256,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature (higher = more random)")
    parser.add_argument("--top-p", type=float, default=0.9,
                       help="Nucleus sampling parameter")
    parser.add_argument("--top-k", type=int, default=50,
                       help="Top-k sampling parameter")
    parser.add_argument("--system-prompt", type=str,
                       help="Optional system prompt to set context")
    
    args = parser.parse_args()
    
    # Set default system prompt if not provided
    if not args.system_prompt:
        args.system_prompt = "You are an expert radiologist. Analyze X-ray images based on descriptions and provide detailed, accurate medical assessments."
    
    # Load model from specified source
    if args.hub:
        model, tokenizer = load_model_from_hub(args.hub)
    elif args.local:
        model, tokenizer = load_model_from_local(args.local)
    else:
        # Try local first, then fall back to Hub
        try:
            model, tokenizer = load_model_from_local()
        except FileNotFoundError:
            print("Local model not found, trying Hugging Face Hub...")
            model, tokenizer = load_model_from_hub()
    
    # Run in specified mode
    if args.interactive:
        interactive_session(
            model, 
            tokenizer,
            system_prompt=args.system_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
    elif args.prompt:
        # Generate a single response
        response, generation_time = generate_response(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            system_prompt=args.system_prompt
        )
        
        print("\n" + "="*50)
        print(f"Prompt: {args.prompt}")
        print("="*50)
        print(f"\nResponse: {response}")
        print(f"\n[Generated in {generation_time:.2f}s]")
    else:
        # Default to interactive mode
        interactive_session(
            model, 
            tokenizer,
            system_prompt=args.system_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )

if __name__ == "__main__":
    main()