#!/usr/bin/env python
# 11_use_model_api.py
# Purpose: Demonstrate how to use the Hugging Face API with the fine-tuned model

import os
import sys
import requests
import json
import time
from dotenv import load_dotenv
import argparse

def get_api_credentials():
    """
    Get Hugging Face API credentials from environment variables
    
    Returns:
    --------
    api_token : str
        Hugging Face API token
    model_id : str
        Model ID to use for inference
    """
    # Load environment variables
    load_dotenv()
    
    api_token = os.getenv("HF_TOKEN")
    hf_username = os.getenv("HF_USERNAME")
    hf_model_name = os.getenv("HF_MODEL_NAME")
    
    if not api_token:
        print("❌ HF_TOKEN not found in environment variables.")
        print("Please set it in the .env file.")
        sys.exit(1)
    
    if not hf_username or not hf_model_name:
        print("⚠️ HF_USERNAME or HF_MODEL_NAME not found in environment variables.")
        print("You will need to specify the model ID with --model-id.")
        return api_token, None
    
    model_id = f"{hf_username}/{hf_model_name}"
    return api_token, model_id

def query_model_api(api_token, model_id, prompt, system_prompt=None, 
                   max_new_tokens=256, temperature=0.7, top_p=0.9):
    """
    Query the Hugging Face Inference API
    
    Parameters:
    -----------
    api_token : str
        Hugging Face API token
    model_id : str
        Model ID to use for inference
    prompt : str
        Prompt to generate a response for
    system_prompt : str
        Optional system prompt to set context
    max_new_tokens : int
        Maximum number of tokens to generate
    temperature : float
        Sampling temperature (higher = more random)
    top_p : float
        Nucleus sampling parameter
    
    Returns:
    --------
    response : str
        The generated response
    """
    API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {api_token}"}
    
    # Prepare the prompt
    if system_prompt:
        formatted_prompt = f"<start_of_turn>system\n{system_prompt}<end_of_turn>\n<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    else:
        formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    
    # Prepare the API payload
    payload = {
        "inputs": formatted_prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "return_full_text": False,
        },
        "options": {
            "use_cache": False,
            "wait_for_model": True,
        }
    }
    
    # Make the API request
    print(f"Sending request to {model_id}...")
    start_time = time.time()
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors
        end_time = time.time()
        
        result = response.json()
        
        # Handle different response formats
        if isinstance(result, list) and len(result) > 0:
            if 'generated_text' in result[0]:
                text = result[0]['generated_text']
            else:
                text = str(result[0])
        else:
            text = str(result)
        
        print(f"✅ Response received in {end_time - start_time:.2f}s")
        return text
        
    except requests.exceptions.HTTPError as err:
        if response.status_code == 503:
            print("⚠️ Model is loading... Retrying in 5 seconds")
            time.sleep(5)
            return query_model_api(api_token, model_id, prompt, system_prompt, 
                                 max_new_tokens, temperature, top_p)
        else:
            print(f"❌ HTTP Error: {err}")
            print(f"Response: {response.text}")
            sys.exit(1)
    
    except requests.exceptions.RequestException as err:
        print(f"❌ Request Error: {err}")
        sys.exit(1)
    
    except json.JSONDecodeError:
        print(f"❌ Error decoding JSON response: {response.text}")
        sys.exit(1)

def interactive_api_session(api_token, model_id, system_prompt=None,
                          max_new_tokens=256, temperature=0.7, top_p=0.9):
    """
    Run an interactive session using the Hugging Face API
    
    Parameters:
    -----------
    api_token : str
        Hugging Face API token
    model_id : str
        Model ID to use for inference
    system_prompt : str
        Optional system prompt to set context
    max_new_tokens : int
        Maximum number of tokens to generate
    temperature : float
        Sampling temperature (higher = more random)
    top_p : float
        Nucleus sampling parameter
    """
    print("\n" + "="*50)
    print(f"🩻 Interactive X-ray Analysis Session using API")
    print(f"  Model: {model_id}")
    print(f"  Type your prompt or 'exit' to quit")
    print("="*50 + "\n")
    
    if system_prompt:
        print(f"System prompt: {system_prompt}\n")
    
    while True:
        user_input = input("You > ")
        
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Exiting interactive session")
            break
        
        print("\nGenerating...")
        response = query_model_api(
            api_token,
            model_id,
            user_input,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        print(f"\nModel > {response}")
        print("-"*50)

def main():
    """Main function to parse arguments and run the appropriate mode"""
    parser = argparse.ArgumentParser(description="Use the fine-tuned X-ray model via Hugging Face API")
    
    # Model ID
    parser.add_argument("--model-id", type=str,
                       help="Hugging Face model ID (username/repo_name)")
    
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
    parser.add_argument("--system-prompt", type=str,
                       help="Optional system prompt to set context")
    
    args = parser.parse_args()
    
    # Set default system prompt if not provided
    if not args.system_prompt:
        args.system_prompt = "You are an expert radiologist. Analyze X-ray images based on descriptions and provide detailed, accurate medical assessments."
    
    # Get credentials
    api_token, default_model_id = get_api_credentials()
    
    # Use specified model ID or default
    model_id = args.model_id or default_model_id
    if not model_id:
        print("❌ Model ID not specified and not found in environment variables.")
        parser.print_help()
        sys.exit(1)
    
    # Run in specified mode
    if args.interactive:
        interactive_api_session(
            api_token,
            model_id,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
    elif args.prompt:
        # Generate a single response
        response = query_model_api(
            api_token,
            model_id,
            args.prompt,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        print("\n" + "="*50)
        print(f"Prompt: {args.prompt}")
        print("="*50)
        print(f"\nResponse: {response}")
    else:
        # Default to interactive mode
        interactive_api_session(
            api_token,
            model_id,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )

if __name__ == "__main__":
    main()