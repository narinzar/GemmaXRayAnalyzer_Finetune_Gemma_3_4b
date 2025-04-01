#!/usr/bin/env python
# use_model_api.py
# Purpose: Demonstrate how to use the fine-tuned model from the Hugging Face API

import os
import requests
import argparse
from dotenv import load_dotenv

def use_inference_api(prompt, max_tokens=256, temperature=0.7, top_p=0.9):
    """
    Use the Hugging Face Inference API to generate a response from the model.
    
    Args:
        prompt: The text prompt for the model
        max_tokens: Maximum number of tokens to generate
        temperature: Controls randomness (higher = more random)
        top_p: Nucleus sampling parameter
    
    Returns:
        The model's response
    """
    # Load environment variables
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    hf_username = os.getenv("HF_USERNAME")
    model_name = os.getenv("HF_MODEL_NAME", "GemmaXRayAnalyzer_Finetune_Gemma_3_4b")
    
    if not hf_token or not hf_username:
        raise ValueError("HF_TOKEN or HF_USERNAME not set in .env file")
    
    # Format the prompt with Gemma's chat template
    formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    
    # Set up the API URL and headers
    api_url = f"https://api-inference.huggingface.co/models/{hf_username}/{model_name}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    # Prepare the payload
    payload = {
        "inputs": formatted_prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "return_full_text": False
        }
    }
    
    # Make the API request
    print("Sending request to Hugging Face Inference API...")
    response = requests.post(api_url, headers=headers, json=payload)
    
    # Check if the request was successful
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            generated_text = result[0].get("generated_text", "")
            return generated_text
        return str(result)
    else:
        error_msg = f"API request failed with status code {response.status_code}: {response.text}"
        print(error_msg)
        return error_msg

def load_direct_model(prompt, max_tokens=256, temperature=0.7, top_p=0.9):
    """
    Load the model directly using the Transformers library.
    
    Args:
        prompt: The text prompt for the model
        max_tokens: Maximum number of tokens to generate
        temperature: Controls randomness (higher = more random)
        top_p: Nucleus sampling parameter
    
    Returns:
        The model's response
    """
    try:
        # Import required libraries
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        # Load environment variables
        load_dotenv()
        hf_username = os.getenv("HF_USERNAME")
        model_name = os.getenv("HF_MODEL_NAME", "GemmaXRayAnalyzer_Finetune_Gemma_3_4b")
        
        if not hf_username:
            raise ValueError("HF_USERNAME not set in .env file")
        
        # Model ID
        model_id = f"{hf_username}/{model_name}"
        
        # Format the prompt with Gemma's chat template
        formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        print(f"Loading model from {model_id}...")
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Tokenize input
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        # Generate
        print("Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )
        
        # Decode and return
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the model's response
        if "<start_of_turn>model\n" in response:
            response = response.split("<start_of_turn>model\n")[-1].strip()
        
        return response
        
    except ImportError:
        return "Error: Required libraries not installed. Please run 'pip install transformers torch'."
    except Exception as e:
        return f"Error loading or using model: {str(e)}"

def main():
    """Main function to parse arguments and call the appropriate method"""
    parser = argparse.ArgumentParser(description="Use the fine-tuned X-ray model via API or direct loading")
    parser.add_argument("--prompt", type=str, required=True, help="The prompt for the model")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--method", choices=["api", "direct"], default="api", 
                        help="Method to use: 'api' for Inference API, 'direct' for loading the model")
    
    args = parser.parse_args()
    
    if args.method == "api":
        response = use_inference_api(args.prompt, args.max_tokens, args.temperature)
    else:
        response = load_direct_model(args.prompt, args.max_tokens, args.temperature)
    
    print("\n" + "=" * 50)
    print("Model Response:")
    print("=" * 50)
    print(response)

if __name__ == "__main__":
    main()