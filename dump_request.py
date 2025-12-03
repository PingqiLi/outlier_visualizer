import requests
import json
import time
import argparse
from transformers import AutoTokenizer

def dump_activations(text_file, model_path):
    url = "http://localhost:8000/v1/completions"
    headers = {"Content-Type": "application/json"}
    
    # Load tokenizer
    print(f"Loading tokenizer from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # Read text file
    if text_file:
        print(f"Reading text from {text_file}...")
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                raw_text = f.read()
        except Exception as e:
            print(f"Error reading text file: {e}")
            return
    else:
        print("No text file provided. Using dummy text.")
        raw_text = "Hello " * 2000

    # Tokenize and truncate
    print("Tokenizing and truncating to 2000 tokens...")
    tokens = tokenizer.encode(raw_text)
    if len(tokens) < 2000:
        print(f"Warning: Text is too short ({len(tokens)} tokens). Repeating to fill...")
        while len(tokens) < 2000:
            tokens += tokens
    
    # Truncate to exactly 2000 tokens
    tokens = tokens[:2000]
    prompt = tokenizer.decode(tokens)
    print(f"Prepared prompt with {len(tokens)} tokens.")
    
    data = {
        "model": model_path, 
        "prompt": prompt,
        "max_tokens": 1, # Only generate 1 token to trigger prefill + 1 step, then stop
        "temperature": 0,
        "ignore_eos": True
    }
    
    print("Sending dump request...")
    try:
        # Use stream=True to handle potential timeouts or long responses
        with requests.post(url, headers=headers, json=data, stream=True) as response:
            if response.status_code == 200:
                print("Request successful! Dumping started...")
                # Consume the stream to ensure the request completes on the server side
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        pass # Just consume
                print("Dump request completed.")
            else:
                print(f"Request failed with status {response.status_code}")
                print(response.text)
    except Exception as e:
        print(f"Error sending request: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send dump request with real text.")
    parser.add_argument("--text_file", type=str, default=None, help="Path to text file for prompt")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model weights (for tokenizer)")
    
    args = parser.parse_args()
    
    # Wait a bit for the server to be fully ready if running immediately after startup
    time.sleep(5)
    dump_activations(args.text_file, args.model_path)
