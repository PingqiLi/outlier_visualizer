import requests
import json

def send_dump_request():
    url = "http://localhost:8017/v1/completions"
    
    # 构造Prompt，长度适中，确保生成能达到2000 token
    # 用户要求：greedy decoding, ignore eos, 保证每次激活sequence length为2000
    payload = {
        "model": "/workspace/weights/Qwen3-30B", # 请根据实际启动的模型名称调整
        "prompt": "The quick brown fox jumps over the lazy dog.",
        "max_tokens": 2000,
        "min_tokens": 2000,     # 强制生成至少2000 token
        "ignore_eos": True,     # 忽略EOS，确保生成长度
        "temperature": 0.0,     # Greedy decoding
        "top_p": 1.0,
        "stream": True          # 开启流式传输，避免 504 Timeout
    }

    headers = {
        "Content-Type": "application/json"
    }

    print(f"Sending request to {url}...")
    print(f"Payload: {json.dumps(payload, indent=2)}")

    try:
        response = requests.post(url, headers=headers, json=payload, stream=True)
        response.raise_for_status()
        
        print("\nRequest successful! Receiving stream...")
        generated_text = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    data_str = decoded_line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        text = data['choices'][0]['text']
                        generated_text += text
                        print(f"\rGenerated length: {len(generated_text)}", end="", flush=True)
                    except json.JSONDecodeError:
                        continue
        
        print(f"\n\nTotal generated text length: {len(generated_text)}")
        # print(f"Generated text: {generated_text[:100]}...")
        
    except Exception as e:
        print(f"\nRequest failed: {e}")
        if 'response' in locals():
            # print(f"Response content: {response.text}")
            pass

if __name__ == "__main__":
    send_dump_request()
