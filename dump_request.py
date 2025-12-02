import requests
import json

def send_dump_request():
        print(f"\n\nTotal generated text length: {len(generated_text)}")
        # print(f"Generated text: {generated_text[:100]}...")
        
    except Exception as e:
        print(f"\nRequest failed: {e}")
        if 'response' in locals():
            # print(f"Response content: {response.text}")
            pass

if __name__ == "__main__":
    send_dump_request()
