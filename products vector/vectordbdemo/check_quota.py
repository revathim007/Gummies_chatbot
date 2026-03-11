import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

try:
    print("Testing Embeddings (text-embedding-3-small)...")
    res_emb = client.embeddings.create(input="Gummy wellness", model="text-embedding-3-small")
    print("Embeddings OK.")
    
    print("Testing Chat (gpt-4o-mini)...")
    res_chat = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hi!"}],
        max_tokens=5
    )
    print(f"Chat OK: {res_chat.choices[0].message.content}")
    print("--- QUOTA SUCCESS: API is FULLY WORKING ---")
except Exception as e:
    print(f"--- QUOTA FAILURE: {e} ---")
