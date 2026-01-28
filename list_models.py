"""
List Available Gemini Models
This will show you exactly which models your API key can access
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

print("Available Gemini Models:")
print("=" * 60)

try:
    models = genai.list_models()
    
    for model in models:
        # Check if model supports generateContent
        if 'generateContent' in model.supported_generation_methods:
            print(f"\n✅ {model.name}")
            print(f"   Display Name: {model.display_name}")
            print(f"   Description: {model.description[:100]}...")
    
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)
print("\nUse one of the model names above (the full 'models/...' path)")
