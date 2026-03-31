"""
Lesson 01 - Model Comparison in LangChain
This example shows how to compare different AI models.
"""

import os
import time

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

MODEL = "qwen3-max"
URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
KEY = "sk-c34720d12dbb45f1aafcc5af5a7237cd"

def compare_models():
    print("🔬 Comparing AI Models\n")

    prompt = "Explain recursion in programming in one sentence."
    models = [MODEL, "qwen-plus"]

    for model_name in models:
        print(f"\n📊 Testing: {model_name}")
        print("─" * 50)

        model = ChatOpenAI(
            model=model_name,
            base_url=URL,
            api_key=KEY,
        )

        start_time = time.time()
        response = model.invoke(prompt)
        duration = (time.time() - start_time) * 1000

        print(f"Response: {response.content}")
        print(f"⏱️  Time: {duration:.0f}ms")

    print("\n✅ Comparison complete!")
    print("\n💡 Key Observations:")
    print("   - gpt-4o is more capable and detailed")
    print("   - gpt-4o-mini is faster and uses fewer resources")
    print("   - Choose based on your needs: speed vs. capability")


if __name__ == "__main__":
    compare_models()
