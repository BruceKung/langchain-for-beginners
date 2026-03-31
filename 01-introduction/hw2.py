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

    prompt = "Explain the difference between machine learning and deep learning."
    models = ["qwen-plus", "qwen-max"]

    answer_collect = "";

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

        print(f"Model: {model_name}")
        print(f"Response: {response.content}")
        print(f"⏱️  Time: {duration:.0f}ms")
        print(f"answer length: {len(response.content)}")

        answer_collect += model_name + ": " + response.content + "\n"

    print(answer_collect)

    prompt = "Compare the performance of the answer of the two models." + answer_collect
    model = ChatOpenAI(
        model=MODEL,
        base_url=URL,
        api_key=KEY,
    )

    response = model.invoke(prompt)
    print(response.content)

if __name__ == "__main__":
    compare_models()
