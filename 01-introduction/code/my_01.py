"""
Lesson 01 - Hello World with LangChain
This example demonstrates a basic LLM call using ChatOpenAI.
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

MODEL = "qwen3-max"
URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
KEY = "sk-c34720d12dbb45f1aafcc5af5a7237cd"

def main():
    print("🦜🔗 Hello LangChain!\n")

    # Create a chat model instance
    model = ChatOpenAI(
        model=MODEL,
        base_url=URL,
        api_key=KEY
    )

    # Make your first AI call!
    response = model.invoke("What is LangChain in one sentence?")

    print("🤖 AI Response:", response.content)
    print("\n✅ Success! You just made your first LangChain call!")

if __name__ == "__main__":
    main()
