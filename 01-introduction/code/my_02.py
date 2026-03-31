"""
Lesson 01 - Message Types in LangChain
This example demonstrates how to use different message types (SystemMessage, HumanMessage).
"""

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

MODEL = "qwen3-max"
URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
KEY = "sk-c34720d12dbb45f1aafcc5af5a7237cd"

def main():
    print("🎭 Understanding Message Types\n")

    model = ChatOpenAI(
        model=MODEL,
        base_url=URL,
        api_key=KEY
    )

    # Using structured messages for better control
    messages = [
        SystemMessage(content="You are a helpful AI assistant who explains things simply."),
        HumanMessage(content="Explain quantum computing to a 10-year-old."),
    ]

    response = model.invoke(messages)

    print("🤖 AI Response:\n")
    print(response.content)
    print("\n✅ Notice how the SystemMessage influenced the response style!")

if __name__ == "__main__":
    main()
