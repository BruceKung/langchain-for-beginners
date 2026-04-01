import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model

MODEL = "qwen3-max"
URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
KEY = "sk-c34720d12dbb45f1aafcc5af5a7237cd"

def main():

    test_temps = [0, 0.5, 1, 1.5, 2]
    for temp in test_temps:
        print(f"Temperature: {temp}")
        print("-" * 80)
        model = init_chat_model(
            model=MODEL,
            temperature=temp,
            model_provider="openai",
            api_key=KEY,
            base_url=URL,
        )

        try:
            print(f"temp: {temp}")
            for i in range(1, 3):
                print(f"Try {i}:")
                response = model.invoke("write a horror story in one sentence")
                print(response.content)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()