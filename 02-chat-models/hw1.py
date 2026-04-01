import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model

MODEL = "qwen3-max"
URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
KEY = "sk-c34720d12dbb45f1aafcc5af5a7237cd"

def main():
    print("🎭 Understanding Message Types\n")

    model = init_chat_model(
        model=MODEL,
        model_provider="openai",
        api_key=KEY,
        base_url=URL,
    )

    messages = [
        SystemMessage(content="You are a helpful AI assistant who explains things simply."),
    ]
    print(f"Hello I'm a AI chatbot. How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Goodbye!")
            break
        
        messages.append(HumanMessage(content=user_input))
        result = ""
        usage = None
        for chunk in model.stream(messages):
            result += chunk.content
            print(chunk.content, end="", flush=True)
            if hasattr(chunk, 'usage_metadata'):
                usage = chunk.usage_metadata
        print()
        messages.append(AIMessage(content=result))
        print(f"Conversation length: {len(messages)}")

        if usage:
            print("Token Breakdown:")
            print(f"  Prompt tokens:     {usage.get('input_tokens', 'N/A')}")
            print(f"  Completion tokens: {usage.get('output_tokens', 'N/A')}")
            print(f"  Total tokens:      {usage.get('total_tokens', 'N/A')}")
        #else:
        #    print("⚠️  Token usage information not available in response metadata.")


if __name__ == "__main__":
    main()