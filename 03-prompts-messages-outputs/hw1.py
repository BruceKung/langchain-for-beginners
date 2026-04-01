import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
import json

MODEL = "qwen3-max"
URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
KEY = "sk-c34720d12dbb45f1aafcc5af5a7237cd"

def main():

    model = init_chat_model(
        model=MODEL,
        model_provider="openai",
        api_key=KEY,
        base_url=URL,
    )

    examples = [
        {"input": "Premium wireless headphones with noise cancellation, $199", "output": json.dumps({"name": "Premium wireless headphones with noise cancellation", "price": "$199", "category": "Electronics", "highlight": "Noise cancellation"})},
        {"input": "Organic cotton t-shirt in blue, comfortable fit, $29.99", "output": json.dumps({"name": "Organic cotton t-shirt in blue, comfortable fit", "price": "$29.99", "category": "Clothing", "highlight": "Comfortable fit"})},
        {"input": "Gaming laptop with RTX 4070, 32GB RAM, $1,499", "output": json.dumps({"name": "Gaming laptop with RTX 4070, 32GB RAM", "price": "$1,499", "category": "Electronics", "highlight": "RTX 4070"})},
    ]

    example_template = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])

    few_shot_template = FewShotChatMessagePromptTemplate(
        example_prompt=example_template,
        examples=examples,
    )

    message_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that converts product descriptions into a specific JSON format."),
        few_shot_template,
        ("human", "{input}"),
    ])

    chain = message_template | model
    
    response  = chain.invoke({"input": "Premium wireless headphones with noise cancellation, $199"})
    print(response.content)
    print(json.loads(response.content))

if __name__ == "__main__":
    main()