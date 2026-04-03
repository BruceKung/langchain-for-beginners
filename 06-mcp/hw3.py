import os
import asyncio
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from typing import AsyncIterable, Optional
from typing import Literal
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

MODEL = "qwen3-max"
URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
KEY = "sk-c34720d12dbb45f1aafcc5af5a7237cd"

client = MultiServerMCPClient(
    {
        "context7": {
            "transport": "streamable_http",
            "url": "https://mcp.context7.com/mcp",
        },
        "my-calculator": {
            "transport": "stdio",
            "command": "python",
            "args": ["/Users/ruichenggong/Projects/langchain-for-beginners/06-mcp/my_server.py"],
        },
    }
)

async def main():
    tools = await client.get_tools()
    model = init_chat_model(
        model=MODEL,
        base_url=URL,
        api_key=KEY,
        model_provider="openai",
    )
    agent = create_agent(model, tools)
    messages = [
        SystemMessage(content="You are a helpful assistant that can answer questions using the tools provided."),
    ]

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Goodbye!")
            break
        response = await agent.ainvoke({"messages": [HumanMessage(content=user_input)]})
        last_message = response["messages"][-1]
        print(f"Agent: {last_message.content}")

if __name__ == "__main__":
    asyncio.run(main())