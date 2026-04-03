import os
import asyncio
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from typing import Optional
from typing import Literal
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

MODEL = "qwen3-max"
URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
KEY = "sk-c34720d12dbb45f1aafcc5af5a7237cd"

client = MultiServerMCPClient(
        {
            "context7": {
                "transport": "streamable_http",
                "url": "https://mcp.context7.com/mcp",
            },
        }
    )

async def main():
    tools = await client.get_tools()
    print(tools)

    model = init_chat_model(
        model=MODEL,
        base_url=URL,
        api_key=KEY,
        model_provider="openai",
    )

    agent = create_agent(model, tools)

    messages = [
        SystemMessage(content="You are a helpful assistant that can answer questions using the tools provided."),
        HumanMessage(content="how do I use Express.js middleware?"),
    ]

    try:
        response = await agent.ainvoke({"messages": messages})
        last_message = response["messages"][-1]
        print(f"🤖 Agent: {last_message.content}\n")
    except Exception as e:
        print(f"Error: {e}")
    

if __name__ == "__main__":
    asyncio.run(main())