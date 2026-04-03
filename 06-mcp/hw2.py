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

@tool()
def calculator(expression: str) -> str:
    """Perform mathematical calculations.
    Args:
        expression: The mathematical expression to evaluate, e.g., '125 * 8' or '50 + 25'
    Returns:
        The result of the calculation as a string.
    """
    print(f"calculator called with expression: {expression}")
    result = eval(expression, {"__builtins__": {}}, {})
    return str(result)

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
    config = {"configurable": {"thread_id": "user-123"}}
    model = init_chat_model(
        model=MODEL,
        base_url=URL,
        api_key=KEY,
        model_provider="openai",
    )

    messages = [
        SystemMessage(content="You are a helpful assistant that can answer questions using the tools provided."),
    ]

    memory = MemorySaver()

    agent = create_agent(model, tools + [calculator], checkpointer=memory)

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Goodbye!")
            break
        response = await agent.ainvoke({"messages": [HumanMessage(content=user_input)]}, config)
        last_message = response["messages"][-1]
        print(f"Agent: {last_message.content}")

if __name__ == "__main__":
    asyncio.run(main())