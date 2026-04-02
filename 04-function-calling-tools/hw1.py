import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from typing import Optional
from typing import Literal

MODEL = "qwen3-max"
URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
KEY = "sk-c34720d12dbb45f1aafcc5af5a7237cd"

class WeatherInput(BaseModel):
    city: str = Field(description="City name")
    units: Optional[Literal["celsius", "fahrenheit"]] = Field(default="fahrenheit", description="Temperature unit")

@tool(args_schema=WeatherInput)
def get_weather(city: str, units: Literal["celsius", "fahrenheit"] = "celsius") -> str:
    """Get current weather information for a city. Returns temperature and weather conditions.
    Use this when the user asks about weather, temperature, or conditions in a specific location."""
    # Simulated weather data for various cities
    weather_data = {
        "Tokyo": {"temp_f": 75, "temp_c": 24, "condition": "partly cloudy"},
        "Paris": {"temp_f": 64, "temp_c": 18, "condition": "sunny"},
        "London": {"temp_f": 59, "temp_c": 15, "condition": "rainy"},
        "New York": {"temp_f": 72, "temp_c": 22, "condition": "clear"},
        "Seattle": {"temp_f": 62, "temp_c": 17, "condition": "cloudy"},
        "Sydney": {"temp_f": 79, "temp_c": 26, "condition": "sunny"},
        "Mumbai": {"temp_f": 88, "temp_c": 31, "condition": "humid and hot"},
    }
    city = city.strip().title()
    temp = weather_data.get(city).get("temp_c" if units == "celsius" else "temp_f")
    condition = weather_data.get(city).get("condition")
    return f"The temperature in {city} is {temp} {units} and the weather is {condition}"

def main():
    model = init_chat_model(
        model=MODEL,
        model_provider="openai",
        api_key=KEY,
        base_url=URL,
    )
    
    model_with_tools = model.bind_tools([get_weather])
    
    messages = [
        SystemMessage(content="You are a helpful assistant that can answer questions about the weather."),
    ]
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Goodbye!")
            break
        messages.append(HumanMessage(content=user_input))
        while True:
            response = model_with_tools.invoke(messages)
            messages.append(AIMessage(content=response.content, tool_calls=response.tool_calls))
            if response.tool_calls and len(response.tool_calls) > 0:
                for tool_call in response.tool_calls:
                    print(f"Tool call: {tool_call}")
                    tool_result = get_weather.invoke(tool_call["args"])
                    messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"], name=tool_call["name"]))
            else:
                break
        print(messages[-1].content)

if __name__ == "__main__":
    main()