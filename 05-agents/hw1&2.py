import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from typing import Optional
from typing import Literal
from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain.agents.middleware.types import ModelResponse
from langgraph.checkpoint.memory import MemorySaver
from typing import Any, Callable
from langchain.agents import create_agent

BASE_MODEL = "qwen-plus"
ADVANCED_MODEL = "qwen3-max"
URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
KEY = "sk-c34720d12dbb45f1aafcc5af5a7237cd"

class CalculatorInput(BaseModel):
    expression: str = Field(description="The mathematical expression to evaluate")

@tool(args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """Perform mathematical calculations."""
    result = eval(expression, {"__builtins__": {}}, {})
    return str(result)

class DistanceInput(BaseModel):
    from_city: str = Field(description="Source city name")
    to_city: str = Field(description="Target city name")
    units: Optional[Literal["miles", "kilometers"]] = Field(default="miles", description="Distance unit")

@tool(args_schema=DistanceInput)
def distance_calculator(from_city: str, to_city: str, units: str = "miles") -> str:
    """Calculate the distance between two cities in miles or kilometers."""
    #simulate distance between cities
    distances = {
        "New York": {"London": 5585, "Paris": 5837, "Tokyo": 10850, "Sydney": 15993},
        "London": {"New York": 5585, "Paris": 344, "Tokyo": 9562, "Sydney": 17015},
        "Paris": {"New York": 5837, "London": 344, "Tokyo": 9714, "Rome": 1430},
        "Tokyo": {"New York": 10850, "London": 9562, "Paris": 9714, "Sydney": 7823},
        "Sydney": {"New York": 15993, "London": 17015, "Tokyo": 7823, "Paris": 16965},
        "Rome": {"Paris": 1430, "London": 1434, "New York": 6896, "Tokyo": 9853},
    }
    from_city = from_city.strip().title()
    to_city = to_city.strip().title()
    if from_city not in distances:
        supported = ", ".join(distances.keys())
        return f"Error: Unknown city '{from_city}'. Supported cities: {supported}"
    if to_city not in distances:
        supported = ", ".join(distances.keys())
        return f"Error: Unknown city '{to_city}'. Supported cities: {supported}"
    distance_km = distances[from_city].get(to_city)
    if not distance_km:
        supported = ", ".join(distances[from_city].keys())
        return f"Error: Distance not available between {from_city} and {to_city}. Available destinations from {from_city}: {supported}"
    units = units or "miles"
    if units == "kilometers":
        distance = distance_km
    else:
        distance = distance_km * 0.621371
    return f"The distance from {from_city} to {to_city} is approximately {distance} {units}"

class ConverterInput(BaseModel):
    value: float = Field(description="The numeric value to convert")
    from_unit: str = Field(description="Source unit, e.g., 'km', 'miles', 'USD'")
    to_unit: str = Field(description="Target unit, e.g., 'km', 'miles', 'EUR'")

@tool(args_schema=ConverterInput)
def converter(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between units: kilometers to miles (and vice versa),
    USD to EUR (and vice versa). Use when you need to convert measurements or currencies."""
    conversions = {
        "km": {"miles": {"rate": 0.621371, "unit": "miles"}},
        "miles": {"km": {"rate": 1.60934, "unit": "kilometers"}},
        "usd": {"eur": {"rate": 0.92, "unit": "EUR"}},
        "eur": {"usd": {"rate": 1.09, "unit": "USD"}},
    }
    print(f"converter called with value: {value}, from_unit: {from_unit}, to_unit: {to_unit}")
    from_lower = from_unit.lower()
    to_lower = to_unit.lower()
    if from_lower not in conversions or to_lower not in conversions.get(from_lower, {}):
        return f"Error: Cannot convert from {from_unit} to {to_unit}. Available conversions: km↔miles, USD↔EUR"
    conversion = conversions[from_lower][to_lower]
    result = value * conversion["rate"]
    return f"{value} {from_unit} equals {result:.2f} {conversion['unit']}"

class ComparisonInput(BaseModel):
    value1: float = Field(description="First value to compare")
    value2: float = Field(description="Second value to compare")

@tool(args_schema=ComparisonInput)
def comparison_tool(value1: float, value2: float) -> str:
    """Compare two values and return the result."""
    if value1 > value2:
        return f"{value1} is greater than {value2}"
    elif value1 < value2:
        return f"{value1} is less than {value2}"
    else:
        return f"{value1} is equal to {value2}"

base_model = init_chat_model(
    model=BASE_MODEL,
    model_provider="openai",
    base_url=URL,
    api_key=KEY,
)

advanced_model = init_chat_model(
    model=ADVANCED_MODEL,
    model_provider="openai",
    base_url=URL,
    api_key=KEY,
)

class DynamicModelMiddleware(AgentMiddleware):
    def __init__(self, msg_length_threshold: int = 10):
        self.msg_length_threshold = msg_length_threshold
        self.base_model = base_model
        self.advanced_model = advanced_model

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        last_msg = request.state["messages"][-1].content or ""
        print(f"last message: {last_msg}")
        if len(last_msg) > self.msg_length_threshold:
            print("use advanced model")
            request = request.override(model=self.advanced_model)
        else:
            print("use base model")
            request = request.override(model=self.base_model)
        return handler(request)

class ToolInfoMiddleware(AgentMiddleware):
    def wrap_tool_call(self, request: Any, handler: Callable[[Any], ToolMessage]) -> ToolMessage:
        tool_name = request.tool_call.get("name", "unknown")
        print(f"tool name: {tool_name}")
        return handler(request)

def main():
    memory = MemorySaver()

    agent = create_agent(
        model=base_model,
        checkpointer=memory,
        tools=[calculator, distance_calculator, converter, comparison_tool],
        middleware=[DynamicModelMiddleware(msg_length_threshold=50), ToolInfoMiddleware()],
    )

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Goodbye!")
            break
        response = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config={"configurable": {"thread_id": "user-grc"}}
        )
        last_message = response["messages"][-1]
        print(f"Agent: {last_message.content}")

if __name__ == "__main__":
    main()
