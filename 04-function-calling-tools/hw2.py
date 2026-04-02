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

class CurrencyInput(BaseModel):
    amount: float = Field(description="The amount to convert")
    from_currency: str = Field(description="Source currency code (e.g., 'USD', 'EUR', 'GBP', 'JPY')")
    to_currency: str = Field(description="Target currency code (e.g., 'USD', 'EUR', 'GBP', 'JPY')")

@tool(args_schema=CurrencyInput)
def currency_converter(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert amounts between different currencies (USD, EUR, GBP, JPY, AUD, CAD).
    Use this when the user wants to convert money from one currency to another
    or asks about exchange rates."""
    # Simulated exchange rates (relative to USD)
    rates = {
        "USD": 1.0,
        "EUR": 0.92,
        "GBP": 0.79,
        "JPY": 149.5,
        "AUD": 1.53,
        "CAD": 1.36,
    }
    from_rate = rates.get(from_currency.upper())
    to_rate = rates.get(to_currency.upper())
    if not from_rate:
        supported = ", ".join(rates.keys())
        return f"Error: Unknown currency '{from_currency}'. Supported currencies: {supported}"
    if not to_rate:
        supported = ", ".join(rates.keys())
        return f"Error: Unknown currency '{to_currency}'. Supported currencies: {supported}"
    amount_in_usd = amount / from_rate
    result = amount_in_usd * to_rate
    return f"{amount} {from_currency.upper()} equals approximately {result:.2f} {to_currency.upper()}"

class DistanceInput(BaseModel):
    from_city: str = Field(description="Source city name")
    to_city: str = Field(description="Target city name")
    units: Optional[Literal["miles", "kilometers"]] = Field(default="miles", description="Distance unit")

@tool(args_schema=DistanceInput)
def distance_calculator(from_city: str, to_city: str, units: str = "miles") -> str:
    """Calculate the distance between two cities in miles or kilometers.
    Use this when the user asks about distance between locations,
    how far apart cities are, or travel distances."""
    # Simulated distances between major cities (in kilometers)
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

class TimeZoneInput(BaseModel):
    city: str = Field(description="City name")
    units: Optional[Literal["hours", "minutes"]] = Field(default="hours", description="Time unit")

@tool(args_schema=TimeZoneInput)
def get_time_zone(city: str, units: str = "hours") -> str:
    """Get the time zone for a city.
    Use this when the user asks about the time zone of a specific location."""
    # Simulated time zones for various cities
    time_zones = {
        "New York": "UTC-5",
        "London": "UTC+1",
        "Paris": "UTC+2",
        "Tokyo": "UTC+9",
        "Sydney": "UTC+10",
        "Rome": "UTC+2",
    }
    city = city.strip().title()
    if city not in time_zones:
        supported = ", ".join(time_zones.keys())
        return f"Error: Unknown city '{city}'. Supported cities: {supported}"
    time_zone = time_zones[city]
    if units == "minutes":
        time_zone = time_zone + ":00"
    return f"The time zone for {city} is {time_zone}"

tool_map = {
    "currency_converter": currency_converter,
    "distance_calculator": distance_calculator,
    "get_time_zone": get_time_zone,
}
    

def main():
    model = init_chat_model(
        model=MODEL,
        model_provider="openai",
        api_key=KEY,
        base_url=URL,
    )
    
    model_with_tools = model.bind_tools([currency_converter, distance_calculator, get_time_zone])
    
    messages = [
        SystemMessage(content="You are a helpful assistant that can answer questions about currency, distance, and time zone."),
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
                    tool_result = tool_map[tool_call["name"]].invoke(tool_call["args"])
                    messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"], name=tool_call["name"]))
            else:
                break
        print(messages[-1].content)

if __name__ == "__main__":
    main()