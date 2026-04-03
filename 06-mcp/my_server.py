import math
import sys

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-calculator")

@mcp.tool()
def calculate(expression: str) -> str:
    """Perform mathematical calculations."""
    result = eval(expression, {"__builtins__": {}}, {})
    return str(result)

if __name__ == "__main__":
    mcp.run()