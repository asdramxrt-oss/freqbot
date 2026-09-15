"""Research package for the evidence-backed trading system."""

from research_lib.strategies.profitable_turtle import (
    ProfitableTurtleParams,
    ProfitableTurtleEngine,
    TurtleEngine as ProfitableTurtleEngine,
)

__version__ = "0.1.0"

__all__ = [
    "ProfitableTurtleParams",
    "ProfitableTurtleEngine",
    "ProfitableTurtleEngine as TurtleEngine",
]