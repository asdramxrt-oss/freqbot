"""Strategy implementations for Freqtrade research."""

from research_lib.strategies.carry import CarryParams, CarryEngine
from research_lib.strategies.directional import DirectionalResult, run_panel_directional, build_variant_grid
from research_lib.strategies.ou import OUParams, OUEngine
from research_lib.strategies.pairs import PairsParams, PairsEngine
from research_lib.strategies.pdf_pipeline import PDFParams, PDFEngine
from research_lib.strategies.regime_hmm import RegimeHMMParams, RegimeHMHEngine
from research_lib.strategies.router import RouterParams, RouterEngine
from research_lib.strategies.turtle import TurtleParams, TurtleEngine
from research_lib.strategies.turtle_sim import TurtleSimParams, TurtleSimEngine
from research_lib.strategies.unified_engine import UnifiedEngineParams, UnifiedEngine
from research_lib.strategies.unified_router import UnifiedRouterParams, UnifiedRouterEngine
from research_lib.strategies.volspike_alpha import VolSpikeAlphaParams, generate_alpha_signals as generate_volspike_signals
from research_lib.strategies.profitable_turtle import ProfitableTurtleParams, ProfitableTurtleEngine, TurtleEngine as ProfitableTurtleEngine

__all__ = [
    "CarryParams",
    "CarryEngine",
    "DirectionalResult",
    "run_panel_directional",
    "build_variant_grid",
    "OUParams",
    "OUEngine",
    "PairsParams",
    "PairsEngine",
    "PDFParams",
    "PDFEngine",
    "RegimeHMMParams",
    "RegimeHMHEngine",
    "RouterParams",
    "RouterEngine",
    "TurtleParams",
    "TurtleEngine",
    "TurtleSimParams",
    "TurtleSimEngine",
    "UnifiedEngineParams",
    "UnifiedEngine",
    "UnifiedRouterParams",
    "UnifiedRouterEngine",
    "VolSpikeAlphaParams",
    "generate_volspike_signals",
    "ProfitableTurtleParams",
    "ProfitableTurtleEngine",
    "TurtleEngine as ProfitableTurtleEngine",
]