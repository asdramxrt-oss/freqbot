"""Production entry layer for the research stack (the "bot").

This package wires every node of the architecture diagram into one runnable,
configuration-driven trading bot:

    RAW MARKET DATA
          |
          v
    Data Normalizer            -> research_lib.data.loader
          |
          v
    Information Bars           -> research_lib.data.infobars
          |
          v
    Feature Engine             -> research_lib.features.engine / volatility / fracdiff / hurst
          |
          +--> Turtle Engine   -> research_lib.strategies.turtle
          +--> OU Engine       -> research_lib.strategies.ou
          |
          v
    Regime Engine (HMM)        -> research_lib.strategies.unified_router.filtered_hmm_router
          |
          v
    Signal Router              -> research_lib.strategies.unified_router (+ meta-model gate)
          |
          v
    Triple Barrier             -> research_lib.ml.triple_barrier
          |
          v
    Meta Model (RF/XGBoost)    -> research_lib.ml.metalabel
          |
          v
    Bet Sizer                  -> research_lib.ml.metalabel.BetSizer
          |
          v
    Risk Engine                -> research_lib.risk.engine.CentralRiskEngine
          |
          v
    Execution Engine           -> research_lib.execution.engine.ExecutionEngine
          |
          +--> Redis Broker     -> research_lib.execution.redis_lock
          +--> Lock Adapter     -> research_lib.execution.redis_lock.DistributedLock

Everything above already existed as validated research modules. This package adds
the orchestration layer that was missing: a config, a broker adapter, a bar-by-bar
live/paper loop, and a CLI.
"""

from research_lib.bot.config import BotConfig, load_config
from research_lib.bot.broker import PaperBroker, LiveBroker
from research_lib.bot.live import BotResult, TradingBot, run_bot

__all__ = [
    "BotConfig",
    "load_config",
    "PaperBroker",
    "LiveBroker",
    "TradingBot",
    "BotResult",
    "run_bot",
]
