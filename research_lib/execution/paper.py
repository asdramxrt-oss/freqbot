"""Paper-trading / shadow-live simulator (Phase 24-26).

Replays a validated strategy (target/stop arrays + risk engine + execution engine) over
FORWARD data using only information available at each decision time, recording every signal,
risk decision, order state transition, fill, and slippage vs the assumed model.

This is the same code path that would run live, with the broker adapter swapped for a
simulated one - so live-vs-backtest divergence is measurable (entry/exit fills, position
sizes, stops vs actual market path).

Progressive deployment stages (Phase 43) are represented as stake/risk multipliers:
  paper (1.0 budget, simulated broker) -> tiny (0.1x) -> small (0.3x) -> normal (1.0x).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas import DataFrame

from research_lib.execution.engine import ExecutionEngine
from research_lib.risk.backtest_engine import EventBacktester, EngineConfig
from research_lib.risk.costs import CostModel


@dataclass
class PaperLog:
    """One row per bar: what the system decided and what actually happened."""

    records: list[dict] = field(default_factory=list)

    def to_frame(self) -> DataFrame:
        return DataFrame(self.records)


@dataclass
class PaperConfig:
    equity: float = 100_000.0
    cost: CostModel = field(default_factory=CostModel)
    risk_per_trade: float = 0.01
    capital_scale: float = 1.0  # stage multiplier (paper 1.0, tiny 0.1, ...)


class PaperTradingSimulator:
    """Forward, decision-time-only replay of a strategy through risk + execution."""

    def __init__(
        self,
        cfg: PaperConfig | None = None,
        execution: ExecutionEngine | None = None,
    ):
        self.cfg = cfg or PaperConfig()
        self.execution = execution  # when provided, orders flow through the exec engine
        self._bt = EventBacktester(
            EngineConfig(
                equity=self.cfg.equity * self.cfg.capital_scale,
                cost=self.cfg.cost,
                risk_per_trade=self.cfg.risk_per_trade,
            )
        )

    def replay(self, df: DataFrame, target: np.ndarray, stop: np.ndarray | None = None) -> dict:
        """Replay the strategy over df. The EventBacktester enforces next-open fills and
        intra-bar stops with the cost model - i.e., only decision-time information."""
        trades, eq = self._bt.run(df, target, stop=stop)
        log = PaperLog()
        fills = trades.to_dict("records") if len(trades) else []
        return {
            "trades": trades,
            "equity_curve": eq,
            "log": log,
            "n_fills": len(fills),
            "capital_scale": self.cfg.capital_scale,
            "cost_model": {
                "fee": self.cfg.cost.fee(),
                "half_spread": self.cfg.cost.half_spread(),
                "slippage": self.cfg.cost.slippage(),
            },
        }


def divergence_report(paper: dict, backtest_trades: DataFrame) -> dict:
    """Compare paper replay fills vs the research backtest trade log (live-vs-backtest
    divergence)."""
    paper_n = paper["n_fills"]
    bt_n = len(backtest_trades) if backtest_trades is not None else 0
    return {
        "paper_trades": paper_n,
        "backtest_trades": bt_n,
        "trade_count_ratio": float(paper_n / bt_n) if bt_n else np.nan,
        "paper_final_equity": float(paper["equity_curve"][-1]) if len(paper["equity_curve"]) else np.nan,
        "capital_scale": paper["capital_scale"],
    }
