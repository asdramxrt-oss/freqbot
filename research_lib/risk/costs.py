"""Transaction-cost model (Phase 20 foundation, used by every backtest).

A strategy that only works under ideal costs is rejected. All research backtests run through
this cost model; the stress harness multiplies the cost components (1.5x / 2x / 3x).

Components:
  * maker/taker fee  (fraction of notional)
  * spread           (half-spread applied adversarially on entry AND exit)
  * slippage         (additional adverse price move per fill)
  * funding          (per-period funding rate, spot -> 0; futures -> applied on open value)
  * latency          (signal-to-order delay in bars; 0 => fill at NEXT bar open)

The model returns an execution price for a given bar and side. All backtests in this project
default to market-on-next-bar-open fills at the adverse side of the spread with slippage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from pandas import DataFrame

Side = Literal["long", "short"]


@dataclass
class CostModel:
    maker_fee: float = 0.0010
    taker_fee: float = 0.0010
    spread_bps: float = 1.0       # full spread in bps (half applied per fill)
    slippage_bps: float = 0.5     # adverse slippage per fill
    funding_rate: float = 0.0     # per-bar funding (spot: 0)
    latency_bars: int = 0         # 0 => act on the next bar open after the signal

    def half_spread(self) -> float:
        return self.spread_bps / 1e4 / 2.0

    def slippage(self) -> float:
        return self.slippage_bps / 1e4

    def fee(self) -> float:
        """Research backtests fill on next-bar open; taker fee is the conservative choice."""
        return self.taker_fee

    def one_way_cost(self) -> float:
        """Total fractional cost of one fill (fee + half spread + slippage)."""
        return self.fee() + self.half_spread() + self.slippage()

    def round_trip_cost(self) -> float:
        """Fractional cost of a full round trip (entry + exit)."""
        return 2.0 * self.one_way_cost()

    def execution_price(self, ref_price: float, side: Side) -> float:
        """Adverse execution price for a fill at ``ref_price`` (next-bar-open style).

        Buying (long entry / short exit) pays up: ref * (1 + half_spread + slippage).
        Selling (short entry / long exit) receives less: ref * (1 - half_spread - slippage).
        """
        adverse = 1.0 + self.half_spread() + self.slippage()
        if side == "long":
            return ref_price * adverse
        return ref_price * (2.0 - adverse)
