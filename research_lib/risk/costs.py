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
Role = Literal["maker", "taker"]


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

    def fee(self, role: Role = "taker") -> float:
        """Fee rate for one fill. Research backtests fill on next-bar open; the taker fee
        is the conservative default. Post-only (maker) fills pay ``maker_fee``."""
        return self.maker_fee if role == "maker" else self.taker_fee

    def one_way_cost(self, role: Role = "taker") -> float:
        """Total fractional cost of one fill. A maker fill earns the half-spread (posted
        quote) and has no slippage, so its cost is just the maker fee minus nothing:
        the price improvement is modelled in ``execution_price``/``maker_quote``."""
        if role == "maker":
            return self.maker_fee
        return self.fee() + self.half_spread() + self.slippage()

    def round_trip_cost(self, role: Role = "taker") -> float:
        """Fractional cost of a full round trip (entry + exit)."""
        return 2.0 * self.one_way_cost(role)

    def maker_quote(self, ref_price: float, side: Side) -> float:
        """Limit price a post-only order would rest at: join the bid (buy) / ask (sell).

        Buying posts below the reference mid; selling posts above it. If the market later
        trades through this price the order fills at the quote (price improvement vs the
        taker path, which pays the half-spread and slippage).
        """
        offset = self.half_spread()
        if side == "long":
            return ref_price * (1.0 - offset)
        return ref_price * (1.0 + offset)

    def execution_price(self, ref_price: float, side: Side, role: Role = "taker") -> float:
        """Execution price for a fill at ``ref_price`` (next-bar-open style).

        Taker (default): adverse - buying pays up: ref * (1 + half_spread + slippage);
        selling receives less: ref * (1 - half_spread - slippage).
        Maker: the resting ``maker_quote`` - favourable by construction, with fill risk
        handled by the broker (the bar must trade through the quote).
        """
        if role == "maker":
            return self.maker_quote(ref_price, side)
        adverse = 1.0 + self.half_spread() + self.slippage()
        if side == "long":
            return ref_price * adverse
        return ref_price * (2.0 - adverse)
