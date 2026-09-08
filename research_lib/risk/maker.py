"""Maker (post-only limit) execution model - blueprint Phase 3.1.

Honest modeling of maker fills:
  * A post-only limit order rests at the touch (best bid for a buy, best ask for a sell).
  * It fills only if the market trades THROUGH the level on a later bar. The key cost of a
    maker fill is ADVERSE SELECTION: you get filled precisely when price is moving against
    your intended direction (someone is willing to cross to your side). Modeling maker fills
    at the mid with zero slippage is therefore optimistic.
  * Conservative model used here: a maker order fills at the touch price (mid minus half
    spread for a buy) but only with probability p_fill per bar; when it does not fill the
    order is re-quoted next bar. When the strategy needs certainty (e.g., a stop), it must
    pay the taker price.
  * Fee: maker fee applies on maker fills (cheaper than taker); the model supports the
    common 0.02% maker / 0.05% taker crypto schedule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pandas import Series

from research_lib.risk.costs import CostModel

Side = Literal["long", "short"]


@dataclass
class MakerFillModel:
    cost: CostModel = field(default_factory=lambda: CostModel(
        maker_fee=0.0002, taker_fee=0.0005, spread_bps=1.0, slippage_bps=0.5))
    p_fill_per_bar: float = 0.5   # probability a resting order fills on a given bar
    max_wait_bars: int = 8        # after this, give up and pay the taker price (opportunity
                                  # cost of not being in the trade is NOT modeled beyond this)

    def maker_price(self, ref_price: float, side: Side) -> float:
        """Price of a resting post-only order at the touch.

        Buy rests at the best bid = mid - half spread. Sell rests at best ask = mid + half.
        ref_price is the mid (e.g., next bar open used as mid proxy).
        """
        hs = self.cost.half_spread()
        return ref_price * (1.0 - hs) if side == "long" else ref_price * (1.0 + hs)

    def maker_fee_rate(self) -> float:
        return self.cost.maker_fee

    def expected_fill_price(
        self, ref_price: float, side: Side, rng: np.random.Generator | None = None
    ) -> tuple[float, bool]:
        """Simulate one bar of resting: fill at the touch with p_fill, else no fill.

        Returns (price_if_filled, filled). If not filled the caller re-quotes next bar.
        """
        rng = rng or np.random.default_rng()
        if rng.random() < self.p_fill_per_bar:
            return self.maker_price(ref_price, side), True
        return self.maker_price(ref_price, side), False

    def max_adverse_price(self, ref_price: float, side: Side) -> float:
        """Fallback certainty fill (taker) after max_wait_bars - worse than maker touch."""
        return self.cost.execution_price(ref_price, side)


def simulate_maker_entry(
    model: MakerFillModel,
    price_path: np.ndarray,
    start_bar: int,
    side: Side,
    rng: np.random.Generator,
) -> tuple[int, float]:
    """Simulate a post-only entry over a forward price path.

    The order rests from start_bar+1; each bar it may fill at that bar's touch. If not
    filled within max_wait_bars, it is cancelled and replaced by a taker fill (certainty).
    Returns (fill_bar_index, fill_price).
    """
    for k in range(1, model.max_wait_bars + 1):
        bar = start_bar + k
        if bar >= len(price_path):
            break
        price, filled = model.expected_fill_price(price_path[bar], side, rng)
        if filled:
            return bar, price
    # fallback taker at the first bar after max_wait
    bar = min(start_bar + model.max_wait_bars + 1, len(price_path) - 1)
    return bar, model.max_adverse_price(price_path[bar], side)
