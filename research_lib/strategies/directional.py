"""Directional strategy evaluation harness (multi-coin, futures).

Why this exists: the project's directional books all failed (ALPHA_DISCOVERY_REPORT.md)
- but they failed as SHORT-HORIZON, LONG-ONLY, SINGLE-COIN, TAKER-cost strategies.
The three failure causes are separately testable:
  1. horizon  - 15m momentum is noise; long-horizon trend is a distinct hypothesis
  2. side     - spot was long-only; perp futures allow shorts (bear regimes tradeable)
  3. universe - one coin's trend is regime beta; cross-sectional relative strength is
                market-neutral by construction

This harness evaluates slow trend + cross-sectional variants with the project's
standard execution model (EventBacktester: signal at close t, fill at open t+1,
adverse taker cost), per-coin sub-equity sizing, and a portfolio equity curve.
Folds/gates live in validation/directional_acceptance.py. A variant only counts as
"successful" if it passes EVERY gate - and the full grid is always reported, winner
or not (multiple-testing honesty).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas import DataFrame

from research_lib.risk.backtest_engine import EngineConfig, EventBacktester
from research_lib.risk.costs import CostModel

TRADING_DAYS_PER_YEAR = 365


# ---------------------------------------------------------------------------
# signals (all causal: bar t uses data strictly before/at close t, filled t+1)
# ---------------------------------------------------------------------------
def donchian_target(
    close: pd.Series,
    lookback: int,
    exit_lookback: int,
    allow_short: bool,
) -> np.ndarray:
    """Turtle-style channel breakout position target in {-1, 0, +1}.

    +1 while in a long trade (entered on a close above the prior ``lookback`` high,
    exited on a close below the prior ``exit_lookback`` low); -1 mirror when
    ``allow_short``. Channels exclude the current bar.
    """
    c = close.astype(float)
    hi = c.rolling(lookback).max().shift(1)
    lo = c.rolling(lookback).min().shift(1)
    x_hi = c.rolling(exit_lookback).max().shift(1)
    x_lo = c.rolling(exit_lookback).min().shift(1)

    pos = np.zeros(len(c))
    state = 0
    cv, hiv, lov, xhiv, xlov = (
        c.to_numpy(),
        hi.to_numpy(),
        lo.to_numpy(),
        x_hi.to_numpy(),
        x_lo.to_numpy(),
    )
    for t in range(len(c)):
        if np.isnan(hiv[t]):
            pos[t] = 0
            continue
        if state == 0:
            if cv[t] > hiv[t]:
                state = 1
            elif allow_short and cv[t] < lov[t]:
                state = -1
        elif state == 1:
            if cv[t] < xlov[t]:
                state = -1 if (allow_short and cv[t] < lov[t]) else 0
            elif allow_short and cv[t] < lov[t]:
                state = -1
        else:  # state == -1
            if cv[t] > xhiv[t]:
                state = 1 if cv[t] > hiv[t] else 0
            elif cv[t] > hiv[t]:
                state = 1
        pos[t] = state
    return pos


def apply_regime_filter(
    target: np.ndarray,
    close: pd.Series,
    sma_window: int,
    mode: str,
) -> np.ndarray:
    """Long-run trend regime filter (the canonical SMA-200 rule, no fitted parameters).

    mode 'long_only_above': longs allowed only above the SMA, flat below.
    mode 'long_short':      longs above, shorts below (full time-series momentum).
    Anything the raw target wanted against the regime is suppressed to 0.
    """
    sma = close.astype(float).rolling(sma_window).mean()
    above = (close.astype(float) > sma).to_numpy()
    out = np.zeros(len(target))
    for t in range(len(target)):
        if np.isnan(sma.iloc[t]):
            continue
        if mode == "long_only_above":
            out[t] = target[t] if (target[t] > 0 and above[t]) else 0
        elif mode == "long_short":
            if target[t] > 0 and above[t]:
                out[t] = 1
            elif target[t] < 0 and not above[t]:
                out[t] = -1
            else:
                out[t] = 0
        else:
            raise ValueError(f"unknown regime filter mode: {mode}")
    return out


def cross_sectional_target(
    closes: DataFrame,
    lookback: int,
    rebalance_every: int,
    n_long: int,
    n_short: int,
) -> dict[str, np.ndarray]:
    """Relative-strength rotation over a panel of closes (columns = symbols).

    Every ``rebalance_every`` bars, rank coins by their past ``lookback`` return
    (data <= t): long the top ``n_long``, short the bottom ``n_short`` (0 = flat).
    Holdings persist until the next rebalance.
    """
    rets = closes / closes.shift(lookback) - 1.0
    targets = {sym: np.zeros(len(closes)) for sym in closes.columns}
    for t in range(lookback, len(closes)):
        if (t - lookback) % rebalance_every != 0:
            if t > 0:
                for sym in targets:
                    targets[sym][t] = targets[sym][t - 1]
            continue
        # rebalance bar: reset everything first, then assign the new basket
        for sym in targets:
            targets[sym][t] = 0.0
        r = rets.iloc[t].dropna()
        if len(r) < n_long + n_short:
            continue
        ranked = r.sort_values(ascending=False)
        for sym in ranked.index[:n_long]:
            targets[sym][t] = 1.0
        if n_short:
            for sym in ranked.index[-n_short:]:
                targets[sym][t] = -1.0
    return targets


# ---------------------------------------------------------------------------
# portfolio backtest
# ---------------------------------------------------------------------------
@dataclass
class DirectionalResult:
    name: str
    trades: DataFrame          # all coins tagged with 'symbol'
    equity: np.ndarray         # portfolio equity per bar
    dates: np.ndarray
    per_coin_net: dict[str, float]
    net_pnl: float
    annualised_return: float
    max_drawdown: float
    n_trades: int
    cost_role: str = "taker"

    def summary(self) -> dict:
        return {
            "name": self.name,
            "net_pnl": self.net_pnl,
            "annualised_return": self.annualised_return,
            "max_drawdown": self.max_drawdown,
            "n_trades": self.n_trades,
            "per_coin_net": self.per_coin_net,
            "cost_role": self.cost_role,
        }


def run_panel_directional(
    name: str,
    panels: dict[str, DataFrame],
    targets: dict[str, np.ndarray],
    cost: CostModel,
    equity: float = 100_000.0,
    risk_per_trade: float = 0.01,
) -> DirectionalResult:
    """Run each coin through the EventBacktester with its own equity/N sub-account.

    ``panels``/``targets`` must be bar-aligned across coins (same index grid).
    Signals are shifted by one bar here so a decision at close t fills at open t+1,
    matching the engine's convention (target[t] is filled at open t).
    """
    syms = sorted(panels)
    n = len(syms)
    sub_equity = equity / n
    all_trades: list[DataFrame] = []
    min_len = min(len(panels[s]) for s in syms)
    eq_sum = None
    dates = None
    per_coin_net: dict[str, float] = {}

    for sym in syms:
        df = panels[sym].reset_index(drop=True).iloc[:min_len]
        tgt = np.asarray(targets[sym], dtype=float)
        shifted = np.roll(tgt, 1)
        shifted[0] = 0.0
        bt = EventBacktester(
            EngineConfig(equity=sub_equity, cost=cost, risk_per_trade=risk_per_trade, allow_short=True)
        )
        trades, eq = bt.run(df, shifted)
        trades["symbol"] = sym
        all_trades.append(trades)
        per_coin_net[sym] = float(eq[-1] - sub_equity) if len(eq) else 0.0
        eq_sum = eq if eq_sum is None else eq_sum + eq
        dates = df["date"].to_numpy()

    trades = (
        pd.concat(all_trades, ignore_index=True) if all_trades else DataFrame()
    )
    if len(trades):
        trades = trades.sort_values("exit_time").reset_index(drop=True)

    eq = eq_sum if eq_sum is not None else np.full(2, equity)
    peak = np.maximum.accumulate(eq)
    max_dd = float(((peak - eq) / peak).max()) if len(eq) else 0.0
    span_days = max(
        (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).total_seconds() / 86400.0, 1.0
    ) if dates is not None and len(dates) > 1 else 1.0
    growth = max(float(eq[-1] / equity), 1e-9)
    return DirectionalResult(
        name=name,
        trades=trades,
        equity=eq,
        dates=dates if dates is not None else np.array([]),
        per_coin_net=per_coin_net,
        net_pnl=float(eq[-1] - equity),
        annualised_return=float(growth ** (TRADING_DAYS_PER_YEAR / span_days) - 1.0),
        max_drawdown=max_dd,
        n_trades=len(trades),
        cost_role="taker",
    )


# ---------------------------------------------------------------------------
# pre-registered variant grid
# ---------------------------------------------------------------------------
def resample_panel(perp_1h: DataFrame, rule: str) -> DataFrame:
    """1h perp frame -> 4h/1d OHLCV (right-labelled, no partial bar leakage)."""
    df = perp_1h.set_index("date")
    agg = (
        df.resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )
    return agg


def build_variant_grid(
    panels_1h: dict[str, DataFrame],
) -> dict[str, tuple[dict[str, DataFrame], dict[str, np.ndarray]]]:
    """The full pre-registered grid. Every variant here gets reported, pass or fail."""
    panels_4h = {s: resample_panel(df, "4h") for s, df in panels_1h.items()}
    panels_1d = {s: resample_panel(df, "1d") for s, df in panels_1h.items()}

    grid: dict[str, tuple[dict[str, DataFrame], dict[str, np.ndarray]]] = {}

    grid["ts_trend_4h_d20_lf"] = (
        panels_4h,
        {s: donchian_target(df["close"], 20, 10, allow_short=False) for s, df in panels_4h.items()},
    )
    grid["ts_trend_4h_d60_lf"] = (
        panels_4h,
        {s: donchian_target(df["close"], 60, 30, allow_short=False) for s, df in panels_4h.items()},
    )
    grid["ts_trend_1d_d20_lf"] = (
        panels_1d,
        {s: donchian_target(df["close"], 20, 10, allow_short=False) for s, df in panels_1d.items()},
    )
    grid["ts_trend_1d_d20_ls"] = (
        panels_1d,
        {s: donchian_target(df["close"], 20, 10, allow_short=True) for s, df in panels_1d.items()},
    )
    closes_1d = DataFrame({s: df["close"].to_numpy() for s, df in panels_1d.items()})
    grid["xs_mom_1d_l30_top2_flat"] = (
        panels_1d,
        cross_sectional_target(closes_1d, lookback=30, rebalance_every=7, n_long=2, n_short=0),
    )
    grid["xs_mom_1d_l30_top2_bottom2"] = (
        panels_1d,
        cross_sectional_target(closes_1d, lookback=30, rebalance_every=7, n_long=2, n_short=2),
    )

    # regime-filtered variants: the fold analysis showed the 2022 bear is what kills
    # long/flat trend. The canonical fix is the SMA-200 rule (4h: 1200 bars) - longs
    # only above, shorts (perp) below. Theory-driven, zero fitted parameters.
    sma_4h = 1200
    grid["ts_trend_4h_d60_sma200_lf"] = (
        panels_4h,
        {
            s: apply_regime_filter(
                donchian_target(df["close"], 60, 30, allow_short=False),
                df["close"], sma_4h, "long_only_above",
            )
            for s, df in panels_4h.items()
        },
    )
    grid["ts_trend_4h_d60_sma200_ls"] = (
        panels_4h,
        {
            s: apply_regime_filter(
                donchian_target(df["close"], 60, 30, allow_short=True),
                df["close"], sma_4h, "long_short",
            )
            for s, df in panels_4h.items()
        },
    )
    return grid
