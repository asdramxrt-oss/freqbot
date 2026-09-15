"""Feature + primary-signal assembly (PDF: Information Bars -> Feature Engine -> engines).

This module turns raw per-pair panels into the two things the router needs:

1. A **feature frame** for the regime instrument (the router's anchor, BTC/USDT by default):
   Yang-Zhang volatility, ATR, RSI, fractional differentiation, Hurst - all causal.
2. **Primary signals** from the three alpha engines:
   * Turtle breakout signals on the anchor instrument (``TurtleEngine.generate``).
   * OU mean-reversion signals on each pair (``OUEngine`` with acceptance gates).
   * Pairs spread signals on a log-ratio (``log_ratio_series`` + ``causal_local_z``).

Nothing here is new maths - every computation delegates to the validated research modules.
The only addition is the glue that maps their output onto the bot's per-bar loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from research_lib.bot.config import BotConfig


# ---------------------------------------------------------------------------
# Feature frame
# ---------------------------------------------------------------------------
def build_feature_frame(df: DataFrame, cfg: BotConfig) -> DataFrame:
    """Attach the causal feature columns used by the router and the meta model.

    Columns added (all causal - computed from data <= t):
      atr_20            - Wilder ATR, the volatility scale for stops and barriers
      rsi_14            - Wilder RSI
      vol_yz_20         - Yang-Zhang volatility (24/7 crypto degenerates to Rogers-Satchell)
      vol_pct_500       - trailing percentile of vol_yz_20 (VolatilityEngine)
      vol_regime        - LOW / NORMAL / HIGH / EXTREME bucket
      hurst_256         - rolling DFA Hurst exponent
      fracdiff_close    - fractionally differenced close (d frozen on the first train slice)
    """
    from research_lib.features.engine import atr, rsi
    from research_lib.features.hurst import rolling_hurst
    from research_lib.features.volatility import VolatilityEngine
    from research_lib.features.fracdiff import fit_fracdiff_transform

    out = df.copy()

    out["atr_20"] = atr(out, 20).replace(0.0, np.nan)
    out["rsi_14"] = rsi(out, 14)

    vol = VolatilityEngine(method="yang_zhang", window=20).compute(out)
    out["vol_yz_20"] = vol["volatility"].to_numpy()
    out["vol_pct_500"] = vol["volatility_percentile"].to_numpy()
    out["vol_regime"] = vol["volatility_regime"].to_numpy()

    # Hurst is O(n) DFA passes and costs ~1 minute on 120k bars; it is a *filter* feature
    # used only by the meta model, so it is opt-in (ft.features.use_hurst in config).
    if getattr(cfg.data, "use_hurst", False):
        try:
            out["hurst_256"] = rolling_hurst(out["close"], window=256).to_numpy()
        except Exception:  # noqa: BLE001 - Hurst is a filter, never a hard dependency
            out["hurst_256"] = np.nan
    else:
        out["hurst_256"] = np.nan

    # Fractional differentiation: d is selected on the first 20% of bars only (train slice)
    # then frozen and applied to the whole frame - the same discipline the research used.
    try:
        train_n = max(int(len(out) * 0.2), 256)
        fd = fit_fracdiff_transform(out["close"].iloc[:train_n], window=100)
        out["fracdiff_close"] = fd.transform(out["close"]).to_numpy()
        out["fracdiff_d"] = fd.d
    except Exception:  # noqa: BLE001 - optional feature
        pass

    return out


# ---------------------------------------------------------------------------
# Primary signals
# ---------------------------------------------------------------------------
@dataclass
class PrimarySignal:
    """One entry candidate produced by a primary engine, before routing/sizing."""

    strategy_id: str            # "turtle" | "ou" | "pairs"
    symbol: str
    bar_index: int              # index in the anchor frame; entry fills at open[bar_index+1]
    signal_time: pd.Timestamp   # the bar whose close triggered the signal
    side: int                   # +1 long / -1 short
    entry_ref: float            # reference price = open of the fill bar
    stop_price: float | None
    note: str = ""


def turtle_signals(df: DataFrame, cfg: BotConfig, symbol: str) -> list[PrimarySignal]:
    """Turtle breakout entry signals for one instrument (trend book).

    ``TurtleEngine.generate`` returns a per-bar frame with columns
    ``date / target / stop / N / dc_upper / dc_lower``. ``target[t]`` is the position the
    strategy wants to hold DURING bar t, entered at ``open[t]`` - so the fill bar is the
    row itself and the triggering bar is ``t-1``. We keep the flat -> long/short entries.
    """
    from research_lib.strategies.turtle import TurtleEngine, TurtleParams

    params = TurtleParams(
        system=1,
        entry_n=cfg.strategy.turtle_entry_n,
        atr_n=cfg.strategy.turtle_atr_n,
        stop_mult=cfg.strategy.turtle_stop_mult,
        pyramid_max_units=cfg.strategy.turtle_max_units,
        pyramid_step_n=cfg.strategy.turtle_pyramid_step_n,
    )
    signals_df, _log = TurtleEngine(params).generate(df)

    out: list[PrimarySignal] = []
    if signals_df is None or len(signals_df) == 0:
        return out

    target = signals_df["target"].to_numpy(dtype=int)
    stops = signals_df["stop"].to_numpy(dtype=float)
    dates = signals_df["date"].to_numpy()
    opens = df["open"].to_numpy(dtype=float)
    n = len(df)

    prev = 0
    for t in range(min(len(target), n)):
        side = int(target[t])
        if side != 0 and prev == 0:
            ref = float(opens[t])
            if not np.isfinite(ref) or ref <= 0:
                prev = side
                continue
            stop = float(stops[t]) if t < len(stops) else np.nan
            out.append(
                PrimarySignal(
                    strategy_id="turtle",
                    symbol=symbol,
                    bar_index=t,
                    signal_time=pd.Timestamp(dates[t - 1] if t > 0 else dates[t]),
                    side=side,
                    entry_ref=ref,
                    stop_price=float(stop) if np.isfinite(stop) else None,
                )
            )
        prev = side
    return out


def ou_signals_for_pair(df: DataFrame, cfg: BotConfig, symbol: str) -> list[PrimarySignal]:
    """OU mean-reversion entry signals via the validated OUEngine.

    The engine fits the OU model on the warmup slice (walk-forward discipline) and emits
    per-bar targets. We convert the target transitions into entry signals. Unlike Turtle,
    the engine's ``target[t]`` is decided from ``z[t-1]`` and filled at ``open[t]``, so the
    signal bar is ``t-1`` and the fill bar is ``t``.
    """
    from research_lib.strategies.ou import OUEngine, OUParams

    params = OUParams(
        z_entry=cfg.strategy.ou_z_entry,
        z_exit=cfg.strategy.ou_z_exit,
        half_life_timeout_mult=cfg.strategy.ou_half_life_timeout_mult,
        min_sample=cfg.strategy.ou_min_sample,
    )
    x = Series(np.log(df["close"].to_numpy(dtype=float)), index=df.index)
    warmup = min(int(cfg.data.warmup_bars), max(len(df) - 2, 1))
    result = OUEngine(params).run(df, x, fit_start=0, fit_end=warmup)
    model = result.model
    if not model.valid:
        return []

    out: list[PrimarySignal] = []
    target = result.signals["target"].to_numpy(dtype=int)
    z_col = result.signals["z"].to_numpy(dtype=float) if "z" in result.signals else None
    opens = df["open"].to_numpy(dtype=float)
    dates = df["date"].to_numpy()
    n = len(df)

    prev = 0
    for t in range(1, n):
        side = int(target[t])
        if side != 0 and prev == 0:
            ref = float(opens[t])
            if not np.isfinite(ref) or ref <= 0:
                prev = side
                continue
            stop = float(np.exp(model.mu - side * 2.0 * model.sigma))
            z = float(z_col[t - 1]) if z_col is not None and np.isfinite(z_col[t - 1]) else np.nan
            out.append(
                PrimarySignal(
                    strategy_id="ou",
                    symbol=symbol,
                    bar_index=t - 1,
                    signal_time=pd.Timestamp(dates[t - 1]),
                    side=side,
                    entry_ref=ref,
                    stop_price=stop,
                    note=f"z={z:.2f}" if np.isfinite(z) else "",
                )
            )
        prev = side
    return out


def pairs_signals(a: DataFrame, b: DataFrame, cfg: BotConfig) -> list[PrimarySignal]:
    """Market-neutral pairs entry signals from the causal local-mean spread z-score."""
    from research_lib.strategies.pairs import PairsParams, causal_local_z, log_ratio_series

    out: list[PrimarySignal] = []
    frame = log_ratio_series(a, b)
    if frame is None or len(frame) < 10:
        return out

    params = PairsParams(
        est_window_bars=int(cfg.strategy.pairs_window),
        z_entry=cfg.strategy.pairs_z_entry,
        z_exit=cfg.strategy.pairs_z_exit,
        max_hold_bars=int(cfg.strategy.pairs_max_hold_bars),
    )
    z = causal_local_z(frame, params)

    closes = frame["close_a"].to_numpy(dtype=float)
    opens = frame["open_a"].to_numpy(dtype=float)
    dates = frame["date"].to_numpy()
    n = len(frame)

    prev = 0
    for t in range(1, n - 1):
        zt = z[t]
        if not np.isfinite(zt):
            continue
        side = 0
        if prev == 0:
            if zt <= -params.z_entry:
                side = 1
            elif zt >= params.z_entry:
                side = -1
        if side != 0:
            ref = float(opens[t + 1])
            if np.isfinite(ref) and ref > 0:
                out.append(
                    PrimarySignal(
                        strategy_id="pairs",
                        symbol=f"{cfg.strategy.pairs_a}|{cfg.strategy.pairs_b}",
                        bar_index=t,
                        signal_time=pd.Timestamp(dates[t]),
                        side=side,
                        entry_ref=ref,
                        stop_price=None,
                        note=f"z={zt:.2f}",
                    )
                )
        prev = side if side != 0 else prev
    return out
