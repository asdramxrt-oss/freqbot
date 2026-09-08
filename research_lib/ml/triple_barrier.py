"""Triple-barrier labeling (Phase 10).

Creates event-based labels for primary-strategy signals:
  * event_start = signal timestamp (the bar whose close produced the signal)
  * profit-taking barrier, stop-loss barrier, vertical time barrier
  * Barriers are volatility-scaled (ATR multiples) by default.
  * Label rule (first barrier touched after the event start):
        +1  profit barrier hit first
        -1  stop barrier hit first
         0  vertical (time) barrier hit first
  * Only POST-event data is used to label; the label is never available at signal time.
  * Records MFE/MAE and realized return for the event.

Events are used for meta-labeling: y = (label == +1) as the meta target, with the original
ternary label preserved for analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


@dataclass
class TripleBarrierParams:
    pt_mult: float = 2.0       # profit barrier = entry + pt_mult * vol_scale
    sl_mult: float = 2.0       # stop barrier = entry - sl_mult * vol_scale
    vertical_bars: int = 48    # vertical barrier in bars
    vol_scale_col: str = "atr_20"  # column in the frame to scale barriers (per-side)
    use_vol_scale: bool = True


@dataclass
class BarrierEvent:
    event_start: pd.Timestamp
    side: int
    pt: float
    sl: float
    vertical_barrier: pd.Timestamp
    label: int
    realized_return: float
    mfe: float
    mae: float
    first_touch_time: pd.Timestamp | None
    barrier_hit: str  # 'pt' | 'sl' | 'vertical'

    def to_dict(self) -> dict:
        return {
            "event_start": self.event_start,
            "side": self.side,
            "pt": self.pt,
            "sl": self.sl,
            "vertical_barrier": self.vertical_barrier,
            "label": self.label,
            "realized_return": self.realized_return,
            "mfe": self.mfe,
            "mae": self.mae,
            "first_touch_time": self.first_touch_time,
            "barrier_hit": self.barrier_hit,
        }


class TripleBarrierLabeler:
    """Label primary-signal events with the triple-barrier method.

    Input: a canonical OHLCV frame with a volatility-scale column, plus an events DataFrame
    with columns (date=signal bar timestamp, side=+1/-1). Only the signal bar's close is
    used as the reference price (entry at close of the signal bar is the standard research
    assumption; live fills would use the next open - see backtest engine).
    """

    def __init__(self, params: TripleBarrierParams | None = None):
        self.p = params or TripleBarrierParams()

    def label_events(
        self, df: DataFrame, events: DataFrame
    ) -> tuple[DataFrame, list[BarrierEvent]]:
        closes = df["close"].to_numpy(dtype=float)
        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        dates = df["date"].to_numpy()
        n = len(df)
        date_to_idx = {d: i for i, d in enumerate(dates)}
        vol_col = self.p.vol_scale_col if self.p.use_vol_scale else None

        rows: list[dict] = []
        events_out: list[BarrierEvent] = []
        for _, ev in events.iterrows():
            t0 = date_to_idx.get(ev["date"])
            if t0 is None or t0 >= n - 1:
                continue
            side = int(ev["side"])
            if side not in (1, -1):
                continue
            ref = closes[t0]
            if vol_col is not None and vol_col in df.columns:
                vol = df[vol_col].iloc[t0]
                if not np.isfinite(vol) or vol <= 0:
                    vol = ref * 0.01
            else:
                vol = ref * 0.01
            if side == 1:
                pt = ref + self.p.pt_mult * vol
                sl = ref - self.p.sl_mult * vol
            else:
                pt = ref - self.p.pt_mult * vol
                sl = ref + self.p.sl_mult * vol
            t1 = min(t0 + self.p.vertical_bars, n - 1)
            vert_date = dates[t1]

            # scan forward for first barrier touch
            label = 0
            barrier_hit = "vertical"
            touch_time = vert_date
            mfe = 0.0
            mae = 0.0
            for t in range(t0 + 1, t1 + 1):
                if side == 1:
                    mfe = max(mfe, highs[t] - ref)
                    mae = max(mae, ref - lows[t])
                    if highs[t] >= pt:
                        label, barrier_hit, touch_time = 1, "pt", dates[t]
                        break
                    if lows[t] <= sl:
                        label, barrier_hit, touch_time = -1, "sl", dates[t]
                        break
                else:
                    mfe = max(mfe, ref - lows[t])
                    mae = max(mae, highs[t] - ref)
                    if lows[t] <= pt:
                        label, barrier_hit, touch_time = 1, "pt", dates[t]
                        break
                    if highs[t] >= sl:
                        label, barrier_hit, touch_time = -1, "sl", dates[t]
                        break
            # realized return at the barrier that was hit
            if barrier_hit == "pt":
                ret = (pt - ref) / ref * side
            elif barrier_hit == "sl":
                ret = (sl - ref) / ref * side
            else:
                ret = (closes[t1] - ref) / ref * side
            # MAE/MFE are expressed on the *side-aligned* basis
            mae_side = mae if side == 1 else mfe
            mfe_side = mfe if side == 1 else mae
            rec = BarrierEvent(
                event_start=ev["date"],
                side=side,
                pt=float(pt),
                sl=float(sl),
                vertical_barrier=vert_date,
                label=label,
                realized_return=float(ret),
                mfe=float(mfe_side),
                mae=float(mae_side),
                first_touch_time=touch_time,
                barrier_hit=barrier_hit,
            )
            rows.append(rec.to_dict())
            events_out.append(rec)
        out = DataFrame(rows)
        return out, events_out


def meta_target(labels: DataFrame) -> Series:
    """Meta-model target: 1 if the primary trade succeeded (pt first), else 0."""
    return (labels["label"] == 1).astype(int).reset_index(drop=True)
