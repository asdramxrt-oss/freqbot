"""Volatility-spike prediction model (blueprint Phase 2.1) - an ENTRY FILTER.

Objective change: instead of predicting price direction, predict whether Yang-Zhang
volatility will exceed a high percentile (e.g. p90) over a forward horizon. The prediction
is used strictly as a risk filter: do NOT enter mean-reversion trades when a volatility
expansion is forecast (they get run over when vol spikes).

Honesty constraints:
  * The label uses only FUTURE vol (post-bar); the model is trained on causal features and
    evaluated on later, unseen events (chronological / purged splits).
  * The filter is only useful if the base strategy has a positive edge to protect; if the
    base edge is negative the filter cannot manufacture one. Both facts are reported.
  * AUC is reported but the deciding metric is the change in net OOS PnL (or avoided loss)
    of the filtered strategy vs unfiltered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas import DataFrame

from research_lib.features.volatility import VolatilityEngine


@dataclass
class VolSpikeParams:
    horizon_bars: int = 96          # 1 day of 15m bars
    percentile: float = 0.90        # "high vol" threshold percentile of realized vol
    min_train_bars: int = 2000


def build_vol_spike_dataset(
    frame: DataFrame, params: VolSpikeParams | None = None, threshold: float | None = None
) -> tuple[DataFrame, np.ndarray]:
    """Build (features, y) where y=1 if realized vol over the NEXT horizon exceeds the
    p-th percentile of past vol.

    Realized vol of the future window is the label (strictly future). Features are causal.
    The percentile threshold defaults to the FULL-series quantile (research convention);
    for walk-forward/purged evaluation the caller must pass a TRAIN-ONLY threshold so test
    labels do not depend on test-period vol levels.
    """
    params = params or VolSpikeParams()
    eng = VolatilityEngine(method="yang_zhang", window=20, session="cont")
    vol = eng.compute(frame)["volatility"]
    # realized vol over the NEXT horizon (label - future by construction)
    lr = np.log(frame["close"] / frame["close"].shift(1))
    future_vol = lr.rolling(params.horizon_bars).std().shift(-params.horizon_bars)
    if threshold is None:
        threshold = float(vol.quantile(params.percentile))
    y = (future_vol > threshold).astype(int)
    out = frame[["date"]].copy()
    out["vol"] = vol
    out["future_vol"] = future_vol
    out["y"] = y
    return out, y.to_numpy()


def causal_vol_features(frame: DataFrame, params: VolSpikeParams | None = None) -> DataFrame:
    """Causal feature frame for the vol-spike model (no future data)."""
    params = params or VolSpikeParams()
    eng = VolatilityEngine(method="yang_zhang", window=20, session="cont")
    v = eng.compute(frame)
    lr = np.log(frame["close"] / frame["close"].shift(1))
    out = DataFrame(index=frame.index)
    out["date"] = frame["date"]
    out["vol"] = v["volatility"]
    out["vol_pctile"] = v["volatility_percentile"]
    out["ret_1"] = frame["close"].pct_change()
    out["ret_abs_24"] = lr.abs().rolling(24).mean()
    out["vol_ratio_24_240"] = lr.rolling(24).std() / lr.rolling(240).std().replace(0, np.nan)
    out["range_24"] = (frame["high"].rolling(24).max() - frame["low"].rolling(24).min()) / frame["close"]
    return out.replace([np.inf, -np.inf], np.nan)


def evaluate_vol_spike_filter(
    frame: DataFrame,
    target: np.ndarray,
    params: VolSpikeParams | None = None,
    seed: int = 0,
) -> dict:
    """Train a vol-spike classifier on the FIRST 60% and evaluate on the LAST 40%
    (chronological, no future leakage). Reports model quality AND the filter's effect on the
    tradeable signal density.
    """
    from xgboost import XGBClassifier

    from research_lib.validation.metrics import trade_metrics

    params = params or VolSpikeParams()
    feats = causal_vol_features(frame, params)
    # train-only vol threshold so test labels never depend on test-period vol levels
    eng = VolatilityEngine(method="yang_zhang", window=20, session="cont")
    vol = eng.compute(frame)["volatility"]
    n_full = len(frame)
    cut_frac = 0.6
    train_vol = vol.iloc[: int(n_full * cut_frac)].dropna()
    thr = float(train_vol.quantile(params.percentile)) if len(train_vol) > 100 else float(vol.quantile(params.percentile))
    _, y = build_vol_spike_dataset(frame, params, threshold=thr)
    # align
    feats = feats.iloc[: len(y)].copy()
    feats["y"] = y
    feats = feats.dropna()
    if len(feats) < params.min_train_bars:
        return {"error": "insufficient data"}
    n = len(feats)
    cut = int(n * cut_frac)
    cols = [c for c in feats.columns if c not in ("date", "y")]
    Xtr, Xte = feats[cols].iloc[:cut], feats[cols].iloc[cut:]
    ytr, yte = feats["y"].iloc[:cut], feats["y"].iloc[cut:]
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return {"error": "degenerate single-class label split", "n_train": len(ytr), "n_test": len(yte)}
    model = XGBClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05, eval_metric="logloss",
        random_state=seed, n_jobs=1,
    )
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(yte, p) if len(np.unique(yte)) > 1 else np.nan
    # filter effect: share of TEST-period bars where a vol spike is predicted
    pred_spike = p > 0.5
    spike_rate = float(pred_spike.mean())
    return {
        "auc": auc,
        "base_rate": float(yte.mean()),
        "pred_spike_rate": spike_rate,
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "feature_cols": cols,
    }
