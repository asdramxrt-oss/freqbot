"""Unified regime router (unified_strategy_plan.md Phases 1-2).

Design: ML never picks trade direction. Two validated models decide WHICH strategy may
trade:
  * HMM regime classifier -> per-bar causal regime label
  * XGBoost vol-spike forecaster -> per-bar P(vol spike), used as a kill switch

Execution matrix:
    regime=TREND (confirmed trend) & no spike  -> Turtle allowed
    regime=LOW_VOL_MR/NEUTRAL (range) & no spike -> Pairs allowed
    spike predicted (p > threshold) or HIGH_VOL/UNCERTAIN -> no trade (HALT)

CRITICAL causality detail:
  The existing HMMRegimeEngine predicts each test BLOCK with hmmlearn's Viterbi (forward-
  backward over the whole block), so the state at bar t can depend on bars AFTER t within
  the block. For routing we cannot use that. This module performs FILTERED inference: the
  model is fit on data strictly before the block (expanding walk-forward), and each bar's
  state is inferred from that single observation only (predict_proba on one row -> forward
  filtered distribution, no backward smoothing). Characterization of states (which state is
  "trend") is computed on the TRAIN window only.

Entry gating convention (matches every simulator in this project):
  gate[t] is consulted when an entry would fill at open[t]; gate[t] must be computable from
  data <= t-1. The router therefore emits a gate array with gate[t] = regime_allowed[t-1].
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from research_lib.strategies.regime_hmm import (
    HMMParams,
    StateCharacterization,
    characterize_states,
)
from research_lib.ml.volspike import (
    VolSpikeParams,
    build_vol_spike_dataset,
    causal_vol_features,
)
from research_lib.features.volatility import VolatilityEngine


# ---------------------------------------------------------------------------
# Causal filtered HMM router
# ---------------------------------------------------------------------------
@dataclass
class RouterConfig:
    hmm: HMMParams = field(default_factory=lambda: HMMParams(n_states=3, n_iter=150))
    refit_bars: int = 1344        # refit HMM every ~14 days of 15m bars
    min_train_bars: int = 4000
    # regime -> allowed strategy
    turtle_regimes: tuple[str, ...] = ("TREND",)
    pairs_regimes: tuple[str, ...] = ("LOW_VOL_MR", "NEUTRAL")
    vol_spike: VolSpikeParams = field(default_factory=lambda: VolSpikeParams(horizon_bars=96))
    vol_kill_p: float = 0.80      # HALT when P(spike) > this
    vol_refit_bars: int = 2688    # refit XGB every ~4 weeks
    train_window: int | None = None  # None = expanding; else rolling N-bar HMM/XGB fit window


@dataclass
class RouterOutput:
    regime: np.ndarray          # per-bar regime label (object)
    p_spike: np.ndarray         # per-bar P(spike) (NaN before first model)
    turtle_gate: np.ndarray     # causal entry gate for Turtle (gate[t] usable at open t)
    pairs_gate: np.ndarray      # causal entry gate for Pairs
    halt_reason: np.ndarray     # per-bar reason string when halted


def _obs(frame: DataFrame) -> np.ndarray:
    """Same observation matrix as the regime engine: rolling log return + realized vol."""
    ret = np.log(frame["close"] / frame["close"].shift(1)).to_numpy(dtype=float)
    lr = pd.Series(ret)
    vol = lr.rolling(24).std().to_numpy(dtype=float)
    return np.column_stack([ret, vol])


def filtered_hmm_router(
    frame: DataFrame, cfg: RouterConfig, seed: int = 42
) -> RouterOutput:
    """Walk-forward router with FILTERED (causal) HMM state inference + XGB vol-spike.

    The frame is the instrument whose regime we classify (e.g. BTC 15m). For every test
    block we fit the HMM on expanding history, characterize states on that history, then
    infer each test bar's state from that bar's observation alone.
    """
    from hmmlearn.hmm import GaussianHMM

    obs = _obs(frame)
    n = len(frame)
    dates = frame["date"].to_numpy()
    vol_all = VolatilityEngine(method="yang_zhang", window=20).compute(frame)["volatility"]
    feats_all = causal_vol_features(frame, cfg.vol_spike)

    regime = np.full(n, "UNCERTAIN", dtype=object)
    p_spike = np.full(n, np.nan)
    turtle_allowed = np.zeros(n, dtype=bool)
    pairs_allowed = np.zeros(n, dtype=bool)
    chars_for_block: dict[int, dict[int, StateCharacterization]] = {}

    finite = np.isfinite(obs[:, 0]) & np.isfinite(obs[:, 1])
    first_finite = int(np.argmax(finite)) if finite.any() else 0
    start = first_finite + cfg.min_train_bars
    block_end = 0

    while start < n - 1:
        hist = np.arange(first_finite, start)
        hist = hist[finite[hist]]
        if cfg.train_window is not None:
            hist = hist[-cfg.train_window:]
        if len(hist) < cfg.min_train_bars // 2:
            start += cfg.refit_bars
            continue
        # standardize on train stats only
        mu = np.nanmean(obs[hist], axis=0)
        sd = np.nanstd(obs[hist], axis=0)
        sd[sd < 1e-12] = 1.0
        z_hist = (obs[hist] - mu) / sd
        model = GaussianHMM(
            n_components=cfg.hmm.n_states,
            covariance_type="diag",
            n_iter=cfg.hmm.n_iter,
            random_state=seed,
            min_covar=1e-4,
        ).fit(z_hist)
        state_seq = model.predict(z_hist)
        chars = characterize_states(model, z_hist, state_seq, cfg.hmm)
        char_map = {c.state_id: c for c in chars}
        chars_for_block[start] = char_map

        # XGB vol-spike on the same expanding history (threshold from train vol only)
        import xgboost as xgb

        thr = float(vol_all[frame["date"] < dates[start]].dropna().quantile(cfg.vol_spike.percentile))
        hist_df = frame.iloc[hist].reset_index(drop=True)
        fh = causal_vol_features(hist_df, cfg.vol_spike)
        cols = [c for c in fh.columns if c != "date"]
        dh, _ = build_vol_spike_dataset(hist_df, cfg.vol_spike, threshold=thr)
        keep = fh.dropna(subset=cols).index
        yh = dh["y"].loc[keep]
        xmodel = None
        if len(keep) > 1200 and len(np.unique(yh)) == 2:
            xmodel = xgb.XGBClassifier(
                n_estimators=150, max_depth=3, learning_rate=0.05,
                eval_metric="logloss", random_state=seed, n_jobs=1,
            )
            xmodel.fit(fh.loc[keep, cols], yh)

        # ---- emit over the block with FILTERED single-bar inference ----
        block_end = min(start + cfg.refit_bars, n - 1)
        for t in range(start, block_end):
            if not finite[t]:
                continue
            row = ((obs[t] - mu) / sd).reshape(1, -1)
            try:
                # CAUSALITY NOTE (important): we call predict_proba on a SINGLE row. Despite
                # the "forward-backward" name, hmmlearn's predict_proba on a 1-row input has
                # no future observations to smooth over, so the result depends only on x_t -
                # this is what makes the router causal. Do NOT "optimize" this into one
                # predict_proba(block) call: on a multi-row block, _score smooths each bar
                # using later bars in the block (verified: only ~36% argmax agreement with
                # the per-row form on real GARCH data), which would introduce look-ahead and
                # silently invalidate every walk-forward result in this project.
                post = model.predict_proba(row)[0]
                sid = int(np.argmax(post))
            except Exception:  # noqa: BLE001
                continue
            ch = char_map.get(sid)
            regime[t] = ch.regime_label if ch else "UNCERTAIN"
            turtle_allowed[t] = regime[t] in cfg.turtle_regimes
            pairs_allowed[t] = regime[t] in cfg.pairs_regimes
            # vol spike probability
            if xmodel is not None:
                frow = feats_all.iloc[[t]].dropna(subset=cols)
                if len(frow):
                    p_spike[t] = float(xmodel.predict_proba(frow[cols])[:, 1][0])
        start = block_end

    # ---- causal entry gates: gate[t] (fill at open t) uses regime[t-1] ----
    turtle_gate = np.zeros(n, dtype=bool)
    pairs_gate = np.zeros(n, dtype=bool)
    halt_reason = np.full(n, "", dtype=object)
    for t in range(1, n):
        prev_ok_turtle = turtle_allowed[t - 1]
        prev_ok_pairs = pairs_allowed[t - 1]
        pk = p_spike[t - 1]
        spike = np.isfinite(pk) and pk > cfg.vol_kill_p
        if spike:
            halt_reason[t] = "vol_spike"
        elif regime[t - 1] in ("HIGH_VOL", "UNCERTAIN"):
            halt_reason[t] = f"regime_{regime[t - 1]}"
        turtle_gate[t] = prev_ok_turtle and not spike
        pairs_gate[t] = prev_ok_pairs and not spike
    return RouterOutput(
        regime=regime,
        p_spike=p_spike,
        turtle_gate=turtle_gate,
        pairs_gate=pairs_gate,
        halt_reason=halt_reason,
    )
