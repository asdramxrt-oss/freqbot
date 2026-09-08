"""Gaussian HMM regime engine (Phase 8) with walk-forward fitting and characterization.

Critical discipline:
  * NEVER fit the HMM on the full history and backtest historically (state-label leakage).
  * Use expanding/rolling walk-forward fitting: fit on TRAIN window -> predict TEST window ->
    expand -> refit -> next TEST.
  * HMM state numbers are arbitrary. States are characterized post-fit by their measured
    statistics (mean return, volatility, trend strength, autocorrelation, duration,
    transition matrix) and mapped to economic regime labels only when the statistics support
    the mapping (LOW_VOL_MR / TREND / HIGH_VOL / UNCERTAIN).

The engine emits a regime label and the posterior probabilities per bar, using ONLY the model
fit on data before that bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

try:
    from hmmlearn.hmm import GaussianHMM

    _HAS_HMM = True
except Exception:  # pragma: no cover
    _HAS_HMM = False


@dataclass
class HMMParams:
    n_states: int = 3
    train_days: int = 180
    refit_days: int = 30
    n_iter: int = 200
    seed: int = 42
    # characterization thresholds
    trend_min_adx: float = 25.0
    high_vol_pct: float = 0.75


@dataclass
class StateCharacterization:
    state_id: int
    mean_return: float
    volatility: float
    autocorr: float
    trend_strength: float  # mean |return| ratio vs vol -> persistence proxy
    duration_bars: float
    regime_label: str  # LOW_VOL_MR / TREND / HIGH_VOL / NEUTRAL

    def to_dict(self) -> dict:
        return {
            "state_id": self.state_id,
            "mean_return": self.mean_return,
            "volatility": self.volatility,
            "autocorr": self.autocorr,
            "trend_strength": self.trend_strength,
            "duration_bars": self.duration_bars,
            "regime_label": self.regime_label,
        }


@dataclass
class RegimeResult:
    labels: Series  # regime label per bar (string) or "UNCERTAIN"
    state_id: Series  # raw HMM state id per bar (-1 before first fit)
    probabilities: DataFrame  # posterior per state
    characterizations: list[StateCharacterization]
    params: HMMParams


def characterize_states(
    model: GaussianHMM, obs: np.ndarray, state_seq: np.ndarray, params: HMMParams
) -> list[StateCharacterization]:
    """Map arbitrary HMM state ids to economic regime labels using measured statistics."""
    chars: list[StateCharacterization] = []
    n_states = model.n_components
    vols = []
    for s in range(n_states):
        cov = model.covars_[s]
        vols.append(float(np.sqrt(cov[0, 0]) if cov.ndim == 2 else np.sqrt(cov)))
    vol_threshold = float(np.quantile(vols, params.high_vol_pct))
    for s in range(n_states):
        mask = state_seq == s
        seg = obs[mask, 0] if mask.sum() > 0 else np.array([])
        mu = float(seg.mean()) if len(seg) else 0.0
        sd = float(seg.std(ddof=1)) if len(seg) > 2 else 0.0
        ac = float(pd.Series(seg).autocorr(lag=1)) if len(seg) > 5 else 0.0
        vol = vols[s]
        # trend strength proxy: |mean| vs vol over a typical horizon (like a t-stat of drift)
        trend = abs(mu) / (vol + 1e-12)
        # expected duration from the transition matrix
        trans = model.transmat_[s]
        dur = 1.0 / (1.0 - trans[s]) if trans[s] < 1.0 else 1e6
        # label logic, purely from measured statistics:
        #   HIGH_VOL   : vol above the high-vol quantile threshold
        #   LOW_VOL_MR : low vol AND negative return autocorrelation (mean-reverting texture)
        #   TREND      : otherwise when |drift| is material (persistent directional move)
        #   NEUTRAL    : everything else (no tradable characterization)
        if vol >= vol_threshold:
            label = "HIGH_VOL"
        elif ac < -0.05 and vol < vol_threshold:
            label = "LOW_VOL_MR"
        elif trend > 0.02:
            label = "TREND"
        else:
            label = "NEUTRAL"
        chars.append(
            StateCharacterization(
                state_id=s,
                mean_return=mu,
                volatility=vol,
                autocorr=ac,
                trend_strength=trend,
                duration_bars=float(dur),
                regime_label=label,
            )
        )
    return chars


class HMMRegimeEngine:
    def __init__(self, params: HMMParams | None = None):
        self.params = params or HMMParams()

    def _observation_matrix(self, df: DataFrame) -> np.ndarray:
        """Observations: rolling log return + realized vol (both causal)."""
        ret = np.log(df["close"] / df["close"].shift(1)).to_numpy(dtype=float)
        lr = pd.Series(ret)
        vol = lr.rolling(24).std().to_numpy(dtype=float)
        obs = np.column_stack([ret, vol])
        return obs

    @staticmethod
    def _standardize(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Z-score each column on the TRAIN slice (fit on train, transform test)."""
        finite = np.isfinite(obs).all(axis=1)
        mu = np.nanmean(obs[finite], axis=0)
        sd = np.nanstd(obs[finite], axis=0)
        sd[sd < 1e-12] = 1.0
        return (obs - mu) / sd, mu, sd

    def run(self, df: DataFrame) -> RegimeResult:
        """Walk-forward fit/predict over the frame.

        Training windows slide by ``refit_days``; a model fit on [0, train_end) predicts the
        following ``refit_days``. Bars before the first completed training window are marked
        UNCERTAIN (state_id -1).
        """
        if not _HAS_HMM:
            raise RuntimeError("hmmlearn not installed")
        obs_raw = self._observation_matrix(df)
        n = len(df)
        dates = df["date"].to_numpy()
        # bar index -> (state_id, posterior row)
        state_out = np.full(n, -1, dtype=int)
        prob_out = np.full((n, self.params.n_states), np.nan)
        char_by_state: dict[int, list[StateCharacterization]] = {}

        # walk-forward: fit on expanding train, predict a refit_days block.
        # Observations are z-scored using ONLY the train slice (scaler frozen per fit), so
        # test transforms never see their own statistics.
        finite = np.isfinite(obs_raw[:, 0]) & np.isfinite(obs_raw[:, 1])
        first_finite = int(np.argmax(finite)) if finite.any() else 0
        min_train = max(int(self.params.train_days * 24 * 3600 / _bar_seconds(df)), 200)
        step = max(int(self.params.refit_days * 24 * 3600 / _bar_seconds(df)), 20)
        train_end = first_finite + min_train
        while train_end < n:
            train_idx = np.arange(first_finite, train_end)
            train_idx = train_idx[finite[train_idx]]
            if len(train_idx) < min_train // 2:
                train_end = min(train_end + step, n)
                if train_end == n:
                    break
                continue
            # standardize using train stats only
            z_train, mu, sd = self._standardize(obs_raw[train_idx])
            model = GaussianHMM(
                n_components=self.params.n_states,
                covariance_type="diag",
                n_iter=self.params.n_iter,
                random_state=self.params.seed,
                min_covar=1e-4,
            ).fit(z_train)
            # characterize on the (standardized) training observations
            state_seq_train = model.predict(z_train)
            chars = characterize_states(model, z_train, state_seq_train, self.params)
            char_by_state[train_end] = chars
            # predict the following block with the frozen scaler + model
            test_end = min(train_end + step, n)
            test_idx = np.arange(train_end, test_end)
            valid = test_idx[finite[test_idx]]
            if len(valid):
                z_test = (obs_raw[valid] - mu) / sd
                sp = model.predict(z_test)
                pp = model.predict_proba(z_test)
                for j, gi in enumerate(valid):
                    state_out[gi] = int(sp[j])
                    prob_out[gi, :] = pp[j]
            train_end = test_end
        # map states to regime labels using the characterization of the fit that predicted
        # each bar (nearest preceding train_end)
        labels = np.array(["UNCERTAIN"] * n, dtype=object)
        trained_ends = sorted(char_by_state.keys())
        chars_for: dict[int, StateCharacterization] = {}
        for end in trained_ends:
            for c in char_by_state[end]:
                chars_for[c.state_id] = c
        for t in range(n):
            if state_out[t] < 0:
                continue
            # find the characterization active for this bar: the fit whose test block covers t
            active = None
            for end in reversed(trained_ends):
                if t >= end:
                    active = char_by_state[end]
                    break
            if active is None:
                continue
            sid = state_out[t]
            match = next((c for c in active if c.state_id == sid), None)
            labels[t] = match.regime_label if match else "UNCERTAIN"
        result = RegimeResult(
            labels=Series(labels, index=df.index),
            state_id=Series(state_out, index=df.index),
            probabilities=DataFrame(prob_out, index=df.index),
            characterizations=chars_for and list(chars_for.values()) or [],
            params=self.params,
        )
        return result


def _bar_seconds(df: DataFrame) -> int:
    if len(df) < 2:
        return 3600
    delta = df["date"].iloc[1] - df["date"].iloc[0]
    return int(delta.total_seconds())
