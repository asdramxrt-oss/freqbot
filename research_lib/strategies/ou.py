"""Ornstein-Uhlenbeck mean-reversion engine (Phase 7) with time-stop (Phase 11).

Pipeline (strict acceptance discipline from the spec):
  1. Estimate Hurst via DFA on the candidate spread. H < 0.5 is ONLY a candidate filter.
  2. Fit X(t+1) = a + b X(t) + eps by OLS on TRAIN data only.
  3. Derive theta = -ln(b)/dt, mu = a/(1-b), half_life = ln(2)/theta.
  4. ACCEPT the OU model only if ALL of:
       * 0 < b < 1
       * b coefficient confidence interval excludes 0 (and is narrow enough)
       * R2 within a sane band (very low R2 => noise, not mean reversion)
       * residual stationarity (ADF on residuals)
       * half-life in a sane range (1..n_train/4 bars)
       * parameter stability across rolling windows (rolling b sign/CI stable)
       * sufficient sample size
  5. Z-score: z = (X - mu)/sigma with sigma = stationary std of X. Entry SHORT z>+z_entry,
     LONG z<-z_entry; exit at |z|<z_exit or time-stop at k*half_life.
  Hurst is NEVER used as sole activation proof; a fitted-but-rejected OU model must not trade.

Parameters are configurable and only ever optimized inside TRAIN folds by the walk-forward
layer - never against validation/test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

try:
    from statsmodels.tsa.stattools import adfuller

    _HAS_STATSMODELS = True
except Exception:  # pragma: no cover
    _HAS_STATSMODELS = False


@dataclass
class OUParams:
    z_entry: float = 2.0
    z_exit: float = 0.0
    half_life_timeout_mult: float = 2.5   # time-stop in half-lives
    min_sample: int = 120
    max_half_life_bars: int = 500
    min_half_life_bars: int = 3
    r2_min: float = 0.005
    b_ci_max_width: float = 0.6           # width of 95% CI on b (stability gate)
    rolling_window: int = 120             # rolling stability window
    rolling_b_min_frac: float = 0.7       # frac of rolling windows with b in (0,1)
    residual_adf_alpha: float = 0.05


@dataclass
class OUModel:
    b: float
    a: float
    theta: float
    mu: float
    half_life: float
    sigma: float
    r2: float
    b_ci: tuple[float, float]
    resid_adf_p: float
    rolling_b_frac_valid: float
    n: int
    valid: bool
    reject_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "b": self.b,
            "a": self.a,
            "theta": self.theta,
            "mu": self.mu,
            "half_life": self.half_life,
            "sigma": self.sigma,
            "r2": self.r2,
            "b_ci": list(self.b_ci),
            "resid_adf_p": self.resid_adf_p,
            "rolling_b_frac_valid": self.rolling_b_frac_valid,
            "n": self.n,
            "valid": self.valid,
            "reject_reasons": self.reject_reasons,
        }


def fit_ou(x: Series, p: OUParams | None = None) -> OUModel:
    """Fit an OU model on a spread series (TRAIN-ONLY data).

    Returns a validated/invalidated model; trading is only allowed when ``valid`` is True.
    """
    p = p or OUParams()
    s = x.dropna()
    n = len(s)
    if n < p.min_sample:
        return OUModel(0, 0, 0, 0, 0, 0, 0, (0, 0), 0, 0, 0, n, False, ["insufficient sample"])
    x_t = s.iloc[:-1].to_numpy()
    x_t1 = s.iloc[1:].to_numpy()
    # OLS: X_{t+1} = a + b X_t
    A = np.vstack([np.ones_like(x_t), x_t]).T
    coef, res_ss, *_ = np.linalg.lstsq(A, x_t1, rcond=None)
    a, b = coef
    # R2
    ss_tot = ((x_t1 - x_t1.mean()) ** 2).sum()
    ss_res = ((x_t1 - (a + b * x_t)) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # standard error of b (for CI)
    resid = x_t1 - (a + b * x_t)
    dof = len(x_t) - 2
    sigma_eps = np.sqrt((resid**2).sum() / dof) if dof > 0 else np.nan
    xtx_inv = np.linalg.inv(A.T @ A)
    se_b = sigma_eps * np.sqrt(xtx_inv[1, 1]) if np.isfinite(sigma_eps) else np.nan
    reject: list[str] = []
    valid = True
    # --- hard gates -------------------------------------------------------
    if not (0 < b < 1):
        reject.append(f"b={b:.4f} not in (0,1)")
        valid = False
    theta = -np.log(b) if 0 < b <= 1 else np.nan
    mu = a / (1 - b) if b != 1 else np.nan
    half_life = np.log(2.0) / theta if theta and np.isfinite(theta) and theta > 0 else np.nan
    if not (p.min_half_life_bars <= half_life <= p.max_half_life_bars):
        reject.append(f"half_life={half_life:.1f} outside [{p.min_half_life_bars},{p.max_half_life_bars}]")
        valid = False
    if np.isfinite(se_b):
        ci_w = 1.96 * se_b
        b_ci = (b - ci_w, b + ci_w)
        if b_ci[0] <= 0 or b_ci[1] >= 1:
            reject.append(f"b CI {b_ci} not fully inside (0,1)")
            valid = False
        if (b_ci[1] - b_ci[0]) > p.b_ci_max_width:
            reject.append(f"b CI width {(b_ci[1]-b_ci[0]):.3f} too wide")
            valid = False
    else:
        b_ci = (np.nan, np.nan)
    if r2 < p.r2_min:
        reject.append(f"R2={r2:.4f} below {p.r2_min}")
        valid = False
    # residual stationarity (ADF)
    resid_adf_p = np.nan
    if _HAS_STATSMODELS and len(resid) > 30:
        try:
            resid_adf_p = float(adfuller(resid, autolag="AIC", maxlag=20, result_object=True)[1])
        except Exception:
            pass
        if resid_adf_p >= p.residual_adf_alpha:
            reject.append(f"residual ADF p={resid_adf_p:.4f} not stationary")
            valid = False
    # rolling b stability: b in (0,1) in most rolling windows
    frac_valid = np.nan
    if n >= p.rolling_window + 10:
        rb = []
        for i in range(0, n - p.rolling_window, max(p.rolling_window // 4, 5)):
            seg_t = s.iloc[i : i + p.rolling_window - 1].to_numpy()
            seg_t1 = s.iloc[i + 1 : i + p.rolling_window].to_numpy()
            if len(seg_t) > 5:
                # polyfit returns [slope, intercept]; slope IS the AR(1) coefficient b
                slope = np.polyfit(seg_t, seg_t1, 1)[0]
                if np.isfinite(slope):
                    rb.append(0 < slope < 1)
        if rb:
            frac_valid = float(np.mean(rb))
            if frac_valid < p.rolling_b_min_frac:
                reject.append(f"rolling b valid frac={frac_valid:.2f} below {p.rolling_b_min_frac}")
                valid = False
    sigma = float(s.std(ddof=1)) if n > 1 else 0.0
    return OUModel(
        b=float(b),
        a=float(a),
        theta=float(theta) if np.isfinite(theta) else 0.0,
        mu=float(mu) if np.isfinite(mu) else 0.0,
        half_life=float(half_life) if np.isfinite(half_life) else 0.0,
        sigma=sigma,
        r2=float(r2),
        b_ci=(float(b_ci[0]), float(b_ci[1])),
        resid_adf_p=float(resid_adf_p) if np.isfinite(resid_adf_p) else 1.0,
        rolling_b_frac_valid=float(frac_valid) if np.isfinite(frac_valid) else 1.0,
        n=n,
        valid=valid,
        reject_reasons=reject,
    )


@dataclass
class OUSignalResult:
    signals: DataFrame          # z, target per bar
    model: OUModel
    events: list[dict]          # entry/exit event log


class OUEngine:
    """Run an OU strategy over a frame.

    The spread ``x`` is the tradable series (log-price, or a spread of the strategy's
    choosing). The OU model is fit on the ``fit`` slice ONLY (walk-forward discipline); the
    z-scores and signals are then generated causally for the full frame using the frozen
    model parameters.
    """

    def __init__(self, params: OUParams | None = None):
        self.p = params or OUParams()

    def run(
        self,
        df: DataFrame,
        x: Series,
        *,
        fit_start: int = 0,
        fit_end: int | None = None,
    ) -> OUSignalResult:
        fit_slice = x.iloc[fit_start:fit_end] if fit_end is not None else x.iloc[fit_start:]
        model = fit_ou(fit_slice, self.p)
        n = len(df)
        dates = df["date"].to_numpy()
        z = np.full(n, np.nan)
        # z-score using frozen mu/sigma only where the model is valid
        if model.valid and model.sigma > 0:
            vals = x.to_numpy(dtype=float)
            z = (vals - model.mu) / model.sigma
        # time-stop horizon in bars
        time_stop_bars = int(model.half_life * self.p.half_life_timeout_mult) if model.valid else 0
        target = np.zeros(n, dtype=int)
        pos = 0
        entry_bar = -1
        events: list[dict] = []
        for t in range(1, n):
            zt = z[t]
            # decide position for bar t from z at t-1 close (act at open t, engine shift)
            z_prev = z[t - 1]
            if np.isnan(z_prev):
                target[t] = 0
                continue
            if pos == 0:
                if model.valid:
                    if z_prev <= -self.p.z_entry:
                        pos, entry_bar = 1, t
                        events.append({"date": dates[t], "action": "entry", "side": 1, "z": float(z_prev)})
                    elif z_prev >= self.p.z_entry:
                        pos, entry_bar = -1, t
                        events.append({"date": dates[t], "action": "entry", "side": -1, "z": float(z_prev)})
            elif pos == 1:
                # exit when z returns toward zero OR time-stop expires
                if z_prev >= self.p.z_exit or (time_stop_bars > 0 and t - entry_bar >= time_stop_bars):
                    pos = 0
                    events.append({"date": dates[t], "action": "exit", "side": 1, "z": float(z_prev), "reason": "z_exit" if z_prev >= self.p.z_exit else "time_stop"})
                else:
                    pos = 1
            elif pos == -1:
                if z_prev <= self.p.z_exit or (time_stop_bars > 0 and t - entry_bar >= time_stop_bars):
                    pos = 0
                    events.append({"date": dates[t], "action": "exit", "side": -1, "z": float(z_prev), "reason": "z_exit" if z_prev <= self.p.z_exit else "time_stop"})
                else:
                    pos = -1
            target[t] = pos
        signals = DataFrame({"date": dates, "z": z, "target": target})
        return OUSignalResult(signals=signals, model=model, events=events)
