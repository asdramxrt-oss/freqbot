"""Meta-modeling pipeline (Phases 11-13).

Architecture (spec-critical):
    PRIMARY SIGNAL -> EVENT (triple barrier) -> META MODEL -> P(success) -> bet size

The ML model NEVER generates the primary signal; it only estimates P(this primary trade
succeeds | features). Features are the causal engine frame aligned to each event start.

Phase 11 (model), Phase 12 (calibration), Phase 13 (bet sizing) live here together because
they form one pipeline; the validation layer (purged k-fold / CPCV / walk-forward) wraps
this pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

try:
    from sklearn.calibration import CalibratedClassifierCV, IsotonicRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.model_selection import cross_val_predict
    from sklearn.isotonic import IsotonicRegression as _ISO  # noqa: F401  (re-export guard)
    from xgboost import XGBClassifier

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover
    _HAS_SKLEARN = False

from scipy.stats import norm

# ---------------------------------------------------------------------------
# feature/event alignment
# ---------------------------------------------------------------------------
def build_event_dataset(
    features: DataFrame,
    labels: DataFrame,
) -> tuple[DataFrame, np.ndarray, DataFrame]:
    """Align labeled events to their causal feature rows (features AT the event start).

    features: engine frame with a 'date' column (one row per bar).
    labels:   triple-barrier label frame with 'event_start' column.
    Returns (X, y, meta):
      X    - feature rows at each event start (index reset, 'date' preserved)
      y    - meta target: 1 if the primary trade succeeded (label == +1), else 0
      meta - the label rows (event_start, side, realized_return, barrier_hit, ...)
    """
    fdate = pd.to_datetime(features["date"], utc=True)
    idx = {d: i for i, d in enumerate(fdate)}
    feat_pos: list[int] = []
    label_pos: list[int] = []
    for j, d in enumerate(labels["event_start"]):
        i = idx.get(pd.Timestamp(d))
        if i is not None:
            feat_pos.append(i)
            label_pos.append(j)
    if not feat_pos:
        empty = features.iloc[0:0].reset_index(drop=True)
        return empty, np.empty(0, dtype=int), labels.iloc[0:0].reset_index(drop=True)
    X = features.iloc[feat_pos].reset_index(drop=True)
    lab = labels["label"].to_numpy()
    y = np.array([1 if lab[j] == 1 else 0 for j in label_pos], dtype=int)
    meta = labels.iloc[label_pos].reset_index(drop=True)
    return X, y, meta


def meta_target_from_labels(labels: DataFrame) -> Series:
    return (labels["label"] == 1).astype(int).reset_index(drop=True)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
def default_models(seed: int = 42) -> dict[str, object]:
    return {
        "xgb": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
        ),
        "rf": RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=20,
            random_state=seed,
            n_jobs=1,
        ),
    }


# ---------------------------------------------------------------------------
# calibration (Phase 12)
# ---------------------------------------------------------------------------
class ProbabilityCalibrator:
    """Isotonic calibration fit on TRAIN probabilities only (never test)."""

    def __init__(self, method: str = "isotonic", cv: int = 3):
        self.method = method
        self.cv = cv

    def fit(self, y: np.ndarray, p_train: np.ndarray) -> None:
        if self.method == "isotonic":
            self._cal = IsotonicRegression(out_of_bounds="clip")
        else:  # sigmoid
            from sklearn.linear_model import LogisticRegression

            self._cal = LogisticRegression()
            p_train = p_train.reshape(-1, 1)
        self._cal.fit(p_train, y)

    def calibrate(self, p: np.ndarray) -> np.ndarray:
        if self.method == "isotonic":
            return self._cal.predict(p)
        return self._cal.predict_proba(p.reshape(-1, 1))[:, 1]

    def evaluate(self, y: np.ndarray, p: np.ndarray) -> dict:
        """Calibration diagnostics on a validation set (probabilities pre-calibration)."""
        return {
            "brier_raw": float(brier_score_loss(y, p)),
            "brier_calibrated": float(brier_score_loss(y, self.calibrate(p))),
            "auc_raw": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
            "mean_pred": float(p.mean()),
            "base_rate": float(y.mean()),
        }


# ---------------------------------------------------------------------------
# Fractional Kelly sizing (blueprint Phase 3.2)
# ---------------------------------------------------------------------------
def kelly_fraction(p_win: float, win_loss_ratio: float) -> float:
    """Full Kelly fraction: f* = p - (1-p)/b where b = win/loss ratio (gross).

    Only meaningful when p_win and the payoff ratio are estimated OUT-OF-SAMPLE from
    calibrated probabilities and realized trade outcomes. Returns <= 0 (no bet) when the
    edge is not positive.
    """
    if not (0 < p_win < 1) or win_loss_ratio <= 0:
        return 0.0
    f = p_win - (1.0 - p_win) / win_loss_ratio
    return max(f, 0.0)


def fractional_kelly_size(
    p_win: float,
    win_loss_ratio: float,
    kelly_frac: float = 0.25,
    max_position_frac: float = 1.0,
) -> float:
    """Fraction of the risk budget to deploy under fractional Kelly.

    f_kelly = kelly_frac * f*  (quarter-Kelly default for variance control).
    Returns the fraction in [0, max_position_frac]; 0 means no trade.
    """
    f_star = kelly_fraction(p_win, win_loss_ratio)
    return min(kelly_frac * f_star, max_position_frac)


@dataclass
class BetSizerParams:
    min_confidence: float = 0.52     # below this, no trade
    max_confidence_mult: float = 2.0  # cap on the m multiplier
    base_risk_per_trade: float = 0.01  # full-confidence risk budget


class BetSizer:
    """Maps a calibrated probability to a bet-size multiplier.

    z = (p - 0.5) / sqrt(p(1-p))
    m = 2 * Phi(z) - 1        (confidence multiplier in [-1, 1])
    Q_final = min(Q_risk * m, Q_max)

    p ~ 0.5  => m ~ 0 (no trade)
    p -> 1   => m -> 1 (full budget)
    p < 0.5  => m < 0 (signal inverted confidence - rejected unless explicitly allowed)
    """

    def __init__(self, params: BetSizerParams | None = None):
        self.p = params or BetSizerParams()

    def multiplier(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        z = (p - 0.5) / np.sqrt(p * (1.0 - p))
        m = 2.0 * norm.cdf(z) - 1.0
        return np.clip(m, 0.0, self.p.max_confidence_mult)

    def approved(self, p: np.ndarray) -> np.ndarray:
        """Trade only when calibrated confidence exceeds the minimum threshold."""
        return (p >= self.p.min_confidence) & (p <= 1 - 1e-6)

    def size_units(
        self,
        p: np.ndarray,
        base_risk: float | None = None,
        risk_budget: float | None = None,
    ) -> np.ndarray:
        """Fraction of the per-trade risk budget to deploy (0..1)."""
        m = self.multiplier(p)
        budget = risk_budget if risk_budget is not None else base_risk or self.p.base_risk_per_trade
        approved = self.approved(p)
        q = np.where(approved, m, 0.0)
        return q  # fraction of budget; the risk engine converts to units
