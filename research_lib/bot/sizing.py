"""Meta-model + bet sizing (PDF: Triple Barrier -> Meta Model -> Bet Sizer).

The PDF's contract is precise and worth restating, because it is the part most trading
bots get wrong:

    "ML never picks trade direction. Two validated models decide WHICH strategy may trade."

So this module does exactly two jobs:

1. **Primary side** comes from Turtle/OU/Pairs - never from ML.
2. **Meta model** answers a binary question about an already-generated primary signal:
   *"given this event, is the primary side worth taking?"* It is trained on triple-barrier
   labels (pt/sl/vertical) of historical primary events, with a strictly chronological
   train/test split (no shuffling, no leakage).
3. **Bet sizer** converts the meta model's calibrated probability into a fraction of the
   per-trade risk budget (``BetSizer``, confidence-multiplier form of fractional Kelly).

When the meta model cannot be trained honestly - too few events, or AUC <= 0.5 on the
held-out tail - it is **disabled** and every primary signal passes at full size. This
mirrors the research finding: the meta model did not beat 0.5 AUC OOS on Turtle or Pairs
events, so the correct production behaviour is to not pretend it has an edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas import DataFrame

from research_lib.bot.config import BotConfig
from research_lib.bot.features import PrimarySignal


@dataclass
class MetaModelReport:
    """Honest record of what the meta model did, for the run report."""

    enabled: bool = False
    n_events: int = 0
    n_train: int = 0
    n_test: int = 0
    test_auc: float | None = None
    base_rate: float | None = None
    model_name: str = ""
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "n_events": self.n_events,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "test_auc": self.test_auc,
            "base_rate": self.base_rate,
            "model_name": self.model_name,
            "note": self.note,
        }


@dataclass
class MetaModel:
    """Calibrated meta-probability model over primary events."""

    cfg: BotConfig
    model: object | None = None
    calibrator: object | None = None
    report: MetaModelReport = field(default_factory=MetaModelReport)

    # ------------------------------------------------------------------
    def fit(self, df: DataFrame, signals: list[PrimarySignal]) -> MetaModelReport:
        """Train the meta model on triple-barrier labels of historical primary events."""
        from research_lib.ml.metalabel import ProbabilityCalibrator, meta_target_from_labels
        from research_lib.ml.triple_barrier import TripleBarrierLabeler, TripleBarrierParams

        if not self.cfg.meta.enable_meta_model:
            self.report = MetaModelReport(enabled=False, note="meta model disabled by config")
            return self.report
        if len(signals) < int(self.cfg.meta.min_events):
            self.report = MetaModelReport(
                enabled=False,
                n_events=len(signals),
                note=f"too few primary events ({len(signals)} < {self.cfg.meta.min_events})",
            )
            return self.report

        tb = TripleBarrierLabeler(
            TripleBarrierParams(
                pt_mult=self.cfg.meta.pt_mult,
                sl_mult=self.cfg.meta.sl_mult,
                vertical_bars=self.cfg.meta.vertical_bars,
                vol_scale_col=self.cfg.meta.vol_scale_col,
            )
        )
        # label_events expects a DataFrame with a `date` column (the signal bar) and `side`
        events = pd.DataFrame(
            {
                "date": [pd.Timestamp(s.signal_time) for s in signals],
                "side": [int(s.side) for s in signals],
                "entry_price": [float(s.entry_ref) for s in signals],
            }
        )
        try:
            labels, _events = tb.label_events(df, events)
        except Exception as e:  # noqa: BLE001 - labelling must never kill the bot
            self.report = MetaModelReport(
                enabled=False, n_events=len(signals), note=f"labelling failed: {e}"
            )
            return self.report
        if labels is None or len(labels) < self.cfg.meta.min_events:
            self.report = MetaModelReport(
                enabled=False,
                n_events=0 if labels is None else len(labels),
                note="insufficient labelled events",
            )
            return self.report

        y = meta_target_from_labels(labels).to_numpy(dtype=int)
        X = _meta_features(df, labels)
        mask = np.isfinite(X).all(axis=1)
        X, y = X[mask], y[mask]
        if len(y) < self.cfg.meta.min_events or len(np.unique(y)) < 2:
            self.report = MetaModelReport(
                enabled=False, n_events=int(len(y)), note="single-class or insufficient events"
            )
            return self.report

        split = int(len(y) * float(self.cfg.meta.train_fraction))
        split = max(min(split, len(y) - 20), 20)
        X_tr, y_tr = X[:split], y[:split]
        X_te, y_te = X[split:], y[split:]

        from research_lib.ml.metalabel import default_models
        from sklearn.metrics import roc_auc_score

        best_name, best_auc, best_model, best_p = "", -np.inf, None, None
        aucs: dict[str, float] = {}
        for name, mdl in default_models(seed=42).items():
            try:
                mdl.fit(X_tr, y_tr)
                p = mdl.predict_proba(X_te)[:, 1]
                auc = float(roc_auc_score(y_te, p)) if len(np.unique(y_te)) == 2 else np.nan
            except Exception:  # noqa: BLE001
                continue
            if np.isfinite(auc):
                aucs[name] = auc
            if np.isfinite(auc) and auc > best_auc:
                best_name, best_auc, best_model, best_p = name, auc, mdl, p

        base_rate = float(y_te.mean()) if len(y_te) else None
        # Require a real margin over chance, not just "> 0.5". Picking the best of
        # two models on the test set is itself selection bias, so we also insist
        # that EVERY fitted model clears the bar (agreement) before enabling.
        min_auc = float(getattr(self.cfg.meta, "min_test_auc", 0.55))
        all_models_beat = bool(aucs) and all(a >= min_auc for a in aucs.values())
        if best_model is None or not np.isfinite(best_auc) or best_auc < min_auc or not all_models_beat:
            self.report = MetaModelReport(
                enabled=False,
                n_events=int(len(y)),
                n_train=int(len(y_tr)),
                n_test=int(len(y_te)),
                test_auc=None if best_model is None else float(best_auc),
                base_rate=base_rate,
                model_name=best_name,
                note=(
                    f"meta model has no reliable OOS edge "
                    f"(best AUC {best_auc:.3f} < {min_auc:.2f}, per-model {aucs}) "
                    f"- disabled, primaries pass at full size"
                ),
            )
            return self.report

        # refit on all labelled events, then calibrate with cross-validated probabilities
        try:
            cal = ProbabilityCalibrator(method="isotonic", cv=3)
            from sklearn.model_selection import cross_val_predict

            p_oof = cross_val_predict(
                best_model, X, y, cv=3, method="predict_proba"
            )[:, 1]
            cal.fit(y, p_oof)
            best_model.fit(X, y)
            self.model, self.calibrator = best_model, cal
        except Exception:  # noqa: BLE001 - fall back to the uncalibrated model
            best_model.fit(X, y)
            self.model, self.calibrator = best_model, None

        self.report = MetaModelReport(
            enabled=True,
            n_events=int(len(y)),
            n_train=int(len(y_tr)),
            n_test=int(len(y_te)),
            test_auc=float(best_auc),
            base_rate=base_rate,
            model_name=best_name,
            note=(
                f"meta model enabled: every model cleared OOS AUC >= {min_auc:.2f} "
                f"(best {best_name} {best_auc:.3f}, per-model {aucs})"
            ),
        )
        return self.report

    # ------------------------------------------------------------------
    def predict(self, df: DataFrame, signal: PrimarySignal) -> float:
        """Calibrated P(primary side is worth taking) for one signal. 0.5 when disabled."""
        if self.model is None:
            return 0.5
        row = _signal_features(df, signal)
        if row is None or not np.isfinite(row).all():
            return 0.5
        try:
            p = float(self.model.predict_proba(row.reshape(1, -1))[:, 1][0])
            if self.calibrator is not None:
                p = float(self.calibrator.calibrate(np.array([p]))[0])
            return p
        except Exception:  # noqa: BLE001
            return 0.5


# ---------------------------------------------------------------------------
# Feature extraction shared by fit/predict (must stay in the same order)
# ---------------------------------------------------------------------------
_META_COLS = ["atr_pct", "rsi_14", "vol_pct", "hurst", "dist_entry"]


def _row_from_bar(bar: pd.Series, entry_price: float) -> np.ndarray | None:
    def g(col: str) -> float:
        v = bar.get(col, np.nan)
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    close = g("close")
    atr = g("atr_20")
    if not np.isfinite(close) or close <= 0 or not np.isfinite(atr) or atr <= 0:
        return None
    return np.array(
        [
            atr / close,                                   # normalized volatility
            np.nan_to_num(g("rsi_14"), nan=50.0) / 100.0,
            np.nan_to_num(g("vol_pct_500"), nan=0.5),
            np.nan_to_num(g("hurst_256"), nan=0.5),
            (entry_price - close) / atr if np.isfinite(entry_price) else 0.0,
        ],
        dtype=float,
    )


def _meta_features(df: DataFrame, labels: DataFrame) -> np.ndarray:
    """Feature matrix aligned to the labelled events (one row per event).

    ``labels`` is the frame returned by ``TripleBarrierLabeler.label_events`` with columns
    event_start / side / pt / sl / label / ... . The reference entry price is not stored, so
    we reconstruct it from the barrier geometry: for a long, pt - sl = (pt_mult + sl_mult)
    * vol, so the mid of pt and sl recovers the entry exactly. We fall back to the signal
    bar close when the barriers are degenerate.
    """
    # Compare on integer nanoseconds so tz-aware and tz-naive timestamps can't
    # blow up: np.searchsorted on datetime64 compares tz-naive values, which
    # raises against a tz-aware panel. Normalising both sides to UTC int64 is
    # unambiguous and fast. NOTE: astype("int64") keeps the column's own unit
    # (pyarrow feathers are millisecond!) - convert to ns explicitly or every
    # lookup silently clamps to the last bar.
    dates = (
        (pd.to_datetime(df["date"], utc=True) - pd.Timestamp("1970-01-01", tz="UTC"))
        // pd.Timedelta(1, "ns")
    ).to_numpy()
    rows: list[np.ndarray] = []

    for rec in labels.to_dict("records"):
        ts = pd.Timestamp(rec["event_start"])
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        idx = int(np.searchsorted(dates, np.int64(ts.value), side="left"))
        idx = min(max(idx, 0), len(df) - 1)
        bar = df.iloc[idx]
        side = int(rec.get("side") or 1)
        pt, sl = rec.get("pt"), rec.get("sl")
        entry = float(bar["close"])
        if pt is not None and sl is not None and np.isfinite(pt) and np.isfinite(sl):
            # pt/sl are symmetric around entry for both sides -> mid recovers entry
            entry = (float(pt) + float(sl)) / 2.0
        row = _row_from_bar(bar, entry)
        if row is None:
            rows.append(np.full(len(_META_COLS), np.nan))
            continue
        row = row.copy()
        row[4] *= float(np.sign(side) or 1.0)
        rows.append(row)
    return np.vstack(rows) if rows else np.empty((0, len(_META_COLS)))


def _signal_features(df: DataFrame, signal: PrimarySignal) -> np.ndarray | None:
    idx = min(max(int(signal.bar_index), 0), len(df) - 1)
    bar = df.iloc[idx]
    row = _row_from_bar(bar, float(signal.entry_ref))
    if row is None:
        return None
    row = row.copy()
    row[4] *= float(np.sign(signal.side) or 1.0)
    return row


# ---------------------------------------------------------------------------
# Bet sizing
# ---------------------------------------------------------------------------
@dataclass
class BetSizerConfig:
    min_confidence: float = 0.52
    max_confidence_mult: float = 2.0
    base_risk_per_trade: float = 0.01


class BetSizerAdapter:
    """Thin wrapper over research ``BetSizer`` producing a risk-budget fraction in [0, 1]."""

    def __init__(self, cfg: BotConfig):
        from research_lib.ml.metalabel import BetSizer, BetSizerParams

        self.sizer = BetSizer(
            BetSizerParams(
                min_confidence=cfg.meta.min_confidence,
                max_confidence_mult=cfg.meta.max_confidence_mult,
                base_risk_per_trade=cfg.meta.base_risk_per_trade,
            )
        )

    def fraction(self, p: float, meta_enabled: bool) -> float:
        """Fraction of the per-trade risk budget. 1.0 when the meta model is disabled."""
        if not meta_enabled:
            return 1.0
        if not np.isfinite(p):
            return 0.0
        q = float(self.sizer.size_units(np.array([p]))[0])
        return float(np.clip(q, 0.0, 1.0))
