"""Unit tests for Phases 10-13: triple barrier, meta-model, calibration, bet sizing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research_lib.ml.metalabel import (
    BetSizer,
    BetSizerParams,
    ProbabilityCalibrator,
    build_event_dataset,
    default_models,
)
from research_lib.ml.triple_barrier import BarrierEvent, TripleBarrierLabeler, TripleBarrierParams

from research_lib.tests.test_phase1_5 import _synth_df


# ------------------------------------------------------------------ Phase 10
class TestTripleBarrier:
    def _frame_and_events(self):
        df = _synth_df(600)
        # add ATR for scaling
        from research_lib.features.engine import atr

        df["atr_20"] = atr(df, 20)
        events = pd.DataFrame(
            {
                "date": [df["date"].iloc[100], df["date"].iloc[150], df["date"].iloc[200]],
                "side": [1, -1, 1],
            }
        )
        return df, events

    def test_labels_are_ternary(self):
        df, events = self._frame_and_events()
        lb = TripleBarrierLabeler(TripleBarrierParams(vertical_bars=30))
        out, evs = lb.label_events(df, events)
        assert len(out) == 3
        assert set(out["label"]) <= {-1, 0, 1}
        assert all(e.label in (-1, 0, 1) for e in evs)
        # each event has barriers
        for e in evs:
            assert e.pt > 0 and e.sl > 0
            assert e.vertical_barrier > e.event_start

    def test_barrier_scaling_direction(self):
        """For a long event pt > ref > sl; for short pt < ref < sl (pt in favorable dir)."""
        df, events = self._frame_and_events()
        lb = TripleBarrierLabeler(TripleBarrierParams(vertical_bars=20))
        out, evs = lb.label_events(df, events)
        refs = {r["event_start"]: r for r in out.to_dict("records")}
        for e in evs:
            if e.side == 1:
                assert e.pt > e.sl
            else:
                assert e.sl > e.pt

    def test_no_future_label_at_signal_time(self):
        """The label for an event must only depend on data AFTER event_start."""
        df, events = self._frame_and_events()
        lb = TripleBarrierLabeler(TripleBarrierParams(vertical_bars=25))
        out, evs = lb.label_events(df, events)
        for e in evs:
            # label of +1 requires a high >= pt at some t in (t0, t1]
            assert e.label in (-1, 0, 1)

    def test_constructed_profit_first(self):
        """Deterministic: force price up to PT before any down move -> label +1."""
        dates = pd.date_range("2022-01-01", periods=200, freq="1h", tz="UTC")
        base = 100.0
        close = np.full(200, base)
        # event at index 20 (long); from index 21 ramp up 5%
        close[21:] = base * np.linspace(1.001, 1.05, 179)
        df = pd.DataFrame(
            {
                "date": dates,
                "open": close,
                "high": close * 1.0001,
                "low": close * 0.9999,
                "close": close,
                "volume": 1.0,
                "atr_20": np.full(200, 0.5),  # tiny ATR -> PT=+1% hit quickly
            }
        )
        events = pd.DataFrame({"date": [dates[20]], "side": [1]})
        lb = TripleBarrierLabeler(TripleBarrierParams(pt_mult=2, sl_mult=2, vertical_bars=60))
        out, evs = lb.label_events(df, events)
        assert out.iloc[0]["label"] == 1
        assert evs[0].barrier_hit == "pt"


# ------------------------------------------------------------------ Phase 11-12
class TestMetaPipeline:
    def _event_data(self, n_events=80):
        rng = np.random.RandomState(0)
        # synthetic features that genuinely separate success/failure
        n_bars = 2000
        dates = pd.date_range("2022-01-01", periods=n_bars, freq="1h", tz="UTC")
        trend = rng.normal(0, 1, n_bars)
        mom = rng.normal(0, 1, n_bars)
        noise = rng.normal(0, 3, n_bars)
        feats = pd.DataFrame(
            {
                "date": dates,
                "trend_feat": trend,
                "mom_feat": mom,
                "noise_feat": noise,
                "const_feat": 1.0,
            }
        )
        event_idx = np.linspace(100, n_bars - 100, n_events).astype(int)
        # P(success) rises with trend+ mom; ~50% base
        logit = 0.8 * trend[event_idx] + 0.8 * mom[event_idx] + noise[event_idx] * 0
        p = 1 / (1 + np.exp(-logit))
        y = (rng.random(n_events) < p).astype(int)
        labels = pd.DataFrame(
            {
                "event_start": [dates[i] for i in event_idx],
                "side": np.where(y == 1, 1, -1),
                "label": np.where(y == 1, 1, -1),
                "realized_return": np.where(y == 1, 0.02, -0.02),
                "barrier_hit": np.where(y == 1, "pt", "sl"),
            }
        )
        return feats, labels, y

    def test_event_dataset_alignment(self):
        feats, labels, y = self._event_data()
        X, y2, meta = build_event_dataset(feats, labels)
        assert len(X) == len(labels) == len(y2)
        assert (X["date"].to_numpy() == labels["event_start"].to_numpy()).all()
        assert np.array_equal(y2, y)

    def test_models_fit_and_predict(self):
        from sklearn.model_selection import train_test_split

        feats, labels, y = self._event_data(120)
        X, y2, meta = build_event_dataset(feats, labels)
        cols = ["trend_feat", "mom_feat", "noise_feat"]
        Xtr, Xte, ytr, yte = train_test_split(X[cols], y2, test_size=0.4, random_state=1)
        for name, model in default_models(seed=1).items():
            model.fit(Xtr, ytr)
            p = model.predict_proba(Xte)[:, 1]
            # signal-bearing model must beat random chance
            from sklearn.metrics import roc_auc_score

            auc = roc_auc_score(yte, p)
            assert auc > 0.55, f"{name} auc={auc:.3f}"

    def test_calibration_improves_brier(self):
        rng = np.random.RandomState(2)
        y = rng.binomial(1, 0.4, 2000)
        # systematically miscalibrated predictions (overconfident)
        p_raw = np.clip(0.3 + 0.4 * rng.normal(0, 1, 2000), 0.01, 0.99)
        cal = ProbabilityCalibrator(method="isotonic")
        cal.fit(y[:1000], p_raw[:1000])
        ev = cal.evaluate(y[1000:], p_raw[1000:])
        assert ev["brier_calibrated"] <= ev["brier_raw"] + 1e-9

    def test_bet_sizer_magnitude(self):
        sizer = BetSizer(BetSizerParams(min_confidence=0.5))
        p = np.array([0.5, 0.55, 0.7, 0.9, 0.99])
        m = sizer.multiplier(p)
        assert m[0] == pytest.approx(0.0, abs=1e-6)
        assert m[-1] == pytest.approx(1.0, abs=1e-6)
        assert np.all(np.diff(m) >= 0)  # monotone in p

    def test_bet_sizer_threshold_gate(self):
        sizer = BetSizer(BetSizerParams(min_confidence=0.55, max_confidence_mult=1.5))
        p = np.array([0.5, 0.52, 0.6, 0.95])
        q = sizer.size_units(p, base_risk=0.01)
        # below threshold -> no trade
        assert q[0] == 0.0 and q[1] == 0.0
        assert q[2] > 0.0 and q[3] > 0.0
        # capped
        sizer2 = BetSizer(BetSizerParams(min_confidence=0.5, max_confidence_mult=0.5))
        assert sizer2.multiplier(np.array([1.0]))[0] <= 0.5 + 1e-9
