"""Unit tests for Phases 15-17: purged K-fold, CPCV, walk-forward, metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research_lib.validation.cpcv import CPCV
from research_lib.validation.metrics import (
    equity_curve_from_trades,
    series_metrics,
    summarize_backtest,
    trade_metrics,
)
from research_lib.validation.purged import PurgedKFold
from research_lib.validation.walkforward import WFConfig, WalkForward


def _events(n=500, horizon=48, spacing=3):
    """Events with t0 increasing and a label horizon (t1)."""
    t0 = np.arange(n) * spacing
    return pd.DataFrame({"t0": t0, "t1": t0 + horizon})


def _events_sparse(n=600, horizon=24):
    """Sparse events: no label overlap between neighbors (all folds survive purging)."""
    return _events(n, horizon=horizon, spacing=100)


# ------------------------------------------------------------------ Phase 15
class TestPurgedKFold:
    def test_overlapping_train_samples_removed(self):
        """THE critical test: a train sample whose label horizon overlaps the test block
        must be purged (not present in train)."""
        events = _events(300)
        pkf = PurgedKFold(n_splits=3, embargo_bars=10)
        folds = pkf.split(events)
        for train_idx, test_idx in folds:
            test_t0 = events["t0"].iloc[test_idx].min()
            test_t1 = events["t1"].iloc[test_idx].max() + 10  # embargo
            for tr in train_idx:
                s0 = events["t0"].iloc[tr]
                s1 = events["t1"].iloc[tr]
                # overlap would mean s1 >= test_start and s0 <= test_end
                assert not (s1 >= test_t0 and s0 <= test_t1), (
                    f"train sample {tr} overlaps embargoed test block"
                )

    def test_folds_cover_all_samples(self):
        events = _events(300)
        pkf = PurgedKFold(n_splits=5, embargo_bars=5)
        folds = pkf.split(events)
        assert len(folds) == 5
        seen = np.concatenate([t for _, t in folds])
        assert len(np.unique(seen)) == 300

    def test_time_ordered_folds(self):
        events = _events(300, spacing=100)  # sparse: no cross-fold horizon overlap
        pkf = PurgedKFold(n_splits=4, embargo_bars=0)
        folds = pkf.split(events)
        # fold test blocks are contiguous in ordered sample space; with sparse events they
        # are also strictly time-ordered (later fold min t0 > earlier fold max t1)
        prev_end = -1
        for _, test_idx in folds:
            start = events["t0"].iloc[test_idx].min()
            assert start > prev_end
            prev_end = events["t1"].iloc[test_idx].max()

    def test_embargo_expands_gap(self):
        events = _events(100, spacing=10)
        pkf0 = PurgedKFold(n_splits=2, embargo_bars=0)
        pkfE = PurgedKFold(n_splits=2, embargo_bars=100)
        f0 = pkf0.split(events)
        fE = pkfE.split(events)
        # embargoed fold trains on fewer samples (more purged)
        assert len(fE[0][0]) <= len(f0[0][0])


# ------------------------------------------------------------------ Phase 16
class TestCPCV:
    def test_combination_count(self):
        cv = CPCV(n_groups=6, n_test_groups=2)
        combos = cv.combinations()
        assert len(combos) == 15  # C(6,2)

    def test_splits_purge_overlaps(self):
        events = _events_sparse(600)
        cv = CPCV(n_groups=6, n_test_groups=2, embargo_bars=10)
        splits = cv.split(events)
        assert len(splits) == 15  # sparse events: all combos retain >=10 train samples
        for s in splits:
            # purge condition: a train sample must not overlap ANY test sample's embargoed
            # horizon (this is the true event-aware leak condition)
            for tr in s["train_idx"]:
                s0, s1 = events["t0"].iloc[tr], events["t1"].iloc[tr]
                for te in s["test_idx"]:
                    t0s = events["t0"].iloc[te]
                    t1s = events["t1"].iloc[te] + 10
                    assert not (s1 >= t0s and s0 <= t1s), "train overlaps test horizon"

    def test_dense_overlap_reduces_train_set(self):
        """Under heavy label overlap, purging removes more training samples per combo than
        under sparse overlap - the purging is doing real work."""
        dense = CPCV(n_groups=6, n_test_groups=2, embargo_bars=10).split(
            _events(600, horizon=48, spacing=3)
        )
        sparse = CPCV(n_groups=6, n_test_groups=2, embargo_bars=10).split(
            _events(600, horizon=24, spacing=100)
        )
        # compare the same combo (0,1) in both
        d = next(s for s in dense if s["combo"] == (0, 1))
        sp = next(s for s in sparse if s["combo"] == (0, 1))
        assert len(d["train_idx"]) < len(sp["train_idx"])

    def test_full_coverage_paths(self):
        """Every path must cover all 6 groups exactly once with disjoint test sets."""
        cv = CPCV(n_groups=6, n_test_groups=2)
        paths = cv.paths()
        assert len(paths) == 15  # perfect matchings of K6
        for path in paths:
            groups = [g for combo in path for g in combo]
            assert sorted(groups) == [0, 1, 2, 3, 4, 5]
            assert len(set(groups)) == 6

    def test_build_paths_from_trades(self):
        events = _events_sparse(600)
        cv = CPCV(n_groups=6, n_test_groups=2, embargo_bars=5)
        splits = cv.split(events)
        # synthesize one trade per split
        results = []
        for s in splits:
            df = pd.DataFrame(
                {
                    "entry_time": [pd.Timestamp("2022-01-01", tz="UTC")],
                    "exit_time": [pd.Timestamp("2022-01-02", tz="UTC")],
                    "side": [1],
                    "entry_price": [100.0],
                    "exit_price": [101.0],
                    "units": [1.0],
                    "notional": [100.0],
                    "pnl_net": [1.0],
                    "ret_net": [0.01],
                    "fees_paid": [0.2],
                    "mfe": [1.5],
                    "mae": [0.3],
                    "exit_reason": ["signal"],
                    "bars_held": [24],
                }
            )
            results.append((s["combo"], df))
        paths = cv.build_paths(results)
        assert len(paths) == 15
        # each path aggregates 3 splits (one per pair of groups)
        assert (paths["trades"].map(len) == 3).all()


# ------------------------------------------------------------------ Phase 17
class TestWalkForward:
    def test_folds_are_sequential_and_disjoint(self):
        cfg = WFConfig(train_bars=300, val_bars=100, test_bars=100, step_bars=100)
        n = 300 + 100 + 100 + 100 * 3  # 4 folds
        frame = pd.DataFrame({"date": pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")})

        def fp(frame, tr, va, te):
            trades = pd.DataFrame(
                {
                    "entry_time": [frame["date"].iloc[te[0]]],
                    "exit_time": [frame["date"].iloc[te[-1]]],
                    "side": [1], "entry_price": [1.0], "exit_price": [1.01],
                    "units": [1.0], "notional": [100.0], "pnl_net": [1.0],
                    "ret_net": [0.01], "fees_paid": [0.0], "mfe": [0.01],
                    "mae": [0.0], "exit_reason": ["signal"], "bars_held": [200],
                }
            )
            return {"dummy": 1}, trades

        wf = WalkForward(cfg)
        folds = wf.run(frame, fp)
        assert len(folds) == 4
        # test windows must be strictly after their own train windows and non-overlapping
        for f in folds:
            assert f.test_idx.min() > f.train_idx.max()
        for i in range(len(folds) - 1):
            assert folds[i].test_idx.max() < folds[i + 1].test_idx.min()

    def test_train_never_sees_test(self):
        """The closure receives indices; verify the frame slice handed for fitting cannot
        include test rows (causality guard at the engine level)."""
        cfg = WFConfig(train_bars=200, val_bars=50, test_bars=50, step_bars=50)
        n = 200 + 50 + 50 + 50
        frame = pd.DataFrame({"date": range(n)})
        seen: list = []

        def fp(frame, tr, va, te):
            seen.append((tr.max(), te.min()))
            return {}, pd.DataFrame()

        WalkForward(cfg).run(frame, fp)
        for tr_max, te_min in seen:
            assert te_min > tr_max


# ------------------------------------------------------------------ metrics
class TestMetrics:
    def _trades(self, n=100):
        rng = np.random.RandomState(0)
        return pd.DataFrame(
            {
                "entry_time": pd.date_range("2022-01-01", periods=n, freq="1D", tz="UTC"),
                "exit_time": pd.date_range("2022-01-02", periods=n, freq="1D", tz="UTC"),
                "side": 1,
                "entry_price": 100.0,
                "exit_price": 100.0 + rng.normal(0.5, 2, n),
                "units": 10.0,
                "notional": 1000.0,
                "pnl_net": rng.normal(20, 100, n),
                "ret_net": rng.normal(0.002, 0.01, n),
                "fees_paid": 2.0,
                "mfe": 50.0,
                "mae": 20.0,
                "exit_reason": "signal",
                "bars_held": 24,
            }
        )

    def test_trade_metrics_sane(self):
        tr = self._trades()
        m = trade_metrics(tr)
        assert m["n_trades"] == 100
        assert 0 <= m["win_rate"] <= 1
        assert m["profit_factor"] > 0
        assert np.isfinite(m["expectancy"])

    def test_series_metrics_deterministic(self):
        # steadily compounding equity -> positive Sharpe, no drawdown beyond 0
        eq = pd.Series(100 * 1.001 ** np.arange(500))
        m = series_metrics(eq, timeframe="1h")
        assert m["sharpe"] > 0
        assert m["total_return"] > 0
        assert m["max_dd"] > -0.01

    def test_summarize(self):
        tr = self._trades(50)
        s = summarize_backtest(tr, None, timeframe="1d", start_equity=10_000)
        assert s["n_trades"] == 50
        assert "sharpe" in s and "max_dd" in s and "turnover" in s
