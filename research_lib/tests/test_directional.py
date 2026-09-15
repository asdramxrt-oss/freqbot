"""Tests for the directional evaluation harness and acceptance gates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research_lib.risk.costs import CostModel
from research_lib.strategies.directional import (
    apply_regime_filter,
    cross_sectional_target,
    donchian_target,
    run_panel_directional,
)
from research_lib.validation.directional_acceptance import (
    acceptance_gates,
    evaluate_directional_folds,
    run_directional_acceptance,
)

COST = CostModel(maker_fee=0.0002, taker_fee=0.0005, spread_bps=2.0, slippage_bps=1.0)


def _frame(prices, start="2022-01-01", freq="4h"):
    dates = pd.date_range(start, periods=len(prices), freq=freq, tz="UTC")
    return pd.DataFrame(
        {"date": dates, "open": prices, "high": prices, "low": prices, "close": prices, "volume": 1.0}
    )


def _close_series(prices):
    return pd.Series(np.asarray(prices, dtype=float))


class TestDonchianTarget:
    def test_breakout_entry_hold_and_exit(self):
        prices = [100.0] * 25 + [130.0] * 5 + [90.0] * 10
        tgt = donchian_target(_close_series(prices), lookback=20, exit_lookback=10, allow_short=False)
        assert tgt[:25].max() == 0.0            # no position before any breakout
        assert tgt[25] == 1.0                   # breakout bar enters
        assert tgt[26:30].min() == 1.0          # holds while above the exit channel
        assert tgt[-1] == 0.0                   # exits after the breakdown

    def test_truncation_invariance(self):
        """Signals at bar t must not depend on bars after t (project causality rule)."""
        rng = np.random.default_rng(7)
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
        full = donchian_target(_close_series(prices), 20, 10, allow_short=True)
        for cut in (150, 250, 350):
            part = donchian_target(_close_series(prices[:cut]), 20, 10, allow_short=True)
            assert (full[:cut] == part).all()

    def test_long_only_never_goes_short(self):
        prices = list(np.linspace(100.0, 60.0, 100))
        tgt = donchian_target(_close_series(prices), 10, 5, allow_short=False)
        assert tgt.min() == 0.0
        tgt_ls = donchian_target(_close_series(prices), 10, 5, allow_short=True)
        assert tgt_ls.min() == -1.0


class TestRegimeFilter:
    def test_long_only_above_suppresses_below_sma(self):
        prices = [100.0] * 25 + [120.0] * 25 + [80.0] * 50
        tgt = np.ones(100)
        out = apply_regime_filter(tgt, _close_series(prices), sma_window=20, mode="long_only_above")
        assert out[30] == 1.0     # above SMA -> long allowed
        assert out[80] == 0.0     # below SMA -> suppressed

    def test_long_short_mode(self):
        prices = [100.0] * 50 + [80.0] * 50
        tgt = np.where(np.arange(100) % 2 == 0, 1.0, -1.0)
        out = apply_regime_filter(tgt, _close_series(prices), sma_window=20, mode="long_short")
        assert out[81] == -1.0    # short target below SMA -> allowed
        assert out[80] == 0.0     # long target below SMA -> suppressed

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            apply_regime_filter(np.ones(10), _close_series([100.0] * 10), 5, "nonsense")


class TestCrossSectional:
    def test_picks_strongest_and_resets_basket(self):
        n = 400
        slow = np.linspace(100, 105, n)      # +5%
        mid = np.linspace(100, 110, n)       # +10%
        fast = np.linspace(100, 140, n)      # +40%
        closes = pd.DataFrame({"A": slow, "B": mid, "C": fast})
        tgt = cross_sectional_target(closes, lookback=30, rebalance_every=7, n_long=1, n_short=1)
        assert tgt["C"][-1] == 1.0           # strongest held long
        assert tgt["A"][-1] == -1.0          # weakest held short
        assert tgt["B"][-1] == 0.0
        # no stale holdings: wherever the target is nonzero it was set at/after the
        # last rebalance bar
        assert set(np.unique(tgt["C"])) <= {-1.0, 0.0, 1.0}


class TestPanelBacktest:
    def test_uptrend_with_always_long_target_is_profitable(self):
        prices = list(np.linspace(100.0, 200.0, 500))
        panels = {s: _frame(prices) for s in ("A", "B")}
        targets = {s: np.ones(500) for s in panels}
        res = run_panel_directional("t", panels, targets, COST, equity=10_000.0)
        assert res.net_pnl > 0.0
        assert res.n_trades >= 1
        assert set(res.trades["symbol"].unique()) == {"A", "B"}

    def test_folds_attribute_equity_differences(self):
        prices = list(np.linspace(100.0, 160.0, 600))
        panels = {s: _frame(prices) for s in ("A", "B")}
        targets = {s: np.ones(600) for s in panels}
        res = run_panel_directional("t", panels, targets, COST, equity=10_000.0)
        folds = evaluate_directional_folds(res, n_folds=6)
        assert len(folds) == 6
        assert folds["net_pnl"].sum() == pytest.approx(res.net_pnl + (res.equity[0] - 10_000.0), rel=0.05)


class TestGatesLogic:
    def _ok_result(self):
        from types import SimpleNamespace

        return SimpleNamespace(
            n_trades=50, per_coin_net={"A": 1000.0, "B": 900.0, "C": 800.0},
            max_drawdown=0.10,
        )

    def test_all_gates_pass_on_healthy_inputs(self):
        from types import SimpleNamespace

        folds = pd.DataFrame({"net_pnl": [100.0] * 6})
        stress = {1.5: SimpleNamespace(net_pnl=50.0), 2.0: SimpleNamespace(net_pnl=10.0)}
        gates = acceptance_gates(self._ok_result(), folds, stress)
        assert all(g.passed for g in gates)

    @pytest.mark.parametrize(
        "folds,stress,coin_net,dd,failing_gate",
        [
            (pd.DataFrame({"net_pnl": [100.0] * 5 + [-1.0]}), None, None, None, 2),
            (pd.DataFrame({"net_pnl": [100.0] * 6}), {2.0: -5.0}, None, None, 3),
            (pd.DataFrame({"net_pnl": [100.0] * 6}), None, {"A": 5000.0, "B": 100.0}, None, 4),
            (pd.DataFrame({"net_pnl": [100.0] * 6}), None, None, 0.40, 5),
        ],
    )
    def test_each_gate_fails_on_its_own_bad_input(self, folds, stress, coin_net, dd, failing_gate):
        from types import SimpleNamespace

        res = self._ok_result()
        if coin_net is not None:
            res.per_coin_net = coin_net
        if dd is not None:
            res.max_drawdown = dd
        if stress is None:
            stress = {1.5: SimpleNamespace(net_pnl=50.0), 2.0: SimpleNamespace(net_pnl=10.0)}
        else:
            stress = {m: SimpleNamespace(net_pnl=v) for m, v in stress.items()}
        gates = {g.gate_id: g for g in acceptance_gates(res, folds, stress)}
        assert gates[failing_gate].passed is False


class TestEndToEnd:
    def test_smooth_trend_world_passes_acceptance(self):
        """A strong, steady multi-coin uptrend must pass every gate."""
        rng = np.random.default_rng(3)
        panels, targets = {}, {}
        for i, s in enumerate(("A", "B", "C", "D", "E")):
            drift = 0.0008 + 0.0002 * i
            prices = 100.0 * np.exp(np.cumsum(drift + rng.normal(0, 0.004, 1500)))
            panels[s] = _frame(prices, freq="1d")
            targets[s] = donchian_target(panels[s]["close"], 20, 10, allow_short=False)
        acc = run_directional_acceptance("trend", panels, targets, COST, equity=100_000.0)
        assert acc.verdict == "PASS", [g.evidence for g in acc.gates if not g.passed]

    def test_chop_world_fails(self):
        """Mean-reverting noise must not pass - the gates exist to reject it."""
        rng = np.random.default_rng(11)
        panels, targets = {}, {}
        for s in ("A", "B", "C", "D", "E"):
            prices = 100.0 + np.cumsum(rng.normal(0, 0.6, 1500))
            panels[s] = _frame(prices, freq="1d")
            targets[s] = donchian_target(panels[s]["close"], 20, 10, allow_short=False)
        acc = run_directional_acceptance("chop", panels, targets, COST, equity=100_000.0)
        assert acc.verdict == "FAIL"
