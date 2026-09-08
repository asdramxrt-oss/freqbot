"""Unit tests for Phases 18-20: Monte Carlo, ablation, cost stress, param robustness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research_lib.risk.costs import CostModel
from research_lib.validation.ablation import cost_stress, cost_stress_verdict, run_ablation
from research_lib.validation.montecarlo import MCConfig, MonteCarlo, parameter_grid_robustness, plateau_acceptance


def _trades(n=200, seed=0, win_rate=0.5):
    rng = np.random.RandomState(seed)
    pnl = np.where(rng.random(n) < win_rate, rng.uniform(50, 300, n), -rng.uniform(50, 250, n))
    dates = pd.date_range("2022-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame(
        {
            "entry_time": dates,
            "exit_time": dates + pd.Timedelta(hours=8),
            "side": 1,
            "entry_price": 100.0,
            "exit_price": 100.0 + pnl / 10,
            "units": 10.0,
            "notional": 1000.0,
            "pnl_net": pnl,
            "ret_net": pnl / 1000.0,
            "fees_paid": 2.0,
            "mfe": 200.0,
            "mae": 100.0,
            "exit_reason": "signal",
            "bars_held": 2,
        }
    )


class TestMonteCarlo:
    def test_deterministic_with_seed(self):
        tr = _trades()
        a = MonteCarlo(MCConfig(n_sims=50, seed=7)).run(tr, 100_000)
        b = MonteCarlo(MCConfig(n_sims=50, seed=7)).run(tr, 100_000)
        pd.testing.assert_frame_equal(a, b)

    def test_summary_bounds(self):
        tr = _trades(n=300, win_rate=0.55)
        s = MonteCarlo(MCConfig(n_sims=200, seed=1)).summarize(tr, 100_000)
        assert 0 <= s["p_pnl_negative"] <= 1
        assert 0 <= s["p_ruin"] <= 1
        assert s["pnl_p5"] <= s["pnl_median"] <= s["pnl_p95"]
        assert s["dd_p5"] <= s["dd_median"] <= s["dd_p95"]  # dd negative: p5 worst

    def test_losing_trades_have_high_p_negative(self):
        tr = _trades(n=300, win_rate=0.1)  # mostly losers
        s = MonteCarlo(MCConfig(n_sims=100, seed=2)).summarize(tr, 100_000)
        assert s["p_pnl_negative"] > 0.5


class TestParamRobustness:
    def test_grid_and_plateau(self):
        def run_fn(params):
            # profit shrinks as lookback moves away from the "true" 20
            lookback = params["lookback"]
            return {"sharpe": 1.0 - 0.5 * abs(lookback - 20) / 20.0, "total_pnl": 1000 - 200 * abs(lookback - 20)}

        grid = parameter_grid_robustness(run_fn, {}, {"lookback": [18, 19, 20, 21, 22]})
        assert len(grid) == 5
        acc = plateau_acceptance(grid, metric="total_pnl", threshold_frac=1.0)
        assert acc["accepted"]

    def test_collapse_outside_single_point_rejected(self):
        def run_fn(params):
            lb = params["lookback"]
            # only works at exactly 20
            return {"total_pnl": 1000 if lb == 20 else -500}

        grid = parameter_grid_robustness(run_fn, {}, {"lookback": [19, 20, 21]})
        acc = plateau_acceptance(grid, metric="total_pnl", threshold_frac=0.5)
        assert not acc["accepted"]


class TestAblation:
    def test_run_ablation(self):
        def base():
            return _trades(n=100, seed=1), None

        def full():
            return _trades(n=100, seed=2), None

        res = run_ablation({"base": base, "full": full}, timeframe="4h", start_equity=10_000)
        assert len(res) == 2
        assert res[0].metrics["n_trades"] == 100


class TestCostStress:
    def test_costs_scale_pnl_down(self):
        """Higher costs must monotonically reduce net profit for a fixed trade set."""
        tr = _trades(n=200, seed=3, win_rate=0.6)

        def run_fn(cost: CostModel):
            # apply the cost model's round-trip cost to each trade's pnl
            pnl = tr["pnl_net"].to_numpy() - tr["notional"].to_numpy() * cost.round_trip_cost()
            out = tr.copy()
            out["pnl_net"] = pnl
            return out, None

        rows = cost_stress(run_fn, timeframe="4h", start_equity=100_000)
        assert len(rows) == 4
        pnl = rows["total_pnl"].tolist()
        assert pnl[0] > pnl[1] > pnl[2] > pnl[3]

    def test_verdict_rejects_collapse(self):
        tr = _trades(n=100, seed=4)

        def run_fn(cost: CostModel):
            out = tr.copy()
            out["pnl_net"] = tr["pnl_net"].to_numpy() - tr["notional"].to_numpy() * cost.round_trip_cost() * 10
            return out, None

        rows = cost_stress(run_fn, start_equity=100_000)
        v = cost_stress_verdict(rows)
        assert not v["accepted"]

    def test_verdict_accepts_resilient(self):
        tr = _trades(n=100, seed=5, win_rate=0.9)

        def run_fn(cost: CostModel):
            out = tr.copy()
            out["pnl_net"] = tr["pnl_net"].to_numpy() - tr["notional"].to_numpy() * cost.round_trip_cost()
            return out, None

        rows = cost_stress(run_fn, start_equity=100_000)
        v = cost_stress_verdict(rows, tolerance_frac=0.3)
        assert v["accepted"]
