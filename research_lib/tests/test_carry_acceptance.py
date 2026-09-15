"""Tests for the carry walk-forward acceptance bundle."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research_lib.risk.costs import CostModel
from research_lib.strategies.carry import CarryConfig
from research_lib.validation.carry_acceptance import (
    acceptance_gates,
    evaluate_carry_folds,
    run_carry_acceptance,
)


def _spot(prices, start="2022-01-01", freq="1h"):
    dates = pd.date_range(start, periods=len(prices), freq=freq, tz="UTC")
    return pd.DataFrame(
        {"date": dates, "open": prices, "high": prices, "low": prices, "close": prices, "volume": 1.0}
    )


def _funding(rates, start="2022-01-01", freq="8h"):
    dates = pd.date_range(start, periods=len(rates), freq=freq, tz="UTC")
    return pd.DataFrame({"date": dates, "funding_rate": rates})


def _flat_world(n_bars=24500, n_prints=3000, rate=0.0006):
    """~1000 days of flat prices with steady positive funding (spot covers all prints)."""
    return _spot([100.0] * n_bars), _funding([rate] * n_prints)


class TestFolds:
    def test_folds_cover_the_full_print_range(self):
        spot, funding = _flat_world()
        folds = evaluate_carry_folds(spot, funding, CarryConfig(), CostModel(), n_folds=6)
        assert len(folds) == 6
        assert folds["n_prints"].sum() > 2500
        assert folds["start"].is_monotonic_increasing

    def test_every_fold_positive_when_funding_positive(self):
        spot, funding = _flat_world()
        folds = evaluate_carry_folds(spot, funding, CarryConfig(), CostModel(), n_folds=6)
        assert (folds["net_pnl"] > 0).all()

    def test_too_little_data_returns_empty(self):
        spot = _spot([100.0] * 50)
        funding = _funding([0.0006] * 6)
        folds = evaluate_carry_folds(spot, funding, CarryConfig(), CostModel(), n_folds=6)
        assert folds.empty


class TestGates:
    def test_clean_world_passes_all_gates(self):
        spot, funding = _flat_world()
        acc = run_carry_acceptance("TEST/USDT", spot, funding, CarryConfig(), CostModel(), n_folds=6)
        assert acc.verdict == "PASS", acc.summary()
        assert all(g.passed for g in acc.gates)

    def test_one_negative_fold_fails_walk_forward_gate(self):
        spot, funding = _flat_world()
        # make one fold deeply negative (fold 3 of 6)
        rates = funding["funding_rate"].to_numpy().copy()
        rates[1000:1500] = -0.001
        funding = funding.copy()
        funding["funding_rate"] = rates
        acc = run_carry_acceptance("TEST/USDT", spot, funding, CarryConfig(), CostModel(), n_folds=6)
        g = {x.gate_id: x for x in acc.gates}
        assert g[2].passed is False
        assert acc.verdict == "FAIL"

    def test_extreme_costs_fail_cost_stress_gate(self):
        """Gate 3 logic, unit-tested directly: stress results that go negative must fail."""
        from types import SimpleNamespace

        folds = pd.DataFrame(
            {"fold": range(6), "n_prints": [500] * 6, "net_pnl": [100.0] * 6}
        )
        full = SimpleNamespace(liquidated=False, equity_curve=pd.DataFrame(), final_equity=100_000.0)
        stress = {1.5: SimpleNamespace(net_pnl=50.0), 2.0: SimpleNamespace(net_pnl=-10.0)}
        gates = {g.gate_id: g for g in acceptance_gates(full, folds, stress)}
        assert gates[3].passed is False
        # other gates are unaffected by the stress failure
        assert gates[2].passed is True
        # and a healthy stress profile passes
        stress_ok = {1.5: SimpleNamespace(net_pnl=50.0), 2.0: SimpleNamespace(net_pnl=10.0)}
        gates_ok = {g.gate_id: g for g in acceptance_gates(full, folds, stress_ok)}
        assert gates_ok[3].passed is True

    def test_liquidation_fails_tail_gate(self):
        # violent +200% rally liquidates the short leg even with re-hedging jumps
        prices = [100.0] * 4000 + [300.0] * 5000
        spot = _spot(prices)
        funding = _funding([0.0006] * 3000)
        cfg = CarryConfig(leverage=1.0, rebalance=False)
        acc = run_carry_acceptance("TEST/USDT", spot, funding, cfg, CostModel(), n_folds=6)
        g = {x.gate_id: x for x in acc.gates}
        assert g[5].passed is False
        assert acc.verdict == "FAIL"

    def test_insufficient_data_fails_gate_one(self):
        spot = _spot([100.0] * 2000)
        funding = _funding([0.0006] * 100)  # way below min_prints
        acc = run_carry_acceptance("TEST/USDT", spot, funding, CarryConfig(), CostModel(), n_folds=6)
        g = {x.gate_id: x for x in acc.gates}
        assert g[1].passed is False
        assert acc.verdict == "FAIL"


class TestRealData:
    def test_btc_carry_passes_acceptance(self):
        """The assessment's headline claim, mechanically re-verified."""
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        funding_path = root / "research" / "data" / "funding" / "BTC_USDT_USDT_funding.csv"
        spot_path = root / "research" / "data" / "BTC_USDT-1h.feather"
        if not funding_path.exists() or not spot_path.exists():
            pytest.skip("research data not available")
        from research_lib.strategies.carry import load_funding

        funding = load_funding(funding_path)
        spot = pd.read_feather(spot_path)
        cost = CostModel(maker_fee=0.0002, taker_fee=0.0005, spread_bps=2.0, slippage_bps=1.0)
        acc = run_carry_acceptance("BTC/USDT", spot, funding, CarryConfig(post_only=True), cost, n_folds=6)
        assert acc.verdict == "PASS", acc.summary()
