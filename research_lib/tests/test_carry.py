"""Tests for the funding-rate carry backtest (delta-neutral long spot + short perp)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research_lib.risk.costs import CostModel
from research_lib.strategies.carry import CarryConfig, load_funding, run_carry


def _spot(prices: list[float], start: str = "2024-01-01", freq: str = "1h") -> pd.DataFrame:
    """Flat-close synthetic OHLC frame (open == close == price)."""
    dates = pd.date_range(start, periods=len(prices), freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "date": dates,
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": 1.0,
        }
    )


def _funding(rates: list[float], start: str = "2024-01-01", freq: str = "8h") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(rates), freq=freq, tz="UTC")
    return pd.DataFrame({"date": dates, "funding_rate": rates})


class TestFundingMath:
    def test_flat_price_constant_rate_earns_funding(self):
        spot = _spot([100.0] * 100)  # 100h of spot covers all 10 funding prints
        funding = _funding([0.001] * 10)  # 0.1% every 8h
        res = run_carry(spot, funding, CarryConfig(equity=100_000.0), CostModel())
        # notional = equity / 2 (leverage 1); 9 prints collect (first print is the signal)
        expected_gross = 9 * 0.001 * 50_000.0
        assert res.gross_funding == pytest.approx(expected_gross, rel=1e-6)
        assert res.liquidated is False
        # flat price -> delta pnl zero; net = gross - costs
        assert res.net_pnl == pytest.approx(res.gross_funding - res.total_costs, rel=1e-9)
        assert res.n_funding_prints == 9

    def test_negative_funding_costs_the_short(self):
        spot = _spot([100.0] * 100)  # 100h of spot covers all 10 funding prints
        funding = _funding([-0.0005] * 10)
        res = run_carry(spot, funding, CarryConfig(), CostModel())
        assert res.gross_funding < 0.0
        assert res.net_pnl < 0.0  # funding paid AND costs paid

    def test_costs_are_both_legs_both_ways(self):
        spot = _spot([100.0] * 100)  # 100h of spot covers all 10 funding prints
        funding = _funding([0.0] * 10)
        cost = CostModel(maker_fee=0.001, taker_fee=0.001, spread_bps=0.0, slippage_bps=0.0)
        res = run_carry(spot, funding, CarryConfig(equity=100_000.0), cost)
        # one_way = 0.001; entry 2 legs + exit 2 legs on ~50k notional each way
        assert res.total_costs == pytest.approx(4 * 0.001 * 50_000.0, rel=1e-6)
        assert res.net_pnl == pytest.approx(-res.total_costs, rel=1e-9)


class TestPostOnlyLift:
    def test_maker_fills_are_cheaper_than_taker(self):
        """Assessment step 4: post-only execution must lift net carry."""
        spot = _spot([100.0] * 100)  # 100h of spot covers all 10 funding prints
        funding = _funding([0.0006] * 10)
        cost = CostModel(maker_fee=0.0002, taker_fee=0.0005, spread_bps=2.0, slippage_bps=1.0)
        taker = run_carry(spot, funding, CarryConfig(post_only=False), cost)
        maker = run_carry(spot, funding, CarryConfig(post_only=True), cost)
        assert maker.total_costs < taker.total_costs
        assert maker.net_pnl > taker.net_pnl
        # same gross funding stream either way
        assert maker.gross_funding == pytest.approx(taker.gross_funding, rel=1e-9)


class TestLiquidation:
    def test_adverse_move_liquidates_and_stops_funding(self):
        prices = [100.0] * 30 + [180.0] * 30  # +80% mid-stream
        spot = _spot(prices)
        funding = _funding([0.001] * 20)
        cfg = CarryConfig(equity=100_000.0, leverage=2.0)  # margin = notional/2
        res = run_carry(spot, funding, cfg, CostModel())
        assert res.liquidated is True
        assert res.liquidation_time is not None
        # no funding collected after the liquidation print
        total_after_entry = int((pd.Series(funding["date"]) > pd.Timestamp("2024-01-01 00:00:00", tz="UTC")).sum())
        assert res.n_funding_prints < total_after_entry

    def test_leverage_one_never_liquidates_below_100pct_move(self):
        prices = [100.0] * 30 + [150.0] * 30  # +50%
        spot = _spot(prices)
        funding = _funding([0.001] * 20)
        res = run_carry(spot, funding, CarryConfig(leverage=1.0), CostModel())
        assert res.liquidated is False


class TestProxyAndLoader:
    def test_constant_basis_leaves_only_small_residual(self):
        """A coin-neutral hedge under a constant basis ratio leaves residual =
        coins * spot_move * basis_ratio (correct linear-perp behaviour, and small).
        Real perp data will add genuine basis volatility on top."""
        prices = list(np.linspace(100.0, 130.0, 40))
        spot = _spot(prices)
        funding = _funding([0.0004] * 10)
        res = run_carry(spot, funding, CarryConfig(basis_bps=50.0), CostModel())
        # residual must be tiny relative to the 50k notional (50bps of the spot move)
        assert res.equity_curve["basis_mtm"].abs().max() < 0.01 * 50_000.0

    def test_rehedge_before_liquidation_on_gradual_rise(self):
        """A gradual +100% rise at 1x must re-hedge, not liquidate."""
        prices = list(np.linspace(100.0, 200.0, 500))
        spot = _spot(prices)
        funding = _funding([0.0005] * 200)
        res = run_carry(spot, funding, CarryConfig(leverage=1.0, rebalance=True), CostModel())
        assert res.liquidated is False
        assert res.n_rebalances >= 1
        # funding keeps flowing after the re-hedge
        assert res.gross_funding > 0.0

    def test_no_rehedge_when_disabled(self):
        prices = list(np.linspace(100.0, 250.0, 500))  # +150% - exceeds the 1x buffer
        spot = _spot(prices)
        funding = _funding([0.0005] * 200)
        res = run_carry(spot, funding, CarryConfig(leverage=1.0, rebalance=False), CostModel())
        assert res.liquidated is True

    def test_load_funding_parses_csv(self, tmp_path):
        p = tmp_path / "f.csv"
        p.write_text(
            "date,funding_rate\n2024-01-01 08:00:00+00:00,0.0001\n2024-01-01 00:00:00+00:00,0.0002\n",
            encoding="utf-8",
        )
        df = load_funding(p)
        assert list(df.columns) == ["date", "funding_rate"]
        # sorted ascending, UTC
        assert df["date"].is_monotonic_increasing
        assert str(df["date"].dt.tz) == "UTC"
        assert df["funding_rate"].iloc[0] == pytest.approx(0.0002)

    def test_load_funding_rejects_bad_schema(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("a,b\n1,2\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_funding(p)


class TestRealPerpSeries:
    def test_real_perp_produces_basis_pnl(self):
        """With a real perp series the hedge pnl is no longer ~zero: a widening
        perp premium is a LOSS for the short leg (and vice versa)."""
        n = 400
        spot = _spot([100.0] * n)
        funding = _funding([0.0002] * 120)
        # perp starts at spot level, premium widens to 2%
        perp_prices = list(np.linspace(100.0, 102.0, n))
        perp = _spot(perp_prices)
        res = run_carry(spot, funding, CarryConfig(), CostModel(), perp=perp)
        assert res.used_real_perp is True
        hedge = res.equity_curve["basis_mtm"].iloc[-1]
        assert hedge < 0.0  # premium widening hurt the short leg
        # and the proxy run on the same data has ~zero hedge pnl
        proxy = run_carry(spot, funding, CarryConfig(), CostModel())
        assert abs(proxy.equity_curve["basis_mtm"].iloc[-1]) < abs(hedge)

    def test_perp_premium_narrowing_helps_the_short(self):
        n = 400
        spot = _spot([100.0] * n)
        funding = _funding([0.0002] * 120)
        perp = _spot(list(np.linspace(102.0, 100.0, n)))  # premium narrows from 2%
        res = run_carry(spot, funding, CarryConfig(), CostModel(), perp=perp)
        assert res.equity_curve["basis_mtm"].iloc[-1] > 0.0


class TestQuarterlyReport:
    def test_quarterly_net_sums_to_total(self):
        spot = _spot([100.0] * 24 * 200, freq="1h")  # ~200 days -> spans quarters
        funding = _funding([0.0008] * 600, start="2024-01-01", freq="8h")
        res = run_carry(spot, funding, CarryConfig(), CostModel())
        assert len(res.quarterly) >= 2
        assert res.quarterly["net"].sum() == pytest.approx(res.net_pnl, rel=1e-6)
        assert res.quarterly["funding"].sum() == pytest.approx(res.gross_funding, rel=1e-6)


class TestRealDataSanity:
    def test_btc_carry_matches_assessment_ballpark(self):
        """Spot-check against the assessment's BTC numbers (~6.6% gross annualised)."""
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        funding_path = root / "research" / "data" / "funding" / "BTC_USDT_USDT_funding.csv"
        spot_path = root / "research" / "data" / "BTC_USDT-1h.feather"
        if not funding_path.exists() or not spot_path.exists():
            pytest.skip("research data not available")
        funding = load_funding(funding_path)
        spot = pd.read_feather(spot_path)
        spot["date"] = pd.to_datetime(spot["date"], utc=True)
        res = run_carry(spot, funding, CarryConfig(post_only=True), CostModel())
        # gross annualised on committed equity should be in the same ballpark as the
        # assessment's 6.6% on notional (leverage 1 deploys half the equity per leg)
        assert 0.01 < res.annualised_return < 0.10
        assert res.liquidated is False
