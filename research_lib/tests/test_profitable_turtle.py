"""Unit tests for the Profitable Turtle Strategy (Turtle + Vol-spike filter)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research_lib.strategies.profitable_turtle import ProfitableTurtleParams, ProfitableTurtleEngine
from research_lib.risk.backtest_engine import EventBacktester, EngineConfig
from research_lib.risk.costs import CostModel

COST = CostModel(maker_fee=0.0002, taker_fee=0.0005, spread_bps=2.0, slippage_bps=1.0)


def _frame(prices, start="2022-01-01", freq="1h"):
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


class TestProfitableTurtleEngine:
    def test_basic_generation(self):
        """Test that the engine can generate basic trading signals."""
        df = _frame([100.0] * 100 + [110.0] * 50 + [90.0] * 50, freq="1h")

        params = ProfitableTurtleParams()
        engine = ProfitableTurtleEngine(params)
        signals_df, pyramid_log = engine.generate(df)

        assert len(signals_df) == len(df)
        assert "target" in signals_df.columns
        assert "stop" in signals_df.columns
        assert isinstance(pyramid_log, list)

    def test_trend_breakout(self):
        """Test that Turtle trend breakouts are detected."""
        prices = [100.0] * 20 + [120.0] * 30 + [110.0] * 50
        df = _frame(prices, freq="1h")

        params = ProfitableTurtleParams(entry_n=20, exit_lookback=10)
        engine = ProfitableTurtleEngine(params)
        signals_df, _ = engine.generate(df)

        # Expect a long position after breakout
        assert signals_df["target"].iloc[25] == 1.0

    def test_exit_on_opposite_breakout(self):
        """Test that positions exit when opposite breakout occurs."""
        prices = [100.0] * 20 + [120.0] * 30 + [95.0] * 50
        df = _frame(prices, freq="1h")

        params = ProfitableTurtleParams(entry_n=20, exit_lookback=10, exit_on_opposite_breakout=True)
        engine = ProfitableTurtleEngine(params)
        signals_df, _ = engine.generate(df)

        # Should exit when price breaks below Donchian lower
        assert signals_df["target"].iloc[-1] == 0.0

    def test_volspike_filter_disabled(self):
        """Test that vol-spike filter can be disabled."""
        prices = [100.0] * 20 + [120.0] * 30 + [95.0] * 50
        df = _frame(prices, freq="1h")

        params = ProfitableTurtleParams(
            entry_n=20, exit_lookback=10, enable_volspike_filter=False
        )
        engine = ProfitableTurtleEngine(params)
        signals_df, _ = engine.generate(df)

        # Should still trade without vol-spike filter
        assert signals_df["target"].iloc[25] == 1.0

    def test_sma_regime_filter(self):
        """Test that SMA regime filter works correctly."""
        prices = [100.0] * 50 + [80.0] * 50  # Clear trend down
        df = _frame(prices, freq="1h")

        params = ProfitableTurtleParams(
            entry_n=20, exit_lookback=10, regime_sma_window=20
        )
        engine = ProfitableTurtleEngine(params)
        signals_df, _ = engine.generate(df)

        # After SMA drop, longs should be suppressed (long_only_above mode)
        assert signals_df["target"].iloc[80] == 0.0

    def test_pyramiding_state(self):
        """Test pyramiding state machine."""
        prices = [100.0] * 20 + [105.0] * 50  # Strong uptrend
        df = _frame(prices, freq="1h")

        params = ProfitableTurtleParams(
            entry_n=20, exit_lookback=10, pyramid_max_units=4
        )
        engine = ProfitableTurtleEngine(params)
        signals_df, pyramid_log = engine.generate(df)

        # Check that pyramiding events are logged
        adds = [log for log in pyramid_log if log["action"] == "add"]
        assert len(adds) > 0

    def test_causality_invariance(self):
        """Test that signals are causally invariant (no future look-ahead)."""
        rng = np.random.default_rng(7)
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 500)))
        full_df = _frame(prices, freq="1h")

        params = ProfitableTurtleParams()
        engine = ProfitableTurtleEngine(params)
        full_signals, _ = engine.generate(full_df)

        # Test on shorter frames - should be identical in the shared region
        for cut in (150, 250, 350):
            partial_df = full_df.iloc[:cut].reset_index(drop=True)
            partial_signals, _ = engine.generate(partial_df)

            assert (full_signals.iloc[:cut]["target"].to_numpy() == partial_signals["target"].to_numpy()).all()

    def test_system_1_vs_system_2(self):
        """Test that system 2 uses different entry lookback (55)."""
        prices = [100.0] * 20 + [120.0] * 30 + [95.0] * 50
        df = _frame(prices, freq="1h")

        # System 1 - 20-bar
        params1 = ProfitableTurtleParams(system=1, entry_n=20)
        engine1 = ProfitableTurtleEngine(params1)
        sig1, _ = engine1.generate(df)

        # System 2 - 55-bar
        params2 = ProfitableTurtleParams(system=2)
        engine2 = ProfitableTurtleEngine(params2)
        sig2, _ = engine2.generate(df)

        # System 2 should have longer lookback (delayed signals)
        assert sig2["target"].iloc[25] == 0.0
        assert sig1["target"].iloc[25] == 1.0

    def test_comprehensive_integration(self):
        """Test the full profitable turtle strategy with all components."""
        # Create a scenario with Turtle trend and vol-spike filter
        rng = np.random.default_rng(42)

        # Generate prices with trend and some volatility
        prices = []
        for i in range(300):
            if i < 80:
                prices.append(100.0)  # flat
            elif i < 180:
                prices.append(100.0 * (1.001 ** i))  # uptrend
            elif i < 280:
                prices.append(120.0 * (0.998 ** (i - 180)))  # downtrend
            else:
                prices.append(95.0 + rng.normal(0, 2.0))  # volatile

        df = _frame(prices, freq="1h")

        # Use default profitable turtle parameters
        params = ProfitableTurtleParams()
        engine = ProfitableTurtleEngine(params)
        signals_df, pyramid_log = engine.generate(df)

        # Basic integration checks
        assert len(signals_df) == len(df)
        assert "target" in signals_df.columns
        assert "stop" in signals_df.columns
        assert isinstance(pyramid_log, list)

        # Should have some trading activity
        assert signals_df["target"].abs().sum() > 0

    def test_backward_compatibility(self):
        """Test that TurtleEngine alias still works."""
        from research_lib.strategies.profitable_turtle import TurtleEngine

        prices = [100.0] * 20 + [120.0] * 30 + [95.0] * 50
        df = _frame(prices, freq="1h")

        params = ProfitableTurtleParams()
        engine = TurtleEngine(params)
        signals_df, _ = engine.generate(df)

        # Should work exactly the same as ProfitableTurtleEngine
        assert signals_df["target"].iloc[25] == 1.0


class TestProfitableTurtleIntegrationWithBacktester:
    def test_integration_with_event_backtester(self):
        """Test that the strategy works with the EventBacktester."""
        prices = [100.0] * 30 + [110.0] * 40 + [95.0] * 30
        df = _frame(prices, freq="1h")

        params = ProfitableTurtleParams()
        engine = ProfitableTurtleEngine(params)
        signals_df, _ = engine.generate(df)

        # Run with EventBacktester
        config = EngineConfig(equity=100_000.0, cost=COST, risk_per_trade=0.01, allow_short=True)
        bt = EventBacktester(config)

        trades, equity_curve = bt.run(df, signals_df["target"].to_numpy())

        # Basic validation
        assert isinstance(trades, pd.DataFrame) or len(trades) == 0
        assert len(equity_curve) == len(df)

    def test_cost_aware_simulation(self):
        """Test that the strategy accounts for trading costs."""
        prices = [100.0] * 30 + [110.0] * 40 + [95.0] * 30
        df = _frame(prices, freq="1h")

        params = ProfitableTurtleParams()
        engine = ProfitableTurtleEngine(params)
        signals_df, _ = engine.generate(df)

        # Run with different cost models to verify cost awareness
        for maker_fee in [0.0001, 0.0002, 0.0005]:
            for taker_fee in [0.0002, 0.0005, 0.001]:
                cost = CostModel(maker_fee=maker_fee, taker_fee=taker_fee, spread_bps=2.0, slippage_bps=1.0)
                config = EngineConfig(equity=100_000.0, cost=cost, risk_per_trade=0.01, allow_short=True)
                bt = EventBacktester(config)
                trades, equity_curve = bt.run(df, signals_df["target"].to_numpy())

                # Should run without errors and produce valid output
                assert len(equity_curve) == len(df)
    def test_integration_with_event_backtester(self):
        """Test that the strategy works with the EventBacktester."""
        from research_lib.risk.backtest_engine import EventBacktester, EngineConfig

        prices = [100.0] * 30 + [110.0] * 40 + [95.0] * 30
        df = _frame(prices, freq="1h")

        params = ProfitableTurtleParams()
        engine = ProfitableTurtleEngine(params)
        signals_df, _ = engine.generate(df)

        # Run with EventBacktester
        config = EngineConfig(equity=100_000.0, cost=COST, risk_per_trade=0.01, allow_short=True)
        bt = EventBacktester(config)

        trades, equity_curve = bt.run(df, signals_df["target"].to_numpy())

        # Basic validation
        assert isinstance(trades, pd.DataFrame) or len(trades) == 0
        assert len(equity_curve) == len(df)

    def test_cost_aware_simulation(self):
        """Test that the strategy accounts for trading costs."""
        prices = [100.0] * 30 + [110.0] * 40 + [95.0] * 30
        df = _frame(prices, freq="1h")

        params = ProfitableTurtleParams()
        engine = ProfitableTurtleEngine(params)
        signals_df, _ = engine.generate(df)

        # Run with different cost models to verify cost awareness
        for maker_fee in [0.0001, 0.0002, 0.0005]:
            for taker_fee in [0.0002, 0.0005, 0.001]:
                cost = CostModel(maker_fee=maker_fee, taker_fee=taker_fee, spread_bps=2.0, slippage_bps=1.0)
                config = EngineConfig(equity=100_000.0, cost=cost, risk_per_trade=0.01, allow_short=True)
                bt = EventBacktester(config)
                trades, equity_curve = bt.run(df, signals_df["target"].to_numpy())

                # Should run without errors and produce valid output
                assert len(equity_curve) == len(df)