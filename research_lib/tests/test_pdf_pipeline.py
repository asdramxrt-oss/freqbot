"""Tests for the PDF-architecture futures pipeline glue (fast paths only).

The slow stages (filtered HMM router, meta-model fit) are exercised indirectly via
the acceptance runner; here we pin the pure glue: entry-event extraction, panel
aggregation with stop passthrough, and futures-only data handling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research_lib.risk.costs import CostModel
from research_lib.strategies.pdf_pipeline import (
    _REGIMES,
    PDFCoinResult,
    PDFPanelSignals,
    _entry_events,
    _v2_event_row,
    run_panel_pdf,
)

COST = CostModel(maker_fee=0.0002, taker_fee=0.0005, spread_bps=2.0, slippage_bps=1.0)


def _frame(prices, start="2023-01-01", freq="1h"):
    dates = pd.date_range(start, periods=len(prices), freq=freq, tz="UTC")
    return pd.DataFrame(
        {"date": dates, "open": prices, "high": prices, "low": prices, "close": prices, "volume": 1.0}
    )


def _signals(panels, targets, stops=None):
    sigs = PDFPanelSignals()
    for sym, tgt in targets.items():
        n = len(tgt)
        sigs.per_coin[sym] = PDFCoinResult(
            symbol=sym,
            target=np.asarray(tgt, dtype=int),
            stop=np.asarray(stops[sym], dtype=float) if stops else np.full(n, np.nan),
            n_primary_entries=0,
            n_routed_entries=0,
            n_meta_suppressed=0,
            meta_enabled=False,
            meta_note="test",
        )
    return sigs


class TestEntryEvents:
    def test_flat_to_position_transitions_only(self):
        df = _frame([100.0] * 10)
        target = np.array([0, 1, 1, 0, -1, -1, 0, 1, 0, 0])
        stop = np.full(10, 90.0)
        events = _entry_events(target, stop, df, "T")
        assert [(e.bar_index, e.side) for e in events] == [(1, 1), (4, -1), (7, 1)]
        assert all(e.strategy_id == "turtle" for e in events)
        # entry ref is the fill bar's open; signal time is the previous bar
        assert events[0].entry_ref == 100.0
        assert events[0].signal_time == pd.Timestamp(df["date"].iloc[0])

    def test_stop_propagated_when_finite(self):
        df = _frame([100.0] * 5)
        target = np.array([0, 1, 1, 1, 0])
        stop = np.array([np.nan, 95.0, 95.0, 95.0, np.nan])
        (event,) = _entry_events(target, stop, df, "T")
        assert event.stop_price == 95.0


class TestPanelPdf:
    def test_uptrend_always_long_is_profitable(self):
        prices = list(np.linspace(100.0, 200.0, 500))
        panels = {s: _frame(prices) for s in ("A", "B")}
        targets = {s: np.ones(500, dtype=int) for s in panels}
        res = run_panel_pdf("t", panels, _signals(panels, targets), COST, equity=10_000.0)
        assert res.net_pnl > 0.0
        assert set(res.trades["symbol"].unique()) == {"A", "B"}
        assert res.dates is not None and len(res.dates) == 500

    def test_stop_passthrough_exits_at_stop_with_risk_budget(self):
        # sharp reversal against a held long: the explicit stop must cut the trade
        # early and keep the loss near the 1% risk budget (risk-parity sizing means
        # a tighter stop scales units up, so the invariant is the exit, not a
        # smaller absolute loss)
        prices = [100.0] * 10 + list(np.linspace(100.0, 50.0, 90))
        panels = {"A": _frame(prices)}
        target = np.array([0] + [1] * 99)
        res = run_panel_pdf(
            "t", panels, _signals(panels, {"A": target}, stops={"A": np.full(100, 99.0)}),
            COST, equity=10_000.0,
        )
        assert res.n_trades == 1
        trade = res.trades.iloc[0]
        assert trade["exit_reason"] == "stop"
        assert trade["bars_held"] < 20          # cut early, not ridden to the end
        assert res.net_pnl > -0.025 * 10_000    # ~1% risk budget + costs, not -50%

    def test_shorts_allowed_and_pnl_linear(self):
        prices = list(np.linspace(100.0, 50.0, 200))
        panels = {"A": _frame(prices)}
        target = np.array([0] + [-1] * 199)
        res = run_panel_pdf("t", panels, _signals(panels, {"A": target}), COST, equity=10_000.0)
        assert res.net_pnl > 0.0  # linear perp short profits from the decline


class TestV2EventRow:
    def _bar(self):
        return pd.Series(
            {"close": 100.0, "atr_20": 2.0, "rsi_14": 60.0, "vol_pct_500": 0.7, "hurst_256": 0.55}
        )

    def test_panelmeta_shape_and_coin_onehot(self):
        row = _v2_event_row(self._bar(), 101.0, 1, sym_idx=2, n_syms=5, regime=None, p_spike=None)
        assert row is not None and row.shape == (5 + 5,)
        assert np.isfinite(row).all()
        assert row[5:].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]

    def test_regfeat_appends_regime_onehot_and_pspike(self):
        row = _v2_event_row(
            self._bar(), 101.0, 1, sym_idx=0, n_syms=5, regime="TREND", p_spike=0.3
        )
        assert row is not None and row.shape == (5 + 5 + len(_REGIMES) + 1,)
        reg_oh = row[10:10 + len(_REGIMES)]
        assert reg_oh.tolist() == [1.0] + [0.0] * (len(_REGIMES) - 1)
        assert row[-1] == pytest.approx(0.3)

    def test_side_sign_flips_dist_feature(self):
        long_row = _v2_event_row(self._bar(), 101.0, 1, 0, 5, None, None)
        short_row = _v2_event_row(self._bar(), 101.0, -1, 0, 5, None, None)
        assert long_row[4] == pytest.approx(-short_row[4])

    def test_nan_pspike_defaults_to_half(self):
        row = _v2_event_row(self._bar(), 100.0, 1, 0, 5, "NEUTRAL", float("nan"))
        assert row[-1] == pytest.approx(0.5)

    def test_bad_bar_returns_none(self):
        bad = pd.Series({"close": np.nan, "atr_20": 2.0})
        assert _v2_event_row(bad, 100.0, 1, 0, 5, None, None) is None
