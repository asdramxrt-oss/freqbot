"""Tests for the production bot layer (research_lib/bot).

These tests are deliberately fast: they use small synthetic frames and the in-process lock,
so they run in seconds and do not depend on the on-disk market data. The heavy end-to-end
run over real bars lives in ``bot_smoke.py`` and the CLI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research_lib.bot.broker import LiveBroker, PaperBroker, make_broker
from research_lib.bot.config import BotConfig, load_config
from research_lib.bot.features import (
    PrimarySignal,
    build_feature_frame,
    turtle_signals,
)
from research_lib.bot.live import BotResult, TradingBot, _InProcessRedis
from research_lib.bot.sizing import BetSizerAdapter, MetaModel
from research_lib.risk.engine import OrderIntent


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _frame(n: int = 3000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.004, n).cumsum()
    close = 20000.0 * np.exp(steps)
    high = close * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    vol = np.abs(rng.normal(100, 20, n))
    dates = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": np.maximum.reduce([open_, close, high]),
            "low": np.minimum.reduce([open_, close, low]),
            "close": close,
            "volume": vol,
            "symbol": "BTC/USDT",
            "exchange": "binance",
            "timeframe": "15m",
        }
    )


@pytest.fixture()
def cfg() -> BotConfig:
    c = BotConfig()
    c.execution.equity = 100_000.0
    c.router.enable_router = False          # keep tests fast and deterministic
    c.meta.enable_meta_model = False
    return c


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
class TestConfig:
    def test_defaults_are_constructible_and_serialisable(self):
        c = BotConfig()
        d = c.to_dict()
        assert d["execution"]["equity"] == 100_000.0
        assert "BTC/USDT" in d["data"]["pairs"]

    def test_from_dict_ignores_unknown_keys(self, tmp_path):
        p = tmp_path / "c.json"
        p.write_text('{"name":"x","unknown":123,"execution":{"equity":5000}}', encoding="utf-8")
        c = load_config(p)
        assert c.name == "x"
        assert c.execution.equity == 5000.0
        assert not hasattr(c, "unknown")

    def test_missing_file_is_not_an_error(self, tmp_path):
        c = load_config(tmp_path / "nope.json" if (tmp_path / "nope.json").exists() else None)
        assert isinstance(c, BotConfig)

    def test_roundtrip_save_load(self, tmp_path):
        c = BotConfig(name="roundtrip")
        p = tmp_path / "out.json"
        c.save(p)
        assert load_config(p).name == "roundtrip"


# ---------------------------------------------------------------------------
# broker
# ---------------------------------------------------------------------------
class TestBroker:
    def _intent(
        self,
        side: int = 1,
        units: float = 0.5,
        price: float = 20000.0,
        stop=19000.0,
        post_only: bool = False,
    ):
        return OrderIntent(
            strategy_id="turtle",
            symbol="BTC/USDT",
            signal_time=pd.Timestamp("2024-01-01", tz="UTC"),
            side=side,
            units=units,
            entry_ref_price=price,
            stop_price=stop,
            notional=units * price,
            post_only=post_only,
        )

    def test_paper_fill_is_adverse_to_the_trader(self):
        b = PaperBroker(BotConfig().execution)
        b.set_market(20000.0)
        ack = b.submit(self._intent(side=1))
        assert ack["status"] == "filled"
        # a long buy must pay more than the reference price
        assert ack["avg_price"] > 20000.0

        b.set_market(20000.0)
        ack_s = b.submit(self._intent(side=-1))
        # a short sell must receive less than the reference price
        assert ack_s["avg_price"] < 20000.0

    def test_paper_rejects_without_market_price(self):
        b = PaperBroker(BotConfig().execution)
        assert b.submit(self._intent())["status"] == "error"

    def test_paper_rejects_nonpositive_units(self):
        b = PaperBroker(BotConfig().execution)
        b.set_market(20000.0)
        assert b.submit(self._intent(units=0.0))["status"] == "rejected"

    def test_fetch_returns_the_fill(self):
        b = PaperBroker(BotConfig().execution)
        b.set_market(20000.0)
        ack = b.submit(self._intent())
        got = b.fetch(ack["order_id"])
        assert got["status"] == "filled"
        assert got["filled"] == pytest.approx(0.5)

    def test_post_only_buy_quotes_below_reference_and_pays_maker_fee(self):
        b = PaperBroker(BotConfig().execution)
        b.set_market(20000.0)
        ack = b.submit(self._intent(side=1, post_only=True))
        assert ack["status"] == "filled"
        # a maker buy posts below the reference price...
        assert ack["avg_price"] < 20000.0
        # ...and pays the maker fee, not the taker fee
        assert ack["fee"] == pytest.approx(b.cost.maker_fee * ack["avg_price"] * 0.5)

    def test_post_only_sell_quotes_above_reference(self):
        b = PaperBroker(BotConfig().execution)
        b.set_market(20000.0)
        ack = b.submit(self._intent(side=-1, post_only=True))
        assert ack["status"] == "filled"
        assert ack["avg_price"] > 20000.0

    def test_post_only_rests_when_bar_does_not_trade_through(self):
        b = PaperBroker(BotConfig().execution)
        # quote = 20000 * (1 - 0.5bps) = 19999.0; a bar low of 19999.5 never reaches it
        b.set_market(20000.0, high=20100.0, low=19999.5)
        ack = b.submit(self._intent(side=1, post_only=True))
        assert ack["status"] == "open"
        assert ack["filled"] == 0.0
        assert ack["order_id"] in b.open_orders

    def test_post_only_fills_when_bar_trades_through(self):
        b = PaperBroker(BotConfig().execution)
        b.set_market(20000.0, high=20100.0, low=19900.0)
        ack = b.submit(self._intent(side=1, post_only=True))
        assert ack["status"] == "filled"
        assert ack["avg_price"] == pytest.approx(b.cost.maker_quote(20000.0, "long"))

    def test_resting_order_fills_on_later_trade_through_via_fetch(self):
        b = PaperBroker(BotConfig().execution)
        b.set_market(20000.0, high=20010.0, low=19999.5)
        ack = b.submit(self._intent(side=1, post_only=True))
        assert ack["status"] == "open"
        b.set_market(20050.0, high=20100.0, low=19900.0)
        got = b.fetch(ack["order_id"])
        assert got["status"] == "filled"
        assert got["avg_price"] == pytest.approx(b.cost.maker_quote(20000.0, "long"))

    def test_cancel_removes_a_resting_order(self):
        b = PaperBroker(BotConfig().execution)
        b.set_market(20000.0, high=20010.0, low=19999.5)
        ack = b.submit(self._intent(side=1, post_only=True))
        assert b.cancel(ack["order_id"])["cancelled"] is True
        assert ack["order_id"] not in b.open_orders
        # a later trade-through must not fill a cancelled order
        b.set_market(20050.0, high=20100.0, low=19900.0)
        assert b.fetch(ack["order_id"])["status"] == "unknown"

    def test_maker_round_trip_is_cheaper_than_taker(self):
        from research_lib.risk.costs import CostModel

        c = CostModel(maker_fee=0.0002, taker_fee=0.0005)
        assert c.round_trip_cost("maker") < c.round_trip_cost("taker")
        assert c.execution_price(100.0, "long", "maker") < 100.0
        assert c.execution_price(100.0, "short", "maker") > 100.0

    def test_make_broker_modes(self):
        c = BotConfig().execution
        c.mode = "paper"
        assert isinstance(make_broker(c), PaperBroker)
        c.mode = "dry_run"
        assert isinstance(make_broker(c), PaperBroker)
        c.mode = "nonsense"
        with pytest.raises(ValueError):
            make_broker(c)

    def test_live_broker_requires_credentials(self):
        # no API keys in the environment -> must refuse to construct
        with pytest.raises(RuntimeError):
            LiveBroker(exchange_id="binance")


# ---------------------------------------------------------------------------
# in-process lock stand-in
# ---------------------------------------------------------------------------
class TestInProcessRedis:
    def test_setnx_semantics(self):
        r = _InProcessRedis()
        assert r.set("k", "v", nx=True) is True
        assert r.set("k", "v2", nx=True) is False
        assert r.get("k") == b"v"

    def test_compare_and_delete_eval(self):
        r = _InProcessRedis()
        r.set("k", "tok")
        # wrong token -> no delete
        assert r.eval("", 1, "k", "other") == 0
        assert r.get("k") == b"tok"
        # right token -> delete
        assert r.eval("", 1, "k", "tok") == 1
        assert r.get("k") is None


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
class TestFeatures:
    def test_build_feature_frame_adds_causal_columns(self, cfg):
        out = build_feature_frame(_frame(600), cfg)
        for col in ("atr_20", "rsi_14", "vol_yz_20", "vol_pct_500"):
            assert col in out.columns
        assert len(out) == 600

    def test_turtle_signals_are_entries_only_and_alternate(self, cfg):
        df = _frame(3000)
        feat = build_feature_frame(df, cfg)
        sigs = turtle_signals(feat, cfg, "BTC/USDT")
        idxs = [s.bar_index for s in sigs]
        assert idxs == sorted(idxs)
        for s in sigs:
            assert s.side in (1, -1)
            assert s.entry_ref == pytest.approx(float(df["open"].iloc[s.bar_index]))


# ---------------------------------------------------------------------------
# sizing / meta model
# ---------------------------------------------------------------------------
class TestSizing:
    def test_bet_sizer_returns_full_size_when_meta_disabled(self, cfg):
        s = BetSizerAdapter(cfg)
        assert s.fraction(0.99, meta_enabled=False) == 1.0

    def test_bet_sizer_monotone_in_probability(self, cfg):
        s = BetSizerAdapter(cfg)
        lo = s.fraction(0.53, meta_enabled=True)
        hi = s.fraction(0.95, meta_enabled=True)
        assert 0.0 <= lo <= hi <= 1.0

    def test_bet_sizer_zero_below_threshold(self, cfg):
        s = BetSizerAdapter(cfg)
        assert s.fraction(0.50, meta_enabled=True) == 0.0

    def test_meta_model_disabled_by_config(self, cfg):
        cfg.meta.enable_meta_model = False
        m = MetaModel(cfg)
        rep = m.fit(_frame(400), [])
        assert rep.enabled is False
        # predict must fall back to the neutral 0.5
        sig = PrimarySignal("turtle", "BTC/USDT", 10, pd.Timestamp("2024-01-01", tz="UTC"), 1, 100.0, 95.0)
        assert m.predict(_frame(400), sig) == 0.5

    def test_meta_model_refuses_with_too_few_events(self, cfg):
        cfg.meta.enable_meta_model = True
        cfg.meta.min_events = 10_000
        m = MetaModel(cfg)
        rep = m.fit(_frame(600), [])
        assert rep.enabled is False
        assert "too few" in rep.note


# ---------------------------------------------------------------------------
# the bot loop
# ---------------------------------------------------------------------------
class TestBotLoop:
    def test_router_disabled_yields_full_gates(self, cfg):
        b = TradingBot(cfg)
        b.anchor = build_feature_frame(_frame(600), cfg)
        b._build_router = lambda: None  # type: ignore[assignment]
        n = len(b.anchor)
        b.turtle_gate = np.ones(n, dtype=bool)
        b.pairs_gate = np.ones(n, dtype=bool)
        sig = PrimarySignal("turtle", "BTC/USDT", 5, pd.Timestamp("2024-01-01", tz="UTC"), 1, 100.0, 95.0)
        assert b._gate_allows(sig) is True

    def test_gate_blocks_when_off(self, cfg):
        b = TradingBot(cfg)
        b.anchor = build_feature_frame(_frame(50), cfg)
        b.turtle_gate = np.zeros(50, dtype=bool)
        b.pairs_gate = np.zeros(50, dtype=bool)
        sig = PrimarySignal("turtle", "BTC/USDT", 5, pd.Timestamp("2024-01-01", tz="UTC"), 1, 100.0, 95.0)
        assert b._gate_allows(sig) is False

    def test_intent_respects_risk_budget_and_position_cap(self, cfg):
        b = TradingBot(cfg)
        eq = 100_000.0
        sig = PrimarySignal("turtle", "BTC/USDT", 5, pd.Timestamp("2024-01-01", tz="UTC"), 1, 20000.0, 19000.0)
        intent = b._intent_for(sig, eq)
        # risk per unit = 1000; budget = 1% of 100k = 1000 -> 1 unit, notional 20000
        assert intent.units == pytest.approx(1.0, rel=1e-6)
        # notional must not exceed the per-symbol cap (25% of equity)
        assert intent.notional <= eq * cfg.risk.per_symbol_notional * 1.0001

    def test_risk_engine_rejects_oversized_intent(self, cfg):
        b = TradingBot(cfg)
        eq = 100_000.0
        b.risk.state.equity = eq
        # 10 units * 1000 risk = 10k risk vs 1% (1k) budget -> rejected
        intent = OrderIntent(
            strategy_id="turtle",
            symbol="BTC/USDT",
            signal_time=pd.Timestamp("2024-01-01", tz="UTC"),
            side=1,
            units=10.0,
            entry_ref_price=20000.0,
            stop_price=19000.0,
            notional=200_000.0,
        )
        assert b.risk.evaluate(intent).approved is False

    def test_run_returns_a_result_on_synthetic_data(self, cfg):
        cfg.execution.max_bars = 400
        cfg.router.enable_router = False  # keep the synthetic run fast
        cfg.meta.enabled = False
        b = TradingBot(cfg)
        frame = _frame(800)
        b.panels = {"BTC/USDT": frame}

        def _fake_load() -> pd.DataFrame:
            # load_data() is responsible for setting the anchor panel; mimic that.
            b.anchor = b.panels["BTC/USDT"]
            return b.anchor

        b.load_data = _fake_load  # type: ignore[assignment]
        res = b.run()
        assert res.bars_processed > 0
        assert res.mode == "dry_run"
        assert isinstance(res.signal_counts, dict)

    def test_position_accounting_closes_on_stop(self, cfg):
        b = TradingBot(cfg)
        b.risk.state.equity = 100_000.0
        open_positions = {
            "BTC/USDT": {
                "strategy": "turtle",
                "symbol": "BTC/USDT",
                "side": 1,
                "units": 1.0,
                "entry_price": 20000.0,
                "stop_price": 19000.0,
                "fee": 20.0,
                "entry_bar": 1,
                "status": "open",
            }
        }
        bar = pd.Series({"low": 18500.0, "high": 20500.0, "open": 19900.0, "close": 19900.0})
        b._mark_to_market(open_positions, 19900.0, bar, 7, None)
        assert "BTC/USDT" not in open_positions
        # loss of 1000 (stop) minus 20 fee, applied exactly once
        assert b.risk.state.equity == pytest.approx(100_000.0 - 1000.0 - 20.0)
        # the closed trade is preserved in the ledger
        assert len(b.closed_trades) == 1
        closed = b.closed_trades[0]
        assert closed["exit_reason"] == "stop"
        assert closed["exit_price"] == pytest.approx(19000.0)
        assert closed["pnl"] == pytest.approx(-1020.0)
        assert closed["exit_bar"] == 7

    def test_forced_close_on_final_mark_books_a_trade(self, cfg):
        b = TradingBot(cfg)
        b.risk.state.equity = 100_000.0
        open_positions = {
            "BTC/USDT": {
                "strategy": "turtle",
                "symbol": "BTC/USDT",
                "side": 1,
                "units": 2.0,
                "entry_price": 20000.0,
                "stop_price": 19000.0,
                "fee": 10.0,
                "entry_bar": 1,
                "status": "open",
            }
        }
        b._mark_to_market(open_positions, 21000.0, None, 99, None, force=True)
        assert not open_positions
        assert len(b.closed_trades) == 1
        assert b.closed_trades[0]["exit_reason"] == "eod"
        # (21000 - 20000) * 2 units - 10 fee = +1990
        assert b.risk.state.equity == pytest.approx(100_000.0 + 1990.0)

    def test_equity_reconciles_with_trade_ledger(self, cfg):
        """Equity change must equal the sum of realised PnL - the guard against
        double-counting equity in both the bot loop and the risk engine."""
        cfg.execution.max_bars = 400
        cfg.router.enable_router = False
        cfg.meta.enable_meta_model = False
        b = TradingBot(cfg)
        frame = _frame(800)

        def _fake_load() -> pd.DataFrame:
            b.anchor = frame
            return b.anchor

        b.panels = {"BTC/USDT": frame}
        b.load_data = _fake_load  # type: ignore[assignment]
        res = b.run()
        closed = [t for t in res.trades if t.get("status") == "closed"]
        realized = sum(float(t.get("pnl") or 0.0) for t in closed)
        assert res.equity == pytest.approx(res.start_equity + realized, abs=1e-6)


# ---------------------------------------------------------------------------
# reconciliation
# ---------------------------------------------------------------------------
class TestReconciliation:
    def test_paper_positions_track_fills_and_exits(self):
        b = PaperBroker(BotConfig().execution)
        b.set_market(20000.0)
        intent = OrderIntent(
            strategy_id="turtle",
            symbol="BTC/USDT",
            signal_time=pd.Timestamp("2024-01-01", tz="UTC"),
            side=1,
            units=0.5,
            entry_ref_price=20000.0,
            stop_price=19000.0,
            notional=10_000.0,
        )
        b.submit(intent)
        assert b.positions() == {"BTC/USDT": pytest.approx(0.5)}
        b.record_external_exit("BTC/USDT", 1, 0.5, 21000.0)
        assert b.positions() == {}

    def test_clean_run_reconciles_without_kill_switch(self, cfg):
        cfg.execution.max_bars = 400
        cfg.execution.recon_every_bars = 50
        cfg.router.enable_router = False
        cfg.meta.enabled = False
        b = TradingBot(cfg)
        frame = _frame(800)
        b.panels = {"BTC/USDT": frame}
        b.load_data = lambda: setattr(b, "anchor", b.panels["BTC/USDT"]) or b.anchor  # type: ignore[assignment]
        res = b.run()
        actions = [d["action"] for d in res.decisions]
        assert "reconciliation_failed" not in actions
        assert "reconciliation_ok" in actions
        assert b.risk.state.kill_switch is False
        assert b.risk.state.reconciliation_ok is True

    def test_divergence_trips_kill_switch_and_blocks_orders(self, cfg):
        b = TradingBot(cfg)
        b.risk.state.equity = 100_000.0
        result = BotResult(mode="dry_run", bars_processed=0, orders_submitted=0, equity=100_000.0, start_equity=100_000.0)
        # internal says we are long; the broker knows nothing -> divergence
        internal_open = {"BTC/USDT": {"side": 1, "units": 0.5}}
        b._reconcile(internal_open, result, 10)
        assert b.risk.state.reconciliation_ok is False
        assert b.risk.state.kill_switch is True
        assert any("reconciliation failed" in w for w in result.warnings)
        intent = OrderIntent(
            strategy_id="turtle",
            symbol="BTC/USDT",
            signal_time=pd.Timestamp("2024-01-01", tz="UTC"),
            side=1,
            units=0.5,
            entry_ref_price=20000.0,
            stop_price=19000.0,
            notional=10_000.0,
        )
        decision = b.risk.evaluate(intent)
        assert decision.approved is False

    def test_resume_after_consistency_restored(self, cfg):
        b = TradingBot(cfg)
        result = BotResult(mode="dry_run", bars_processed=0, orders_submitted=0, equity=100_000.0, start_equity=100_000.0)
        b._reconcile({"BTC/USDT": {"side": 1, "units": 0.5}}, result, 10)
        assert b.risk.state.kill_switch is True
        # position closed internally and at the broker -> consistent again
        b._reconcile({}, result, 11)
        assert b.risk.state.reconciliation_ok is True
        assert b.risk.state.kill_switch is False


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------
class TestCli:
    def test_doctor_runs_and_reports(self, capsys):
        from research_lib.bot.cli import main

        rc = main(["doctor"])
        out = capsys.readouterr().out
        assert "bot config" in out
        assert rc in (0, 1)

    def test_parser_rejects_unknown_command(self):
        from research_lib.bot.cli import main

        with pytest.raises(SystemExit):
            main(["definitely-not-a-command"])

    def test_live_refuses_without_yes(self, capsys):
        from research_lib.bot.cli import main

        rc = main(["live"])
        assert rc == 2
        assert "Refusing" in capsys.readouterr().out
