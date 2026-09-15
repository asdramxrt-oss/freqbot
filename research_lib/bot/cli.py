"""Command-line interface for the bot.

    python -m research_lib.bot.cli doctor                  # environment + data check
    python -m research_lib.bot.cli backtest                # replay full history
    python -m research_lib.bot.cli dry-run  --bars 2000    # decision-only run, no orders
    python -m research_lib.bot.cli paper    --bars 2000    # paper fills via the broker
    python -m research_lib.bot.cli live                    # live loop (needs credentials)
    python -m research_lib.bot.cli carry                   # funding-carry: research vs production runner
    python -m research_lib.bot.cli directional             # slow-trend perp: research vs production runner

``dry-run`` = gates + sizing + risk, but the broker is never called with real money
(it is the same PaperBroker, the mode label just makes the intent explicit).
``paper`` = identical path with the paper broker filling at next open.
``live`` = the real ccxt broker; refuses to start without API keys.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from research_lib.bot.config import load_config
from research_lib.bot.live import TradingBot


def _cmd_doctor(args: argparse.Namespace) -> int:
    from research_lib.constants import RESEARCH_DATA_DIR, PROJECT_ROOT

    cfg = load_config(args.config)
    problems: list[str] = []
    ok: list[str] = []

    # data availability
    data_dir = Path(cfg.data.data_dir) if cfg.data.data_dir else RESEARCH_DATA_DIR
    for pair in cfg.data.pairs:
        f = data_dir / f"{pair.replace('/', '_')}-{cfg.data.timeframe}.feather"
        tag = "OK   " if f.exists() else "MISSING"
        line = f"{pair} {cfg.data.timeframe}: {tag} {f}"
        (ok if f.exists() else problems).append(line)
        # funding series (carry strategy input; optional for the directional books)
        ff = data_dir / "funding" / f"{pair.replace('/', '_')}_USDT_funding.csv"
        if ff.exists():
            ok.append(f"{pair} funding: OK    {ff}")
        else:
            problems.append(f"optional: {pair} funding missing ({ff}) - carry unavailable")

    # optional deps
    for mod, purpose in [
        ("hmmlearn", "regime HMM router"),
        ("xgboost", "meta model"),
        ("sklearn", "meta model + calibration"),
        ("fakeredis", "in-process lock fallback"),
        ("ccxt", "live trading only"),
    ]:
        try:
            __import__(mod)
            ok.append(f"import {mod} ({purpose})")
        except ImportError:
            tag = "optional" if mod in ("ccxt",) else "REQUIRED"
            problems.append(f"{tag}: {mod} missing ({purpose})")

    print(f"bot config: {args.config or (PROJECT_ROOT / 'user_data' / 'bot_config.json')}")
    print(f"name={cfg.name} version={cfg.version} mode={cfg.execution.mode}")
    print(f"pairs={cfg.data.pairs} timeframe={cfg.data.timeframe} exchange={cfg.data.exchange}")
    print()
    for line in ok:
        print(f"  OK    {line}")
    for line in problems:
        print(f"  ISSUE {line}")
    return 1 if any(p.startswith(("REQUIRED", "MISSING")) for p in problems) else 0


def _run(args: argparse.Namespace, mode: str) -> int:
    cfg = load_config(args.config)
    cfg.execution.mode = mode
    if args.equity is not None:
        cfg.execution.equity = float(args.equity)
    if args.bars is not None:
        cfg.execution.max_bars = int(args.bars)
    if args.no_router:
        cfg.router.enable_router = False
    if args.no_meta:
        cfg.meta.enable_meta_model = False
    if args.pairs:
        cfg.strategy.pairs_enabled = True

    print(f"[{mode}] preparing bot (this loads data, fits HMM + meta model)...")
    bot = TradingBot(cfg)
    until = None if args.full else (args.bars if args.bars else 5000)
    if until is not None:
        print(f"[{mode}] bounded replay: last {int(until):,} bars (use --full for all history)")
    else:
        print(f"[{mode}] full-history replay")
    result = bot.run(until_bar=until)

    print()
    print(result.summary())
    path = bot.write_report(result, args.report)
    print(f"\nreport written: {path}")
    return 0


def _cmd_backtest(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    cfg.execution.mode = "paper"
    if args.no_router:
        cfg.router.enable_router = False
    if args.no_meta:
        cfg.meta.enable_meta_model = False
    if args.pairs:
        cfg.strategy.pairs_enabled = True
    if args.equity is not None:
        cfg.execution.equity = float(args.equity)

    print("[backtest] replaying full history through the live code path...")
    bot = TradingBot(cfg)
    result = bot.run()
    print()
    print(result.summary())
    path = bot.write_report(result, args.report)
    print(f"\nreport written: {path}")
    return 0


def _cmd_live(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    cfg.execution.mode = "live"
    if not args.yes:
        print(
            "Refusing to start live trading without --yes.\n"
            "Live mode sends real orders to the exchange. Re-run with --yes once you are sure."
        )
        return 2
    print("[live] starting live loop (Ctrl-C to stop)...")
    bot = TradingBot(cfg)
    bot.prepare()
    n = len(bot.anchor) if bot.anchor is not None else 0
    print(f"prepared: {n} bars, {len(bot.signals)} primary signals, meta={bot.meta.report.to_dict()}")
    try:
        result = bot.run(start_bar=n - 1)
        print(result.summary())
        print("report:", bot.write_report(result))
    except KeyboardInterrupt:
        print("\nstopped by user")
    return 0


def _cmd_carry(args: argparse.Namespace) -> int:
    """Run the funding-carry trade two ways and report the divergence:
    the research model (run_carry) vs the production execution chain (CarryRunner).
    The short leg uses the real perp series (research/data/futures/) when available."""
    import pandas as pd

    from research_lib.bot.carry_runner import CarryRunner, CarryRunnerConfig
    from research_lib.constants import RESEARCH_DATA_DIR
    from research_lib.risk.costs import CostModel
    from research_lib.strategies.carry import CarryConfig, load_funding, run_carry

    cfg = load_config(args.config)
    data_dir = Path(cfg.data.data_dir) if cfg.data.data_dir else RESEARCH_DATA_DIR
    symbols = cfg.data.pairs if args.symbol in (None, "all") else [args.symbol]
    rc = 0
    for symbol in symbols:
        rc |= _carry_one(args, cfg, data_dir, symbol, multi=len(symbols) > 1)
    return rc


def _carry_one(args: argparse.Namespace, cfg, data_dir: Path, symbol: str, multi: bool = False) -> int:
    import pandas as pd

    from research_lib.bot.carry_runner import CarryRunner, CarryRunnerConfig
    from research_lib.risk.costs import CostModel
    from research_lib.strategies.carry import CarryConfig, load_funding, run_carry

    spot_path = data_dir / f"{symbol.replace('/', '_')}-{args.timeframe}.feather"
    funding_path = data_dir / "funding" / f"{symbol.replace('/', '_')}_USDT_funding.csv"
    perp_path = data_dir / "futures" / f"{symbol.replace('/', '_')}-1h-perp.feather"
    for p in (spot_path, funding_path):
        if not p.exists():
            print(f"MISSING: {p} - cannot run carry for {symbol}")
            return 1

    spot = pd.read_feather(spot_path)
    funding = load_funding(funding_path)
    perp = pd.read_feather(perp_path) if perp_path.exists() else None
    ec = cfg.execution
    cost = CostModel(
        maker_fee=ec.maker_fee, taker_fee=ec.taker_fee,
        spread_bps=ec.spread_bps, slippage_bps=ec.slippage_bps,
    )
    post_only = not args.taker
    role = "maker" if post_only else "taker"

    runner_bot = CarryRunner(
        CarryRunnerConfig(
            symbol=symbol, equity=ec.equity, leverage=args.leverage, post_only=post_only
        ),
        ec,
    )
    runner = runner_bot.run(spot, funding, perp=perp)
    # compare like-for-like: the research model gets the runner's effective fraction
    # (capped by the risk engine's gross-notional margin check)
    research = run_carry(
        spot, funding,
        CarryConfig(
            equity=ec.equity,
            leverage=args.leverage,
            post_only=post_only,
            notional_fraction=runner_bot._eff_fraction,
        ),
        cost,
        perp=perp,
    )

    div = (
        (runner["net_pnl"] - research.net_pnl) / research.net_pnl
        if research.net_pnl
        else float("nan")
    )
    perp_tag = "real-perp" if perp is not None else "spot-proxy"
    print(f"[carry] {symbol}  role={role}  leverage={args.leverage:.1f}  equity={ec.equity:,.0f}  ({perp_tag})")
    print(f"  research model : net={research.net_pnl:>12,.0f}  ann={research.annualised_return:6.2%}"
          f"  gross={research.gross_funding:>10,.0f}  costs={research.total_costs:>7,.0f}"
          f"  rehedges={research.n_rebalances}  liquidated={research.liquidated}")
    print(f"  production run : net={runner['net_pnl']:>12,.0f}  ann={runner['annualised_return']:6.2%}"
          f"  gross={runner['gross_funding']:>10,.0f}  fees={runner['total_fees']:>7,.0f}"
          f"  rehedges={runner['n_rehedges']}  liquidated={runner['liquidated']}")
    print(f"  live-vs-backtest divergence: {div:+.2%}  (kill_switch={runner['kill_switch']},"
          f" recon_ok={runner['reconciliation_ok']}, book_flat={runner['book_flat']})")

    report = {
        "symbol": symbol,
        "role": role,
        "leverage": args.leverage,
        "used_real_perp": perp is not None,
        "research": research.summary(),
        "runner": {k: v for k, v in runner.items() if k not in ("equity_curve", "log")},
        "divergence_pct": float(div * 100.0),
    }
    if args.report:
        rep_path = Path(args.report)
        if multi:
            rep_path = rep_path.with_name(f"{rep_path.stem}_{symbol.replace('/', '')}{rep_path.suffix}")
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        rep_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"report written: {rep_path}")
    return 0


def _cmd_directional(args: argparse.Namespace) -> int:
    """Run the validated slow-trend variant (ts_trend_4h_d60_sma200_ls) two ways:
    the research EventBacktester vs the production execution chain (DirectionalRunner).
    Futures only: signals and marks come from the perp series, never spot."""
    import pandas as pd

    from research_lib.bot.directional_runner import (
        MARGIN_SAFE_NOTIONAL,
        DirectionalRunner,
        DirectionalRunnerConfig,
        build_trend_signals,
    )
    from research_lib.constants import RESEARCH_DATA_DIR
    from research_lib.risk.backtest_engine import EngineConfig, EventBacktester
    from research_lib.risk.costs import CostModel
    from research_lib.strategies.directional import resample_panel

    cfg = load_config(args.config)
    data_dir = Path(cfg.data.data_dir) if cfg.data.data_dir else RESEARCH_DATA_DIR
    symbols = cfg.data.pairs if args.symbol in (None, "all") else [args.symbol]
    rc = 0
    for symbol in symbols:
        rc |= _directional_one(args, cfg, data_dir, symbol, multi=len(symbols) > 1)
    return rc


def _directional_one(args: argparse.Namespace, cfg, data_dir: Path, symbol: str, multi: bool = False) -> int:
    import pandas as pd

    from research_lib.bot.directional_runner import (
        MARGIN_SAFE_NOTIONAL,
        DirectionalRunner,
        DirectionalRunnerConfig,
        build_trend_signals,
    )
    from research_lib.risk.backtest_engine import EngineConfig, EventBacktester
    from research_lib.risk.costs import CostModel
    from research_lib.strategies.directional import resample_panel

    perp_path = data_dir / "futures" / f"{symbol.replace('/', '_')}-1h-perp.feather"
    if not perp_path.exists():
        print(f"MISSING: {perp_path} - cannot run directional for {symbol}")
        return 1

    perp_1h = pd.read_feather(perp_path)
    df4 = resample_panel(perp_1h, "4h")
    ec = cfg.execution
    cost = CostModel(
        maker_fee=ec.maker_fee, taker_fee=ec.taker_fee,
        spread_bps=ec.spread_bps, slippage_bps=ec.slippage_bps,
    )
    rcfg = DirectionalRunnerConfig(
        symbol=symbol, equity=ec.equity, leverage=args.leverage,
        post_only=False,  # divergence is measured against the taker-cost research model
    )
    target, stop = build_trend_signals(df4, rcfg)

    runner_bot = DirectionalRunner(rcfg, ec)
    runner = runner_bot.run(perp_1h)
    bt = EventBacktester(
        EngineConfig(
            equity=rcfg.equity, cost=cost, risk_per_trade=rcfg.risk_per_trade,
            max_leverage=min(rcfg.leverage, MARGIN_SAFE_NOTIONAL), allow_short=True,
        )
    )
    _trades, eq = bt.run(df4, target, stop=stop)
    research_net = float(eq[-1] - rcfg.equity)

    div = (runner["net_pnl"] - research_net) / research_net if research_net else float("nan")
    role = "taker"
    print(f"[directional] {symbol}  role={role}  leverage={args.leverage:.1f}  equity={rcfg.equity:,.0f}  (real-perp)")
    print(f"  research model : net={research_net:>12,.0f}  trades={len(_trades)}")
    print(f"  production run : net={runner['net_pnl']:>12,.0f}  trades={runner['n_trades']}"
          f"  fees={runner['total_fees']:>7,.0f}  max_dd={runner['max_drawdown']:.1%}")
    print(f"  live-vs-backtest divergence: {div:+.2%}  (kill_switch={runner['kill_switch']},"
          f" recon_ok={runner['reconciliation_ok']}, book_flat={runner['book_flat']})")

    report = {
        "symbol": symbol,
        "role": role,
        "leverage": args.leverage,
        "variant": "ts_trend_4h_d60_sma200_ls",
        "research": {"net_pnl": research_net, "n_trades": len(_trades)},
        "runner": {k: v for k, v in runner.items() if k not in ("equity_curve", "log")},
        "divergence_pct": float(div * 100.0),
    }
    if args.report:
        rep_path = Path(args.report)
        if multi:
            rep_path = rep_path.with_name(f"{rep_path.stem}_{symbol.replace('/', '')}{rep_path.suffix}")
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        rep_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"report written: {rep_path}")
    return 0


def _cmd_router_book(args: argparse.Namespace) -> int:
    """Run config D of the router book two ways and report the divergence:
    the research harness (run_router_book_acceptance.run_book) vs the production
    execution chain (RouterBookRunner). Futures only, all 5 coins, both sleeves."""
    import pandas as pd

    from research_lib.bot.router_book_runner import (
        ROUTER_CFG_4H,
        MR_RATIOS,
        TREND_SYMS,
        RouterBookConfig,
        RouterBookRunner,
    )
    from research_lib.constants import RESEARCH_DATA_DIR
    from research_lib.risk.costs import CostModel
    from research_lib.strategies.unified_router import filtered_hmm_router

    cfg = load_config(args.config)
    data_dir = Path(cfg.data.data_dir) if cfg.data.data_dir else RESEARCH_DATA_DIR
    perp = {}
    for s in TREND_SYMS:
        p = data_dir / "futures" / f"{s}_USDT-1h-perp.feather"
        if not p.exists():
            print(f"MISSING: {p}")
            return 1
        perp[s] = pd.read_feather(p)

    print("[router-book] fitting HMM routers (5 coins, 4h)...")
    import research_lib.strategies.directional as _d

    panels_4h = {s: _d.resample_panel(df, "4h") for s, df in perp.items()}
    routers = {s: filtered_hmm_router(panels_4h[s], ROUTER_CFG_4H, seed=42) for s in TREND_SYMS}

    ec = cfg.execution
    rcfg = RouterBookConfig(equity=ec.equity)
    meta_signals = None
    funding_map = None
    if args.meta_kelly:
        import pickle

        from research_lib.constants import PROJECT_ROOT
        from research_lib.strategies.carry import load_funding

        cache = (
            PROJECT_ROOT / "research" / "artifacts"
            / "pdf_signals_pdf_v4_4h_slowtrend_regfeat.pkl"
        )
        if not cache.exists():
            print(f"MISSING: {cache} - run research/run_pdf_pipeline_acceptance.py first")
            return 1
        meta_signals = pickle.loads(cache.read_bytes())
        if meta_signals.meta_model is None:
            print("cached v4 signals have no fitted meta model - cannot use --meta-kelly")
            return 1
        funding_map = {}
        for s in TREND_SYMS:
            f = data_dir / "funding" / f"{s}_USDT_USDT_funding.csv"
            if f.exists():
                fd = load_funding(f).copy()
                fd["date_ns"] = (
                    (pd.to_datetime(fd["date"], utc=True) - pd.Timestamp("1970-01-01", tz="UTC"))
                    // pd.Timedelta(1, "ns")
                )
                fd["f30"] = fd["funding_rate"].rolling(90, min_periods=30).mean()
                funding_map[s] = fd
        rcfg.meta_kelly = True
        print("[router-book] meta-Kelly ON (pooled panel meta, fractional Kelly sizing)")
    runner_bot = RouterBookRunner(rcfg, routers, ec, meta_signals=meta_signals,
                                  funding_map=funding_map)
    runner = runner_bot.run(perp)

    # research twin: identical config through the research harness
    from research.run_router_book_acceptance import COST, MR_COST, run_book

    research = run_book(
        COST, MR_COST, panels_4h, routers,
        trend_gate="volkill", risk=rcfg.risk_per_trade, trend_share=rcfg.trend_share,
    )
    div = (
        (runner["net_pnl"] - research.net_pnl) / research.net_pnl
        if research.net_pnl else float("nan")
    )
    print(f"[router-book] config D  equity={rcfg.equity:,.0f}  risk={rcfg.risk_per_trade:.0%}"
          f"  trend_share={rcfg.trend_share:.0%}  sleeves=5 trend + {len(MR_RATIOS)} MR")
    print(f"  research model : net={research.net_pnl:>12,.0f}  trades={research.n_trades}"
          f"  max_dd={research.max_drawdown:.1%}")
    print(f"  production run : net={runner['net_pnl']:>12,.0f}  trades={runner['n_trades']}"
          f"  fees={runner['total_fees']:>7,.0f}  max_dd={runner['max_drawdown']:.1%}")
    print(f"  live-vs-backtest divergence: {div:+.2%}  (kill_switch={runner['kill_switch']},"
          f" recon_ok={runner['reconciliation_ok']}, book_flat={runner['book_flat']})")

    if args.report:
        report = {
            "config": "D",
            "research": research.summary(),
            "runner": {k: v for k, v in runner.items() if k not in ("equity_curve", "log", "trades")},
            "divergence_pct": float(div * 100.0),
        }
        rep_path = Path(args.report)
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        rep_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"report written: {rep_path}")
    return 0


def _cmd_router_book_live(args: argparse.Namespace) -> int:
    """Run config D against the live market.

    Default is PAPER mode (PaperBroker, no real money): the loop polls the
    exchange for new closed 1h bars, replays the book to derive target
    positions, and diffs them against the (paper) broker. --live requires
    --yes and API credentials in BINANCE_API_KEY / BINANCE_API_SECRET."""
    import logging
    import os

    from research_lib.bot.router_book_live import ROUTER_REFIT_HOURS, RouterBookLive
    from research_lib.bot.router_book_runner import RouterBookConfig
    from research_lib.constants import RESEARCH_DATA_DIR

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    cfg = load_config(args.config)
    data_dir = Path(cfg.data.data_dir) if cfg.data.data_dir else RESEARCH_DATA_DIR
    ec = cfg.execution

    broker = None
    if args.live:
        if not args.yes:
            print("Refusing live mode without --yes. Live sends REAL orders.")
            return 2
        key = os.environ.get("BINANCE_API_KEY")
        secret = os.environ.get("BINANCE_API_SECRET")
        if not key or not secret:
            print("Refusing live mode: set BINANCE_API_KEY and BINANCE_API_SECRET.")
            return 2
        from research_lib.bot.broker import LiveBroker

        broker = LiveBroker(
            market_type="swap", params={"apiKey": key, "secret": secret}
        )
        print("[router-book-live] LIVE broker - real orders will be sent")
    else:
        print("[router-book-live] PAPER mode (no real orders)")

    rcfg = RouterBookConfig(equity=args.equity or ec.equity)
    bot = RouterBookLive(rcfg, data_dir, ec, broker=broker)
    if args.once:
        rep = bot.cycle()
        print(f"cycle: new_bars={rep.new_bars} new_4h={rep.new_4h_bar} refit={rep.router_refit}")
        print(f"target positions: {rep.target}")
        print(f"deltas: {rep.deltas} filled={rep.orders_filled} failed={rep.orders_failed}")
        print(f"recon_ok={rep.reconciliation_ok} kill_switch={rep.kill_switch}")
        return 0 if rep.reconciliation_ok and not rep.kill_switch else 1
    bot.run(poll_seconds=args.poll_seconds)
    return 0


def main(argv: list[str] | None = None) -> int:
    # --config / --report are accepted both before and after the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=None, help="path to bot_config.json")
    common.add_argument("--report", default=None, help="path to write the JSON run report")

    p = argparse.ArgumentParser(prog="research_lib.bot.cli", description="Research trading bot")
    p.add_argument("--config", default=None, help="path to bot_config.json")
    p.add_argument("--report", default=None, help="path to write the JSON run report")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="check data + dependencies", parents=[common])
    d.set_defaults(func=_cmd_doctor)

    def _common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--equity", type=float, default=None)
        sp.add_argument("--no-router", action="store_true", help="disable regime gating")
        sp.add_argument("--no-meta", action="store_true", help="disable the meta model")
        sp.add_argument("--pairs", action="store_true", help="enable the pairs book")

    b = sub.add_parser("backtest", help="replay the full history", parents=[common])
    b.add_argument("--equity", type=float, default=None)
    b.add_argument("--no-router", action="store_true")
    b.add_argument("--no-meta", action="store_true")
    b.add_argument("--pairs", action="store_true")
    b.set_defaults(func=_cmd_backtest)

    for name, help_text in [
        ("dry-run", "decision-only run (no real orders)"),
        ("paper", "paper run with simulated fills"),
    ]:
        sp = sub.add_parser(name, help=help_text, parents=[common])
        sp.add_argument("--bars", type=int, default=None, help="cap bars processed")
        sp.add_argument("--full", action="store_true", help="ignore --bars, run everything")
        _common(sp)
        sp.set_defaults(func=lambda a, m=name.replace("-", "_"): _run(a, m))

    lv = sub.add_parser("live", help="live trading loop", parents=[common])
    lv.add_argument("--yes", action="store_true", help="confirm real orders")
    lv.set_defaults(func=_cmd_live)

    ca = sub.add_parser("carry", help="funding-carry: research model vs production runner", parents=[common])
    ca.add_argument("--symbol", default=None, help="pair to carry (default: first config pair)")
    ca.add_argument("--timeframe", default="1h", help="spot timeframe for marking (default 1h)")
    ca.add_argument("--leverage", type=float, default=1.0, help="perp leverage (default 1.0)")
    ca.add_argument("--taker", action="store_true", help="use taker fills instead of post-only maker")
    ca.set_defaults(func=_cmd_carry)

    di = sub.add_parser("directional", help="slow-trend perp: research model vs production runner", parents=[common])
    di.add_argument("--symbol", default=None, help="pair to trade (default: config pairs)")
    di.add_argument("--leverage", type=float, default=1.0, help="leverage cap (default 1.0, as validated)")
    di.set_defaults(func=_cmd_directional)

    rb = sub.add_parser("router-book", help="config D router book: research vs production runner", parents=[common])
    rb.add_argument("--meta-kelly", action="store_true",
                    help="enable the PDF bet sizer (fractional Kelly on the pooled panel meta model)")
    rb.set_defaults(func=_cmd_router_book)

    rl = sub.add_parser("router-book-live", help="config D live loop (paper by default)", parents=[common])
    rl.add_argument("--live", action="store_true", help="use the real ccxt broker (needs --yes + API keys)")
    rl.add_argument("--yes", action="store_true", help="confirm real orders")
    rl.add_argument("--once", action="store_true", help="run one poll cycle and exit")
    rl.add_argument("--poll-seconds", type=float, default=300.0, help="poll interval (default 300)")
    rl.add_argument("--equity", type=float, default=None)
    rl.set_defaults(func=_cmd_router_book_live)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
