# The Trading Bot — operator's guide

This package (`research_lib/bot/`) is the **production layer** that wires the
`research_lib` research stack into a runnable bot. The research stack implements
the architecture in `Untitled document.pdf`; this layer adds the things a research
library deliberately does not have: a config file, a broker, a live loop, and a CLI.

```
config.json ──> TradingBot ──> Risk ──> Execution ──> Broker ──> (Redis lock)
                   │
                   ├── data: loader ──> information bars ──> feature engine
                   ├── router: walk-forward HMM + vol-spike kill switch
                   ├── primaries: Turtle / OU / Pairs
                   └── meta: triple-barrier ──> meta model ──> bet sizer
```

---

## 1. Read this first: what this bot can and cannot do

The research that produced this stack ran 26 phases of validation and reached a
blunt conclusion, which the bot does **not** hide:

- **There is no validated out-of-sample directional edge.** Neither the Turtle
  breakout book, the OU/mean-reversion book, nor the pairs book produced a
  persistent OOS edge after realistic costs. Backtests that look good are
  within the noise band.
- **The one validated capability is volatility-spike prediction** (AUC ≈ 0.82–0.90).
  The bot therefore uses that model as a **kill switch** — it can turn risk *off* —
  never as a direction picker.
- **ML never picks direction.** This is architectural, not stylistic. Direction
  comes from Turtle/OU/Pairs. The regime classifier decides *which* book may
  trade; the meta model only *gates and sizes*. The meta model self-disables
  when its OOS AUC is not comfortably above chance.

Concretely, on the sample window shipped with the repo the meta model measured
**xgb AUC 0.461 / rf AUC 0.427** — both *below* chance — and the bot correctly
reported `enabled: false` and passed primaries at full size without pretending.

> Treat this as a research instrument and an execution harness. Do not point it
> at an account you are not prepared to lose. See §7 for the safety guarantees.

---

## 2. Install

The repo ships a virtualenv with everything except the optional live-trading dep:

```bash
cd freqtrade-develop/freqtrade-develop

# core + research deps are already installed in .venv
.venv/Scripts/python.exe -m research_lib.bot.cli doctor      # Windows
# .venv/bin/python -m research_lib.bot.cli doctor            # macOS / Linux

# only needed for live mode (talks to a real exchange)
.venv/Scripts/python.exe -m pip install ccxt
```

`doctor` checks your data files and every optional dependency and prints a
green/red report. If it says `MISSING` for a pair, download that data first
(§3).

---

## 3. Data

The bot reads feather files from `research/data/`. The expected filename is
`<PAIR with _ instead of />-<timeframe>.feather`, e.g. `BTC_USDT-15m.feather`.

The repo already contains BTC/ETH/SOL/BNB at 15m/1h/4h. To use other data, either
place a matching feather file in `research/data/`, or point `data.data_dir` at
your own directory in the config.

Required columns: `date, open, high, low, close, volume`.

---

## 4. Configure

Config lives in `user_data/bot_config.json`. Every field is optional — missing
fields fall back to dataclass defaults, and unknown fields are ignored, so a
partial file is valid.

```jsonc
{
  "name": "research_bot",
  "version": "v1",

  "data": {
    "pairs": ["BTC/USDT"],
    "timeframe": "15m",
    "exchange": "binance",
    "warmup_bars": 5000            // history required before trading
    // "use_information_bars": true, "bar_type": "volume"   // optional
  },

  "router": {
    "enable_router": true,
    "hmm_states": 3,
    "hmm_refit_bars": 1344,         // how often the HMM is refit (walk-forward)
    "hmm_min_train_bars": 4000,     // training window floor
    "turtle_regimes": ["TREND"],    // regimes in which the Turtle book may trade
    "vol_kill_p": 0.8               // vol-spike probability that halts trading
  },

  "meta": {
    "enable_meta_model": true,
    "min_events": 120,
    "min_test_auc": 0.55            // EVERY model must clear this or meta disables
  },

  "risk": {
    "per_trade_risk": 0.01,         // 1% of equity at risk per trade
    "max_leverage": 1.0,
    "daily_loss_limit": 0.03,
    "max_drawdown_kill": 0.20
  },

  "execution": {
    "mode": "dry_run",              // dry_run | paper | live
    "equity": 100000.0,
    "maker_fee": 0.0002,
    "taker_fee": 0.0004,
    "spread_bps": 1.0,
    "slippage_bps": 0.5
    // "redis_url": "redis://localhost:6379/0"   // optional, for real locking
  }
}
```

### Choosing a run length

The HMM router refits walk-forward and dominates runtime — roughly **70 s per
6,000 bars**. Use `--bars N` for iteration and `--full` only when you mean it.

---

## 5. Run it

```bash
# 1. environment + data check
.venv/Scripts/python.exe -m research_lib.bot.cli doctor

# 2. decision-only replay over the last 5,000 bars (default)
.venv/Scripts/python.exe -m research_lib.bot.cli dry-run --bars 3000

# 3. paper trading: same path, broker fills at next bar open with costs
.venv/Scripts/python.exe -m research_lib.bot.cli paper --bars 6000 \
    --report user_data/bot_runs/paper.json

# 4. full history
.venv/Scripts/python.exe -m research_lib.bot.cli backtest

# 5. live (refuses to start without --yes)
.venv/Scripts/python.exe -m research_lib.bot.cli live --yes
```

Flags: `--config PATH`, `--report PATH`, `--equity N`, `--bars N`, `--full`,
`--no-router`, `--no-meta`, `--pairs`, `--yes` (live only).

`--no-router` / `--no-meta` are the fastest way to isolate which stage is
responsible for a result.

### Sample output

```
[paper] bounded replay: last 6,000 bars (use --full for all history)

mode=paper bars=6000
signals={'turtle': 203}
orders submitted=2 rejected=37
trades closed=2 open=0 win_rate=0.0% realized_pnl=-243.24
equity 100,000.00 -> 99,756.76 (-0.24%)
router={'enabled': True, 'regime_counts': {'HIGH_VOL': 4032, 'UNCERTAIN': 4025,
        'TREND': 1943, 'NEUTRAL': 1344}, 'turtle_gate_on': 1943,
        'pairs_gate_on': 1344, 'killed_by_vol_spike': 4}
meta={'enabled': False, ..., 'test_auc': 0.461, 'base_rate': 0.5,
      'note': "meta model has no reliable OOS edge (best AUC 0.461 < 0.55, ...)"}
```

Note what the 203 signals became: **2 fills, 37 rejections**. The 37 rejections
are the risk engine doing its job (strategy concentration), not a bug.

---

## 6. Also runnable via freqtrade

`user_data/strategies/ResearchTurtleRouter.py` wraps the same decision stack for
native freqtrade tooling:

```bash
.venv/Scripts/freqtrade backtesting --strategy ResearchTurtleRouter \
    --config user_data/bot_config.json
```

It imports `research_lib` rather than reimplementing it, so the Turtle entry,
regime gate, and ATR stop have exactly one definition. If the gate is off for
every bar it logs a warning explaining which regimes were actually seen.

---

## 7. Files and safety guarantees

| File | Role |
|---|---|
| `config.py` | `BotConfig` + `load_config()`; partial configs valid |
| `broker.py` | `PaperBroker` (next-open fills, adverse pricing), `LiveBroker` (ccxt) |
| `features.py` | feature frame + `PrimarySignal` + Turtle/OU/Pairs signal builders |
| `sizing.py` | triple-barrier → meta model → `BetSizerAdapter` |
| `live.py` | `TradingBot`, `BotResult`, `run_bot()` |
| `cli.py` | the `doctor / backtest / dry-run / paper / live` commands |

**Safety properties the code enforces:**

- **Causality.** Gates consulted at bar `t` use state from `t-1` or earlier.
  The router calls `predict_proba` on a *single row* on purpose — batching it
  into one block call would apply forward-backward smoothing and leak the
  future (measured: only ~36% argmax agreement with the causal form). There is
  an explicit warning comment in `unified_router.py`; do not "optimise" it.
- **One risk choke point.** Every order passes `CentralRiskEngine.evaluate()`
  before execution. Per-trade, per-symbol, portfolio, leverage, daily/weekly
  loss, drawdown, and concentration limits all live there.
- **Equity is owned by the risk engine.** `on_position_closed()` is the only
  place equity moves on a close. The bot loop never touches it (doing so
  double-counts PnL — a bug this repo hit and now guards with a test).
- **Live mode is opt-in twice.** `mode: "live"` *and* `--yes`, and `LiveBroker`
  raises if API credentials are absent. It never fabricates a fill.
- **Idempotency + locking.** Orders carry an idempotency key, tracked in an
  order-state store; a Redis distributed lock (Lua compare-and-delete) guards
  the loop, with an in-process fallback when Redis is not configured.

---

## 8. Tests

```bash
.venv/Scripts/python.exe -m pytest research_lib/tests -c research_lib/pytest.ini -q
```

170 tests, including ~30 covering this bot layer: config loading, broker fills,
the lock, feature/signal generation, meta-model self-disabling, the equity
reconciliation invariant, and the CLI surface.

---

## 9. Credentials (live mode only)

Do **not** commit keys. Set them in the environment and let the broker read them:

```bash
export BINANCE_API_KEY=...
export BINANCE_API_SECRET=...
```

If a key is ever pasted into a chat, a log, or a file, treat it as compromised:
revoke it at the exchange and issue a new pair.

---

## 10. Where to look next

- `research_lib/` — the research implementation and phase reports. Read the
  conclusions before believing any backtest.
- `research_lib/strategies/unified_router.py` — the causality note; the single
  most important comment in the codebase.
- `research_lib/risk/engine.py` — the limit set and the kill switches.
