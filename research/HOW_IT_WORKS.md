# How the Trading Bot Works — As Implemented

**Last updated:** 2026-09-14
**Audience:** anyone who needs to understand, run, or extend this system.
**Read alongside:** `research/BOT_HANDOVER.md` (project state + warnings).

---

## 1. What this system is

A crypto futures trading system built on the design-PDF architecture:

```
RAW MARKET DATA → Data Normalizer → Information Bars → Feature Engine
  (Yang-Zhang, ATR, RSI, fracdiff, Hurst/DFA) → Turtle Engine + OU Engine
  → Regime Engine (walk-forward HMM) → Signal Router → Triple Barrier
  → Meta Model (RF/XGBoost) → Bet Sizer → Risk Engine → Execution Engine
  → Redis Lock / Broker Adapter
```

Every stage exists and is exercised. Two books are production-ready:

| Book | Edge | Status |
|---|---|---|
| **Router book, config D** (trend + mean-reversion, regime-routed) | Slow Donchian trend in TREND regimes; OU pairs dormant but non-toxic | **+$40,477 research / +$40,046 production** on $100k, 12.1% dd, 4/5 gates |
| **Funding carry** (BTC, ETH only) | Delta-neutral funding premium | Validated earlier (all 5 gates) |

The directional book trades **all 5 coins** (BTC, ETH, BNB, SOL, XRP USDT-M
perpetuals) long AND short. The one gate it fails is "every 9.5-month fold
positive" — it loses ~2.7% in 2022-style bear chop. That is strategy-inherent,
documented, and accepted by the project owner.

---

## 2. Repository layout (project code only)

```
research_lib/            # all project code (the pristine upstream freqtrade/ is untouched)
├── data/                # loader, quality checks, information bars
├── features/            # 45 causal features: YZ vol, ATR, RSI, fracdiff, Hurst/DFA
├── strategies/          # turtle.py, ou.py, pairs.py, directional.py, carry.py,
│                        # regime_hmm.py, unified_router.py, pdf_pipeline.py, pairs_sim.py
├── ml/                  # triple_barrier.py, metalabel.py, volspike.py
├── risk/                # engine.py (CentralRiskEngine), backtest_engine.py, costs.py
├── execution/           # engine.py, order_state.py, redis_lock.py, reconciliation.py
├── bot/                 # config.py, features.py, sizing.py, broker.py, live.py,
│                        # carry_runner.py, directional_runner.py, router_book_runner.py, cli.py
├── validation/          # acceptance.py, directional_acceptance.py, carry_acceptance.py,
│                        # purged.py, cpcv.py, walkforward.py, montecarlo.py
└── tests/               # 247 tests (pytest -c research_lib/pytest.ini)

research/                # runners + reports (all acceptance bundles are CI-able)
├── run_directional_acceptance.py      # 8-variant directional grid
├── run_pdf_pipeline_acceptance.py     # full PDF stack, 8 variants (v1..v4)
├── run_router_book_acceptance.py      # Turtle+MR book, profit-max grid (A..D)
├── run_ou_spread_acceptance.py        # OU-on-spreads final battery
├── run_meta_battery.py                # meta-model final battery (B1/B2/B3)
├── run_trend_funding_ablation.py      # vol-kill ablation + funding realism
├── run_ou_halflife_study.py           # MR structural study
├── download_carry_data.py             # perp 1h OHLCV + funding refresher
├── data/futures/                      # {SYM}_USDT-1h-perp.feather (2022-01 → today)
├── data/funding/                      # {SYM}_USDT_USDT_funding.csv (8h prints)
└── reports/                           # every verdict, pass or fail

user_data/
├── bot_config.json                    # pairs, strategy/router/meta/risk/execution config
├── strategies/ResearchTurtleRouter.py # RESEARCH ONLY (see warnings in handover)
└── strategies/DirectionalTrendLS.py   # freqtrade-native slow-trend strategy (futures)
```

---

## 3. The production book: router book config D

### 3.1 Signal flow per sleeve

**TREND sleeves (5 coins, 75% of equity, 15% each)**

1. 1h perp OHLCV → resampled to 4h bars (`directional.resample_panel`).
2. Slow-trend target (`build_trend_signals`):
   - Donchian channel: enter long when close > prior 60-bar high, exit on close
     < prior 30-bar low; mirror for shorts (`donchian_target`).
   - Regime filter: longs only above the 1200-bar SMA (≈200 days), shorts only
     below (`apply_regime_filter`, mode `long_short`).
   - Stop: fixed 2×ATR(20) from entry, held for the episode.
   - Decision at close t → fill at open t+1 (no same-bar look-ahead).
3. **Vol-spike kill** (the PDF's validated veto): an XGBoost model predicts
   P(vol spike) per bar from causal vol features; when P(spike) > 0.80 no new
   trend entry is taken. This cut max drawdown from 12.9% → 9.0% in validation.
4. Sizing: `units = 2% × sleeve_equity / |entry − stop|`, notional capped at
   2× sleeve equity (the validated contract).

**MR sleeves (4 ratios: ETH/BTC, SOL/BTC, BNB/BTC, XRP/BTC, 25% of equity)**

1. Log-ratio spread `log(A/B)` on aligned 4h closes.
2. Causal z-score vs trailing 42-bar (7-day) window; enter at |z| ≥ 2,
   exit at z-cross-0 or 84-bar time stop (`pairs.generate_signals`).
3. OU tradability gate: a causal Ornstein-Uhlenbeck fit on the trailing window
   must confirm a sane half-life, else no trades that block.
4. HMM `pairs_gate`: the regime engine must classify leg A as
   LOW_VOL_MR/NEUTRAL — this is what neutralized MR's toxicity (always-on MR
   lost −$6.8k to −$17.7k; regime-gated MR is ≈ flat, +$148 combined).
5. Dollar-neutral legs, 50% of sleeve equity per leg, maker (post-only) fills.

### 3.2 Production execution chain (every order, every sleeve)

```
OrderIntent → CentralRiskEngine.evaluate (margin, exposure, kill switches)
           → ExecutionEngine.execute (idempotency keys, order state machine)
           → PaperBroker (fills; LiveBroker exists for real trading)
           → PositionReconciler (bot-vs-broker positions every 168 bars;
             divergence trips the risk kill switch — no new orders)
```

Implemented in `research_lib/bot/router_book_runner.py` (`RouterBookRunner`).
Invariants enforced and tested: book ends flat, no naked MR leg (if leg 2
fails, leg 1 is unwound at taker immediately), reconciliation clean.

### 3.3 Verified economics (refreshed data through 2026-09-14)

| Metric | Research harness | Production runner |
|---|---|---|
| Net PnL ($100k) | +$40,477 | +$40,046 |
| Trades | 611 | 581 |
| Max drawdown | 12.1% | 13.4% |
| Divergence | — | **−1.07%** |

Fold profile: −2,412 / −2,815 / +20,474 / +7,194 / +6,908 / +11,127 —
small controlled bleed in bear chop, large gains in trending windows.
The production run takes 30 fewer trades: the real risk engine's margin
checks reject a few entries the idealized harness fills. That is intended.

---

## 4. The supporting engines (what each module actually does)

- **Feature engine** (`features/`): 45 causal features; every feature uses data
  ≤ current bar only (a look-ahead scanner test enforces this).
- **Regime engine** (`strategies/regime_hmm.py`, `unified_router.py`): walk-forward
  Gaussian HMM with FILTERED (causal) state inference — refit every 168 bars on a
  rolling 1998-bar window. Emits per-bar regime (TREND / LOW_VOL_MR / NEUTRAL /
  HIGH_VOL / UNCERTAIN), `turtle_gate`, `pairs_gate`, and P(vol spike).
  Only `filtered_hmm_router` is causal — hmmlearn Viterbi block prediction leaks.
- **Triple barrier + meta model** (`ml/`, `bot/sizing.py`): labels primary events
  (pt/sl/vertical barriers), trains RF+XGB with a chronological split and
  isotonic calibration; **auto-disabled unless EVERY model clears AUC ≥ 0.55
  OOS**. On 2026-09-14 a millisecond/nanosecond timestamp bug in feature
  alignment was found and fixed; with the B3 config (horizon-matched barriers
  6/2/96 + fracdiff/YZ-vol/funding features) the meta signal is weak-but-real
  (purged-CV rf 0.589). It is OFF in production until the v4 acceptance
  confirms it adds economics.
- **Bet sizer** (`bot/sizing.py`, `ml/metalabel.py`): calibrated-probability
  fractional Kelly (`fractional_kelly_size`, quarter-Kelly, causal win/loss
  ratio). Fully wired into the router book runner via
  `RouterBookConfig(meta_kelly=True)` / `cli router-book --meta-kelly`.
  Measured 2026-09-14: ON = +$3,874 / 1.8% dd vs OFF = +$40,046 / 13.4% dd —
  the sizer works exactly as the PDF specifies and the economics still favor
  OFF, so it ships disabled by default (decision documented in the handover).
- **Risk engine** (`risk/engine.py`): per-trade risk, margin availability,
  exposure/leverage caps, daily/weekly loss limits, drawdown kill switch
  (armed at 25% for the router book), funding payment accounting,
  reduce-only awareness.
- **Execution engine** (`execution/`): idempotency keys (no duplicate orders on
  retry), order state machine, Redis lock (fakeredis in-process fallback),
  post-only maker support with resting-order fill logic.

---

## 5. How to run everything

```bash
cd freqtrade-develop

# full test suite (247 tests)
.venv/Scripts/python -m pytest research_lib/tests -c research_lib/pytest.ini -q

# environment + data check
.venv/Scripts/python -m research_lib.bot.cli doctor

# THE PRODUCTION BOOK (config D): research vs production divergence
.venv/Scripts/python -m research_lib.bot.cli router-book --report user_data/bot_runs/router_book_D.json

# directional single-book runner (slow trend, per coin)
.venv/Scripts/python -m research_lib.bot.cli directional --symbol all

# funding carry (BTC/ETH): research vs production
.venv/Scripts/python -m research_lib.bot.cli carry --symbol BTC

# acceptance suites (all exit non-zero on failure)
.venv/Scripts/python research/run_router_book_acceptance.py     # the A..D grid
.venv/Scripts/python research/run_directional_acceptance.py     # 8-variant trend grid
.venv/Scripts/python research/run_pdf_pipeline_acceptance.py    # full PDF stack v1..v4
.venv/Scripts/python research/run_carry_acceptance.py           # carry gates

# refresh market data (perp 1h + funding, 5 coins)
.venv/Scripts/python research/download_carry_data.py

# freqtrade-native (dry-run on futures): use user_data/strategies/DirectionalTrendLS.py
# with a futures config (trading_mode: futures, margin_mode: isolated)
```

---

## 6. Validation discipline (why results here can be trusted)

1. **Causality everywhere**: signals at close t fill at open t+1; features are
   causal; the HMM uses filtered (not smoothed) inference; funding/OU gates use
   trailing windows only.
2. **Pre-registration**: every variant grid (8 directional variants, PDF v1–v4,
   router book A–D, meta battery B1–B3) is declared BEFORE results are seen, and
   the full grid is always reported — a lucky winner cannot hide.
3. **5 acceptance gates**: data sufficiency / every walk-forward fold positive /
   1.5×+2× cost stress / no concentration >50% / drawdown < 25%.
4. **Holdout hygiene**: `research/data_holdout/` is never read in research code;
   directional research frames are hard-trimmed (a leakage incident happened once).
5. **Production parity**: every research result must reproduce through the
   production chain within a few % divergence before it counts.

## 7. Known limitations (honest list)

- Gate 2 fails for the directional book: −1% to −2.8% in 2022-style bear chop,
  in every configuration tested. Accepted as strategy-inherent.
- MR sleeve is a diversifier, not a profit center (+$148 combined).
- Funding drag ≈ −$3.3k over 4.75 years on the net-long-tilted trend book.
- Live data adapter is NOT built — the bot replays local feathers; LiveBroker
  is unexercised. This is the one remaining build item before real deployment.
- Deployment path: paper → 0.1× → 0.3× → full size, reconciliation + alerting
  live at each stage. Do not skip stages.
