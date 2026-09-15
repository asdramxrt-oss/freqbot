# Trading Bot — Implementation Handover

**Audience:** developer or coding AI continuing this project. Read this first.
**Last updated:** 2026-09-13

---

## 1. What this project is

A freqtrade-based crypto trading research project. The target architecture (from the
design PDF, López de Prado style) is:

```
Raw market data → Data Normalizer → Information Bars → Feature Engine
  (Yang-Zhang, ATR, RSI, fracdiff, Hurst/DFA) → Turtle Engine + OU Engine
  → Regime Engine (walk-forward HMM) → Signal Router → Triple Barrier
  → Meta Model (RF/XGBoost) → Bet Sizer → Risk Engine → Execution Engine
  → Redis lock / Broker adapter
```

**Hard-earned research verdict (do not re-litigate):** every *short-horizon,
long-only, single-coin* directional strategy (Turtle, OU, pairs, vol-spike-as-signal)
failed out-of-sample — holdout Sharpe −0.06. The validated positive edges are
**funding-rate carry** (delta-neutral long spot + short perp), which only survives on
**BTC and ETH** once real perp basis risk is included, and — added 2026-09-13 —
**slow long/short trend** (`ts_trend_4h_d60_sma200_ls`: 4h Donchian 60/30 with a
1200-bar SMA regime filter, long above / short below, 5-coin perp panel), the ONLY
variant of the pre-registered 8-variant directional grid to pass all 5 acceptance
gates on data through 2025-06 (worst fold +$580, survives 2× costs, 13.2% max dd).

**RE-VALIDATION 2026-09-14 (data extended to today): the ls variant now FAILS
gate 2 (walk_forward_positive).** With perp data through 2026-09-14 the 6-fold
windows shift and the 2022 bear lands in its own folds: worst fold −$1,956. This
is robust, not windowing luck — every fold scheme 4–12 has negative folds.
Calendar attribution: 2022 −$3,935 / 2023 −$330 / 2024 +$18,355 / 2025 +$1,166 /
2026 YTD +$11,126. Overall net +$26.4k, max dd 12.9%, passes cost stress (2×) and
concentration. Read: it is an honest slow trend-follower — small losses in
bear/chop years, large gains in trend years. Per the pre-registered discipline
that is a FAIL; do not tune parameters to force a pass. Deployment decision
(accept regime risk at reduced size vs keep paper-trading) is OPEN and needs the
project owner's call. See `research/reports/DIRECTIONAL_ACCEPTANCE_REPORT.md`.

## 2. Current validated state

| Coin | Carry verdict | Note |
|---|---|---|
| BTC | **PASS** (all 5 gates) | worst fold +$923, survives 2× costs |
| ETH | **PASS** (all 5 gates) | worst fold +$361 |
| BNB | FAIL | basis risk: worst fold −$3,410, funding dd 9.7% |
| SOL | FAIL | worst fold −$1,565, 68% concentration |
| XRP | FAIL | one fold −$121 (marginal) |

Validation: 6 contiguous walk-forward folds, gates = data sufficiency / walk-forward
positive / 1.5×+2× cost stress / no concentration / tail behaviour. Full suite:
**229 tests passing** (`.venv/Scripts/python -m pytest research_lib/tests -c research_lib/pytest.ini`).

## 3. What is implemented (and where)

### Pipeline stages (pre-existing, research-grade)
- **Data normalizer** — `research_lib/data/loader.py`, `data/quality.py`
- **Information bars** — `research_lib/data/infobars.py` (disabled in config)
- **Feature engine** — `research_lib/features/` (45 causal features, YZ vol, ATR, RSI,
  fracdiff, Hurst/DFA, look-ahead scanner)
- **Turtle + OU engines** — `research_lib/strategies/turtle.py`, `ou.py`
- **Regime engine (HMM)** — `research_lib/strategies/regime_hmm.py`,
  `unified_router.py` (use ONLY `filtered_hmm_router` — see warnings below)
- **Signal router, triple barrier, meta model, bet sizer, risk engine** —
  `research_lib/strategies/router.py`, `research_lib/ml/`, `research_lib/bot/sizing.py`,
  `research_lib/risk/engine.py`
- **Execution engine, order state machine, idempotency** —
  `research_lib/execution/engine.py`, `order_state.py`, `redis_lock.py`

### Added in the latest work cycle (2026-09-13)
- **Post-only (maker) execution** — `risk/costs.py` (`fee(role)`, `maker_quote`,
  `execution_price(role)`), `OrderIntent.post_only`, `ExecutionConfig.post_only`,
  `PaperBroker` resting orders with trade-through fill logic, `LiveBroker` ccxt
  `postOnly` at top-of-book. Assessment step 4 confirmed: maker cuts carry costs ~70%.
- **Reconciliation wired into the live loop** — `bot/live.py::_reconcile` (periodic +
  final), `PaperBroker.positions()/record_external_exit()`, `LiveBroker.positions()`.
  Divergence trips the risk kill switch (fail-safe: NO NEW ORDER).
- **Reduce-only awareness in risk engine** — `risk/engine.py::evaluate`: orders trading
  against the open side are no longer blocked by exposure/leverage/margin checks
  (was silently rejecting carry closes; caught via CLI divergence check).
- **Funding-carry backtest** — `research_lib/strategies/carry.py` (`load_funding`,
  `run_carry`): causal, 8h settlement, both legs costed, re-hedge before liquidation,
  quarterly reporting, **real perp series support** (`perp=`), truncation where the
  shorter of spot/perp series ends.
- **Carry acceptance bundle** — `research_lib/validation/carry_acceptance.py`
  (folds over one continuous position, 5 gates) + `research/run_carry_acceptance.py`
  (CI-able, exits 1 on failure) → `research/reports/CARRY_ACCEPTANCE_REPORT.md`.
- **Production-path carry runner** — `research_lib/bot/carry_runner.py`: same trade
  through OrderIntent → risk → execution engine → broker → reconciliation. Naked-leg
  protection, loud close failures, book-flat invariant. Live-vs-backtest divergence
  ≈ 0% (taker) / +2.4% (maker, quote improvement).
- **CLI** — `python -m research_lib.bot.cli carry [--symbol X/USDT|all] [--taker]
  [--leverage N] [--report out.json]`; `doctor` now checks funding data.
- **Directional retest (2026-09-13, later same day)** —
  `research_lib/strategies/directional.py` (Donchian trend, SMA regime filter,
  cross-sectional momentum, panel portfolio backtest on perp data),
  `research_lib/validation/directional_acceptance.py` (same 5-gate discipline as
  carry), `research/run_directional_acceptance.py` (CI-able; exits 0 only if a
  variant passes) → `research/reports/DIRECTIONAL_ACCEPTANCE_REPORT.md`.
  **1 of 8 pre-registered variants PASSes: `ts_trend_4h_d60_sma200_ls`.** All
  long/flat variants still fail — shorts in bear regimes are what make it work.
- **Full PDF-architecture stack retested on futures only (2026-09-13, evening)** —
  `research_lib/strategies/pdf_pipeline.py` wires the design-PDF chain per coin on
  the 5-coin perp panel (features incl. Hurst/fracdiff → Turtle long+short → causal
  filtered-HMM router + vol-spike kill → triple barrier → AUC-gated meta model →
  bet sizer → risk engine; zero spot data). `research/run_pdf_pipeline_acceptance.py`
  → `research/reports/PDF_PIPELINE_ACCEPTANCE_REPORT.md`. **All 3 variants FAIL
  (1h/4h, turtle entry 20/60).** Meta model found no OOS edge anywhere (AUC ≤ 0.54
  once enough events existed; router leaves too few events elsewhere) — the
  PDF-compliant behaviour is exactly what happened: meta disabled, primaries pass.
  Conclusion: the validated directional bot is the SIMPLE slow trend + SMA-regime
  variant above; the HMM router and meta model subtract value and stay OFF.
- **Data** — `research/download_carry_data.py` fetches perp 1h OHLCV + funding for
  BTC/ETH/BNB/SOL/XRP into `research/data/futures/` and `research/data/funding/`;
  XRP spot 15m downloaded and resampled to 1h.
- **Config** — `user_data/bot_config.json` pairs = the 5 coins.

### Meta-Kelly in production (2026-09-14): functional, and honestly outclassed
The binary 0.52 suppression was degenerate (v4: 0 trades). The PDF-native fix
was implemented: `RouterBookRunner(cfg.meta_kelly=True)` scores every trend
entry with the exported pooled panel meta (B3 features), converts calibrated p
to a fractional-Kelly scale via `ml.metalabel.fractional_kelly_size`
(quarter-Kelly, causal win/loss ratio estimated from the run's own closed
trades, prior 3.0, cap = risk budget). CLI: `router-book --meta-kelly`.
Measured on refreshed data: **meta-Kelly ON: +$3,874, 1.8% dd, 527 trades**
vs **meta OFF (config D): +$40,046, 13.4% dd, 581 trades**. Quarter-Kelly ×
calibrated p (~0.5) shrinks the book ~10×; return per unit of drawdown is
WORSE (2,152 vs 2,988 per dd unit). The bet sizer now works exactly as the
PDF specifies — and the economics still say config D ships with meta OFF.
Capability shipped, default OFF, decision documented.

### v4 meta verdict (2026-09-14, acceptance with ns-fix + B3): meta ON but not useful
- **v3 panelmeta: meta switched ON for the first time in project history**
  (254 pooled events, xgb 0.561 / rf 0.576, both ≥ 0.55 — the ns-fix made the
  difference). It suppressed ~40% of entries: +$13,052 net, **1.8% dd** (vs
  +$19.8k/3.7% unsuppressed) — meta cuts drawdown but subtracts pnl.
- **v4 regfeat: meta ON (xgb 0.615 / rf 0.582) but suppressed ALL 580 entries →
  0 trades, net $0.** The model RANKS (AUC > 0.6) but calibrated probabilities
  never cross the 0.52 confidence threshold on the ungated pool (winner base
  rate too low). AUC ≠ tradable probabilities.
- Final meta answer: repaired, functional, and honestly marginal — usable as a
  drawdown reducer on the gated book, not as a pnl source. Production config D
  correctly ships with meta OFF.

### CONFIG D IMPLEMENTED (2026-09-14): production-path router book
- `research_lib/bot/router_book_runner.py` — all 9 sleeves (5 trend volkill +
  4 MR pairs) through OrderIntent → CentralRiskEngine → ExecutionEngine →
  PaperBroker → PositionReconciler. Naked-leg protection on MR entries,
  book-flat invariant, periodic reconciliation, 25% drawdown kill armed.
- CLI: `.venv/Scripts/python -m research_lib.bot.cli router-book [--report out.json]`
  → research-vs-production divergence report (`user_data/bot_runs/router_book_D.json`).
- **Divergence: −1.07%** (production +$40,046 vs research +$40,477, 581 vs 611
  trades — the real risk engine's margin checks reject a few entries the
  idealized harness takes; kill_switch=False, recon_ok=True, book_flat=True).
  First attempt showed −49% from a wrong sizing cap (single-symbol
  MARGIN_SAFE_NOTIONAL applied where the research contract is leverage×sleeve
  equity) — fixed and pinned by tests.
- Tests: `research_lib/tests/test_router_book_runner.py` (3 smoke tests: all
  sleeves run, book flat, no naked MR leg, clean reconciliation).
- Deployment path per §7 of this handover: acceptance → paper → 0.1× → 0.3× →
  full, with the live data adapter (gap #12) as the remaining build item.

### Router book profit-maximization grid (2026-09-14)
Pre-registered portfolio-construction grid (gate strictness / sleeve weights /
risk budget — no alpha params). The profit-vs-gate-2 frontier, all on refreshed
data, full grid in `reports/ROUTER_BOOK_ACCEPTANCE_REPORT.md`:
| config | net | dd | worst fold | gates |
|---|---|---|---|---|
| A baseline (9 equal, hmm gate, 1% risk) | +$11,133 | 2.2% | −$47 | 3/5 |
| B trend-heavy 75% | +$14,912 | 2.9% | −$418 | 3/5 |
| C volkill trend, 75%, 1% | +$21,886 | 7.3% | −$1,500 | 3/5 |
| **D volkill trend, 75%, 2% risk** | **+$40,477** | **12.1%** | −$2,815 | **4/5** |
D passes every gate except walk-forward-positive (the two 2022 bear folds,
−$2.4k/−$2.8k ≈ −2.7% each). Cost stress +$28.2k at 2×, concentration 38%.
A↔D is the explicit frontier: safer gating buys gate-2 margin at the cost of
profit; D maximizes profit within the 25% dd budget. **Production candidate: D.**

### Router book (2026-09-14): the PDF router design WORKS — Turtle + MR alongside
**9-sleeve version (all 5 coins in BOTH sleeves):** ratios ETH/BTC, SOL/BTC,
BNB/BTC, XRP/BTC added so every coin trades trend + MR. Result: net +$11,133,
max dd 2.2%, **worst fold −$47** (fold 0 2022 POSITIVE +$24) — the regime
router takes the book flat in chop and lets trend run in TREND. MR sleeves
~flat (+$148 combined): regime conditioning neutralized MR toxicity. FAILs
only gate 2 (−$47 ≈ −0.05% of equity) and gate 4 (concentration 53%, XRP
trend). Original 8-sleeve note below.
`research/run_router_book_acceptance.py` → `reports/ROUTER_BOOK_ACCEPTANCE_REPORT.md`.
TREND sleeve (slow trend under turtle_gate, 5 perps) + MR sleeve (OU pairs under
pairs_gate, 3 ratios, maker), 8 equal sleeves, causal HMM router per coin:
**net +$12,323, max dd 2.5%, worst fold −$104** (vs −$2,051 ungated) — the router
keeps the book flat in edgeless regimes and runs trend in TREND regimes, exactly
as designed. The MR sleeve is ~flat (−$513/+$480/−$2): regime conditioning
neutralized MR's toxicity (always-on MR lost −$6.8k to −$17.7k). FAILs gate 2
by only −$104 (−0.1%) and gate 4 (concentration 54%, XRP trend). This is the
PDF's regime-routing thesis validated: route, don't force one book everywhere.

### CRITICAL BUG FOUND + FIXED (2026-09-14): ms/ns timestamp misalignment
`read_feather` yields `datetime64[ms, UTC]`; `_meta_features` (bot/sizing.py) and
the v2/v3 pool builders used `astype("int64")` (keeps ms) against ns
`ts.value` in searchsorted → **every meta-feature lookup silently clamped to the
LAST bar**. All previous perp-data meta AUCs (0.46–0.54, "no edge") were
measurement artifacts. Fixed in `bot/sizing.py:264`, `pdf_pipeline.py`,
`run_meta_battery.py` (explicit ns conversion). With aligned features the meta
signal is weak but REAL (battery, purged CV: B1 0.53, B2 0.55/0.57, **B3 rf
0.589 / xgb 0.552 mean AUC**; unstable: 2 folds < 0.50). **B3 config =
horizon-matched barriers (pt=6/sl=2/vert=96 — the slow trend's actual geometry;
the old 2/2/48 barriers mislabeled trend winners) + expanded features
(fracdiff_close, vol_yz_20, funding level/30d mean).** All signal caches were
bug-contaminated and cleared; full acceptance re-run includes pre-registered
**v4 = v3 + ns-fix + B3** (`pdf_v4_4h_slowtrend_regfeat`).

### OU structural study (2026-09-14): mean-reversion is ABSENT from perp spreads
`research/run_ou_halflife_study.py` → `reports/OU_HALFLIFE_STUDY.md`. Rolling OU
fits at 3d/7d/14d/30d on ETH/BTC, SOL/BTC, BNB/BTC: valid fits only 24–46% of
blocks, and **MR gross pnl is NEGATIVE after both valid AND invalid regimes at
every horizon and ratio** — OU validity carries no forward information. The one
mean-reversion trade that works in this market is the perp/spot BASIS, which is
exactly the validated funding-carry book already in production. OU book closed
with structural proof, not just PnL evidence.

### Gate-2 campaign: EXHAUSTED (2026-09-14)
`research/run_trend_funding_ablation.py` → `reports/TREND_FUNDING_ABLATION_REPORT.md`.
All arms one engine, one cost model, refreshed data, funding files topped up to
2026-09-14:
| arm | net | dd | funding drag | worst fold |
|---|---|---|---|---|
| plain slow trend | +$31,358 | 9.7% | — | −$2,051 |
| plain + funding realism | +$28,026 | 10.1% | −$3,333 | −$2,106 |
| vol-spike kill | +$29,070 | 9.0% | — | −$2,051 |
| vol-spike kill + funding | +$25,823 | 9.4% | −$3,247 | −$2,688 |
| + funding-regime filter (pre-registered final shot) | +$30,967 | 9.2% | — | −$2,315 |
| + both | +$27,713 | 9.6% | −$3,254 | −$3,064 |
Findings: (1) funding realism = −$3.3k drag over 4.75y (net-long book pays
funding); (2) the vol-spike kill is a dd improver (9.0% vs 9.7%), not a net-pnl
improver at identical costs; (3) the funding-regime filter (30d mean vs trailing
1y 10/90th pct, causal, zero price-derived params) does NOT flip the 2022 bear
folds either. Gate 2 fails by −1% to −3% in EVERY configuration — the 2022
shortfall is strategy-inherent. **Decision (owner): amend gate 2 for a slow
trend-follower (e.g. worst fold > −3% equity) or accept 4/5 and paper-deploy.**

### OU book CLOSED (2026-09-14, final pre-registered attempt)
`research/run_ou_spread_acceptance.py` → `reports/OU_SPREAD_ACCEPTANCE_REPORT.md`.
OU mean-reversion on 1h PERP spreads (ETH/BTC, SOL/BTC, BNB/BTC), causal OU
half-life gate ON, with the three levers that were never tested together:
| variant | net | dd |
|---|---|---|
| taker costs | −$17,702 | 21.2% |
| maker (post-only) | −$11,304 | 17.7% |
| maker + vol-spike kill | −$6,837 | 14.1% |
Every lever helped monotonically (maker +$6.4k, kill +$4.5k) but the GROSS spread
prediction is negative-edge — costs were never the binding constraint. OU fails on
absolute price AND on spreads, at 15m and 1h, spot and perp, taker and maker.
Do not reopen without genuinely new information (e.g. order-flow features).

### Added 2026-09-14 (second cycle: PDF-pipeline v2, pre-registered BEFORE results)
- **Enhancement A (`panelmeta`)** — v1's meta model could never switch on (hard HMM
  gate starved every coin below the 120-event minimum; where trained, AUC ≤ 0.54).
  Fix: pool triple-barrier events across all 5 coins into ONE meta model with a coin
  one-hot feature (~500+ events). `pdf_pipeline.build_pdf_panel_signals_v2(mode="panelmeta")`.
- **Enhancement B (`regfeat`)** — blueprint 2.1 verbatim: the HMM regime is a
  FEATURE for the meta model, not a hard gate. Only the validated vol-spike kill
  switch still vetoes entries. Regime one-hot + P(spike) added as meta features.
  `build_pdf_panel_signals_v2(mode="regfeat")`.
- Same honesty contract as v1: chronological 60/40 split on time-sorted pooled
  events, EVERY candidate model must clear AUC ≥ 0.55 on the held-out tail or the
  meta stage is disabled. New variants `pdf_v2_1h_e20_panelmeta` /
  `pdf_v2_1h_e20_regfeat` appended to `research/run_pdf_pipeline_acceptance.py`
  (full grid always reported). Stale v1 signal caches were cleared so all variants
  run on identical refreshed data. 5 new glue tests in `tests/test_pdf_pipeline.py`.
- **RESULTS (2026-09-14, refreshed data, full grid reported): PASS variants NONE.**
  `panelmeta`: starvation mechanically fixed (684 pooled events ≥ 120) but pooled
  AUC 0.506 < 0.55 → meta correctly OFF (variant then identical to v1: −$9.8k).
  `regfeat`: 3,376 pooled events, AUC 0.495 → meta OFF; ungated fast turtle
  overtrades into cost bleed (−$49k at 2× costs). **Conclusion: no meta-labeling
  layer can fix a primary with no edge.** The fast e20 Turtle primary is the
  binding constraint, not the meta stage. The 4h gated variants are net positive
  (+$27.1k e20 / +$28.7k e60, dd 5–6%) but fail fold/concentration gates.
  **v3 RESULTS (2026-09-14):** the PDF stack wrapped around the slow-trend
  primary. Meta STILL correctly OFF on both modes (panelmeta 254 events AUC 0.467;
  regfeat 580 events AUC 0.462) — the meta-veto is now proven on 4 independent
  event pools. Economics: `pdf_v3_4h_slowtrend_regfeat` +$29,070 net, 5.6% ann,
  **9.0% dd, passes 4/5 gates** (only gate 2 fails: the two 2022 bear folds,
  −$2,051/−$1,915) — the vol-spike kill switch is the PDF component that adds
  value (BNB flipped positive +$2,095). `panelmeta` +$19,772, 3.7% dd (hard HMM
  gate cuts both profit and dd). Remaining blocker is strategy-inherent (trend
  followers lose small in bear chop), not a bug.
  **Next candidates:** (1) clean vol-spike-kill ablation at identical costs;
  (2) funding-payment realism on multi-week perp holds (open item from the
  directional harness — may shift 2022 folds either way). Deployment at 4/5 with
  the regime-risk caveat is the project owner's call.

### Added 2026-09-14
- **Production-path directional runner verified** — `research_lib/bot/directional_runner.py`
  (was built after the last handover edit): validated variant through OrderIntent →
  risk → execution → broker → reconciliation. CLI `directional --symbol all` shows
  live-vs-backtest divergence +1.7–2.9% on BTC/ETH/SOL/XRP (BNB +33% but tiny $ base).
- **Freqtrade-native wrapper** — `user_data/strategies/DirectionalTrendLS.py`: the
  validated variant as an `IStrategy` (4h, `can_short=True`, 2×ATR custom stop,
  1%-risk `custom_stake_amount`, leverage 1.0). Signal parity vs the production
  runner: 100% (0/10,303 bars mismatched, BTC). HMM router + meta model stay OFF.
  Not yet backtested through freqtrade's own engine (needs candle download).
- **Data refreshed** — perp 1h for all 5 coins now runs 2022-01-01 → 2026-09-14.

## 4. What remains to be implemented (priority order)

0. **RESOLVE the directional gate-2 FAIL on 2026 data (see verdict above).** Owner
   decision required. Also open: the harness ignores perp funding payments on
   multi-week holds — adding funding realism may shift fold nets either way.

1. **Trade only BTC+ETH carry.** BNB/SOL/XRP fail acceptance on real basis risk.
   Either exclude them from the live book or tighten `rebalance_trigger` and re-validate.
2. **Extend spot price series past 2025-06-03 for BTC/ETH/BNB/SOL** (carry window is
   currently truncated there; the trimmed research feathers must NOT be touched —
   download separate carry-only spot files if needed).
3. **Live data adapters** — the bot replays local feathers; live needs an exchange
   websocket/polling feed (`bot/live.py` currently has no live feed; `LiveBroker`
   exists but is unexercised).
4. **Ops hardening** — restart safety, graceful shutdown, structured logging, alerting
   on reconciliation failure.
5. **Multi-symbol portfolio runner** — one `CarryRunner` per coin with equity
   allocation + portfolio-level risk aggregation (currently single-coin runs).
6. **Freqtrade-native exposure (optional)** — wrap carry as an `IStrategy` for
   freqtrade's own dry-run/live machinery (`user_data/strategies/` currently only has
   the research-only `ResearchTurtleRouter.py`).
7. **Real-money deployment checklist** — acceptance PASS on latest data → paper →
   tiny (0.1×) → small (0.3×) → normal, with reconciliation + alerting live at each stage.

## 5. How to run things

```bash
cd freqtrade-develop
# full test suite (229 tests)
.venv/Scripts/python -m pytest research_lib/tests -c research_lib/pytest.ini -q
# environment + data check
.venv/Scripts/python -m research_lib.bot.cli doctor
# carry: research model vs production runner, divergence report
.venv/Scripts/python -m research_lib.bot.cli carry --symbol all
# acceptance gates for all 5 coins (exits 1 if any FAIL)
.venv/Scripts/python research/run_carry_acceptance.py
# directional variant grid acceptance (exits 0 only if a variant PASSes)
.venv/Scripts/python research/run_directional_acceptance.py
# refresh market data (perp 1h + funding, 5 coins; --spot-15m for XRP spot)
.venv/Scripts/python research/download_carry_data.py --spot-15m
```

## 6. Warnings — do NOT change these (learned the hard way)

- **Holdout hygiene:** never read `research/data_holdout/` in research code; the
  directional research frames are hard-trimmed at 2025-06-01 (leakage incident before).
- **Meta model gate:** auto-disabled at AUC ≤ 0.5 — do not force-enable; directional
  meta-labeling has no signal (AUC 0.39–0.47).
- **HMM causality:** only `unified_router.filtered_hmm_router` is causal; hmmlearn
  Viterbi block prediction leaks.
- **Frozen-leg artifact:** never mark one leg past the end of its price series —
  always truncate runs at the shortest series (created −$20k phantom losses once).
- **Dependency pin:** `joblib<1.6` (breaks hyperopt otherwise).
- Keep custom work out of the pristine upstream `freqtrade/` tree — all project code
  lives in `research_lib/`, `research/`, `user_data/`.
- `ResearchTurtleRouter.py` is RESEARCH ONLY — do not deploy it expecting its
  backtest to repeat.
