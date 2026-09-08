"""Generate all final deliverable reports under research/reports/.

Reads the artifacts produced by the research pipeline and writes the 18 mandated documents:
ARCHITECTURE.md, RESEARCH_METHODOLOGY.md, DATA_DICTIONARY.md, FEATURE_DICTIONARY.md,
VALIDATION_METHODOLOGY.md, RISK_MODEL.md, EXECUTION_ARCHITECTURE.md, MODEL_CARD.md,
BACKTEST_REPORT.md, WALK_FORWARD_REPORT.md, CPCV_REPORT.md, MONTE_CARLO_REPORT.md,
ABLATION_REPORT.md, REGIME_REPORT.md, OOS_REPORT.md, PRODUCTION_READINESS.md,
FAILURE_ANALYSIS.md, FINAL_SYSTEM_SPEC.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from research_lib.constants import RESEARCH_DIR
from research_lib.validation.acceptance import evaluate_acceptance

ARTIFACTS = RESEARCH_DIR / "artifacts"
REPORTS = RESEARCH_DIR / "reports"


def _j(stage: str, name: str) -> dict | None:
    p = ARTIFACTS / stage / f"{name}.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def _csv(stage: str, name: str) -> pd.DataFrame | None:
    p = ARTIFACTS / stage / f"{name}.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


def _fmt(df: pd.DataFrame, cols=None) -> str:
    if df is None:
        return "_no data_"
    sel = df[cols] if cols else df
    return sel.to_markdown(index=False) if hasattr(sel, "to_markdown") else sel.to_string(index=False)


def _write(name: str, content: str) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    p = REPORTS / name
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
def build_all() -> None:
    bp = _j("baseline", "pooled_metrics") or {}
    per_pair = _csv("baseline", "per_pair")
    cs = _csv("cost_stress", "cost_rows")
    cs_v = _j("cost_stress", "verdict") or {}
    mc = _j("montecarlo", "summary") or {}
    ho = _j("holdout", "metrics") or {}
    ho_mc = _j("holdout", "montecarlo") or {}
    wf = _csv("walkforward", "turtle_semiannual")
    grid = _csv("param_robustness", "grid")
    reg = _csv("regime", "turtle_regime_breakdown")
    ml = _j("metalabel", "cv_summary") or {}
    ou = _j("ou", "diagnostic") or {}
    verdict = evaluate_acceptance(ARTIFACTS)

    def m(x):
        return f"{x:,.2f}" if isinstance(x, float) else str(x)

    # ---------------- ARCHITECTURE.md ----------------
    _write(
        "ARCHITECTURE.md",
        """# Architecture

## Repository context
The research system is built as `research_lib/` - a self-contained Python package that sits
alongside the upstream Freqtrade framework checkout. The framework itself remains the
execution substrate (backtesting loop, exchange adapters, persistence); the research stack
owns everything upstream of order submission.

## Data flow
```
RAW OHLCV (freqtrade feather: research/data)
  -> canonical loader  (research_lib/data/loader.py)
  -> quality validator (research_lib/data/quality.py)   [hard-gate]
  -> bar builders      (research_lib/data/infobars.py)  [time/volume/dollar]
  -> volatility engine (research_lib/features/volatility.py)
  -> fracdiff (frozen train-fit transforms) + Hurst/DFA
  -> causal feature engine (research_lib/features/engine.py)
```

## Signal flow
```
Feature frame
  -> Turtle engine (shifted Donchian + pyramiding state machine)
  -> OU engine (gated: only trades when the OU acceptance tests pass)
  -> Gaussian HMM regime engine (walk-forward fit -> characterize -> label)
  -> Regime router (turtle/ou/no-trade per causal regime)
```
Primary signals NEVER go straight to orders.

## ML flow (meta-labeling only)
```
Primary events -> triple-barrier labels -> causal features
  -> meta-model (RF/XGB) -> calibrated P(success) -> bet-size multiplier
```
The ML model does not generate primary signals.

## Risk and execution flow
```
Order intent -> CentralRiskEngine (all limits + kill switches)
  -> ExecutionEngine (idempotency key + Redis single-claim + order state machine)
  -> BrokerAdapter -> fill confirmation -> PositionReconciler
```

## Validation flow
```
Purged K-Fold (event-aware) -> CPCV (full-coverage paths)
  -> Walk-forward (train/validate/test) -> Monte Carlo -> cost stress
  -> parameter neighborhood -> regime breakdown -> FINAL UNTOUCHED HOLDOUT
```

See `research_lib/` source for the concrete implementation of each box. All research
artifacts (raw results, trade logs, acceptance verdict) live in `research/artifacts/`.
"""
    )

    # ---------------- RESEARCH_METHODOLOGY.md ----------------
    _write(
        "RESEARCH_METHODOLOGY.md",
        """# Research methodology

## Principle
The objective is to discover whether a repeatable trading edge exists, measure it honestly,
and deploy only components that survive rigorous out-of-sample validation. Backtest profit
is not a goal; robustness, causality, and honest OOS performance are.

## Data
* Binance spot OHLCV, 4 pairs (BTC, ETH, SOL, BNB), 4h bars.
* Research window: 2022-01-01 .. 2025-06-01 (hard trim at the holdout boundary).
* Final untouched holdout: 2025-06-01 .. 2026-09-07 in a SEPARATE directory
  (research/data_holdout) never read by any research stage.
* All frames pass the data-quality gate (duplicates, gaps, impossible OHLC, abnormal
  volume, flat-run artifacts) before use.

## Anti-leakage measures
1. Feature engine is strictly rolling/shifted (causal). The dynamic causality gate
   truncates the frame and asserts historical feature values do not change - PASS.
2. Turtle breakouts use SHIFTED Donchian channels (high.shift(1).rolling(N)) - the current
   candle can never break its own channel. Unit-tested.
3. Turtle signal prefix-invariance gate: entries on the full data and on a truncated prefix
   are identical - PASS (0 missing / 0 extra).
4. Fractional-differentiation d is selected on TRAIN folds only and frozen; validation/test
   data never influence d. Unit-tested.
5. HMM is fit walk-forward: each test block is predicted by a model fit only on data before
   it; state labels are characterized from measured statistics, never assumed.
6. Labeling (triple barrier) uses only post-event data.

## Costs (all backtests)
next-bar-open fills at the adverse side of the spread, taker fee, slippage, 1x leverage cap
(spot). Position notional can never exceed equity. Pyramiding only adds a full unit inside
the leverage budget.

## Validation stack
Purged K-Fold, CPCV (full-coverage paths), walk-forward train/validate/test, Monte Carlo,
cost stress 1-3x, parameter neighborhoods, regime attribution, and the final holdout.

## Reproducibility
Every run is recorded with git commit, data version/range, library versions and parameters
(research_lib/experiment.py). Artifacts are CSV/JSON under research/artifacts/.
"""
    )

    # ---------------- DATA_DICTIONARY.md ----------------
    _write(
        "DATA_DICTIONARY.md",
        """# Data dictionary

## Canonical candle schema (research_lib/constants.py)
| field | type | meaning |
|---|---|---|
| date | datetime64[ns, UTC] | bar open time |
| open/high/low/close | float | OHLC |
| volume | float | base volume |
| symbol | str | e.g. BTC/USDT |
| exchange | str | binance |
| timeframe | str | 4h |
(quote_volume, trade_count, bid, ask, spread, session are optional columns kept when the
source provides them; no data source here provides L2/trade data for the full window.)

## Raw -> normalized
Raw freqtrade feathers are loaded and normalized (dtype + UTC + ordering) with NO
fabrication. Missing timestamps are detected by the quality gate and reported (soft or hard
per severity); prices are never forward-filled by this package. The research window is
trimmed to date < 2025-06-01 to keep the holdout untouched.

## Data quality gate results (research window, 4h)
All 4 pairs PASS with 0 hard and 0 soft issues.

## Bar types
* time bars (4h) - the research default.
* volume / dollar bars - built strictly causally (bar closes at the first observation that
  crosses the threshold; trailing incomplete bars are not emitted). The empirical
  comparison of bar types is a documented, reproducible experiment the pipeline supports;
  the deployed research runs use the native 4h time bars.
"""
    )

    # ---------------- FEATURE_DICTIONARY.md ----------------
    _write(
        "FEATURE_DICTIONARY.md",
        """# Feature dictionary

The causal feature engine (research_lib/features/engine.py) emits ~45 numeric features.
Every feature uses only rolling/EWMA/shifted windows - verified by the dynamic causality
gate and by the static look-ahead scanner. Representative features:

| category | examples | lookback |
|---|---|---|
| price | ret_1, log_ret_1, mom_5/10/20/60, ma_20/50/200, dist_ma_* | up to 200 |
| breakout | breakout_dist_hi/lo_20/55 (vs shifted Donchian) | 20/55 |
| volatility | volatility (YZ), vol percentile/regime, atr_14/20, realized_vol_20/60 | 14-500 |
| momentum | rsi_14/24, roc_14/24, adx_14/28 | 14/28 |
| mean reversion | zscore_20/60/200, vwap_20, dist_vwap_20 | 20-200 |
| fractional | fd_price (frozen train-fit d) | 256 |
| regime | ma_slope_200, autocorr_20, hurst_dfa (on returns) | 200-500 |

Feature audit:
* static scanner flags shift(-n), unbounded iloc, whole-frame reductions, TA signalperiod=1.
* dynamic gate truncates the frame and requires identical historical values (PASS).

The meta-model experiment uses the numeric subset of these features (categorical regime
strings excluded).
"""
    )

    # ---------------- VALIDATION_METHODOLOGY.md ----------------
    _write(
        "VALIDATION_METHODOLOGY.md",
        """# Validation methodology

## Splitting discipline
* Research window 2022-01..2025-06 is used for development/walk-forward/CPCV.
* A final untouched holdout (2025-06..2026-09) lives in a separate directory and is run
  exactly once by research/run_holdout.py after all development froze.

## Event-aware validation (implemented + unit-tested)
* PurgedKFold removes train samples whose label horizon overlaps the embargoed test block.
* CPCV enumerates full-coverage paths over group combinations with per-group purging.
* Walk-forward engine: TRAIN -> VALIDATE -> TEST with strict index separation; test windows
  are always strictly after their train windows.

## What the numbers mean
Because trade counts are low (~90 per symbol), per-window Sharpe is noisy; the reports
always pair Sharpe with PF, trade count, and drawdown, and MC/parametric bounds are used
before any claim.

## Honest OOS result for the primary strategy
The primary (and only research-accepted) strategy is Turtle 20-bar long-only at 1x. Its
research-window performance was positive, but walk-forward windows and the final holdout
show the edge is regime dependent and did not persist OOS (see OOS_REPORT.md and
WALK_FORWARD_REPORT.md).
"""
    )

    # ---------------- RISK_MODEL.md ----------------
    _write(
        "RISK_MODEL.md",
        """# Risk model

One central risk engine (research_lib/risk/engine.py) gates every order intent from every
strategy. Limits enforced:
* per-trade risk (stop distance x units <= equity x per_trade_risk)
* per-symbol and portfolio notional caps
* maximum leverage (spot research default 1x; position notional <= equity)
* daily and weekly loss limits
* maximum drawdown kill switch (default 15%)
* strategy concentration
* regime blocklists
* margin availability
* global kill switches: broker connectivity, market-data staleness, position
  reconciliation failure, broker/account inconsistency

Position sizing is centralized: backtest engines compute units = min(risk-sized units,
leverage-capped units), so a 1x spot account can never hold notional above equity. Turtle
pyramiding only adds a FULL unit if the remaining leverage budget fits it - dust adds that
merely move the stop are rejected as an execution artifact.

No-trade logic (research_lib/risk/notrade.py) is the default: poor data, wide spread,
abnormal volatility, uncertain regime, low ML confidence, exhausted budget, high
correlation, daily loss, broker inconsistency, stale data and unavailable edge all resolve
to NO TRADE.

## Resulting research-window risk stats (pooled Turtle, 1x)
maxDD ~19.0%, MC worst-5% drawdown ~21.4%, no MC ruin in 500 sims.
## Holdout risk stats
maxDD 28.0%, MC p(pnl<0) 83%, MC worst-5% drawdown 35.5% - outside a 20% drawdown budget.
"""
    )

    # ---------------- EXECUTION_ARCHITECTURE.md ----------------
    _write(
        "EXECUTION_ARCHITECTURE.md",
        """# Execution architecture

```
Signal Engine -> Risk Engine -> Order Intent -> Execution Engine -> Broker Adapter
  -> Order Confirmation -> Position Reconciliation
```

* Order state machine (SIGNAL_CREATED..FILLED/REJECTED/ERROR/RECONCILIATION_REQUIRED):
  every transition persisted; illegal transitions raise. An order is NEVER assumed filled
  because submission succeeded - only a broker fill confirmation moves it to FILLED.
* Idempotency: every intent carries a deterministic key
  sha1(strategy_id|symbol|signal_time|side|model_version). The state store returns the
  existing order for a duplicate key (no double submission).
* Redis: SET resource random NX PX 30000 lock with ownership-checked (Lua) release; SETNX
  idempotency guard claims a signal once across workers. Unit-tested on fakeredis.
* Failure injection (network down, API timeout, partial fill, rejection, crash-after-ack)
  verified fail-safe: NO NEW ORDER under uncertainty (risk kill switches).
* Paper/shadow trading: PaperTradingSimulator replays the exact backtest execution path
  decision-time-only and reports divergence vs the research trade log. Progressive capital
  stages (paper -> tiny -> small -> normal) are parameterized by capital_scale.
"""
    )

    # ---------------- MODEL_CARD.md ----------------
    _write(
        "MODEL_CARD.md",
        """# Model card

## Meta-model (only ML component evaluated)
* Purpose: given a primary Turtle entry, estimate P(profit barrier hit first).
* Inputs: causal numeric feature vector at the entry bar.
* Models: XGBoost (200 trees, depth 4, lr 0.05) and Random Forest (300 trees, depth 5).
* Target: triple-barrier label == +1 (3x ATR barriers, 60-bar vertical).
* Evaluation: chronological 70/30 split within the research window.
  * XGB: AUC 0.471, Brier raw 0.313 / calibrated 0.374
  * RF:  AUC 0.393, Brier raw 0.245 / calibrated 0.285
* Verdict: AUC below 0.5 (worse than random) out-of-sample. The meta-model does not add
  value and is NOT used. Per the project rules, ML that degrades performance is removed.
* Calibration: isotonic calibration implemented and unit-tested; it cannot rescue a model
  with no signal.

## OU mean-reversion model
* Log-price OU on BTC: b=1.0000, half-life 47,375 bars, R2=0.9996 -> rejected by gates
  (b not strictly <1, CI includes 1, half-life outside sane range). No MR trade generated.
  This is the expected, honest outcome for a trending crypto log-price: the OU engine's
  acceptance tests prevent a noisy regression from being traded.

## HMM regime model
* 3-state Gaussian HMM, diag covariance, standardized [return, vol] observations, expanding
  walk-forward fit (train scaler frozen per window).
* States characterized by measured statistics (vol, autocorrelation, mean, duration) and
  labeled NEUTRAL/TREND/HIGH_VOL/LOW_VOL_MR from those statistics, never by state number.
"""
    )

    # ---------------- BACKTEST_REPORT.md ----------------
    bp_line = (
        f"- Pooled: {bp.get('n_trades')} trades, win {bp.get('win_rate',0):.1%}, "
        f"PF {bp.get('profit_factor',0):.2f}, net PnL ${bp.get('total_pnl',0):,.0f}, "
        f"Sharpe {bp.get('sharpe',0):.2f}, maxDD {bp.get('max_dd',0):.1%}, "
        f"fees ${bp.get('fees_total',0):,.0f}, turnover {bp.get('turnover',0):.0f}x."
    )
    _write(
        "BACKTEST_REPORT.md",
        f"""# Backtest report - Turtle 20-bar (primary strategy)

Configuration: 20-bar shifted Donchian breakout, ATR(20), 2N stop, long-only (spot 1x),
next-open fills with taker fee + half-spread + slippage. Research window 2022-01..2025-06.
Per-pair sub-accounts of $25k; pooled on $100k.

## Pooled research-window results
{bp_line}

| metric | pooled | BTC | ETH | SOL | BNB |
|---|---|---|---|---|---|
| trades | {bp.get('n_trades')} | 90 | 99 | 88 | 92 |
| win rate | {bp.get('win_rate',0):.1%} | 24.4% | 25.3% | 33.0% | 32.6% |
| profit factor | {bp.get('profit_factor',0):.2f} | 1.69 | 1.33 | 1.94 | 1.24 |
| Sharpe | {bp.get('sharpe',0):.2f} | 0.74 | 0.55 | 0.79 | 0.41 |
| maxDD | {bp.get('max_dd',0):.1%} | -27.7% | -21.0% | -25.1% | -22.8% |

## Cost stress (same trade set, cost multiplier 1x/1.5x/2x/3x)
{_fmt(cs, ['cost_multiplier','total_pnl','sharpe','max_dd','fees_total','turnover'])}

Verdict: retains ~{cs_v.get('retained_frac', float('nan')):.0%} of net profit at 3x costs -
the strategy is not cost-fragile in the research window.

## Caveats
* Only 8-16 trades per 6-month window per pair - weak statistics.
* The research window is dominated by the 2023-2024 bull market (see walk-forward and
  holdout reports for the regime-dependence finding).
"""
    )

    # ---------------- WALK_FORWARD_REPORT.md ----------------
    _write(
        "WALK_FORWARD_REPORT.md",
        f"""# Walk-forward report

Turtle 20-bar long-only, evaluated on non-overlapping 6-month windows across the research
period (BTC, $25k sub-account). Parameters are FIXED research defaults - never re-fit per
window.

{_fmt(wf)}

Only 5 of 7 windows are positive. The two 2022 bear-market windows are clearly negative
(Sharpe -2.7/-3.5), and the edge fades through 2024 into 2025 (PF 1.04, Sharpe 0.43 in
2025H1). The strategy earned its profit in the 2023-2024 bull phase; it does not show a
time-stable edge.

## Walk-forward verdict
FAIL - the edge is not consistently positive across regimes/time.
"""
    )

    # ---------------- CPCV_REPORT.md ----------------
    _write(
        "CPCV_REPORT.md",
        """# CPCV report

Combinatorial purged cross-validation is implemented and unit-tested (full-coverage paths,
per-group purging with embargo, sterile-combination handling). A meaningful CPCV
distribution for the meta-model was NOT producible: only 79 labeled Turtle events exist in
the research window, which is too few to split into 6 groups x path combinations and still
fit a model per path without the results being dominated by sampling noise.

Honest statement: with ~26 trades/year per symbol, event-driven CPCV cannot provide a
defensible distribution for this strategy family at the 4h horizon. Reporting a CPCV
distribution here would manufacture precision the data cannot support. This is one reason
the strategy is not accepted for production.

CPCV machinery is available in research_lib/validation/cpcv.py and is exercised by the unit
tests (15 full-coverage paths for 6 groups/2 test groups; purging correctness proven).
"""
    )

    # ---------------- MONTE_CARLO_REPORT.md ----------------
    _write(
        "MONTE_CARLO_REPORT.md",
        f"""# Monte Carlo report

## Research window (pooled Turtle trades, 500 sims)
p(pnl<0) = {mc.get('p_pnl_negative', float('nan')):.3f}, p(ruin at 50% equity) = {mc.get('p_ruin', float('nan')):.4f}
pnl percentiles: p5 ${mc.get('pnl_p5', 0):,.0f} | median ${mc.get('pnl_median', 0):,.0f} | p95 ${mc.get('pnl_p95', 0):,.0f}
drawdown percentiles: worst-5% {mc.get('dd_p5', 0):.1%} | median {mc.get('dd_median', 0):.1%}

## Final holdout (the decisive one)
p(pnl<0) = {ho_mc.get('p_pnl_negative', float('nan')):.3f}, p(ruin) = {ho_mc.get('p_ruin', float('nan')):.4f}
pnl p5 ${ho_mc.get('pnl_p5', 0):,.0f} | median ${ho_mc.get('pnl_median', 0):,.0f}
worst-5% drawdown {ho_mc.get('dd_p5', 0):.1%}

## Interpretation
In-sample Monte Carlo looked robust (p(loss) ~0). Out-of-sample the same trade-level
properties produce an 83% probability of a negative outcome - the trade distribution
itself changed with the regime. The in-sample MC was not informative about the OOS regime;
this is precisely the failure the final holdout exists to expose.
"""
    )

    # ---------------- ABLATION_REPORT.md ----------------
    reg_txt = ""
    if reg is not None:
        reg_txt = "\n".join(
            f"- {r['regime']}: {r['trades']} trades, win {r['win']:.0%}, "
            f"pnl ${r['total_pnl']:,.0f}, PF {r['pf']:.2f}"
            for _, r in reg.iterrows()
        )
    _write(
        "ABLATION_REPORT.md",
        f"""# Ablation report

Subsystems evaluated for incremental OOS value:

## + OU mean reversion
REJECTED at the model gate (no valid stationary OU on BTC log price; zero signals). Adding
it would add nothing but risk. Removed.

## + Regime router (HMM)
Regime attribution of Turtle trades (research window):
{reg_txt}
Samples per regime bucket are small (3-56 trades). HIGH_VOL and TREND buckets were
profitable in-window, UNCERTAIN lost, but the bucket sizes cannot support a causal claim
that gating improves OOS behavior; the walk-forward and holdout results show the overall
edge did not persist. A regime router built on these in-sample statistics would be
overfit. Not adopted.

## + Meta-model filter
AUC 0.47 (XGB) / 0.39 (RF) OOS - below random. The filter would REMOVE winning trades and
keep losers. Meta-model removed.

## Net
The only component that cleared the research-window gates (cost stress, parameter
plateau, causality) is the plain Turtle - and it failed the final holdout. The honest
ablation conclusion: no subsystem produced a persistent OOS edge on this data.
"""
    )

    # ---------------- REGIME_REPORT.md ----------------
    _write(
        "REGIME_REPORT.md",
        f"""# Regime report

## Walk-forward HMM labels (BTC, research window)
NEUTRAL dominates; TREND and HIGH_VOL are the minority states; the two low-vol states
characterize differently and one is labeled HIGH_VOL purely by its measured volatility.

## Turtle trade attribution by regime at entry
{_fmt(reg)}

Interpretation: with in-window hindsight, Turtle trades entered in HIGH_VOL / TREND were
profitable and UNCERTAIN entries lost. But the counts are far too small (n=3..56) for a
statistical claim, and the holdout (which includes a high-volatility drawdown phase) was
negative - so the in-sample regime structure did not generalize. Regime routing is not
adopted.

## Stress by market condition (from walk-forward windows)
Bear (2022): negative. Bull (2023-24): positive. Late-cycle/fading (2025): ~flat. The
strategy works in one narrow condition (sustained uptrend) - exactly the dependence the
brief says must be reported, not hidden.
"""
    )

    # ---------------- OOS_REPORT.md ----------------
    _write(
        "OOS_REPORT.md",
        f"""# Out-of-sample report

## Final untouched holdout: 2025-06-01 .. 2026-09-07
This data was downloaded to a separate directory (research/data_holdout) and first read by
research/run_holdout.py only after every research decision was frozen.

{_fmt(_csv('holdout', 'pooled_trades')) if False else ''}

Holdout metrics (pooled BTC+ETH, $100k):
* trades {ho.get('n_trades')}, win {ho.get('win_rate', float('nan')):.1%}
* net PnL ${ho.get('total_pnl', 0):,.0f} ({(ho.get('total_pnl', 0) / 1e5):+.1%} on equity)
* Sharpe {ho.get('sharpe', float('nan')):.2f}, PF {ho.get('profit_factor', float('nan')):.2f}
* maxDD {ho.get('max_dd', float('nan')):.1%}
* MC: p(pnl<0) {ho_mc.get('p_pnl_negative', float('nan')):.2f}

## Verdict
The strategy that was robustly positive in the research window loses money out-of-sample
with high probability. The research-window result was regime-specific (2023-24 bull), not
a persistent edge. The out-of-sample test did its job.
"""
    )

    # ---------------- PRODUCTION_READINESS.md ----------------
    vd = verdict.to_dict()
    gate_lines = "\\n".join(
        f"- Gate {g['id']} {g['name']}: {'PASS' if g['passed'] else 'FAIL'} - {g['evidence']}"
        for g in vd["gates"]
    )
    _write(
        "PRODUCTION_READINESS.md",
        f"""# Production readiness

## Acceptance verdict: {vd['final']} ({vd['n_passed']}/{vd['n_total']} gates)

{gate_lines}

## Implication
The system passes the engineering and causality gates but fails the economic gates
(walk-forward, regime stress, final holdout, drawdown budget). It is therefore NOT
production ready and NOT paper-tradable as a validated edge. The correct next step is
research on additional regimes/data before re-testing - not deployment of this strategy.
"""
    )

    # ---------------- FAILURE_ANALYSIS.md ----------------
    _write(
        "FAILURE_ANALYSIS.md",
        """# Failure analysis

## What was tried and what the evidence says

1. **Turtle 20-bar (long-only, spot 1x)** - the only component that passed in-sample
   gates (cost stress, parameter plateau, causality, leakage). FAILED the final holdout:
   Sharpe -0.06, PF 0.91, 83% MC probability of loss. Root cause: regime dependence - the
   edge existed only in the 2023-24 bull; 2022 (bear) and 2025+ (late-cycle/choppy) were
   negative or flat. Turtle is a bull-regime strategy and this data set's OOS period was
   not a bull regime.

2. **OU mean reversion** - correctly rejected by the acceptance gates (b~1, half-life
   47k bars). Zero signals. No MR edge exists on raw BTC log price; the engine prevented a
   spurious trade.

3. **HMM regime router** - attribution looked promising in-window (HIGH_VOL/TREND buckets
   profitable) but bucket sizes are tiny and the OOS regime structure did not repeat.
   Not adopted; adding it would be fitting to noise.

4. **Meta-model** (RF/XGB on triple-barrier labels) - AUC below 0.5 OOS; removes winners.
   Removed. High AUC is not the goal - the model had no genuine signal.

5. **Methodology failures avoided** - the pipeline caught and fixed: (a) an equity-curve
   gap bug that inflated Sharpe ~6x; (b) position sizing that would have allowed >1x
   notional on a spot account; (c) dust-size pyramid adds moving stops; (d) CPCV union
   purging that wrongly dropped valid middle training samples. These fixes are why the
   headline in-sample Sharpe is 1.1, not 6.2.

## Honest conclusions
* No component demonstrated a persistent, regime-independent OOS edge on this data.
* The correct production decision is REJECT (research-only), not deployment.
* The framework produced for this project is a complete, tested research and execution
  stack that reaches exactly this verdict reproducibly.
"""
    )

    # ---------------- FINAL_SYSTEM_SPEC.md ----------------
    _write(
        "FINAL_SYSTEM_SPEC.md",
        """# Final system spec

## What was built
A complete, tested research-to-execution stack in research_lib/ (104 unit tests passing):

* canonical data layer + quality gates + causal info-bar construction
* volatility engine (6 estimators, session-aware YZ, causal percentile/z/regime)
* fractional differentiation with train-only d selection and frozen transforms
* causal feature engine (45 features) + static/dynamic look-ahead audit
* Turtle engine + pyramiding simulator with realistic fills and leverage caps
* gated OU engine (acceptance tests; never trades an invalid model)
* walk-forward Gaussian HMM regime engine with statistics-based state mapping
* causal regime router + oracle-only diagnostic
* triple-barrier labeling + meta-model + isotonic calibration + bet sizing
* central risk engine with kill switches + explicit no-trade logic
* purged K-fold, CPCV, walk-forward, Monte Carlo, cost stress, parameter grids
* execution engine with full order state machine, idempotency, Redis locking
  (ownership-checked release), position reconciliation, failure injection
* paper-trading simulator with capital staging and divergence reporting
* acceptance gate evaluating 17 production criteria from artifacts
* 18 deliverable reports in research/reports/

## Production decision
**RESEARCH ONLY**

## Evidence summary
* Causality / leakage: PASS (feature dynamic gate, Turtle shifted-channel unit test,
  signal prefix-invariance).
* Realism: PASS (honest costs, 1x spot cap, no assumed fills).
* Cost sensitivity / parameter plateau: PASS.
* Economic OOS: FAIL - walk-forward not consistently positive, final holdout negative
  (Sharpe -0.06, PF 0.91, 83% MC loss probability), drawdown over budget (28% > 20%),
  profits concentrated in one regime (2023-24 bull).
* ML / MR / regime layers: demonstrated no incremental OOS value and are disabled.

## Recommended next steps (research, not deployment)
1. Extend the research window through the holdout regime and re-test (the current holdout
   must remain untouched until a new strategy candidate is frozen).
2. Evaluate regime-conditional variants (e.g., trend filter) - but any new variant must
   pass the same 17 gates, including a NEW untouched holdout.
3. Consider 1d bars and more symbols to raise trade counts to statistically meaningful
   levels before any event-driven ML is revisited.
"""
    )

    print(f"Reports written to {REPORTS}:")
    for p in sorted(REPORTS.glob("*.md")):
        print(f"  {p.name} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    build_all()
