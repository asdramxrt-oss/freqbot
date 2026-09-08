# Blueprint Implementation Report: Pivot to Pairs / ML / Execution

**Date:** 2026-09-08
**Blueprint source:** `bot_success_blueprint.md`
**Scope:** Full implementation and honest evaluation of the blueprint's 4-phase pivot
**Overall finding:** The blueprint's engineering proposals were implemented and validated;
its core alpha assumptions were tested rigorously on real 15m data and are **not supported
by the evidence** on the 2022-2026 window. Final verdict for the new candidate:
**RESEARCH ONLY (8/17 gates)** — all economic gates fail.

---

## Phase 1 - Strategic Pivot (pairs trading, higher frequency)

### Implemented
| Item | Where | Notes |
|---|---|---|
| 15m market data (4 pairs) | `research/data`, `research/data_holdout` | ~120k bars/pair research + holdout, Binance spot |
| Perpetual funding histories | `research/data/funding/` | BTC/ETH perp funding: mean ~+0.006%/8h |
| Log-ratio spread construction | `strategies/pairs.py::log_ratio_series` | inner-join aligned, spread = log(A/B) |
| Causal local-mean z-score | `strategies/pairs.py::causal_local_z` | trailing-window mean/std ONLY; truncation-invariance unit-tested |
| Walk-forward OU tradability gate | `strategies/pairs.py::_quick_ou_valid` | reuses the OU acceptance tests causally per window |
| z-entry / z-exit / time-stop signals | `strategies/pairs.py::generate_signals` | +1 long A/short B, -1 short A/long B |
| Two-leg dollar-neutral backtester | `strategies/pairs_sim.py::PairsSimulator` | per-leg adverse fills at next open, per-leg costs, exact per-leg notional at fill price; sized on CURRENT equity (drawdown de-risks) |
| Tests | `tests/test_pairs.py` (7) | causality, dollar-neutral PnL, MR profitability on synthetic, real-data run |

### Honest evidence (research window 2022-01..2025-06, 15m)
1. **Pre-screen**: every crypto log-ratio (ETH/BTC, SOL/BTC, BNB/BTC, SOL/ETH) trended
   strongly across 2022-2025 (e.g. ETH/BTC from -2.53 to -3.22). Whole-window
   cointegration/OU would have repeated the earlier OU failure — hence the local-mean
   design.
2. **Only 30-38% of 14-day windows pass the OU stationarity gates** — the cointegration
   premise is weak for these pairs.
3. **Every configuration loses** (window 7d-112d, z 2.0-3.0, with/without OU gate):

| window | z | OU gate | pooled pnl (3-4 ratios, $100k book) |
|---|---|---|---|
| 7d | 2.0 | no | -$262k |
| 14d | 2.0 | no | -$212k |
| 28d | 2.0 | no | -$164k |
| 14d | 2.0 | yes | -$169k (gate reduces damage, never creates profit) |
| 112d (ETH/BTC) | 2.0 | no | -$413 (approaches zero as trades shrink) |

4. Gross (pre-fee) is NEGATIVE: pooled gross -$127k on $100k — **this is not a fee/execution
   problem** (the blueprint's fee-bleed hypothesis is falsified for pairs).
5. Sub-period: ETH/BTC loses in every year (2022-2025). Even in 2022 (60% win rate) net is
   negative because reversion wins are small while trend-continuation losses are large.
6. **Final untouched holdout** (2025-06..2026-09): ETH/BTC -$14.3k, SOL/BTC -$11.4k, pooled
   -$25.7k (Sharpe -2.13, PF 0.39). The edge does not exist OOS.

**Conclusion for Phase 1:** The market-neutral pairs pivot does not produce a tradeable edge
on this data. The mechanism is clear from the trade records: when ETH/BTC is 2-sigma
"cheap" in a persistent downtrend, buying the spread catches a falling knife; local
reversion captures only small wiggles while the position carries full trend risk.

---

## Phase 2 - Machine-learning pivot

### Implemented
| Item | Where | Notes |
|---|---|---|
| Volatility-spike prediction model | `ml/volspike.py` | predicts YZ-vol over a horizon exceeding the p90 of TRAIN vol; XGBoost on causal vol features |
| Train-only label threshold | `ml/volspike.py` | test labels never depend on test-period vol levels (no leakage) |
| Honest chronological evaluation | `ml/volspike.py::evaluate_vol_spike_filter` | 60/40 chronological split, AUC + filter-density reported |
| Vol filter applied to pairs & Turtle | research runners | blueprint's stated use: do not enter MR trades when a vol expansion is predicted |
| Tests | `tests/test_blueprint_p23.py` | GARCH learnability (AUC>0.55), label-future-only, causal features |

### Honest evidence
1. **Vol-spike prediction is a REAL OOS capability** (the blueprint's Phase 2.1 instinct is
   directionally correct):

| pair | AUC | base rate |
|---|---|---|
| BTC/USDT 15m | **0.863** | 5.8% |
| ETH/USDT 15m | **0.822** | 6.9% |
| SOL/USDT 15m | **0.897** | 4.3% |

   Unlike direction prediction (AUC <0.5), volatility clustering is genuinely learnable.
2. **As an entry filter on the (negative) pairs strategy it reduces damage but cannot
   create edge**: ETH/BTC unfiltered -$38.5k vs vol-filtered -$14.1k (63% loss reduction,
   blocking 96 entries). Consistent with the blueprint's mechanism (MR entries in vol
   expansions get run over) — but the residual is still negative.
3. **Applied to Turtle (the positive-gross strategy) the filter REDUCES profit**:
   research $99.9k -> $68.1k (blocked entries were net-profitable); holdout -$2.4k -> -$1.8k
   (marginal, still negative). A risk filter is not an alpha source.
4. **Meta-labeling with 10x more events still shows no signal**: 318 pooled pairs events
   (vs 79 for Turtle) -> XGB AUC 0.473, RF AUC 0.505 on a chronological split. The
   blueprint's claim that higher trade frequency (target 500-1500/yr) unlocks the
   meta-model is NOT met by this data (15m pairs yield ~90-200 events/yr, and the model
   still cannot separate winners from losers). A probability filter on the test set reduced
   the loss (-$34k -> -$15.6k) only by trading less — not by finding edge.

**Conclusion for Phase 2:** the ML objective pivot to volatility is validated as a
prediction capability and is a legitimate risk-management input; it is NOT a standalone
edge and does not make either base strategy profitable.

---

## Phase 3 - Execution and position sizing

### Implemented
| Item | Where | Notes |
|---|---|---|
| Maker/post-only execution model | `risk/maker.py` | post-only rests at the touch, fills probabilistically per bar, falls back to taker after max_wait; maker fee < taker fee |
| Honest adverse-selection handling | `risk/maker.py` | maker fills are never assumed free; a fill implies the market traded through your level |
| Fractional Kelly sizing | `ml/metalabel.py` | f* = p - (1-p)/b from calibrated p and realized win/loss; quarter-Kelly default; 0 when no edge |
| Perp funding (carry) model | `research/data/funding/` | net funding on a hedged ETH/BTC book ~ +0.0005%/8h (negligible carry) |
| Tests | `tests/test_blueprint_p23.py` | maker touch pricing, probabilistic fills, taker fallback; Kelly monotonic/zero-edge |

### Honest evidence
1. **Maker savings are real but insufficient**: on the pooled pairs trade set, fees were
   $12.6k; a best-case 60% maker conversion saves ~$3.8k. Gross is -$127k — execution
   optimization cannot rescue a negative-gross strategy.
2. **Fractional Kelly correctly refuses to trade a no-edge system** (f*<=0). Its value
   would only materialize on a calibrated positive-edge system, which neither strategy is.
3. The execution/risk stack (order state machine, Redis locking, reconciliation, kill
   switches, per-leg fills) remains intact and unit-tested from the earlier build.

**Conclusion for Phase 3:** execution improvements are correctly implemented and reduce
costs, but no execution layer can create edge where gross PnL is negative.

---

## Phase 4 - Revised validation

### Implemented / run
* Walk-forward semi-annual windows on the pairs strategy: **all 7 windows (2022-2025)
  negative** on ETH/BTC (unlike Turtle, which was at least positive in the bull windows).
* Final untouched holdout executed once for the pairs candidate.
* The blueprint's proposed gates (WFO on 15m, strict Sharpe>1.5 across regimes, MC
  p(loss)<5%) were applied to the evidence: the pairs candidate fails every economic gate
  before any Sharpe gate is even reached.

---

## Acceptance verdict (17-gate)

* Turtle 20 (previous candidate): RESEARCH ONLY (9/17) — positive in-sample economics,
  fails OOS (regime dependence).
* Pairs local-z (this pivot): RESEARCH ONLY (8/17) — the passing gates are methodological
  (causality, uniform-failure structure, execution realism); **all economic gates fail**
  (walk-forward 0/7, holdout negative, gross negative even at zero cost, drawdown over
  budget).

The single validated positive capability produced by this blueprint work is the
**volatility-spike prediction model (OOS AUC ~0.85)**, which is best used as a risk/filter
input, not as an alpha generator.

## File map (blueprint work)
```
research_lib/strategies/pairs.py, pairs_sim.py     pairs engine + two-leg backtester
research_lib/ml/volspike.py                        vol-spike prediction + filter
research_lib/risk/maker.py                         post-only execution model
research_lib/ml/metalabel.py (Kelly additions)     fractional Kelly sizing
research_lib/tests/test_pairs.py, test_blueprint_p23.py
research/run_pairs.py / run_volspike.py / run_metalabel_pairs.py / run_pairs2.py /
           run_turtle_volfilter.py                 research runners
research/artifacts/pairs/, research/artifacts/blueprint/   evidence
research/artifacts/acceptance/verdict_pairs.json   pairs acceptance verdict
```
