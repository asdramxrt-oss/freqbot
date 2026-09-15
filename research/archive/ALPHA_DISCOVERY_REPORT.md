# Alpha Discovery Report: Trading the Predicted Volatility Spike

**Date:** 2026-09-08
**Brief:** `deepseek_v4_flash_task.md`
**Objective:** Engineer a fundamentally new alpha with mathematically positive gross PnL
using the validated volatility-prediction model (OOS AUC 0.82-0.90) — with the vol spike
as the *signal*, not a filter.
**Outcome:** The brief's 5/7-positive-gross WFO target was **NOT reached**. Four
structurally distinct strategy families were implemented and rigorously tested under strict
walk-forward discipline; all fail. This report documents the evidence chain and why the
negative result is the honest scientific outcome rather than a failure of effort.

---

## 1. Approach

The brief's premise is sound and was honored: the vol-spike model is the project's only
validated predictive capability (per-window OOS AUC 0.86-0.94 reproduced in every WFO
fold), and previous work proved it fails as a *filter* (blocks profitable trend entries,
cannot rescue negative-gross mean reversion). The remaining possibility the brief
identifies is that the spike itself carries tradeable information. We tested that
hypothesis as rigorously as the infrastructure allows.

### Methodology
* Data: 15m Binance BTC/USDT + ETH/USDT, research window 2022-01..2025-06. Final holdout
  (`research/data_holdout`) **never touched**.
* Model: XGBoost predicting P(realized vol over next 96 bars > train-only p-percentile).
  Retrained per WFO window on data **strictly before** the window (expanding); spike
  threshold computed on training history only. No window observation enters training.
* Causality: a decision at close `t` fills at open `t+1` (unit-tested truncation
  invariance). Positions sized by the risk engine at 1x with documented futures-style
  shorts.
* Test windows: six semi-annual windows 2022H2..2025H1. (2022H1 is structurally
  untestable — no prior history exists to train the model, so the brief's "7 regimes"
  is 6 testable windows; reported honestly.)
* Gross = next-open signal PnL with no costs. Net = taker fee + half-spread + slippage.

## 2. Hypotheses tested (all implemented, all walk-forward evaluated)

| ID | Family | Idea | Result (best config) |
|---|---|---|---|
| H1 | Momentum-ignition (`follow`) | predicted spike + recent up-move → long | BTC 3/6 gross-pos; gross ≈ −follow (see below) |
| H2 | Vol-crush fade (`fade`) | predicted spike + recent down-move → long (fade) | ETH 4/6 gross-pos but total gross **+$564** (noise); net −$13k |
| H3 | Asymmetric dip-buy (`dipbuy`) | long only after spike + dip | BTC 2/6 gross-pos |
| H4 | Vol-gated breakout (`breakout`) | range breakout joined only inside predicted-spike windows (direction-agnostic entry) | best 3/6 gross-pos; gross negative in 15/16 configs |

Threshold sweep (p_entry 0.2-0.5), two vol labels (p75, p90), two breakout ranges, both
pairs: **no configuration reached 5/6 positive-gross windows.**

## 3. Why it fails — the evidence chain

### 3.1 Momentum direction is provably uninformative
`fade` and `follow` are **exact mirror images window-by-window** (identical trade sets,
flipped sides; gross_fade = −gross_follow). This is mathematical, not empirical: the
direction rule only picks a *side*, never which spikes to trade. If pre-spike momentum had
directional content, the two rules could not be exact opposites. The BTC fade "+$18k" is
therefore bull-window luck (two windows dominate) and the ETH fade "+$564" is noise.

### 3.2 Directional agreement is at chance with real statistical power
Direct diagnostic: after predicted spikes, measure agreement of the forward move's sign
with (a) medium-term trend (200-bar SMA slope) and (b) short momentum (8-bar):

| pair | label | events | mean \|agreement − 0.5\| (trend) | mean \|agreement − 0.5\| (mom) |
|---|---|---|---|---|
| BTC | p90 | 853 | 0.124 | — |
| BTC | p75 | 5,482 | 0.059 | — |
| ETH | p90 | 1,783 | 0.081 | — |
| ETH | p75 | 10,970 | 0.076 | — |

With 5k-11k events the test has ample power; agreement hugging 0.5 with sign flipping
window-to-window (27-67%) is the signature of **no directional signal**. This is not a
data-quantity problem (the brief's "more trades" hypothesis is thereby also tested and
falsified — p75 provides the volume and still shows nothing).

### 3.3 The residual "edge" is leveraged regime beta
Spike-conditional forward returns track the *window's own drift*: 2022 bear windows
−83/-126 bps after spikes; 2023-24 bull windows +98 to +291 bps; 2025 mixed. A model that
predicts *magnitude* concentrates the market's drift into short windows; trading it
directionally is a leveraged long/short bet on the prevailing regime — the same disease
that killed Turtle, not a new edge. Net of taker costs, every configuration loses because
the gross is ~zero and the model fires often enough to pay fees.

### 3.4 The magnitude forecast cannot be monetized directionally
Crypto spot/futures (this stack) has no instrument to monetize predicted magnitude without
a directional view: no options/straddles/variance swaps. The vol-spike model's correct and
validated role is **risk management** (vol-targeted sizing, exposure gating) — exactly the
filter role already shown not to be an alpha source. This is a structural property of the
signal and the instrument set, not a parameter miss.

## 4. Steps 1-5 of the brief — disposition

* Step 1 (3 hypotheses): done — four families drafted and code-ready (H1-H4 above).
* Step 2 (implement the soundest as a strategy class): done —
  `research_lib/strategies/volspike_alpha.py` + 10 unit tests; integrates with the
  EventBacktester / risk engine.
* Step 3 (WFO over 2022-2025): done — expanding walk-forward, six testable windows,
  per-window gross and net recorded (`research/artifacts/alpha/*`).
* Step 4 (iterate to ≥5/7 positive gross): attempted honestly across four structurally
  distinct families and a wide parameter/threshold/label space. **Not achieved.** Per the
  brief's own "without overfitting" rule, iteration was stopped once the absence of
  directional signal was established with high statistical power — continuing to tune
  would be curve-fitting noise, and the project's standing rules forbid manufacturing a
  pass.
* Step 5 (17 gates / holdout): not reached. The candidate failed walk-forward, so the
  untouched holdout stays untouched (a failed candidate earns no holdout run).

## 5. Acceptance verdict

**volspike_directional_15m → RESEARCH ONLY (8/17 gates)** — the passing gates are purely
methodological (causality, leakage, execution realism, sufficient statistical power,
uniform-failure structure). Every economic gate fails. Full detail in
`research/artifacts/acceptance/verdict_volspike_alpha.json` (written by
`research/run_alpha_final.py`).

## 6. What WAS learned (constructive)

1. The vol-spike model is a **reliable magnitude forecaster** — per-window AUC 0.86-0.94
   reproduced in every fold of every WFO run, at both p75 and p90 labels.
2. Predicted vol spikes **concentrate the market's own drift** — a fact that could inform
   a *regime-conditional* risk overlay, but is not tradeable directionally on this
   instrument set.
3. The direction rules' mirror symmetry and the chance-level agreement diagnostics are
   reusable, honest tests that future alpha candidates should be required to pass before a
   holdout run.

## 7. File map
```
research_lib/strategies/volspike_alpha.py        strategy (H1-H4 families)
research_lib/tests/test_volspike_alpha.py        10 tests (causality, breakout, time stop)
research/run_event_study.py                       pooled conditional-return screen
research/run_event_study_windows.py               regime-decomposed screen
research/run_alpha_wfo.py                         per-window WFO (follow/fade/dipbuy)
research/run_alpha_thresholds.py                  threshold sweep
research/run_alpha_breakout.py                    H4 breakout WFO
research/run_alpha_final.py                       acceptance verdict writer
research/artifacts/alpha/*                        all evidence (json)
research/artifacts/acceptance/verdict_volspike_alpha.json
```
