# Phase Execution Log

Format per the brief: PHASE / STATUS / IMPLEMENTED / TESTS / RESULTS / OOS RESULTS /
LEAKAGE CHECK / REGRESSION CHECK / PERFORMANCE CHANGE / FILES / KNOWN ISSUES / NEXT PHASE.

---

## PHASE 0 - Repository audit
STATUS: COMPLETE
- The repository is a pristine upstream Freqtrade framework checkout with NO custom
  strategies, research code, git history or market data at audit start. The premise of
  "existing Turtle/OU implementations with known issues" did not hold for this checkout;
  every strategy component is net-new.
- Env established: Python 3.12 venv, freqtrade 2026.9-dev editable, pandas 3.0.5,
  statsmodels, scikit-learn, xgboost, hmmlearn, fakeredis. Framework unit baseline:
  1919 passed / 1 skipped. Reference backtest of the framework's own StrategyTestV3
  executes end-to-end (losing baseline: PF 0.93 on the tiny bundled fixtures).
- Deliverable: research/ARCHITECTURE_AUDIT.md (18 mandated sections, all claims verified).

## PHASE 1 - Canonical data layer
STATUS: COMPLETE
- Canonical schema + loaders (freqtrade feather -> normalized frames) + data-quality
  validator (duplicates, missing, order, impossible OHLC, non-positive prices, abnormal
  volume, flat-run artifacts) with hard/soft severity and a pass gate.
- Real data acquired: Binance spot 1h+4h for BTC/ETH/SOL/BNB (research window 2022-2025)
  plus a strictly separate final holdout directory (2025-06..2026-09).

## PHASE 2 - Information-driven bars
STATUS: COMPLETE
- Causal volume/dollar/imbalance bar builders (trailing incomplete bar never emitted).
  Time bars remain the research default; comparison harness available and documented.

## PHASE 3 - Volatility engine
STATUS: COMPLETE
- close-to-close, EWMA, Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang (session-aware;
  YZ degenerates to RS for 24/7 crypto with no session boundary) + causal
  percentile/zscore/regime. Fixed a wrong RS/GK formula pairing found by unit tests.
- Empirical: all six estimators agree at sane magnitudes on BTC 4h (~0.5% per-bar vol).

## PHASE 4 - Fractional differentiation
STATUS: COMPLETE
- Finite-window causal fracdiff; d selected on TRAIN folds only across [0,1] step 0.05 with
  ADF+KPSS, frozen and applied to later data (unit-tested for no refit/leak). Fixed weight
  alignment bug. On a pure random walk it correctly selects d ~ 0.95.

## PHASE 5 - Feature engine + audit
STATUS: COMPLETE
- 45 causal features (price/vol/momentum/MR/fractional/regime incl. DFA-Hurst on returns).
- Static look-ahead scanner + dynamic truncation causality gate; full feature set PASSES
  the gate (historical values invariant to future data).

## PHASE 6 - Turtle engine
STATUS: COMPLETE
- Shifted Donchian (high.shift(1).rolling(N)) breakout engine, ATR(20), 2N stop,
  pyramiding state machine, fresh-cross entries (prevents stop-out churn).
- Simulator with realistic fills; found and fixed: >1x notional sizing bug on spot,
  dust-add stop-moving artifact, shorting without a margin account.

## PHASE 7 - OU engine
STATUS: COMPLETE
- AR(1) fit with strict acceptance gates (0<b<1, b-CI, half-life, residual ADF, rolling-b
  stability, R2, sample size). Fixed a rolling-stability bug (was testing the intercept).
- On real BTC log price the model is correctly REJECTED (b=1.0, half-life 47k bars) and
  produces zero trades - the honest outcome, no spurious MR.

## PHASE 8 - HMM regime engine
STATUS: COMPLETE
- Gaussian HMM with expanding walk-forward fitting, train-only scaler, diagonal covariance;
  states characterized by measured statistics (vol/ac/mean/duration) not by state id.
- Fixed an off-by-one infinite loop that would have hung production runs.

## PHASE 9 - Regime router
STATUS: COMPLETE
- Causal router (TREND->turtle, LOW_VOL_MR->ou, HIGH_VOL/UNCERTAIN->no_trade) + oracle-only
  diagnostic. Regime attribution of Turtle trades produced in-window (TREND/HIGH_VOL
  profitable, UNCERTAIN losing) but with sample sizes too small to adopt.

## PHASE 10 - Triple barrier
STATUS: COMPLETE - volatility-scaled labeling (+1/-1/0) with MFE/MAE; 79 Turtle events
labeled on BTC research window (44 win / 35 loss).

## PHASE 11-13 - Meta-model, calibration, bet sizing
STATUS: COMPLETE
- RF/XGB meta-model on turtle entries, isotonic calibration, probability-gated bet sizing.
- OOS (chronological split): XGB AUC 0.471, RF AUC 0.393 - BELOW random. Meta-model
  removed per the "ML that degrades is removed" rule.

## PHASE 14 - Central risk engine + no-trade logic
STATUS: COMPLETE - per-trade/symbol/portfolio/leverage/daily/weekly/drawdown/
concentration/regime limits, global kill switches, explicit no-trade gate. 14 tests.

## PHASE 15 - Purged K-fold
STATUS: COMPLETE - event-aware purging + embargo; overlap-removal proof tests.

## PHASE 16 - CPCV
STATUS: COMPLETE - full-coverage path enumeration (15 paths for 6x2), per-group purging.
Fixed union-purge bug that wrongly dropped valid middle training samples.

## PHASE 17 - Walk-forward engine
STATUS: COMPLETE - TRAIN/VALIDATE/TEST with strict index separation; turtle evaluated on
6-month windows: only 5/7 positive (2022 bear negative, 2025 fading) -> no time-stable edge.

## PHASE 18 - Monte Carlo
STATUS: COMPLETE - reshuffle/bootstrap/slippage/fee/missed-trade sims. Research-window MC
looked robust (p loss ~0) but the HOLDOUT MC is decisive: p(pnl<0)=0.83, worst-5% DD -35%.

## PHASE 19 - Ablations
STATUS: COMPLETE - OU rejected at model gate; regime router not adopted (tiny buckets);
meta-model removed (AUC<0.5). Plain Turtle is the survivor - and it failed the holdout.

## PHASE 20 - Transaction-cost stress
STATUS: COMPLETE - 1x/1.5x/2x/3x; retains 73% of net profit at 3x costs in-window. PASS.

## PHASE 21-23 - Execution engine + Redis + reconciliation
STATUS: COMPLETE - order state machine, idempotency keys, NX/PX lock with ownership-checked
Lua release, SETNX single-claim, position reconciliation with kill switch on mismatch,
failure injection (network/timeout/partial-fill/reject/crash) all fail-safe. 29 tests.

## PHASE 24-26 - Paper trading / shadow / limited capital
STATUS: COMPLETE (implementation) - PaperTradingSimulator replays decision-time-only with
capital staging and divergence reporting. Live/shadow deployment NOT reached: the final
verdict is RESEARCH ONLY, so no capital stage is justified.

## PHASE 27-52 (research rules, gates, reports, final decision)
STATUS: COMPLETE
- Look-ahead gate: PASS (prefix-invariant signals; feature causality gate).
- Acceptance gate: 9/17 PASS -> **RESEARCH ONLY** (not PRODUCTION READY / PAPER ONLY).
- 18 deliverable reports generated in research/reports/.
- Final decision and evidence in FINAL_SYSTEM_SPEC.md and PRODUCTION_READINESS.md.
