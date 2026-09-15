# Architecture Audit — Phase 0

Audit date: 2026-09-07
Auditor: adversarial quantitative research pass (per spec)
Repo under audit: `freqtrade-develop/freqtrade-develop` (Freqtrade framework source checkout)

> Scope note: The brief presupposed an existing custom trading system with Turtle/OU/HMM/ML
> strategies inside this repository. **This checkout contains none of those.** It is a pristine
> upstream **Freqtrade framework** source tree (version `2026.9-dev`), with an empty `user_data/`
> scaffold (no strategies, no downloaded market data, no research code, no configs, no git repo,
> no venv at audit start). The audit below therefore records exactly what is present and verified,
> and explicitly flags what is **absent** rather than inventing an inventory. Everything claimed
> below was verified by reading the actual source or running the actual commands.

---

## 0. Baseline environment (verified by execution)

| Item | Value |
|---|---|
| Python | 3.12.10 (CPython, 64-bit) |
| pip | 26.2.1 (system), 25.0.1 (venv) |
| freqtrade | `2026.9-dev` (editable install from this tree) |
| pandas / numpy / scipy | 3.0.5 / 2.5.3 / 1.18.1 |
| ccxt / TA-Lib | 4.5.77 / 0.7.1 |
| scikit-learn / optuna / xgboost / joblib | 1.9.0 / 5.0.0 / 3.4.1 / 1.5.3 (downgraded from 1.6 — see Known bugs) |
| pytest / pytest-xdist | 9.1.1 / 3.8.0 |
| Git repo | **none** (`.git` absent; `git status` fails) |
| Market data | none downloaded; only bundled unit-test fixtures in `tests/testdata/` |
| User strategies | **none** (`user_data/strategies/` has only `.gitkeep`) |

### Baseline test result (verified by execution)

Command: `pytest tests/optimize tests/strategy tests/data tests/util tests/leverage tests/freqtradebot tests/persistence tests/plugins -q --timeout=600`

Result: **1919 passed, 1 skipped, 193 warnings** (excludes `tests/exchange_online` — requires live network — and `tests/freqai` — requires datasieve/tensorboard extras; these are framework-optional, not core).

1920 tests are collected for those areas; the single skip is an intentional `@pytest.mark.skip`.

### Baseline backtest result (verified by execution)

The backtest pipeline (config validation → exchange resolution → data load → indicator computation → signal backtest loop → fee model → reporting) executes end-to-end on this machine.

Reference run — StrategyTestV3 (the framework's own test strategy) on bundled spot data:

```
Config : tests/testdata/config.tests.json (exchange=gate, spot, BTC staking)
Data   : ETH/BTC, LTC/BTC, XLM/BTC, ADA/BTC 5m spot (tests/testdata fixtures)
Range  : 2018-01-10 06:35 → 2018-01-30 04:50 (19 days, 5,739 candles)
Trades : 134 (7.05/day)
Win%   : 51.5 (69 win / 9 draw / 56 loss)
Profit factor : 0.93
Expectancy    : -0.10 ratio / ~-0.00002 BTC per trade
Sharpe / Sortino / Calmar (closed): -2.57 / -2.29 / -26.06
Abs profit    : -0.00274 BTC on 1000 BTC start (-0.00%)
Max DD (balance) : 0.0126 BTC (0.00%)
Fees used     : 0.20% (gate worst-case tier)
```

This is a **losing** baseline — useful: any future strategy must beat a clearly-negative reference, and the framework's own test strategy is not a profit benchmark, just a pipeline sanity check.

---

## 1. Existing architecture

The repository is the **Freqtrade** crypto trading framework. It is a platform, not a strategy.
Its architecture is a single-process bot plus research CLI commands:

```
freqtrade/                         # framework package
├── main.py                        # CLI entry point
├── freqtradebot.py                # live/dry-run bot loop (process → refresh → analyze → trade)
├── worker.py                      # process supervisor / thread wrapper
├── commands/                      # CLI subcommands (backtesting, hyperopt, freqai, lookahead, ...)
├── configuration/                 # config load + jsonschema validation + TimeRange
├── data/                          # dataprovider, history loaders, OHLCV converter, metrics
│   ├── dataprovider.py            # strategy data access + causal slice guard (backtest)
│   ├── converter/converter.py     # ohlcv_to_dataframe, clean, fill-missing, trim
│   └── history/datahandlers/      # feather/parquet/json readers (idatahandler pattern)
├── strategy/                      # IStrategy interface + decorators + validation
│   └── interface.py               # populate_indicators/entry/exit + 20+ callbacks
├── optimize/                      # backtesting.py, hyperopt/, analysis/(lookahead, recursive)
│   ├── backtesting.py             # vector+row-loop hybrid backtester
│   ├── hyperopt/                  # optuna-based optimizer
│   └── hyperopt_loss/             # sharpe/sortino/calmar/maxdd/etc objectives
├── freqai/                        # ML training/inference subsystem (walk-forward)
│   ├── freqai_interface.py        # train/predict/backtest loops
│   ├── data_kitchen.py            # windowing, feature/label slicing, pipelines
│   ├── data_drawer.py             # model persistence
│   └── prediction_models/         # LGBM/XGBoost/RF/sklearn classifiers+regressors, RL
├── exchange/                      # ccxt adapter layer (per-exchange subclasses)
├── plugins/                       # pairlists + protections (risk circuit breakers)
├── persistence/                   # SQLAlchemy Trade/Order models, sqlite
├── rpc/                           # telegram / webhook / api_server / external message bus
├── leverage/                      # interest + liquidation price math
├── wallets.py                     # balance / stake sizing / position exposure
└── enums/                         # ExitType, TradingMode, SignalDirection, ...
```

Notable absences for this task's purposes:

- **No custom strategies** anywhere (`user_data/strategies/` empty).
- **No research/ directory** (this file creates the first one).
- **No Turtle / OU / HMM / meta-labeling / CPCV / walk-forward / Monte-Carlo code exists** in the
  tree. The task's "existing research has identified weaknesses in previous MR implementations"
  cannot refer to this checkout. There is no prior MR implementation here to audit or preserve.
- **No git repository**, so there is no commit history, no branch discipline, and nothing to
  revert to. (Recommendation: `git init` at project root before Phase 1 so every phase is
  committed and reproducible.)

---

## 2. Data flow (verified in code)

```
Exchange REST/WS ──ccxt──▶ freqtrade/exchange/exchange.py
        │  download-data command
        ▼
user_data/data/<exchange>/[futures/]<pair>-<tf>.feather   (raw OHLCV per candle-type)
        │  history/datahandlers
        ▼
freqtrade/data/converter/converter.py::ohlcv_to_dataframe
        │   * floor date to timeframe grid
        │   * groupby(date) to drop duplicate ticks  (clean_ohlcv_dataframe)
        │   * drop last (incomplete) candle by default
        │   * fill missing candles → ffill close, synthesize OHLC=close, volume=0
        ▼
DataProvider (dp) ──▶ strategy.populate_indicators() on the FULL requested window
        ▼
Backtesting._get_ohlcv_as_lists: signals shifted by 1 candle, startup trimmed
        ▼
row loop: entries → order fills → exits (ROI/stoploss/signal) → fee/wallet updates
```

Key verified facts:

- Candle timestamps are floored to the timeframe grid; duplicate timestamps are aggregated away.
- **Missing candles are back-filled by forward-filling close and synthesizing OHLC = close,
  volume = 0** (`ohlcv_fill_up_missing_data`). This is "repair" data and is NOT distinguished
  from genuine data downstream — a data-quality risk (flat-price phantom candles can depress
  volatility and create fake support/resistance; fills can occur on fabricated candles).
- In backtest mode `DataProvider.get_pair_dataframe` slices informative/historical frames to
  `date < cutoff` where cutoff derives from the current decision time — the framework
  deliberately prevents dp-based look-ahead in callbacks (verified in `dataprovider.py:371-400`).
- `backtesting.py` shifts entry/exit signal columns by exactly one row before the row loop
  (`_get_ohlcv_as_lists`), so a signal computed on candle *t* is acted on at candle *t+1* open —
  the framework's core causality mechanism.
- **But indicators themselves are computed once over the full window** (all candles present),
  which the framework's own `lookahead-analysis` docs explicitly warn is the main place a
  strategy can "cheat" (e.g. whole-frame `.mean()`, `shift(-n)`, `iloc[]`, TA-lib signalperiod=1).

### Data-quality checks present
- Duplicate timestamps: aggregated (silently).
- Out-of-order: not explicitly checked; groupby+sort normalizes.
- Zero/negative prices: not validated at ingestion.
- Impossible OHLC (low>high etc.): not validated.
- Abnormal volume / gaps: only the fill-missing logger emits a pct-missing warning.
- **No canonical multi-instrument schema with bid/ask/spread/trade_count/quote_volume/session.**
  Freqtrade candles are OHLCV only. Trades (orderflow) are an optional separate stream
  (`use_public_trades`, `data/converter/orderflow.py`).

---

## 3. Signal flow (verified)

```
populate_indicators()      → columns + feature_engineering_* (freqai) 
populate_entry_trend()     → enter_long/enter_short/enter_tag
populate_exit_trend()      → exit_long/exit_short/exit_tag
        │
        ▼  backtest: shift(1) → action at next candle open
        ▼  live:     act on latest closed candle (get_entry_signal reads last row)
Backtesting._enter_trade / FreqtradeBot.execute_entry
        │
        ▼
wallet stake sizing → exchange order → fill at open (backtest) / orderbook (live)
```

Exits: ROI table, fixed/custom stoploss, trailing, exit signals, custom_exit, force/emergency,
liquidation (futures). Custom hooks exist for entry price, exit price, stake amount, position
adjustment (DCA), and leverage — all strategy-accessible.

---

## 4. ML flow — FreqAI (verified)

```
strategy.feature_engineering_expand_all/basic/standard → columns prefixed %-
strategy.set_freqai_targets()                          → columns prefixed &-
        │
        ▼
FreqaiDataKitchen splits timerange into sliding windows:
  TRAIN (train_period_days) → immediately-following TEST (backtest_period_days)
  window slides forward (split_timerange: start += bt_period each iteration)
        │
        ▼ per window: fit model on TRAIN only → predict on TEST → append
        ▼ features normalized via datasieve pipeline fit on TRAIN, transform on TEST
model persistence: user_data/models/<identifier>/sub-train-<pair>_<ts>
```

FreqAI is walk-forward by construction (train window strictly precedes test window), which is
the right *shape*. Risks observed in code:

- **Label creation can leak** if `set_freqai_targets` uses `shift(-n)` beyond the train cut —
  the strategy author owns target causality; the kitchen slices train at `date < tr_train.stopdt`
  *after* targets are computed (verified at `freqai_interface.py:347-359`), so a `shift(-k)` label
  whose horizon crosses the train/backtest boundary will incorporate post-boundary information
  into the last training rows. FreqAI's buffer/NaN handling mitigates but does not enforce.
- `buffer_train_data_candles` trims both ends of the TRAIN window *after* indicators are built —
  indicator warm-up is the intent; but the stop-trim also silently drops the most recent train
  rows, which is safe only if labels are NaN there.
- HMM/CPCV/purged-k-fold/meta-labeling: **not present** in FreqAI; the sklearn `train_test_split`
  path (`data_kitchen.make_train_test_datasets`) is used only when a model needs a held-out split
  inside the train window (`shuffle=False` default — good; but `shuffle_after_split` option exists
  and, if enabled, shuffles *within train* — acceptable; the option `reverse_train_test_order`
  exists and would be a leak if misused — it is off by default).

---

## 5. Risk flow (verified)

Three layers exist:

1. **Strategy-level**: `stoploss`, `minimal_roi`, `trailing_stop*`, `max_open_trades`,
   `position_adjustment_enable` (DCA caps via `max_entry_position_adjustment`).
2. **Protections plugin** (`plugins/protections/`): CooldownPeriod, StoplossGuard,
   LowProfitPairs, MaxDrawdown. These are *reactive* circuit breakers on recent trade outcomes,
   not ex-ante risk budgeting.
3. **Wallet/position limits** (`wallets.py`): `tradable_balance_ratio`, `max_open_trades`
   stake splitting, `stake_amount` (fixed/unlimited), min/max stake validation,
   `amend_last_stake_amount`, leverage caps and liquidation price tracking (futures).

Absent / to be built:

- Central risk engine as a single choke point is **not** present; risk is distributed across
  strategy callbacks + wallets + protections. No per-symbol/per-strategy exposure limits, no
  daily/weekly loss limits, no correlation or concentration limits, no portfolio-level
  value-at-risk, no global kill-switch beyond protections, no stale-data / broker-state /
  reconciliation interlocks.

---

## 6. Execution flow (verified)

```
Strategy signal → FreqtradeBot.create_trade / handle_trade / adjust_position
  → exchange.create_order (limit/market/stoploss)  [ccxt]
  → Order row persisted (sqlite) with status from ccxt order dict
  → periodic sync: update_open_orders → fetch_order → reconcile DB row
  → stoploss_on_exchange handled as conditional order where supported
  → position reconciliation = Trade/Order DB vs exchange.fetch_orders on startup + loop
```

Verified properties:

- Order **state** is a persisted status string mirroring ccxt (`open/closed/canceled/...`),
  not an explicit state machine with guarded transitions.
- Entry orders are limit by default; fills in backtest assume the open price of the signal+1
  candle (or stop/ROI boundary prices within the candle via `_get_close_rate_for_stoploss`).
  Slippage is not modeled beyond price-side/orderbook assumptions; a single flat `fee` (worst
  of maker/taker from exchange, or config `fee`) is applied to every fill. **No spread model,
  no latency model, no partial-fill model, no funding-cost model in spot; futures funding rates
  are loaded as data and applied in futures backtests.**
- Idempotency keys, distributed locking, order-intent layering, broker-adapter separation,
  position-reconciliation state machine: **none present** (single-process bot makes many of
  these unnecessary *for live freqtrade* — but the task's distributed architecture is absent).

---

## 7. Known bugs / environment issues found during audit (verified)

1. **`joblib>=1.6` breaks freqtrade hyperopt**: `freqtrade/optimize/hyperopt/hyperopt_optimizer.py`
   imports `from joblib.externals import cloudpickle`, which joblib 1.6 removed. Reproduced as
   `ImportError`. Fixed in this environment by pinning `joblib<1.6` (1.5.3). This is a dependency
   hygiene issue in the project (requirements allow joblib>=1.2.0).
2. **Empty pytest logs when run via `cmd` background redirect** on this host (tooling artifact,
   not a project bug) — worked around by foreground runs.
3. **Config schema strictness**: minimal backtest configs must include full `telegram` and
   `api_server` blocks even when disabled (jsonschema `required`), which is friction for
   generating research configs (worked around via existing test configs).
4. `tests/strategy/test_strategy_parameters.py` requires `optuna` even for non-hyperopt runs
   (import-time dependency) — an environment gap, not a code bug; resolved by installing
   `optuna`/`scikit-learn`.
5. **Backfill synth-candles** (`ohlcv_fill_up_missing_data`) silently fabricates OHLC from
   previous close with volume 0 — a latent data-quality hazard for any strategy that trades
   breakouts or volume (see Data flow).

---

## 8. Look-ahead risks (framework-level, ranked)

1. **Indicators computed over the full backtest window** — the single biggest vector. Any
   strategy use of expanding (non-rolling) aggregates, `shift(-n)`, `iloc` row access,
   `pandas` future-aware ops, or TA-lib `signalperiod=1` leaks. The framework ships
   `freqtrade lookahead-analysis` (tests/optimize/analysis/lookahead.py) which detects this by
   truncating data and re-comparing signals; it is a *post-hoc detector*, not a prevention.
2. **Custom exit/entry price callbacks and `custom_stoploss`** receive the current candle's
   OHLC (`row`) plus `current_rate`; a strategy returning prices derived from the same candle's
   high/low can create intra-candle look-ahead (the framework has guards for some ROI cases but
   custom pricing is largely author responsibility).
3. **FreqAI target horizon crossing the train/backtest boundary** (see ML flow).
4. **`startup_candle_count` under-declared** → indicators are NaN/partial at the start of the
   backtest window and the first rows are trimmed by the framework; if a strategy declares too
   few, early signals are computed on partially-warmed indicators that would not exist live.
5. **Detail-timeframe (intra-candle) data** — when `timeframe_detail` is used, exit logic can
   see finer-grained candles *within* the current main candle; framework code uses only the
   detail rows up to the current sub-candle time, but strategy custom-pricing code operating on
   detail frames must be careful.

Framework mitigations that ARE present: signal `shift(1)`, dp slice cutoff in backtest,
last-candle drop on ingestion, startup-candle trim, lookahead/recursive analysis commands.

## 9. Leakage risks (beyond look-ahead)

- **Selection / multiple-testing**: hyperopt optimizes directly on the backtest window with no
  embargo, no purged CV, no untouched holdout concept built in. Running hyperopt repeatedly and
  keeping the best epoch is textbook overfitting; the framework provides
  `hyperopt-loss` functions but no honest out-of-sample gate.
- **FreqAI model caching** (`check_if_backtest_prediction_is_valid`) reuses saved predictions
  keyed by train-window stop time; if features/strategy change but the identifier doesn't, stale
  predictions silently replay (mitigation: `--freqai-backtest-live-models` and identifiers).
- **Indicator leakage via informative pairs** if the informative timeframe is *smaller* than the
  strategy timeframe — the framework forbids this at strategy load (verified
  `interface.py:172-176`).
- **Fee/cost optimism**: single worst-case flat fee, no spread/slippage/latency/funding-in-spot;
  any strategy tuned on this backtester will look better than live.
- **Survivorship**: depends on data vendor; framework does not track delisted pairs for you.

## 10. Survivorship-bias risks

- Pairlists like VolumePairList/PerformanceFilter (live only) inherently select currently-active,
  recently-strong pairs. In backtest these are banned (verified `backtesting.py:248-252`), but a
  researcher using today's whitelist to backtest history is doing implicit survivorship bias.
- No delisting/adjustment handling in the bundled data.

## 11. Overfitting risks

- Hyperopt on the full window without nested validation (highest risk).
- FreqAI hyperopt of `train_period_days`/feature lists against the backtest window.
- No parameter-perturbation or plateau analysis tools.
- No CPCV/purged/embargo utilities anywhere in the tree.
- Strategy parameters with `optimize=True` are tuned on whatever window is passed; nothing
  prevents repeated peeking.

## 12. Dead code

- `freqtrade/strategy/interface.py` deprecated shims (`populate_buy_trend`/`populate_sell_trend`,
  `check_buy_timeout`, `check_sell_timeout`, `custom_sell`) exist for v2/v1 back-compat — retained
  by design (interface versioning), not dead.
- Deprecated FreqAI data-pipeline functions (`data_cleaning_train/predict`, `normalize_data`)
  still present with deprecation warnings — transitional, framework-maintained.
- No custom dead code exists (nothing custom exists).

## 13. Duplicate code

- FreqAI keeps deprecated + new pipeline paths side-by-side (intentional migration support).
- Exchange subclasses duplicate small fee/market helpers by design (ccxt per-exchange overrides).
- No custom duplication to report.

## 14. Implemented but not wired

- `reduce_df_footprint` (memory optimization) is wired only for non-live modes.
- `use_public_trades` orderflow path exists but requires exchange + data support; orderflow
  features are experimental.
- FreqAI `ReinforcementLearner` models require the RL extras and are not exercised in default
  test runs.
- Several example configs (`config_examples/`) and templates exist for reference only.
- Nothing else of consequence is unwired — the framework is coherent.

## 15. Claimed to work but does not (in this checkout / environment)

- Nothing in the framework is misrepresented; but note the task's premise — prior Turtle/OU/MR
  "existing implementations with wiring/stop-loss/config-switch issues" — **cannot be found here**.
  If those existed in a different project/directory, they are not part of this repository and
  should not be treated as present. (Verified: `grep -ri "turtle\|ornstein\|hurst\|OU mean"`
  across the tree returns only an unrelated `test_optimize_reports.py` match.)
- The framework's own sample strategy (StrategyTestV3) backtests **negative** here, so no claim
  of out-of-the-box profitability is being made or validated.

## 16. Retain

- Freqtrade's causal signal shift(1) + dp slice cutoff machinery (don't re-invent).
- Backtester row loop, fee/wallet/position handling, precision rounding.
- `lookahead-analysis` and `recursive-analysis` tools — reuse as leakage regression gates.
- FreqAI walk-forward sliding-window trainer — reuse its window *shape* for any ML integration.
- Protections plugins (cooldown/stoploss-guard/maxdd/low-profit) — reuse inside a risk engine.
- Exchange adapter + ccxt layer; persistence models.
- Existing test harness (1919 green tests) as the regression floor.

## 17. Redesign / build (the actual work for later phases)

Everything the task lists as missing must be *built* (not refactored):

1. **Canonical normalized data layer** with strict quality checks and raw/transformed separation
   (keep the freqtrade datahandlers for I/O; add a normalized, validated research frame).
2. **Information-driven bars** (volume/dollar/imbalance) — causal construction + comparison.
3. **Volatility engine** (close-to-close, EWMA, Parkinson, Garman-Klass, Rogers-Satchell,
   Yang-Zhang with explicit session handling for 24/7 crypto).
4. **Fractional differentiation** with train-only d-selection and frozen weights.
5. **Causal feature engine + feature audit** (automatic future-reference detection).
6. **Turtle engine** (shifted Donchian, 2N stop, pyramiding state machine).
7. **OU engine** (DFA/Hurst filter, b∈(0,1), half-life, time stop).
8. **Gaussian HMM regime engine** with expanding walk-forward fitting and
   statistics-based state characterization.
9. **Regime router** (router must not be allowed to see future state).
10. **Triple-barrier labeling + meta-model** (post-signal, probability-calibrated).
11. **Purged K-fold / embargo / CPCV / walk-forward / untouched final holdout.**
12. **Central risk engine** (per-trade/per-symbol/portfolio/daily/DD limits, kill switch) —
    the single order-intent gate.
13. **Transaction-cost model** (spread + slippage + fees + latency + partial fills) + stress.
14. **Monte Carlo / parameter robustness / ablation / regime attribution tooling.**
15. **Execution layer** (idempotency, order state machine, reconciliation) only if/when the
    freqtrade live path is insufficient — else keep freqtrade as broker adapter.
16. **Reporting + experiment-ID reproducibility layer.**
17. **Git init** at project root + commit-per-phase discipline.

## 18. Remove

- Nothing in the framework itself should be removed (it is the retained substrate).
- `user_data/baseline_backtest.json` (audit scratch) should be deleted or moved under
  `research/`; it is not part of the deliverable.

---

## Bottom line for Phase 0

- The substrate is a **healthy, well-tested framework** (1919 green core tests) with solid
  causal signal timing, but it is a *toolkit*, not a strategy.
- **There is no existing strategy/research to preserve or repair in this repo.** The honest
  conclusion is that every strategy-phase deliverable (Turtle, OU, HMM, router, meta-model,
  validation machinery, risk engine) is net-new work on top of freqtrade.
- The most dangerous framework-native risks to design against are (a) full-window indicator
  computation, (b) hyperopt without honest OOS, (c) label horizon crossing train/backtest
  boundaries, (d) fabricated backfill candles, and (e) the optimistic single-fee cost model.
- Next action per the brief: proceed to Phase 1 (canonical data layer) — nothing may be
  "optimized" or "tuned" before the data contract, feature audit, and validation harness exist.
