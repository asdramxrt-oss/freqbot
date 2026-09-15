# DeepSeek V4 Flash: Alpha Discovery & Implementation Brief

**Target Agent:** DeepSeek V4 Flash (1M Context, Max Reasoning Effort)
**Objective:** Engineer a fundamentally new trading alpha with a mathematically positive gross PnL, leveraging existing infrastructure and our validated volatility-prediction model.

## 1. Codebase & Context Ingestion
You are stepping into a structurally perfect algorithmic trading pipeline that currently lacks a profitable alpha.
* **Read Infrastructure:** Ingest `research_lib/` to understand the existing execution stack, which includes an event-aware Walk-Forward Optimization (WFO) engine, a maker/post-only execution model (`risk/maker.py`), and 122 passing unit tests.
* **Analyze the Failures:** Read `IMPLEMENTATION_REPORT.md` and `BLUEPRINT_REPORT.md`. Note that the Turtle 20-bar strategy failed out-of-sample due to regime dependence, and the market-neutral pairs strategy failed entirely with a negative gross PnL (-$127k).
* **Extract the Golden Asset:** Analyze `research_lib/ml/volspike.py`. Our XGBoost model successfully predicts volatility expansions with an OOS AUC of 0.82–0.90 on 15m data.

## 2. Alpha Generation Directives
Your task is to formulate a new strategy that relies on validated predictive power rather than curve-fitted price trends.
* **Trade the Volatility Spike:** Our previous tests proved that using the vol-spike model purely as a defensive filter blocks profitable entries on trend strategies and cannot rescue negative-gross mean reversion. You must design a strategy where the volatility spike *is* the signal.
* **Examples to Explore:** 
  * Momentum-ignition trading: Entering directional trades instantly upon a predicted volatility spike on 15m bars, capturing the initial price expansion.
  * Volatility-crush mean reversion: Fading extreme price moves only *after* the predicted volatility regime has crested.
* **Fee Optimization:** The strategy must remain viable after taker fees, though you should utilize the existing maker/post-only logic to capture the spread where possible.

## 3. Step-by-Step Execution Plan
* **Step 1:** Draft 3 distinct, code-ready alpha hypotheses based on trading volatility expansions.
* **Step 2:** Implement the most mathematically sound hypothesis as a Python strategy class integrating with `PairsSimulator` or `TurtleSimulator`.
* **Step 3:** Execute the Walk-Forward Optimization (WFO) over the 2022–2025 research window. 
* **Step 4:** Iterate on the signal logic until the strategy achieves a strictly positive gross PnL across at least 5 out of 7 semi-annual WFO regimes.
* **Step 5:** Ensure the finalized strategy passes the 17 acceptance gates (`research_lib/validation/acceptance.py`) without overfitting. Do not touch the `data_holdout` directory during optimization.
