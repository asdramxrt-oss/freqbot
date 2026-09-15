# Unified Alpha Execution Plan: Integrating Trend, Mean Reversion, and ML

**Objective:** Combine the Turtle (Trend) and Pairs (Mean Reversion) strategies using our validated Machine Learning models, creating an adaptive, multi-strategy trading engine that survives across all market regimes.

## 1. The Strategy Paradox (And the Solution)
The empirical evidence from the research stack is clear:
*   **Turtle 20-bar:** Fails in choppy/bear markets; succeeds only in sustained bull regimes.
*   **Pairs Mean Reversion:** Bleeds capital during persistent trends; only works in tight, ranging markets.
*   **The ML Failure:** Predicting price direction (up/down) failed entirely (AUC < 0.50).
*   **The ML Success:** Predicting volatility magnitude and market regimes succeeded brilliantly (AUC 0.82-0.90).

**The Solution:** Do not combine the strategies linearly. Instead, use the ML models as a **Regime Router**. The ML layer will not pick the direction of the trade; it will pick *which strategy is allowed to trade*.

---

## 2. The Machine Learning Regime Router

We will deploy two ML models in tandem to act as the brain of the trading bot.

### A. The Hidden Markov Model (HMM) Classifier
*   **Purpose:** Classify the current market state into distinct regimes (e.g., Bull-Trend, Bear-Trend, Low-Vol Sideways).
*   **Input Features:** Rolling Fractional Differentiation of price, Yang-Zhang volatility, and spread divergence.
*   **Output:** A categorical state classification.

### B. The XGBoost Volatility Forecaster
*   **Purpose:** Predict the imminent expansion or crush of market volatility.
*   **Input Features:** Causal volatility metrics and order book imbalances.
*   **Output:** The probability that volatility will exceed the 90th percentile in the next `N` bars.

---

## 3. The Execution Matrix (How the Strategies Interact)

The `CentralRiskEngine` will read the outputs from the ML Router and dynamically allocate capital to either Turtle or Pairs based on the current regime. 

| ML Regime Classification | Volatility Forecast | Active Strategy | Rationale |
| :--- | :--- | :--- | :--- |
| **Confirmed Trend** | High / Expanding | **Turtle Breakout** | Momentum is strong; breakouts are likely to follow through. Pairs trading is disabled to prevent catching falling knives. |
| **Sideways / Ranging** | Low / Contracting | **Pairs (Mean Reversion)** | Price is mean-reverting. Turtle is disabled to prevent stop-out churn (whipsaw). |
| **Regime Transition** | Spike Predicted | **HALT TRADING** | A volatility spike is imminent, meaning the regime is shifting. Both strategies are paused until the HMM confirms the new state. |

---

## 4. Implementation Steps (Engineering Roadmap)

### Phase 1: Reprogramming the HMM Router
1.  Navigate to `research_lib/strategies/regime_hmm.py`.
2.  Expand the state classification logic to confidently label 'Trend' vs 'Ranging' states.
3.  Ensure the state transitions are causal and do not look ahead.

### Phase 2: Building the Dynamic Allocation Engine
1.  Create a new central routing script (e.g., `strategies/unified_router.py`).
2.  Import `TurtleSimulator`, `PairsSimulator`, the HMM model, and the XGBoost Vol-Spike model.
3.  Write logic that dynamically shifts the `capital_fraction` allocated to each strategy:
    *   *If HMM = Trend:* Turtle Capital = 100%, Pairs Capital = 0%.
    *   *If HMM = Range:* Turtle Capital = 0%, Pairs Capital = 100%.

### Phase 3: The Volatility Kill-Switch
1.  Integrate the Vol-Spike model (`ml/volspike.py`) directly into the `CentralRiskEngine`.
2.  If the XGBoost model outputs a >80% probability of a volatility expansion, trigger a hard `NO-TRADE` gate for the Pairs strategy to prevent catastrophic losses against a sudden trend.

### Phase 4: Walk-Forward Validation
1.  Run the newly unified strategy through the 6-month Walk-Forward Optimization (WFO) loops using 15m data.
2.  **Success Metric:** The combined portfolio must achieve a mathematically positive gross PnL across at least 5 out of the 6 testable market regimes, surviving the exact periods where the individual strategies previously failed.
