# Sprint 8 — Advanced Alpha Research Engine

## Four new modules

### 1. alpha/alpha_attribution.py
Measures individual alpha contribution to portfolio performance.
Uses Marginal Sharpe Contribution (MSC) — the industry-standard
metric for understanding which signals genuinely add value.

Formula:
    MSC_i = w_i * (Σ^-1 * μ)_i / portfolio_sharpe

where:
    w_i = weight of signal i in portfolio
    Σ   = signal covariance matrix
    μ   = signal expected returns vector

This answers: "If I remove this signal, how much Sharpe do I lose?"

### 2. alpha/meta_alpha_model.py
Predicts future IC from past signal behaviour.
Uses signal features (IC history, decay, regime performance, turnover)
to predict whether a signal will keep working.

Feature vector per signal:
    mean_ic, ic_ir, ic_stability, decay_rate, capacity,
    regime_ic[0..4], epochs_alive

Target: next-period IC

Models:
    - Ridge regression (fast, interpretable)
    - Gradient boosting (captures non-linear feature interactions)

### 3. simulation/synthetic_market.py
Generates synthetic market scenarios for alpha stress-testing.
Goes beyond historical scenarios — creates parametric simulations
of market conditions not present in the historical data.

Scenarios:
    - GBM baseline (lognormal prices)
    - Jump diffusion (fat tails)
    - Mean-reverting (Ornstein-Uhlenbeck)
    - Regime-switching (HMM-like)
    - Correlated shocks (systemic crisis)
    - Inflation spiral (trending with macro overlay)

### 4. alpha/formula_engine.py
Enables miners to submit alpha as a formula string rather than
precomputed signal values. The engine parses and evaluates formulas
against the feature store, creating a combinatorial search space.

Formula grammar:
    factor     = zscore(momentum_20d)
    combination = rank(momentum_20d) - rank(volatility_60d)
    conditional = regime_signal * momentum_5d

Operators: zscore, rank, decay, neutralize, clip, lag
Inputs:    any FeatureStore feature name

This is how WorldQuant's WebSim and Two Sigma's platform work —
miners describe transformations, not values.

## Integration contracts
- All modules accept pd.DataFrame (prices or returns)
- All results are JSON-serialisable dataclasses
- Zero modifications to existing Macro8 engine
- Zero modifications to existing subnet modules
