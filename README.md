# Macro8

Regime-adaptive, multi-horizon alpha engine with Bittensor subnet integration.

---

## Quick start

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install
pip install -r requirements.txt

# 3. Put your price data in the root folder
#    File: calibrated_prices.csv
#    Format: dates as rows (YYYY-MM-DD), tickers as columns, adjusted close prices
#    Minimum: 500 rows. Universe used in development: SPY QQQ IWM TLT GLD DBC EEM FXI VNQ HYG

# 4. Verify
python -m macro8_subnet.local_simulation --fast    # 2-3 min

# 5. Run
python -m macro8_subnet.execution.live_runner --mode once
```

---

## Architecture

```
GPMiner               discover candidate signal formulas (GP evolution)
AdaptiveEnsemble      regime-scoped pools + prob-weighted allocation
RegimeTransitionModel predict P(calm / normal / stress)
PortfolioEvaluator    2D grid scoring: 4 horizons × 4 capital tiers
CapitalEngine         multi-horizon bucket allocation + softmax reallocation
ConstraintSolver      kill-switch, position limits, stress-adjusted costs
TradeExecutor         stress-aware execution
LiveRunner / PaperTrader   daily pipeline + capital feedback
Validator             trust decay, spike penalty, diversity enforcement
```

### Signal pools

| Pool | Signals | Activated by |
|------|---------|-------------|
| Calm/normal | `momentum_20d − volatility_20d`, `momentum_60d` | P(calm) + P(normal) |
| Stress | `market_corr_60d × iwm_spy_20d − volatility_20d` | P(stress) |
| Normal specialist | `vol_ratio` (vol5/vol60) | P(normal) > 0.60 |

### 2D evaluation grid

Each signal scored across 16 cells: 4 time horizons × 4 capital scales.

| | $1k | $10k | $100k | $1M |
|---|---|---|---|---|
| **1d** | daily rebal | → | → | → |
| **7d** | weekly | → | → | → |
| **30d** | monthly | → | → | → |
| **90d** | quarterly | → | → | → |

Cell score = `sharpe_at_freq(h) × capital_viability(c)`. Turnover penalty uses the frequency-specific turnover, not daily.

### Key numbers (1246-day OOS 2018–2023)

| Metric | Value |
|--------|-------|
| Gross Sharpe | +0.47 |
| Net @2bps | +0.38 |
| Max drawdown | −7.2% |
| Portfolio turnover | 33×/yr |
| Stress Sharpe | +1.35 |
| Normal Sharpe | −0.05 |
| Calm Sharpe | +0.10 |
| Walk-forward (3 splits) | +0.59 / +0.42 / +0.94 |
| Regime noise ±20% | −0.014 Sharpe degradation |

---

## File structure

```
Macro8/
├── requirements.txt
├── setup.py
├── README.md
├── calibrated_prices.csv       ← your price data goes here
├── .vscode/
│   ├── settings.json
│   └── launch.json
└── macro8_subnet/
    ├── alpha/
    │   ├── feature_store.py          38 causal features
    │   ├── batch_evaluator.py        FeatureTensor, FormulaEncoder
    │   ├── gp_miner.py               GP formula discovery
    │   ├── ic_scorer.py              Information coefficient
    │   ├── portfolio_intelligence.py AdaptiveEnsemble, regime pools
    │   ├── portfolio_evaluator.py    2D grid + frequency-aware scoring
    │   └── regime_prediction.py      RegimeTransitionModel
    ├── evaluation/
    │   └── transaction_costs.py
    ├── execution/
    │   ├── capital_engine.py         Multi-horizon bucket allocation
    │   ├── engine.py                 ConstraintSolver, TradeExecutor
    │   └── live_runner.py            PaperTrader, CLI, capital feedback
    ├── neurons/
    │   ├── miner.py
    │   └── validator.py              Trust decay, spike penalty, diversity
    ├── protocol/
    │   └── synapse.py
    ├── tests/
    │   ├── test_sprint29.py          Regime prediction
    │   ├── test_sprint30.py          Execution engine
    │   ├── test_sprint31.py          Live pipeline
    │   └── test_sprint32.py          Integration
    └── local_simulation.py           7-layer integration check
```

---

## CLI commands

```bash
# Today's positions + regime probabilities
python -m macro8_subnet.execution.live_runner --mode once

# Full 5-year backtest
python -m macro8_subnet.execution.live_runner --mode backtest

# Daily paper trading (run from cron)
python -m macro8_subnet.execution.live_runner --mode paper

# Force retrain
python -m macro8_subnet.execution.live_runner --mode retrain

# Integration checks
python -m macro8_subnet.local_simulation --fast     # ~2 min
python -m macro8_subnet.local_simulation            # ~8 min

# Tests
python -m pytest macro8_subnet/tests/ -q
```

---

## Daily cron (paper trading)

```bash
# Add to crontab — runs 6pm Mon–Fri
0 18 * * 1-5 cd /path/to/Macro8 && .venv/bin/python -m macro8_subnet.execution.live_runner --mode paper >> ~/.macro8/paper.log 2>&1
```

---

## Bittensor deployment

```bash
# Install bittensor
pip install bittensor>=10.0.0

# Register wallet
btcli wallet new_coldkey --wallet.name validator
btcli wallet new_hotkey --wallet.name validator --wallet.hotkey default

# Register on testnet (netuid 263)
btcli s register --netuid 263 --wallet.name validator --subtensor.network test

# Run validator
python -m macro8_subnet.neurons.validator \
  --subtensor.network test \
  --netuid 263 \
  --wallet.name validator \
  --wallet.hotkey default

# Run miner
python -m macro8_subnet.neurons.miner \
  --subtensor.network test \
  --netuid 263 \
  --wallet.name miner \
  --wallet.hotkey default
```

---

## Validator scoring pipeline

Each epoch, miners are scored through five stages:

1. **IC + portfolio composite** — BatchEvaluator scores each formula on multi-horizon IC and 2D grid (4 horizons × 4 capital tiers)
2. **Time-decayed trust** — `Trust_t = 0.9 × Trust_{t-1} + 0.1 × Score_t`
3. **Anti-overfitting penalty** — if current score > mean + 2σ of history, fractional reduction (max 50%)
4. **Diversity enforcement** — miners with correlated signal vectors (|corr| > 0.85) share a reward budget (max 40% reduction)
5. **Capital feedback** — miners who submit positions get their realised portfolio return fed back into trust (20% weight)

---

## Capital engine

Four parallel rebalancing buckets per signal:

| Bucket | Rebalances every | Natural signal |
|--------|-----------------|----------------|
| 1d | 1 trading day | Fast signals |
| 7d | 5 trading days | Medium signals |
| 30d | 21 trading days | `momentum_20d − vol` |
| 90d | 63 trading days | `momentum_60d` (+0.89 Sharpe) |

Weekly softmax reallocation shifts capital toward whichever bucket is currently earning best Sharpe. EMA smoothing (α=0.20) and 5% floor prevent over-reaction.

---

## Design decisions

**Why `momentum_60d` not `zscore_20d`:** zscore_20d has 1.26/day turnover = 6.4%/yr cost at 2bps, killing its +0.43 gross Sharpe. momentum_60d has 0.17/day, OOS calm Sharpe +0.83.

**Why keyword-based pool routing:** Market_corr variants have training calm Sharpe +1.10 (2009–2018 bull market) but OOS calm -1.55. Training Sharpe is non-stationary across macro regimes. Signal keywords are causal.

**Why 70/30 global blend:** Regime model errors at ±20% noise cost only 0.014 Sharpe. The 30% equal-weight floor prevents full routing failure.

**Why frequency-aware scoring:** `momentum_60d` daily Sharpe = +0.12; quarterly Sharpe = +0.89. The same signal is 7× better at the right cadence. Scoring at daily frequency permanently undervalues slow signals.

**Why softmax not argmax for frequency weights:** Smoother, less overfit, and calibrated to the same softmax logic the capital engine uses.
