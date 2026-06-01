# Macro8 — Deployment Guide

**Decentralized Alpha Research Network on Bittensor**

Macro8 is a Bittensor subnet where miners compete to discover predictive
alpha signals and validators score them using out-of-sample IC evaluation.

---

## Architecture

```
Miners                         Validators
  │                               │
  │  AlphaSubmissionSynapse       │
  │ ─────────────────────────────►│
  │   {formulas, role, epoch}     │
  │                               │  BatchEvaluator
  │                               │  IC scoring (~6k formulas/sec)
  │                               │  Hypothesis engine update
  │                               │  Role-stratified rewards
  │                               │
  │◄─────────────────────────────│
  │   {ic_scores, reward_signal}  │
  │                               │  subtensor.set_weights()
                                  │
                              Bittensor
                              (on-chain weights → TAO emissions)
```

---

## Quick Start

### 1. Install

```bash
# Python 3.10+
pip install bittensor>=10.0
pip install -e .          # installs macro8-miner and macro8-validator entry points

# Or from source:
pip install numpy pandas scikit-learn scipy yfinance requests bittensor
```

### 2. Create wallets

```bash
# Owner wallet (for subnet registration)
btcli wallet new_coldkey --wallet.name owner

# Validator wallet
btcli wallet new_coldkey --wallet.name validator
btcli wallet new_hotkey  --wallet.name validator --wallet.hotkey default

# Miner wallet
btcli wallet new_coldkey --wallet.name miner
btcli wallet new_hotkey  --wallet.name miner --wallet.hotkey default
```

### 3. Get testnet TAO (free)

```bash
# Join Bittensor Discord and request testnet TAO from #faucet channel
# Discord: https://discord.gg/bittensor
# Or use the testnet faucet: https://test.taostats.io/faucet
```

### 4. Register subnet (testnet)

```bash
btcli subnet create \
    --wallet.name owner \
    --wallet.hotkey default \
    --subtensor.network test

# Note the assigned netuid — update netuid_config.py:
# NETUID = <your_netuid>
```

### 5. Register neurons

```bash
# Register validator
btcli subnet register \
    --netuid <NETUID> \
    --wallet.name validator \
    --wallet.hotkey default \
    --subtensor.network test

# Register miner
btcli subnet register \
    --netuid <NETUID> \
    --wallet.name miner \
    --wallet.hotkey default \
    --subtensor.network test
```

---

## Running the Network

### Validator

```bash
# Basic (testnet):
python -m macro8_subnet.neurons.validator \
    --subtensor.network test \
    --netuid <NETUID>

# With real data (set your free FRED API key):
export FRED_API_KEY=your_free_key_here   # https://fred.stlouisfed.org/docs/api/api_key.html
python -m macro8_subnet.neurons.validator \
    --subtensor.network test \
    --netuid <NETUID> \
    --wallet.name validator \
    --wallet.hotkey default

# Dry-run (no network, for testing):
python -c "
from macro8_subnet.neurons.validator import Macro8Validator
v = Macro8Validator()
v._dry_run(n_epochs=3)
"
```

### Miner — Signal Role (default)

```bash
# Submit alpha formula strings
python -m macro8_subnet.neurons.miner \
    --subtensor.network test \
    --netuid <NETUID> \
    --wallet.name miner \
    --wallet.hotkey default \
    --role signal \
    --n_formulas 500

# Dry-run:
python -c "
from macro8_subnet.neurons.miner import Macro8Miner
m = Macro8Miner()
m._dry_run(n_epochs=3)
"
```

### Miner — Other Roles

```bash
# Risk model miner: submit covariance model parameters
python -m macro8_subnet.neurons.miner --role risk

# Portfolio miner: submit constraint sets
python -m macro8_subnet.neurons.miner --role portfolio

# Meta miner: predict which signals will succeed next epoch
python -m macro8_subnet.neurons.miner --role meta
```

---

## Configuration

Edit `netuid_config.py` for all subnet parameters:

```python
NETUID           = 1       # your registered netuid
DEFAULT_NETWORK  = "test"  # "test" | "finney" (mainnet)
MIN_IC_THRESHOLD = 0.015   # minimum IC to earn rewards
ROLE_BUDGETS = {
    "signal":    0.40,     # 40% of emissions to formula miners
    "strategy":  0.20,
    "risk":      0.15,
    "portfolio": 0.15,
    "meta":      0.10,
}
```

---

## Data Sources

The validator automatically uses real market data when available:

| Source | Data | API Key |
|--------|------|---------|
| **yfinance** | 20+ years daily prices for 20 ETFs | None needed |
| **FRED** | VIX, yields, CPI, employment | Free at stlouisfed.org |
| **Synthetic** | Calibrated fallback (Student-t, GARCH, regimes) | None |

Set `FRED_API_KEY` environment variable for macro data:
```bash
export FRED_API_KEY=your_key  # Free: https://fred.stlouisfed.org/docs/api/api_key.html
```

---

## Scoring System

### How IC is computed

The validator evaluates each submitted formula on the **most recent 20% of data** (out-of-sample), ensuring miners cannot overfit to historical data the validator has already seen.

```
Formula string
    ↓
FormulaEncoder → weight vector [n_features]
    ↓
FeatureTensor einsum → signal [time × assets]
    ↓
Rank-correlation with forward returns → IC
    ↓
Lifecycle adjustment (EXPERIMENTAL=0.3x, PRODUCTION=1.0x, RETIRED=0.0x)
    ↓
Role-stratified softmax → reward weight
    ↓
subtensor.set_weights()
```

### Anti-gaming measures

| Measure | What it prevents |
|---------|-----------------|
| Out-of-sample window | Curve-fitting to historical data |
| Orthogonality filter (ρ>0.9) | Correlated signal spam |
| Rate limiting (50 formulas/epoch) | Brute-force submission |
| Lifecycle scoring | Reward farming with stale signals |
| EMA smoothing (20 epochs) | Reward manipulation via one-off submissions |

---

## Miner Strategies

### What earns high rewards

1. **Novel signals** — formulas that discover IC not already in the library
2. **Hypothesis-grounded signals** — formulas supporting testable market theories
3. **Stable IC** — signals that show consistent IC across market regimes
4. **Diverse formulas** — different feature combinations rather than variations of the same idea

### Example high-reward formulas

```python
# Momentum with volatility adjustment
"rank(momentum_20d) - rank(volatility_20d)"

# Regime-conditional momentum
"regime_signal * rank(momentum_60d)"

# Cross-asset momentum divergence
"cross_momentum - relative_vol"

# Decay-weighted short-term momentum
"decay(rank(momentum_5d), halflife=5)"

# Latent market structure (after representation learning)
"rank(latent_pca_0) - rank(volatility_60d)"
```

### Formula syntax

```
Features:   momentum_5d, momentum_20d, momentum_60d,
            volatility_10d, volatility_20d, volatility_60d,
            zscore_20d, zscore_60d, rsi_14,
            cross_momentum, relative_vol, regime_signal

Operators:  rank(x), zscore(x), decay(x, halflife=N),
            neutralize(x), clip(x), lag(x, n=N),
            sign(x), abs(x)

Arithmetic: x + y, x - y, x * y
```

---

## Health Monitoring

```bash
# Check subnet state
btcli subnet metagraph --netuid <NETUID> --subtensor.network test

# Check weights set by validator
btcli wallet overview --wallet.name validator --subtensor.network test

# Monitor miner rewards
btcli wallet overview --wallet.name miner --subtensor.network test

# Run full test suite
cd /path/to/Macro8
python -m pytest macro8_subnet/tests/ -q
# Expected: 647 passed
```

---

## Hardware Requirements

| Role | RAM | CPU | Notes |
|------|-----|-----|-------|
| Validator | 16GB | 4+ cores | Runs IC evaluation engine |
| Miner (signal) | 8GB | 2+ cores | Formula generation |
| Miner (risk/portfolio) | 4GB | 2 cores | Lightweight |

GPU not required for testnet. For mainnet with large formula batches (100k+/epoch), a GPU with CUDA accelerates the representation learning engine.

---

## Mainnet Registration

```bash
# Create subnet on mainnet (costs ~1 TAO):
btcli subnet create \
    --wallet.name owner \
    --wallet.hotkey default

# Register validator (costs registration fee):
btcli subnet register \
    --netuid <NETUID> \
    --wallet.name validator \
    --wallet.hotkey default

# Stake TAO to become a validator:
btcli stake add \
    --wallet.name validator \
    --wallet.hotkey default \
    --amount 100   # minimum stake varies
```

---

## Troubleshooting

**"Not registered on netuid"**
```bash
btcli subnet register --netuid <NETUID> --wallet.name <NAME>
```

**"No axons in metagraph"** — No miners registered yet. Register a miner first.

**"Proxy error / 403"** — Network restriction in current environment.
Deploy to a machine with unrestricted internet access.

**Low IC scores** — Normal at launch. IC builds as the hypothesis engine
accumulates evidence. Expect meaningful differentiation after 10–20 epochs.

**Validator not setting weights** — Check:
1. Validator is registered: `btcli subnet metagraph`
2. Validator has stake
3. Weights version key matches: see `WEIGHTS_VERSION_KEY` in `netuid_config.py`

---

## Development Workflow

```bash
# 1. Run full test suite (should be 647 passed)
python -m pytest macro8_subnet/tests/ -q

# 2. Dry-run validator locally (no network needed)
python -c "
from macro8_subnet.neurons.validator import Macro8Validator
v = Macro8Validator()
v._dry_run(n_epochs=5)
"

# 3. Dry-run miner locally
python -c "
from macro8_subnet.neurons.miner import Macro8Miner
m = Macro8Miner()
m._dry_run(n_epochs=3)
"

# 4. Test the data pipeline
python -c "
from macro8_subnet.data.market_data_fetcher import MarketDataFetcher
f = MarketDataFetcher()
r = f.fetch_for_session(n_assets=8, n_years=10)
print(r.summary())
"

# 5. Deploy to testnet
python -m macro8_subnet.neurons.validator --subtensor.network test --netuid 263
```

---

*Macro8 v1.0.0 | 49 production modules | 647 tests passing*
