# Sprint 11 — Signal Prediction Market + Distributed Validators

## System A: Signal Prediction Market

### Core mechanic

Each signal in the alpha library becomes a market asset.
Miners stake on signals they believe will have positive IC next epoch.
The market aggregates distributed beliefs into a confidence score.
Portfolio weights = IC_score × market_confidence.

### Information theory basis

The prediction market solves the fundamental problem:
  - IC scoring measures past performance
  - Market prices measure *forward-looking* beliefs

Combined signal weight = α * IC_history + β * market_confidence

This is Bayesian updating: prior (IC history) + evidence (market beliefs).

### Market mechanics

Scoring rule: quadratic scoring rule (proper, incentive-compatible)
  reward = 2 * p_i * outcome - p_i^2
  where p_i = stake fraction, outcome = 1 if IC > 0 else 0

Proper scoring rules make truthful reporting the optimal strategy.
Miners cannot gain by misreporting beliefs.

Market price per signal = stake-weighted average confidence
  price = Σ(stake_i * confidence_i) / Σ(stake_i)

Position types:
  LONG  = betting IC will be positive next epoch
  SHORT = betting IC will be negative (or zero)

Settlement: after each epoch, ICs are observed and positions settled.

### New modules

#### market/signal_market.py  (Blockchain Engineer)
  - SignalPosition dataclass (uid, signal, direction, stake, confidence)
  - MarketBook: tracks all open positions per signal
  - price_signal(): aggregate stakes → market price [0,1]
  - settle_epoch(): compare predictions to actual ICs, compute P&L

#### market/market_rewards.py  (Blockchain Engineer)
  - QuadraticScorer: proper scoring rule implementation
  - compute_predictor_rewards(): stake-weighted accuracy → TAO weights
  - Handles long/short positions, partial accuracy, zero-stake edge cases

#### market/market_integrator.py  (Simulation Engineer)
  - market_weighted_ics(): blend IC history with market prices
  - signal_confidence_vector(): market prices as portfolio weights modifier
  - Called by MultiAgentLoop between IC scoring and portfolio construction

---

## System B: Distributed Validators

### The decentralisation problem

Currently one validator type runs everything: IC scoring, covariance
evaluation, Sharpe measurement, forecast scoring. This is centralised
evaluation — the validator is a single point of failure and trust.

### Solution: validator specialisation + consensus

Four validator types, each responsible for one evaluation domain.
Validators vote on rewards for their domain. Final rewards = weighted
average of validator votes.

```
Signal Validator    → IC scoring → votes on signal miner rewards
Risk Validator      → covariance eval → votes on risk miner rewards
Portfolio Validator → Sharpe eval → votes on portfolio miner rewards
Meta Validator      → forecast scoring → votes on meta miner rewards
```

Consensus mechanism:
  - Each validator produces a reward vector for its domain
  - Final reward = stake-weighted average across validators
  - Disagreeing validators are penalised (Brier score of their reward calls)

### New modules

#### validators/validator_types.py  (Protocol Engineer)
  - ValidatorRole enum (SIGNAL, RISK, PORTFOLIO, META)
  - ValidatorSubmission: validator's reward proposal for its domain
  - ValidatorRegistry: maps roles to evaluation modules

#### validators/consensus.py  (Blockchain Engineer)
  - RewardProposal: one validator's reward votes for a domain
  - ConsensusEngine: stake-weighted average of validator proposals
  - DisagreementPenalty: penalise validators whose votes differ from consensus
  - final_rewards(): merge domain-specific votes → global reward vector

---

## Build order

1. market/signal_market.py      — Blockchain Engineer
2. market/market_rewards.py     — Blockchain Engineer
3. market/market_integrator.py  — Simulation Engineer
4. validators/validator_types.py — Protocol Engineer
5. validators/consensus.py      — Blockchain Engineer
6. QA — tests for all 5 modules

## Interface contracts

- All inputs/outputs are pd.DataFrame or JSON-serialisable dataclasses
- market_integrator.py calls ONLY existing ic_scorer + alpha_library
- consensus.py calls ONLY existing role_rewards.py
- Zero modifications to any existing module
