# Sprint 10 — Multi-Agent Research Roles

## The core insight

The existing system is a single-role network: every miner does the
same thing (submit formulas). This sprint differentiates miners into
five specialised agent roles, each improving a different pipeline stage.

## The five agent roles

```
Role              Submits                    Validated by
──────────────────────────────────────────────────────────
SignalMiner       alpha formulas             IC scorer
StrategyMiner     signal combination weights rolling backtest + composite scorer
RiskMiner         covariance + vol forecasts prediction accuracy vs realised
PortfolioMiner    portfolio constraints      Sharpe improvement over baseline
MetaMiner         IC predictions             prediction accuracy (IC-IC corr)
```

## Architecture: role-aware research loop

The existing `ResearchLoop` handles only `SignalMiner` submissions.
Sprint 10 adds a `MultiAgentLoop` that:
  1. Routes each submission to the correct evaluator
  2. Computes role-specific rewards
  3. Combines role outputs into the final portfolio
  4. Feeds results back to the meta model

```
MultiAgentLoop.run_epoch(submissions)
       │
       ├─ SignalMiner submissions → ICScorer → AlphaLibrary
       │
       ├─ StrategyMiner submissions → CompositeScorer → StrategyLibrary
       │
       ├─ RiskMiner submissions → RiskEvaluator → CovarianceStore
       │
       ├─ PortfolioMiner submissions → PortfolioEvaluator → WeightStore
       │
       └─ MetaMiner submissions → MetaEvaluator → MetaAlphaModel
                │
                ▼
         MultiAgentPortfolio
                │
                ▼
         MultiAgentRewards (role-specific reward weights)
```

## New modules

### agents/agent_roles.py  (Architect)
Defines the five AgentRole enum, AgentSubmission dataclass (extends
FormulaSubmission with role tag), and the AgentRegistry that maps
roles → evaluators.

### agents/strategy_miner_agent.py  (Backend Developer)
StrategyMiner submissions: miners submit signal combination weights
(which signals from the library to combine, and in what proportion).
Evaluated by the existing CompositeScorer (rolling + stress + hidden).

### agents/risk_miner_agent.py  (Simulation Engineer)
RiskMiner submissions: miners submit a covariance model (shrinkage
parameter + factor model specification). Evaluated by measuring how
well the submitted model predicts next-period realised volatility.

### agents/portfolio_miner_agent.py  (Simulation Engineer)
PortfolioMiner submissions: miners submit portfolio constraint sets
(max weight, min weight, sector limits, turnover limit). Evaluated by
how much the submitted constraints improve the portfolio Sharpe vs
unconstrained baseline.

### agents/meta_miner_agent.py  (Scoring Engineer)
MetaMiner submissions: miners submit IC predictions for library
signals. Evaluated by correlation between predicted IC and realised IC.
The best MetaMiner's predictions guide the evolution engine.

### agents/multi_agent_loop.py  (Backend Developer)
The top-level orchestrator. Accepts a mixed list of AgentSubmission
objects, routes each to the correct evaluator, collects all results,
builds the final portfolio using all role outputs, computes
role-stratified rewards, and returns a MultiEpochReport.

### agents/role_rewards.py  (Blockchain Engineer)
Role-specific reward calculation. Each role has its own scoring
formula and reward weight in the total TAO emission:

    SignalMiner   : 30% of epoch rewards (IC score)
    StrategyMiner : 25% of epoch rewards (composite score)
    RiskMiner     : 20% of epoch rewards (vol prediction accuracy)
    PortfolioMiner: 15% of epoch rewards (Sharpe improvement)
    MetaMiner     : 10% of epoch rewards (IC prediction accuracy)

## Key design decisions

1. All evaluators call EXISTING modules — no new evaluation logic.
   - StrategyMiner → calls research_loop._step_ic_scoring() indirectly
   - RiskMiner     → calls synthetic_market for realised vol
   - PortfolioMiner → calls portfolio_optimizer
   - MetaMiner     → calls meta_alpha_model

2. Role submissions are optional — an epoch with only SignalMiners
   works exactly like the existing ResearchLoop (backward compatible).

3. Reward emission is role-stratified but normalised globally:
   - each role's rewards sum to the role's budget fraction
   - total across all roles always sums to 1.0

4. The MultiAgentLoop wraps (not replaces) ResearchLoop.

## Interface contracts
- AgentSubmission has all fields of FormulaSubmission + role: AgentRole
- MultiEpochReport has all fields of EpochReport + role_results: dict
- All new types are JSON-serialisable
- Zero modifications to existing modules
