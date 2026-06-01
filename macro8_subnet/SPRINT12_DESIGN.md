# Sprint 12 — Hypothesis Engine

## The insight

Current system: searches formula space (syntactic)
Hypothesis system: searches economic explanation space (semantic)

A formula like rank(momentum_20d) - rank(volatility_60d) is a
black box. A hypothesis like "momentum works in low-volatility regimes"
is a testable scientific claim.

The hypothesis engine makes signals *interpretable* — each formula
becomes evidence for or against a market theory. The system learns
which economic principles produce alpha, not just which formulas work.

## New modules (one package)

### alpha/hypothesis_engine.py

Contains all hypothesis types and logic in a self-contained module:

  HypothesisCategory   — enum (MOMENTUM, MEAN_REVERSION, MACRO, RISK,
                                CARRY, VOLATILITY, CROSS_ASSET, REGIME)

  HypothesisRecord     — the core scientific object:
      id, statement, category, supporting_signals, IC_history,
      regime_ic (IC broken down by market regime),
      confidence_score, posterior, prior, epoch_born, epoch_last_updated

  BayesianUpdater      — updates hypothesis confidence using Bayes:
      posterior = prior × likelihood(IC_observation) / normaliser
      Uses Beta distribution conjugate prior for binary outcome
      (IC > 0 = hypothesis "works", IC ≤ 0 = doesn't work)

  HypothesisLibrary    — persistent store of all hypotheses:
      add(), get(), update_with_ic(), retire()
      query_by_category(), query_by_confidence()
      rank_by_confidence() → sorted list

  HypothesisEvolution  — guides formula evolution using hypothesis strength:
      suggest_features(hypothesis) → list of relevant FeatureStore features
      hypothesis_bias(evo_population) → weight formulas toward strong hypotheses
      generate_from_hypothesis(hypothesis) → seed formula strings

  HypothesisSubmission — miner submission schema:
      miner_uid, hypothesis_statement, category, supporting_formula, tags

  HypothesisReport     — per-epoch update summary:
      n_updated, n_retired, top_5_by_confidence, knowledge_entries

## Scientific design

Confidence updating: Beta-Binomial conjugate model
  prior: Beta(α=1, β=1) — flat (uninformed)
  likelihood: IC > threshold → success (+1); IC ≤ threshold → failure (+0)
  posterior: Beta(α + successes, β + failures)
  confidence_score = posterior_mean = α / (α + β)

This gives:
  - New hypothesis: confidence = 0.5 (no evidence)
  - After 10 positives, 5 negatives: confidence = 11/17 ≈ 0.65
  - After 50 positives, 5 negatives: confidence = 51/57 ≈ 0.89

Regime-conditional IC:
  IC is tracked separately per Regime enum value
  This reveals "momentum works but only in low-vol trending markets"

Hypothesis retirement:
  - confidence < 0.30 AND min_epochs ≥ 10 → retire
  - conflicting hypothesis with higher confidence → archive

## Integration points

1. Miners submit HypothesisSubmission (new role or attached to SIGNAL)
2. ResearchLoop.run_epoch() → calls hypothesis_engine.update_with_ic() after IC scoring
3. AlphaEvolution uses hypothesis_engine.hypothesis_bias() to weight formula mutations
   toward features relevant to high-confidence hypotheses
4. MetaMiner can predict hypothesis confidence changes (not just signal IC)
5. EpochReport includes HypothesisReport (new field)

## Build order

1. HypothesisCategory enum
2. HypothesisRecord dataclass
3. BayesianUpdater
4. HypothesisLibrary
5. HypothesisEvolution
6. HypothesisSubmission + HypothesisReport
7. Integration helpers for ResearchLoop
8. QA — full test suite

## Contracts
- All types JSON-serialisable
- No modification to existing modules
- HypothesisLibrary is the only stateful object (persists across epochs)
- BayesianUpdater is stateless (pure function)
