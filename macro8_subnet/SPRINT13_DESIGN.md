# Sprint 13 — FormulaRecord + Bi-directional Research Graph

## The problem with Sprint 12

Sprint 12's HypothesisLibrary links signals to hypotheses through
signal_names stored on the record. That is:

    HypothesisRecord.supporting_signals = ["f_momentum_uid0", ...]

But there is no FormulaRecord object — the formula string itself is
stateless. The evidence chain breaks at the formula level:

    hypothesis.confidence ← IC
                              ↑
                          signal name
                              ↑
                          formula string   ← no structured object here

## The fix: FormulaRecord + ResearchGraph

FormulaRecord is the missing middle tier:
    formula_id          deterministic hash of formula string
    formula_string      "rank(momentum_20d) - rank(volatility_60d)"
    hypothesis_ids      list of hypotheses this formula supports
    ic_history          list of IC observations
    msc_history         list of MSC (attribution) observations
    regime_ic           dict[regime → list[IC]]
    miner_uid           who submitted it
    epoch_born          when it entered the system
    generation          evolution generation (0 = miner-submitted)
    parent_ids          list of parent formula_ids (for evolved formulas)

ResearchGraph owns the bi-directional mapping:
    formula_to_hypotheses   dict[formula_id → set[hypothesis_id]]
    hypothesis_to_formulas  dict[hypothesis_id → set[formula_id]]

    link(formula_id, hypothesis_id) — create the bi-directional edge
    evidence_for(hypothesis_id, ic) — update hypothesis + all linked formulas
    evidence_from_formula(formula_id, ic, regime) — update formula +
                          propagate to linked hypotheses

## Evidence flow (correct)

    Formula evaluated → IC observed
           ↓
    FormulaRecord.ic_history.append(IC)
           ↓
    For each linked hypothesis_id:
        HypothesisRecord: Bayesian update α or β
           ↓
    HypothesisRecord.confidence_score updated

Multi-formula evidence aggregation:
    If hypothesis H has 5 formulas, and 4 get IC > threshold this epoch:
    → 4 Bayesian successes this epoch (one per supporting formula)
    → confidence rises faster than with one formula

## Knowledge graph queries

The ResearchGraph enables queries that were impossible before:
    "Which hypotheses have ≥3 supporting formulas?"
    "Which formulas support multiple hypotheses?"
    "What is the average IC of formulas supporting H?"
    "Show me the formula evolution tree for H"
    "Which hypotheses share a formula?"

## New module: alpha/research_graph.py

Contains:
    FormulaRecord           dataclass
    ResearchGraph           the bi-directional graph with evidence propagation
    FormulaLibrary          stores FormulaRecord objects (parallel to AlphaLibrary)
    KnowledgeGraph          high-level query interface (wraps both libraries)

## Build order
1. FormulaRecord dataclass
2. FormulaLibrary (store + CRUD)
3. ResearchGraph (bi-directional mapping + evidence propagation)
4. KnowledgeGraph (query layer)
5. QA — full test suite

## Contracts
- FormulaRecord is JSON-serialisable
- ResearchGraph.propagate_evidence() calls existing BayesianUpdater
- No modifications to hypothesis_engine.py
- No modifications to any other existing module
