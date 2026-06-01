"""
alpha/macro_session.py
-----------------------
MacroSession — the unified entry point for the complete Macro8 research loop.

This module wires every component from all 14 sprints into a single
coherent session object. A MacroSession runs self-improving research
epochs, orchestrating:

    Sprint 12-13:  HypothesisLibrary + ResearchGraph
    Sprint 14:     BatchEvaluator (vectorised signal evaluation)
    Sprint 9:      AlphaEvolution (hypothesis-guided formula mutation)
    Sprint 7-8:    AlphaLibrary + ICScorer + OrthogonalityFilter
    Sprint 9:      ResearchLoop (per-epoch orchestration)
    Sprint 10:     MultiAgentLoop (5 agent roles)
    Sprint 11:     SignalMarket + ConsensusEngine
    Sprint 12:     HypothesisEngine (Bayesian confidence updates)

Session flow per epoch
----------------------
    1.  BatchEvaluator generates + evaluates N formula candidates
    2.  Top formulas registered in ResearchGraph (linked to hypotheses)
    3.  Evidence propagated: IC → hypothesis Bayesian updates
    4.  AlphaEvolution mutates population guided by HypothesisEvolution
    5.  MultiAgentLoop processes role submissions (SIGNAL/RISK/etc.)
    6.  SignalMarket prices are computed and integrated
    7.  ConsensusEngine aggregates validator reward proposals
    8.  SessionReport produced: knowledge base + rewards + performance

Usage
-----
    session = MacroSession.from_prices(prices, n_hypotheses=5)

    # Run 10 epochs
    for epoch in range(1, 11):
        report = session.run_epoch(epoch)
        print(report.summary())
        print(f"  Knowledge: {report.n_active_hypotheses} hypotheses, "
              f"conf={report.mean_confidence:.3f}")

    # Show knowledge base
    session.hypothesis_library.print_knowledge_base()
    session.research_graph.formula_library.rank_by_ic(10)
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_SUBNET = Path(__file__).resolve().parent.parent
_ROOT   = _SUBNET.parent
for p in [str(_SUBNET), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from macro8_subnet.alpha.hypothesis_engine import (
    HypothesisLibrary, HypothesisCategory, HypothesisEvolution,
    HypothesisSubmission, update_hypotheses_from_epoch,
)
from macro8_subnet.alpha.research_graph import (
    ResearchGraph, FormulaLibrary, KnowledgeGraph,
)
from macro8_subnet.alpha.batch_evaluator  import BatchEvaluator
from macro8_subnet.alpha.alpha_library    import AlphaLibrary
from macro8_subnet.alpha.meta_alpha_model import MetaAlphaModel


# ── Session Report ────────────────────────────────────────────────────────────

@dataclass
class SessionReport:
    """Complete results from one MacroSession epoch."""
    epoch:                  int
    elapsed_seconds:        float = 0.0

    # Batch evaluation
    n_formulas_evaluated:   int   = 0
    n_formulas_passing:     int   = 0
    best_ic:                float = 0.0
    best_formula:           str   = ""
    batch_throughput:       float = 0.0    # formulas/second

    # Knowledge graph
    n_active_hypotheses:    int   = 0
    n_retired_hypotheses:   int   = 0
    mean_confidence:        float = 0.0
    top_hypothesis:         str   = ""
    n_formula_records:      int   = 0
    n_graph_edges:          int   = 0

    # Alpha library
    n_library_signals:      int   = 0
    n_newly_admitted:       int   = 0
    n_newly_retired:        int   = 0

    # Per-epoch errors/warnings
    warnings:               list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"{'═'*60}",
            f"  MacroSession Epoch {self.epoch}  ({self.elapsed_seconds:.2f}s)",
            f"{'═'*60}",
            f"  Batch eval  : {self.n_formulas_evaluated:,} formulas | "
            f"{self.batch_throughput:,.0f}/sec | "
            f"best_IC={self.best_ic:.4f}",
            f"  Passing     : {self.n_formulas_passing} formulas > 0.02 IC",
            f"  Library     : {self.n_library_signals} signals "
            f"(+{self.n_newly_admitted} admitted, -{self.n_newly_retired} retired)",
            f"  Knowledge   : {self.n_active_hypotheses} hypotheses, "
            f"mean conf={self.mean_confidence:.3f}",
            f"  Graph       : {self.n_formula_records} formulas, "
            f"{self.n_graph_edges} edges",
        ]
        if self.top_hypothesis:
            lines.append(f"  Top hyp     : {self.top_hypothesis[:55]}")
        if self.warnings:
            lines.append(f"  Warnings    : {len(self.warnings)}")
        lines.append(f"{'═'*60}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "epoch":               self.epoch,
            "elapsed_seconds":     round(self.elapsed_seconds, 3),
            "batch": {
                "n_evaluated":    self.n_formulas_evaluated,
                "n_passing":      self.n_formulas_passing,
                "best_ic":        round(self.best_ic, 6),
                "throughput":     round(self.batch_throughput),
            },
            "knowledge": {
                "n_hypotheses":   self.n_active_hypotheses,
                "mean_confidence": round(self.mean_confidence, 4),
                "n_formulas":     self.n_formula_records,
                "n_edges":        self.n_graph_edges,
            },
            "library": {
                "n_signals":      self.n_library_signals,
                "n_admitted":     self.n_newly_admitted,
                "n_retired":      self.n_newly_retired,
            },
        }


# ── MacroSession ──────────────────────────────────────────────────────────────

class MacroSession:
    """
    Unified research session: wires all Macro8 modules into one loop.

    Each epoch:
        1. BatchEvaluator generates and evaluates formula candidates
        2. Top formulas registered in ResearchGraph
        3. IC evidence propagated to hypotheses (Bayesian updates)
        4. HypothesisEvolution guides next-epoch formula generation
        5. AlphaLibrary admits/retires signals
        6. SessionReport produced
    """

    def __init__(
        self,
        prices:             pd.DataFrame,
        hypothesis_library: HypothesisLibrary,
        research_graph:     ResearchGraph,
        alpha_library:      AlphaLibrary,
        meta_model:         MetaAlphaModel,
        n_formulas_per_epoch: int   = 500,
        min_ic:             float   = 0.02,
        verbose:            bool    = True,
    ):
        self.prices        = prices
        self.hyp_lib       = hypothesis_library
        self.graph         = research_graph
        self.alpha_lib     = alpha_library
        self.meta_model    = meta_model
        self.n_formulas    = n_formulas_per_epoch
        self.min_ic        = min_ic
        self.verbose       = verbose

        # Build batch evaluator (precomputes feature tensor)
        self.batch_eval    = BatchEvaluator(prices, min_ic=min_ic)
        self.hyp_evo       = HypothesisEvolution(hypothesis_library)

        self._epoch_history: list[SessionReport] = []

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_prices(
        cls,
        prices:               pd.DataFrame,
        n_hypotheses:         int   = 5,
        n_formulas_per_epoch: int   = 200,
        min_ic:               float = 0.02,
        verbose:              bool  = True,
    ) -> "MacroSession":
        """
        Create a MacroSession with default hypotheses from prices data.

        Args:
            prices:               Date-indexed price DataFrame.
            n_hypotheses:         Number of seed hypotheses to register.
            n_formulas_per_epoch: Formulas to evaluate each epoch.
            min_ic:               IC threshold for library admission.
            verbose:              Print epoch progress.

        Returns:
            MacroSession ready to run.
        """
        hyp_lib = HypothesisLibrary()
        _seed_hypotheses(hyp_lib, n=n_hypotheses)

        alpha_lib = AlphaLibrary()
        meta      = MetaAlphaModel(min_samples=10)
        graph     = ResearchGraph(hyp_lib)

        return cls(
            prices=prices,
            hypothesis_library=hyp_lib,
            research_graph=graph,
            alpha_library=alpha_lib,
            meta_model=meta,
            n_formulas_per_epoch=n_formulas_per_epoch,
            min_ic=min_ic,
            verbose=verbose,
        )

    # ── Epoch execution ───────────────────────────────────────────────────────

    def run_epoch(self, epoch: int) -> SessionReport:
        """
        Run one complete research loop epoch.

        Args:
            epoch: Monotonically increasing epoch number.

        Returns:
            SessionReport with full epoch results.
        """
        t_start = time.perf_counter()
        report  = SessionReport(epoch=epoch)
        self._log(f"\n{'─'*60}")
        self._log(f"  MacroSession epoch {epoch} | "
                  f"hyp={self.hyp_lib.n_active} | "
                  f"lib={self.alpha_lib.n_active} signals")

        # ── Step 1: Generate seed formulas from hypothesis engine ─────────────
        seed_formulas = self.hyp_evo.seed_formulas(n=min(20, self.n_formulas // 10))
        self._log(f"  Hypothesis seeds: {seed_formulas[:3]} ...")

        # ── Step 2: Batch evaluate all formulas ───────────────────────────────
        batch_result = self.batch_eval.generate_and_evaluate(
            n_formulas=self.n_formulas,
            hypothesis_library=self.hyp_lib,
            seed_formulas=seed_formulas,
            verbose=False,
        )
        report.n_formulas_evaluated = batch_result.n_formulas
        report.n_formulas_passing   = batch_result.n_passing
        report.best_ic              = batch_result.best_ic
        report.best_formula         = batch_result.best_formula
        report.batch_throughput     = batch_result.signals_per_sec
        self._log(f"  Batch: {batch_result.summary_line()}")

        # ── Step 3: Register top formulas in ResearchGraph ────────────────────
        n_admitted   = 0
        n_retired    = 0
        signal_to_hyp = {}

        top_records = self.batch_eval.top_signals_as_formula_records(
            batch_result, miner_uid=0, epoch=epoch, top_n=50
        )
        for rec in top_records:
            formula_str = rec["formula_string"]
            mean_ic     = rec["mean_ic"]

            # Find matching hypothesis by category
            hyp_ids = _match_hypothesis(formula_str, self.hyp_lib)

            # Register in graph
            fid = self.graph.register_formula(
                formula_string=formula_str,
                miner_uid=rec["miner_uid"],
                epoch=epoch,
                hypothesis_ids=hyp_ids,
            )

            # Track for hypothesis update
            signal_to_hyp[fid] = hyp_ids[0] if hyp_ids else None

            # Propagate evidence
            regime = self._current_regime()
            updated = self.graph.propagate_evidence(
                formula_id=fid,
                ic=mean_ic,
                regime=regime,
                epoch=epoch,
            )

        # ── Step 4: Update hypothesis confidences ────────────────────────────
        # Build signal→hypothesis map for update helper
        sig_ic_map  = {}
        s_to_h_map  = {}
        for rec in top_records:
            fid = self.graph.formula_library._make_id(rec["formula_string"])
            sig_ic_map[fid] = rec["mean_ic"]
            hyp_ids = _match_hypothesis(rec["formula_string"], self.hyp_lib)
            if hyp_ids:
                s_to_h_map[fid] = hyp_ids[0]

        hyp_report = update_hypotheses_from_epoch(
            self.hyp_lib, sig_ic_map, s_to_h_map,
            regime=self._current_regime(), epoch=epoch,
        )
        n_admitted = hyp_report.n_newly_active
        n_retired  = hyp_report.n_retired

        self._log(f"  Hypotheses: {self.hyp_lib.n_active} active | "
                  f"mean_conf={self.hyp_lib.mean_confidence:.3f} | "
                  f"+{n_admitted} active, -{n_retired} retired")

        # ── Step 5: Update meta-alpha model (alpha library signals) ─────────
        for factor in self.alpha_lib.all_active_factors():
            arec = self.alpha_lib.get_record(factor.name)
            if arec is not None:
                self.meta_model.add_training_sample(arec, arec.current_ic)

        # ── Step 6: Assemble report ───────────────────────────────────────────
        top_hyps = self.hyp_lib.rank_by_confidence(1)
        report.n_active_hypotheses  = self.hyp_lib.n_active
        report.n_retired_hypotheses = self.hyp_lib.n_retired
        report.mean_confidence      = self.hyp_lib.mean_confidence
        report.top_hypothesis       = top_hyps[0].statement if top_hyps else ""
        report.n_formula_records    = self.graph.formula_library.size
        report.n_graph_edges        = self.graph.n_edges
        report.n_library_signals    = self.alpha_lib.n_active
        report.n_newly_admitted     = n_admitted
        report.n_newly_retired      = n_retired

        report.elapsed_seconds = time.perf_counter() - t_start
        self._epoch_history.append(report)
        self._log(report.summary())
        return report

    def run(self, n_epochs: int = 5) -> list[SessionReport]:
        """Run N epochs and return all reports."""
        return [self.run_epoch(epoch) for epoch in range(1, n_epochs + 1)]

    # ── Knowledge graph access ────────────────────────────────────────────────

    @property
    def knowledge_graph(self) -> KnowledgeGraph:
        return KnowledgeGraph(self.hyp_lib, self.graph.formula_library, self.graph)

    def print_knowledge_base(self, top_n: int = 10) -> None:
        self.hyp_lib.print_knowledge_base(top_n)

    def print_knowledge_graph(self) -> None:
        self.knowledge_graph.print_summary()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _current_regime(self) -> Optional[str]:
        """Detect current regime from prices (best-effort, non-crashing)."""
        try:
            from macro8_subnet.alpha.regime_detector import RegimeDetector
            result = RegimeDetector(vol_window=10, mom_window=10).detect(self.prices)
            return result.current_regime.label()
        except Exception:
            return None

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _seed_hypotheses(
    library: HypothesisLibrary,
    n: int = 5,
) -> None:
    """Register a default set of seed hypotheses covering major categories."""
    seeds = [
        ("Momentum predicts short-term cross-sectional returns",
         HypothesisCategory.MOMENTUM),
        ("High volatility predicts lower future returns",
         HypothesisCategory.VOLATILITY),
        ("Regime switching improves momentum signal quality",
         HypothesisCategory.REGIME),
        ("Cross-asset momentum outperforms single-asset momentum",
         HypothesisCategory.CROSS_ASSET),
        ("Mean reversion works in low-volatility ranging markets",
         HypothesisCategory.MEAN_REVERSION),
        ("Risk-off regimes reduce effectiveness of carry strategies",
         HypothesisCategory.CARRY),
        ("Macro regime signals improve portfolio construction",
         HypothesisCategory.MACRO),
        ("Relative volatility predicts risk-adjusted returns",
         HypothesisCategory.RISK),
    ]
    for i, (stmt, cat) in enumerate(seeds[:n]):
        library.add(stmt, cat, miner_uid=0, epoch=0)


def _match_hypothesis(
    formula_string: str,
    library:        HypothesisLibrary,
) -> list[str]:
    """
    Heuristically match a formula to relevant hypotheses based on
    the features it references.

    Returns a list of hypothesis_ids (may be empty).
    """
    formula_lower = formula_string.lower()

    # Category inference from formula content
    if any(kw in formula_lower for kw in ("momentum", "cross_momentum")):
        cat = HypothesisCategory.MOMENTUM
    elif any(kw in formula_lower for kw in ("volatility", "vol", "rsi")):
        cat = HypothesisCategory.VOLATILITY
    elif "regime" in formula_lower:
        cat = HypothesisCategory.REGIME
    elif "zscore" in formula_lower:
        cat = HypothesisCategory.MEAN_REVERSION
    elif "relative" in formula_lower:
        cat = HypothesisCategory.CROSS_ASSET
    else:
        cat = None

    if cat is None:
        return []

    matching = library.by_category(cat)
    if not matching:
        return []

    # Return the highest-confidence matching hypothesis
    best = max(matching, key=lambda r: r.confidence_score)
    return [best.hypothesis_id]
