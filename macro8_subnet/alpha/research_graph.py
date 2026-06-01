"""
alpha/research_graph.py
------------------------
The Macro8 Research Graph — bi-directional mapping between formulas
and market hypotheses.

This module adds the missing middle tier to the knowledge architecture:

    HypothesisRecord (scientific claim)
          ↕  bi-directional
    FormulaRecord    (testable implementation)
          ↓
    Signal data → IC evidence → Bayesian update

Every formula is a node. Every hypothesis is a node. The edges
between them carry evidence: when a formula is evaluated, the IC
observation propagates upward to all linked hypotheses.

Key objects
-----------
    FormulaRecord      one formula string with its full performance history
    FormulaLibrary     persistent store for FormulaRecord objects
    ResearchGraph      bi-directional formula↔hypothesis mapping +
                       evidence propagation engine
    KnowledgeGraph     high-level query interface combining both libraries

Evidence propagation
--------------------
    formula evaluated → IC = 0.042
           ↓
    formula.ic_history.append(0.042)
           ↓
    for each linked hypothesis H:
        BayesianUpdater().update(H, IC=0.042, regime=current_regime)
           ↓
    H.confidence_score updated

Multi-formula bonus: a hypothesis supported by 3 formulas that all
show positive IC gets 3 Bayesian successes per epoch — its confidence
grows faster than a hypothesis with only one formula. This incentivises
miners to submit diverse implementations of the same idea.

Knowledge graph queries (examples)
-----------------------------------
    graph.formulas_for(hypothesis_id)     → list[FormulaRecord]
    graph.hypotheses_for(formula_id)      → list[HypothesisRecord]
    graph.most_supported_hypotheses(n)    → sorted by n_supporting_formulas
    graph.formula_evolution_tree(id)      → tree of parent → child formulas
    graph.shared_formulas(h1_id, h2_id)  → formulas linking two hypotheses
    graph.knowledge_summary()            → full structured summary dict

Usage
-----
    # Setup (one instance per validator session)
    hyp_lib   = HypothesisLibrary()
    graph     = ResearchGraph(hyp_lib)

    # When a miner submits a formula with a hypothesis:
    formula_id = graph.register_formula(
        formula_string="rank(momentum_20d)",
        miner_uid=0, epoch=1,
        hypothesis_ids=["abc123"],     # links to hypothesis
    )

    # After IC evaluation:
    graph.propagate_evidence(
        formula_id=formula_id,
        ic=0.042,
        msc=0.018,
        regime="Low-Vol Trending",
        epoch=2,
    )
    # → FormulaRecord updated
    # → HypothesisRecord.alpha_param += 1 (success)
    # → hypothesis.confidence_score rises

    # Query:
    kg = KnowledgeGraph(hyp_lib, graph.formula_library)
    kg.most_supported_hypotheses(5)
"""

from __future__ import annotations

import hashlib
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

_SUBNET = Path(__file__).resolve().parent.parent
_ROOT   = _SUBNET.parent
for p in [str(_SUBNET), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from macro8_subnet.alpha.hypothesis_engine import (
    HypothesisLibrary, HypothesisRecord, BayesianUpdater,
)


# ═══════════════════════════════════════════════════════════════════════════
# FormulaRecord — the middle tier
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FormulaRecord:
    """
    One alpha formula with its complete performance history.

    A FormulaRecord tracks everything observable about a formula:
    its IC across epochs, its Marginal Sharpe Contribution (MSC),
    its regime-conditional performance, and its provenance (which
    miner submitted it, which generation of evolution produced it,
    which parent formulas it was derived from).

    Multiple FormulaRecords can support the same hypothesis.
    One FormulaRecord can support multiple hypotheses.
    This is a many-to-many relationship, managed by ResearchGraph.

    Attributes
    ----------
    formula_id       : str — deterministic SHA-256 hash of formula_string
    formula_string   : str — the actual formula text
    miner_uid        : int — who submitted or evolved this formula
    miner_hotkey     : str
    epoch_born       : int — when first evaluated
    generation       : int — 0 = miner-submitted, N = Nth evolution generation
    parent_ids       : list[str] — formula_ids this was derived from
    hypothesis_ids   : list[str] — hypotheses this formula supports
    ic_history       : list[float] — IC per epoch
    msc_history      : list[float] — Marginal Sharpe Contribution per epoch
    regime_ic        : dict[str, list[float]] — IC by regime
    is_retired       : bool — True if IC consistently below threshold
    """
    formula_id:      str
    formula_string:  str
    miner_uid:       int
    miner_hotkey:    str                   = ""
    epoch_born:      int                   = 0
    generation:      int                   = 0      # 0 = original, N = evolved
    parent_ids:      list[str]             = field(default_factory=list)
    hypothesis_ids:  list[str]             = field(default_factory=list)
    ic_history:      list[float]           = field(default_factory=list)
    msc_history:     list[float]           = field(default_factory=list)
    regime_ic:       dict[str, list[float]] = field(default_factory=dict)
    is_retired:      bool                  = False

    # ── Computed properties ───────────────────────────────────────────────────

    @property
    def mean_ic(self) -> float:
        return float(np.mean(self.ic_history)) if self.ic_history else 0.0

    @property
    def mean_msc(self) -> float:
        return float(np.mean(self.msc_history)) if self.msc_history else 0.0

    @property
    def ic_stability(self) -> float:
        """Fraction of epochs with IC > 0."""
        if not self.ic_history:
            return 0.0
        return float(sum(1 for ic in self.ic_history if ic > 0) / len(self.ic_history))

    @property
    def n_evaluations(self) -> int:
        return len(self.ic_history)

    @property
    def is_evolved(self) -> bool:
        return self.generation > 0

    @property
    def n_supported_hypotheses(self) -> int:
        return len(self.hypothesis_ids)

    def best_regime(self) -> Optional[str]:
        if not self.regime_ic:
            return None
        avgs = {r: float(np.mean(ics)) for r, ics in self.regime_ic.items() if ics}
        return max(avgs, key=avgs.get) if avgs else None

    def ic_in_regime(self, regime: str) -> Optional[float]:
        """Mean IC in a specific regime, or None if no data."""
        ics = self.regime_ic.get(regime)
        return float(np.mean(ics)) if ics else None

    def should_retire(self, min_ic: float = 0.01, min_obs: int = 5) -> bool:
        """True if this formula has enough evidence and consistently poor IC."""
        if self.n_evaluations < min_obs:
            return False
        return self.mean_ic < min_ic and self.ic_stability < 0.3

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "formula_id":       self.formula_id,
            "formula_string":   self.formula_string,
            "miner_uid":        self.miner_uid,
            "epoch_born":       self.epoch_born,
            "generation":       self.generation,
            "parent_ids":       self.parent_ids,
            "hypothesis_ids":   self.hypothesis_ids,
            "n_evaluations":    self.n_evaluations,
            "mean_ic":          round(self.mean_ic,       6),
            "mean_msc":         round(self.mean_msc,      6),
            "ic_stability":     round(self.ic_stability,  4),
            "best_regime":      self.best_regime(),
            "is_retired":       self.is_retired,
            "is_evolved":       self.is_evolved,
        }

    def summary_line(self) -> str:
        hyp_count = len(self.hypothesis_ids)
        return (f"[gen={self.generation}] {self.formula_string[:50]:<50} "
                f"IC={self.mean_ic:.4f} n={self.n_evaluations:3d} "
                f"hyp={hyp_count}")


# ═══════════════════════════════════════════════════════════════════════════
# FormulaLibrary — storage for FormulaRecord objects
# ═══════════════════════════════════════════════════════════════════════════

class FormulaLibrary:
    """
    Persistent store for FormulaRecord objects.

    Parallel to AlphaLibrary (which stores AlphaFactor objects with
    raw signal data). FormulaLibrary stores the lightweight metadata
    records — the performance history without the full signal arrays.
    """

    def __init__(self):
        self._records: dict[str, FormulaRecord] = {}

    # ── Admission ─────────────────────────────────────────────────────────────

    def register(
        self,
        formula_string: str,
        miner_uid:      int,
        miner_hotkey:   str  = "",
        epoch:          int  = 0,
        generation:     int  = 0,
        parent_ids:     list[str] = (),
    ) -> FormulaRecord:
        """
        Register a formula. If it already exists, return the existing record.

        The formula_id is a deterministic hash of the formula string,
        so identical formulas submitted by different miners resolve to
        the same record.
        """
        fid = self._make_id(formula_string)
        if fid in self._records:
            return self._records[fid]

        record = FormulaRecord(
            formula_id=fid,
            formula_string=formula_string,
            miner_uid=miner_uid,
            miner_hotkey=miner_hotkey,
            epoch_born=epoch,
            generation=generation,
            parent_ids=list(parent_ids),
        )
        self._records[fid] = record
        return record

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get(self, formula_id: str) -> Optional[FormulaRecord]:
        return self._records.get(formula_id)

    def get_by_string(self, formula_string: str) -> Optional[FormulaRecord]:
        return self._records.get(self._make_id(formula_string))

    def all_active(self) -> list[FormulaRecord]:
        return [r for r in self._records.values() if not r.is_retired]

    def all_evolved(self) -> list[FormulaRecord]:
        return [r for r in self._records.values() if r.is_evolved]

    def rank_by_ic(self, top_n: int = 10) -> list[FormulaRecord]:
        active = self.all_active()
        return sorted(active, key=lambda r: r.mean_ic, reverse=True)[:top_n]

    def rank_by_msc(self, top_n: int = 10) -> list[FormulaRecord]:
        active = self.all_active()
        return sorted(active, key=lambda r: r.mean_msc, reverse=True)[:top_n]

    def by_hypothesis(self, hypothesis_id: str) -> list[FormulaRecord]:
        return [r for r in self._records.values()
                if hypothesis_id in r.hypothesis_ids]

    # ── Updates ───────────────────────────────────────────────────────────────

    def update_ic(
        self,
        formula_id: str,
        ic:         float,
        msc:        float  = 0.0,
        regime:     Optional[str] = None,
    ) -> Optional[FormulaRecord]:
        """Record a new IC observation for a formula."""
        rec = self._records.get(formula_id)
        if rec is None:
            return None
        rec.ic_history.append(float(ic))
        if msc != 0.0:
            rec.msc_history.append(float(msc))
        if regime:
            if regime not in rec.regime_ic:
                rec.regime_ic[regime] = []
            rec.regime_ic[regime].append(float(ic))
        # Auto-retire if consistently poor
        if rec.should_retire():
            rec.is_retired = True
        return rec

    def retire(self, formula_id: str) -> None:
        rec = self._records.get(formula_id)
        if rec:
            rec.is_retired = True

    # ── Statistics ────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._records)

    @property
    def n_active(self) -> int:
        return len(self.all_active())

    @property
    def n_evolved(self) -> int:
        return len(self.all_evolved())

    @staticmethod
    def _make_id(formula_string: str) -> str:
        return hashlib.sha256(formula_string.strip().encode()).hexdigest()[:12]


# ═══════════════════════════════════════════════════════════════════════════
# ResearchGraph — bi-directional mapping + evidence propagation
# ═══════════════════════════════════════════════════════════════════════════

class ResearchGraph:
    """
    Bi-directional formula↔hypothesis graph with evidence propagation.

    The graph has two node types (FormulaRecord, HypothesisRecord) and
    one edge type (supports: formula → hypothesis). Edges are unweighted
    but evidence propagates along them: evaluating a formula updates
    all linked hypotheses.

    This is the central integration point for the research loop:

        1. Miner submits (formula, hypothesis_id) pair
        2. register_formula() creates the FormulaRecord and the edge
        3. After IC evaluation: propagate_evidence() updates both
        4. KnowledgeGraph queries the resulting graph
    """

    def __init__(
        self,
        hypothesis_library: HypothesisLibrary,
        ic_threshold:       float = 0.02,
    ):
        self.hypothesis_library = hypothesis_library
        self.formula_library    = FormulaLibrary()
        self.ic_threshold       = ic_threshold
        self._updater           = BayesianUpdater(ic_threshold)

        # Bi-directional adjacency
        self._f_to_h: dict[str, set[str]] = defaultdict(set)  # formula_id → hypothesis_ids
        self._h_to_f: dict[str, set[str]] = defaultdict(set)  # hypothesis_id → formula_ids

    # ── Graph construction ────────────────────────────────────────────────────

    def register_formula(
        self,
        formula_string:  str,
        miner_uid:       int,
        epoch:           int         = 0,
        hypothesis_ids:  list[str]   = (),
        miner_hotkey:    str         = "",
        generation:      int         = 0,
        parent_ids:      list[str]   = (),
    ) -> str:
        """
        Register a formula and create edges to linked hypotheses.

        Args:
            formula_string:  The formula text.
            miner_uid:       Submitting miner.
            epoch:           Current epoch.
            hypothesis_ids:  IDs of hypotheses this formula supports.
            miner_hotkey:    Miner's ss58 address.
            generation:      Evolution generation (0 = original).
            parent_ids:      Parent formula IDs (for evolved formulas).

        Returns:
            formula_id (deterministic hash of formula_string).
        """
        rec = self.formula_library.register(
            formula_string, miner_uid, miner_hotkey,
            epoch, generation, parent_ids,
        )
        fid = rec.formula_id

        for hid in hypothesis_ids:
            self.link(fid, hid)

        return fid

    def link(self, formula_id: str, hypothesis_id: str) -> None:
        """
        Create a bi-directional edge between a formula and a hypothesis.

        Safe to call multiple times — idempotent.
        """
        self._f_to_h[formula_id].add(hypothesis_id)
        self._h_to_f[hypothesis_id].add(formula_id)

        # Update FormulaRecord.hypothesis_ids
        frec = self.formula_library.get(formula_id)
        if frec and hypothesis_id not in frec.hypothesis_ids:
            frec.hypothesis_ids.append(hypothesis_id)

    def unlink(self, formula_id: str, hypothesis_id: str) -> None:
        """Remove an edge between formula and hypothesis."""
        self._f_to_h[formula_id].discard(hypothesis_id)
        self._h_to_f[hypothesis_id].discard(formula_id)
        frec = self.formula_library.get(formula_id)
        if frec and hypothesis_id in frec.hypothesis_ids:
            frec.hypothesis_ids.remove(hypothesis_id)

    # ── Evidence propagation ──────────────────────────────────────────────────

    def propagate_evidence(
        self,
        formula_id:  str,
        ic:          float,
        msc:         float         = 0.0,
        regime:      Optional[str] = None,
        epoch:       int           = 0,
    ) -> dict[str, float]:
        """
        Propagate IC evidence from a formula to all linked hypotheses.

        This is the core operation of the research graph. When a formula
        is evaluated and an IC is observed:
            1. FormulaRecord is updated (ic_history, msc_history, regime_ic)
            2. Every linked HypothesisRecord gets a Bayesian update

        Args:
            formula_id:  The evaluated formula.
            ic:          Observed IC value.
            msc:         Observed Marginal Sharpe Contribution.
            regime:      Current market regime name.
            epoch:       Current epoch.

        Returns:
            Dict {hypothesis_id → new_confidence} for all updated hypotheses.
        """
        # Update formula record
        self.formula_library.update_ic(formula_id, ic, msc, regime)

        # Propagate to linked hypotheses
        updated = {}
        for hid in self._f_to_h.get(formula_id, set()):
            hrec = self.hypothesis_library.get(hid)
            if hrec is None:
                continue
            self._updater.update(hrec, ic, regime, epoch)
            updated[hid] = hrec.confidence_score

        return updated

    def propagate_batch(
        self,
        formula_ic_map: dict[str, float],
        msc_map:        Optional[dict[str, float]] = None,
        regime:         Optional[str]              = None,
        epoch:          int                        = 0,
    ) -> dict[str, dict[str, float]]:
        """
        Propagate evidence for multiple formulas at once.

        Args:
            formula_ic_map:  {formula_id: ic} for all evaluated formulas.
            msc_map:         Optional {formula_id: msc}.
            regime:          Current market regime.
            epoch:           Current epoch.

        Returns:
            {formula_id: {hypothesis_id: new_confidence}}
        """
        results = {}
        for fid, ic in formula_ic_map.items():
            msc     = (msc_map or {}).get(fid, 0.0)
            results[fid] = self.propagate_evidence(fid, ic, msc, regime, epoch)
        return results

    # ── Graph queries ─────────────────────────────────────────────────────────

    def formulas_for(self, hypothesis_id: str) -> list[FormulaRecord]:
        """Return all FormulaRecords supporting a hypothesis."""
        fids = self._h_to_f.get(hypothesis_id, set())
        return [r for fid in fids
                if (r := self.formula_library.get(fid)) is not None]

    def hypotheses_for(self, formula_id: str) -> list[HypothesisRecord]:
        """Return all HypothesisRecords a formula supports."""
        hids = self._f_to_h.get(formula_id, set())
        return [r for hid in hids
                if (r := self.hypothesis_library.get(hid)) is not None]

    def n_supporting_formulas(self, hypothesis_id: str) -> int:
        return len(self._h_to_f.get(hypothesis_id, set()))

    def n_supported_hypotheses(self, formula_id: str) -> int:
        return len(self._f_to_h.get(formula_id, set()))

    def shared_formulas(
        self,
        hypothesis_id_a: str,
        hypothesis_id_b: str,
    ) -> list[FormulaRecord]:
        """Return formulas that support both hypotheses."""
        shared_ids = (
            self._h_to_f.get(hypothesis_id_a, set()) &
            self._h_to_f.get(hypothesis_id_b, set())
        )
        return [r for fid in shared_ids
                if (r := self.formula_library.get(fid)) is not None]

    def formula_evolution_tree(self, formula_id: str) -> dict:
        """
        Build the evolution ancestry tree for a formula.

        Returns a nested dict: {formula_id: {ic, children: [...]}}
        """
        def _build(fid: str, visited: set) -> dict:
            if fid in visited:
                return {}
            visited.add(fid)
            rec  = self.formula_library.get(fid)
            node = {
                "formula_id":     fid,
                "formula_string": rec.formula_string if rec else "unknown",
                "mean_ic":        round(rec.mean_ic, 6) if rec else 0.0,
                "generation":     rec.generation if rec else 0,
                "children":       [],
            }
            # Find children: formulas that list this as a parent
            for other in self.formula_library._records.values():
                if fid in other.parent_ids:
                    node["children"].append(_build(other.formula_id, visited))
            return node

        return _build(formula_id, set())

    # ── Statistics ────────────────────────────────────────────────────────────

    @property
    def n_formulas(self) -> int:
        return self.formula_library.size

    @property
    def n_edges(self) -> int:
        return sum(len(hs) for hs in self._f_to_h.values())

    @property
    def n_isolated_formulas(self) -> int:
        """Formulas with no linked hypotheses."""
        return sum(1 for fid in self.formula_library._records
                   if not self._f_to_h.get(fid))

    @property
    def n_isolated_hypotheses(self) -> int:
        """Hypotheses with no supporting formulas."""
        return sum(
            1 for rec in self.hypothesis_library.all_active()
            if not self._h_to_f.get(rec.hypothesis_id)
        )


# ═══════════════════════════════════════════════════════════════════════════
# KnowledgeGraph — high-level query interface
# ═══════════════════════════════════════════════════════════════════════════

class KnowledgeGraph:
    """
    High-level query and summarisation interface for the research graph.

    Combines HypothesisLibrary and FormulaLibrary to answer questions
    that span both tiers of the knowledge architecture.
    """

    def __init__(
        self,
        hypothesis_library: HypothesisLibrary,
        formula_library:    FormulaLibrary,
        graph:              ResearchGraph,
    ):
        self.hypotheses = hypothesis_library
        self.formulas   = formula_library
        self.graph      = graph

    # ── Rich queries ──────────────────────────────────────────────────────────

    def most_supported_hypotheses(self, top_n: int = 5) -> list[dict]:
        """
        Hypotheses ranked by number of supporting formulas.

        Returns a list of dicts with hypothesis metadata + formula count.
        """
        results = []
        for rec in self.hypotheses.all_active():
            n_formulas = self.graph.n_supporting_formulas(rec.hypothesis_id)
            results.append({
                "hypothesis_id":   rec.hypothesis_id,
                "statement":       rec.statement,
                "category":        rec.category.value,
                "confidence":      round(rec.confidence_score, 4),
                "n_formulas":      n_formulas,
                "mean_ic":         round(rec.mean_ic, 6),
            })
        return sorted(results, key=lambda x: x["n_formulas"], reverse=True)[:top_n]

    def most_versatile_formulas(self, top_n: int = 5) -> list[dict]:
        """
        Formulas that support the most hypotheses.
        """
        results = []
        for rec in self.formulas.all_active():
            n_hyp = self.graph.n_supported_hypotheses(rec.formula_id)
            results.append({
                "formula_id":     rec.formula_id,
                "formula_string": rec.formula_string,
                "n_hypotheses":   n_hyp,
                "mean_ic":        round(rec.mean_ic, 6),
                "generation":     rec.generation,
            })
        return sorted(results, key=lambda x: x["n_hypotheses"], reverse=True)[:top_n]

    def best_formulas_for_hypothesis(
        self,
        hypothesis_id: str,
        top_n: int = 5,
    ) -> list[dict]:
        """
        Best (highest IC) formulas supporting a specific hypothesis.
        """
        formulas = self.graph.formulas_for(hypothesis_id)
        results  = [
            {"formula_string": r.formula_string,
             "mean_ic":        round(r.mean_ic, 6),
             "n_evaluations":  r.n_evaluations,
             "generation":     r.generation}
            for r in sorted(formulas, key=lambda r: r.mean_ic, reverse=True)
        ]
        return results[:top_n]

    def regime_performance_matrix(self) -> dict[str, dict[str, float]]:
        """
        Matrix of mean IC by (hypothesis_category × regime).

        Returns {category_value: {regime_name: mean_ic}}.
        """
        matrix: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

        for hrec in self.hypotheses.all_active():
            for regime, ics in hrec.regime_ic.items():
                matrix[hrec.category.value][regime].extend(ics)

        return {
            cat: {
                regime: round(float(np.mean(ics)), 6)
                for regime, ics in regimes.items() if ics
            }
            for cat, regimes in matrix.items()
        }

    def knowledge_summary(self) -> dict:
        """
        Full structured summary of the knowledge graph.
        JSON-serialisable.
        """
        return {
            "n_hypotheses":          self.hypotheses.size,
            "n_active_hypotheses":   self.hypotheses.n_active,
            "n_formulas":            self.formulas.size,
            "n_active_formulas":     self.formulas.n_active,
            "n_evolved_formulas":    self.formulas.n_evolved,
            "n_edges":               self.graph.n_edges,
            "n_isolated_formulas":   self.graph.n_isolated_formulas,
            "n_isolated_hypotheses": self.graph.n_isolated_hypotheses,
            "mean_confidence":       round(self.hypotheses.mean_confidence, 4),
            "most_supported":        self.most_supported_hypotheses(3),
            "most_versatile":        self.most_versatile_formulas(3),
            "regime_matrix":         self.regime_performance_matrix(),
        }

    def print_summary(self) -> None:
        """Print a formatted knowledge graph summary."""
        s = self.knowledge_summary()
        print(f"\n  🔬  MACRO8 KNOWLEDGE GRAPH")
        print(f"  {'─'*55}")
        print(f"  Hypotheses  : {s['n_active_hypotheses']}/{s['n_hypotheses']} active")
        print(f"  Formulas    : {s['n_active_formulas']}/{s['n_formulas']} active "
              f"({s['n_evolved_formulas']} evolved)")
        print(f"  Edges       : {s['n_edges']} (formula→hypothesis links)")
        print(f"  Mean conf   : {s['mean_confidence']:.3f}")
        print(f"\n  Most supported hypotheses:")
        for entry in s["most_supported"]:
            print(f"    [{entry['n_formulas']:2d} formulas] "
                  f"conf={entry['confidence']:.2f}  "
                  f"{entry['statement'][:45]}")
        print(f"\n  Most versatile formulas:")
        for entry in s["most_versatile"]:
            print(f"    [{entry['n_hypotheses']:2d} hyps] "
                  f"IC={entry['mean_ic']:.4f}  "
                  f"{entry['formula_string'][:40]}")
        print()
