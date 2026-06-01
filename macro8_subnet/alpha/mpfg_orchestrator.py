"""
alpha/mpfg_orchestrator.py
---------------------------
Massively Parallel Factor Generation — Top-level orchestrator.

Runs the complete MPFG pipeline:
  1. BatchFormulaGenerator  → large batch of formula strings
  2. ParallelICScorer       → IC for all formulas (vectorized + threaded)
  3. OrthogonalityFilter    → remove near-duplicate signals
  4. ResearchGraph          → register surviving formulas + propagate evidence
  5. HypothesisLibrary      → Bayesian updates per formula
  6. Returns MPFGReport     → complete run summary

This replaces the signal discovery step in ResearchLoop with an
industrially-scaled version that evaluates hundreds of formulas per
epoch instead of one per miner submission.

Design intent
-------------
MPFGOrchestrator does NOT replace ResearchLoop — it augments it.
The typical usage is:

    In ResearchLoop.run_epoch():
        1. Process miner formula submissions (existing logic)
        2. Run MPFGOrchestrator for autonomous batch discovery
        3. Merge results into the alpha library

This separates:
    Miner-submitted signals   → incentivised, human-curated
    MPFG-discovered signals   → systematic, algorithmic

Both enter the same alpha library.

Usage
-----
    prices   = load_prices(...)
    fs       = FeatureStore(prices)
    hyp_lib  = HypothesisLibrary()
    form_lib = FormulaLibrary()
    rg       = ResearchGraph(hyp_lib)

    orch = MPFGOrchestrator(prices, fs, hyp_lib, form_lib, rg)

    report = orch.run(
        epoch=5,
        batch_size=500,
        min_ic=0.02,
        regime="Low-Vol Trending",
    )
    print(f"Discovered {report.n_admitted} new signals")
    print(f"Throughput: {report.formulas_per_sec:.0f} formulas/sec")
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

_SUBNET = Path(__file__).resolve().parent.parent
_ROOT   = _SUBNET.parent
for p in [str(_SUBNET), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from macro8_subnet.alpha.batch_generator      import BatchFormulaGenerator, GenerationStrategy
from macro8_subnet.alpha.vectorized_evaluator  import VectorizedEvaluator
from macro8_subnet.alpha.parallel_scorer       import ParallelICScorer, BatchICResult
from macro8_subnet.alpha.orthogonality         import OrthogonalityFilter
from macro8_subnet.alpha.hypothesis_engine     import HypothesisLibrary
from macro8_subnet.alpha.research_graph        import ResearchGraph, FormulaLibrary


# ── Report types ──────────────────────────────────────────────────────────────

@dataclass
class MPFGReport:
    """Complete report from one MPFG pipeline run."""
    epoch:              int
    batch_size:         int
    n_generated:        int
    n_evaluated:        int
    n_passed_ic:        int
    n_deduped_out:      int    # removed by orthogonality filter
    n_admitted:         int    # entered formula library
    n_hypothesis_updates: int
    top_formulas:       list[dict]       # top N by IC
    elapsed_seconds:    float
    formulas_per_sec:   float
    generation_ms:      float
    evaluation_ms:      float
    filtering_ms:       float

    def summary(self) -> str:
        return (
            f"MPFG Epoch {self.epoch}: "
            f"gen={self.n_generated} | "
            f"eval={self.n_evaluated} | "
            f"pass_ic={self.n_passed_ic} | "
            f"dedup_out={self.n_deduped_out} | "
            f"admitted={self.n_admitted} | "
            f"hyp_updates={self.n_hypothesis_updates} | "
            f"{self.formulas_per_sec:.0f} f/s"
        )

    def to_dict(self) -> dict:
        return {
            "epoch":                self.epoch,
            "batch_size":           self.batch_size,
            "n_generated":          self.n_generated,
            "n_evaluated":          self.n_evaluated,
            "n_passed_ic":          self.n_passed_ic,
            "n_deduped_out":        self.n_deduped_out,
            "n_admitted":           self.n_admitted,
            "n_hypothesis_updates": self.n_hypothesis_updates,
            "elapsed_seconds":      round(self.elapsed_seconds,  3),
            "formulas_per_sec":     round(self.formulas_per_sec, 1),
            "generation_ms":        round(self.generation_ms,    1),
            "evaluation_ms":        round(self.evaluation_ms,    1),
            "filtering_ms":         round(self.filtering_ms,     1),
            "top_formulas":         self.top_formulas[:10],
        }


# ── Orchestrator ──────────────────────────────────────────────────────────────

class MPFGOrchestrator:
    """
    Massively Parallel Factor Generation pipeline orchestrator.

    Runs large-scale systematic formula discovery each epoch:
    generation → vectorized evaluation → deduplication → library admission.
    """

    def __init__(
        self,
        prices:             pd.DataFrame,
        feature_store       = None,
        hypothesis_library: Optional[HypothesisLibrary] = None,
        formula_library:    Optional[FormulaLibrary]    = None,
        research_graph:     Optional[ResearchGraph]     = None,
        n_workers:          int   = 2,
        corr_threshold:     float = 0.90,
        verbose:            bool  = False,
    ):
        """
        Args:
            prices:              Asset price DataFrame.
            feature_store:       FeatureStore for FormulaEngine.
            hypothesis_library:  For hypothesis-guided generation + updates.
            formula_library:     For library crossover + dedup.
            research_graph:      For registering new formulas.
            n_workers:           Parallel worker count.
            corr_threshold:      Orthogonality filter threshold.
            verbose:             Print progress.
        """
        self.prices    = prices
        self.returns   = prices.pct_change().dropna()
        self.verbose   = verbose

        # Component instances
        self._generator = BatchFormulaGenerator(
            hypothesis_library=hypothesis_library,
            formula_library=formula_library,
            feature_store=feature_store,
        )
        self._scorer    = ParallelICScorer(
            returns=self.returns,
            feature_store=feature_store,
            prices=prices,
            n_workers=n_workers,
        )
        self._filter    = OrthogonalityFilter(threshold=corr_threshold)
        self._hyp_lib   = hypothesis_library
        self._form_lib  = formula_library
        self._rg        = research_graph

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        epoch:           int   = 0,
        batch_size:      int   = 500,
        min_ic:          float = 0.02,
        regime:          Optional[str] = None,
        strategy:        GenerationStrategy = GenerationStrategy.MIXED,
        top_k_to_admit:  int   = 50,
    ) -> MPFGReport:
        """
        Run one MPFG epoch.

        Args:
            epoch:          Current epoch number.
            batch_size:     Number of formulas to generate and evaluate.
            min_ic:         Minimum mean IC for library admission.
            regime:         Current market regime (for evidence propagation).
            strategy:       Formula generation strategy.
            top_k_to_admit: Maximum formulas to admit per epoch.

        Returns:
            MPFGReport with complete run statistics.
        """
        t_total = time.perf_counter()
        self._log(f"  🏭  MPFG Epoch {epoch} | batch={batch_size} | min_ic={min_ic}")

        # ── Step 1: Generate formulas ─────────────────────────────────────────
        t0   = time.perf_counter()
        formulas, gen_report = self._generator.generate_with_report(batch_size, strategy)
        gen_ms = (time.perf_counter() - t0) * 1000
        self._log(f"    Gen: {len(formulas)} formulas in {gen_ms:.0f}ms")

        if not formulas:
            return self._empty_report(epoch, batch_size)

        # ── Step 2: Parallel IC evaluation ────────────────────────────────────
        t0       = time.perf_counter()
        batch_ic = self._scorer.evaluate_batch(formulas)
        eval_ms  = (time.perf_counter() - t0) * 1000
        self._log(
            f"    Eval: {batch_ic.n_success}/{batch_ic.n_total} passed | "
            f"{batch_ic.formulas_per_sec:.0f} f/s | "
            f"vec={batch_ic.n_vectorized} thr={batch_ic.n_threaded}"
        )

        # ── Step 3: Filter by minimum IC ──────────────────────────────────────
        all_ics      = batch_ic.all_ics()
        passing      = {f: ic for f, ic in all_ics.items() if ic >= min_ic}
        n_passed_ic  = len(passing)
        self._log(f"    IC filter: {n_passed_ic}/{len(all_ics)} passed min_ic={min_ic}")

        if not passing:
            elapsed = time.perf_counter() - t_total
            return MPFGReport(
                epoch=epoch, batch_size=batch_size,
                n_generated=len(formulas), n_evaluated=batch_ic.n_total,
                n_passed_ic=0, n_deduped_out=0, n_admitted=0,
                n_hypothesis_updates=0, top_formulas=[],
                elapsed_seconds=elapsed,
                formulas_per_sec=len(formulas) / max(elapsed, 1e-6),
                generation_ms=gen_ms, evaluation_ms=eval_ms, filtering_ms=0.0,
            )

        # ── Step 4: Orthogonality deduplication ───────────────────────────────
        t0 = time.perf_counter()
        passing_sorted = sorted(passing.items(), key=lambda x: x[1], reverse=True)
        top_k = dict(passing_sorted[:top_k_to_admit * 3])   # over-select before dedup

        # Build signal proxies for orthogonality check
        # Use IC scores as proxies (fast) — detailed signal check is expensive
        unique_formulas = self._dedup_by_ic_rank(top_k, max_keep=top_k_to_admit)
        n_deduped_out   = n_passed_ic - len(unique_formulas)
        filt_ms         = (time.perf_counter() - t0) * 1000
        self._log(f"    Dedup: {len(unique_formulas)} unique (removed {n_deduped_out})")

        # ── Step 5: Admit to formula library ──────────────────────────────────
        n_admitted = 0
        if self._form_lib is not None:
            for formula, ic in unique_formulas.items():
                rec = self._form_lib.register(formula, miner_uid=-1, epoch=epoch,
                                               generation=0)
                self._form_lib.update_ic(rec.formula_id, ic)
                n_admitted += 1

        # ── Step 6: Propagate evidence to research graph ──────────────────────
        n_hyp_updates = 0
        if self._rg is not None and self._form_lib is not None:
            for formula, ic in unique_formulas.items():
                rec = self._form_lib.get_by_string(formula)
                if rec is None:
                    continue
                updates = self._rg.propagate_evidence(
                    rec.formula_id, ic, regime=regime, epoch=epoch
                )
                n_hyp_updates += len(updates)

        # ── Report ─────────────────────────────────────────────────────────────
        elapsed        = time.perf_counter() - t_total
        top_formulas   = [
            {"formula": f, "mean_ic": round(ic, 6)}
            for f, ic in sorted(unique_formulas.items(), key=lambda x: x[1], reverse=True)[:10]
        ]

        report = MPFGReport(
            epoch=epoch,
            batch_size=batch_size,
            n_generated=len(formulas),
            n_evaluated=batch_ic.n_total,
            n_passed_ic=n_passed_ic,
            n_deduped_out=n_deduped_out,
            n_admitted=n_admitted,
            n_hypothesis_updates=n_hyp_updates,
            top_formulas=top_formulas,
            elapsed_seconds=elapsed,
            formulas_per_sec=len(formulas) / max(elapsed, 1e-6),
            generation_ms=gen_ms,
            evaluation_ms=eval_ms,
            filtering_ms=filt_ms,
        )
        self._log(f"    {report.summary()}")
        return report

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _dedup_by_ic_rank(
        formula_ics: dict[str, float],
        max_keep:    int,
    ) -> dict[str, float]:
        """
        Simple deduplication by IC rank — keep top max_keep formulas
        sorted by IC, skipping near-identical formula strings.

        True signal-level correlation check is expensive for large batches.
        We use formula string similarity as a fast proxy: if two formulas
        differ only by a feature name, they are likely to be correlated.
        """
        sorted_items = sorted(formula_ics.items(), key=lambda x: x[1], reverse=True)
        kept         = {}

        for formula, ic in sorted_items:
            if len(kept) >= max_keep:
                break
            # Simple string-level dedup: skip if very similar to an existing kept formula
            is_near_dup = any(
                _formula_similarity(formula, kept_f) > 0.85
                for kept_f in kept
            )
            if not is_near_dup:
                kept[formula] = ic

        return kept

    def _empty_report(self, epoch: int, batch_size: int) -> MPFGReport:
        return MPFGReport(
            epoch=epoch, batch_size=batch_size,
            n_generated=0, n_evaluated=0,
            n_passed_ic=0, n_deduped_out=0, n_admitted=0,
            n_hypothesis_updates=0, top_formulas=[],
            elapsed_seconds=0.0, formulas_per_sec=0.0,
            generation_ms=0.0, evaluation_ms=0.0, filtering_ms=0.0,
        )

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)


# ── Utility ───────────────────────────────────────────────────────────────────

def _formula_similarity(a: str, b: str) -> float:
    """
    Fast string-level formula similarity ∈ [0, 1].
    Uses character n-gram overlap (trigrams).
    Not a substitute for signal-level correlation — use for fast pre-filtering.
    """
    def trigrams(s: str) -> set:
        s = s.replace(" ", "").lower()
        return {s[i:i+3] for i in range(len(s) - 2)} if len(s) >= 3 else set(s)

    ta, tb  = trigrams(a), trigrams(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    intersection = len(ta & tb)
    union        = len(ta | tb)
    return intersection / union if union > 0 else 0.0
