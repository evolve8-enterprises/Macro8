"""
alpha/research_loop.py
-----------------------
The Macro8 self-improving research loop.

This module orchestrates the complete alpha discovery pipeline as a
repeating epoch cycle. Each epoch the system:

  1. Receives formula submissions from miners
  2. Evaluates signal quality (IC scoring)
  3. Filters duplicates (orthogonality)
  4. Updates the alpha library (admission + lifecycle)
  5. Trains/updates the meta-alpha model
  6. Builds an optimised portfolio from library signals
  7. Measures alpha attribution (MSC per signal)
  8. Stress-tests against synthetic market scenarios
  9. Computes miner rewards from IC + attribution
  10. Produces a complete EpochReport

The loop is self-improving because:
  - The meta model accumulates training data → better IC predictions
  - The library retires weak signals → average quality rises
  - Miners optimise toward IC → signal quality improves
  - Attribution feedback shows which signal types are valued

Design principles
-----------------
  Fault isolation:  each step wraps in try/except, failures
                    produce None results and logged errors —
                    a bad signal never crashes the epoch

  Stateless epochs: all state lives in AlphaLibrary and
                    MetaAlphaModel, not in the loop itself

  Observable:       every step produces a typed result that
                    is included in EpochReport — nothing is
                    silently discarded

Usage
-----
    library = AlphaLibrary()
    model   = MetaAlphaModel()
    loop    = ResearchLoop(prices, library, model)

    for epoch in range(100):
        formulas  = collect_miner_submissions()
        report    = loop.run_epoch(epoch, formulas)
        rewards   = report.miner_rewards
        print(report.summary())
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

from macro8_subnet.alpha.alpha_schema       import AlphaFactor, AlphaEvaluation, AlphaCategory
from macro8_subnet.alpha.alpha_library      import AlphaLibrary
from macro8_subnet.alpha.alpha_lifecycle    import AlphaLifecycleManager
from macro8_subnet.alpha.ic_scorer          import ICScorer
from macro8_subnet.alpha.orthogonality      import OrthogonalityFilter
from macro8_subnet.alpha.meta_alpha_model   import MetaAlphaModel
from macro8_subnet.alpha.portfolio_optimizer import PortfolioOptimizer, OptMethod
from macro8_subnet.alpha.alpha_attribution  import AlphaAttributionEngine
from macro8_subnet.alpha.synthetic_market   import SyntheticMarketSimulator, SimModel
from macro8_subnet.alpha.feature_store      import FeatureStore
from macro8_subnet.alpha.formula_engine     import FormulaEngine


# ── Formula submission type ───────────────────────────────────────────────────

@dataclass
class FormulaSubmission:
    """A miner's formula submission for one epoch."""
    miner_uid:    int
    miner_hotkey: str
    formula:      str
    category:     str   = "unknown"
    description:  str   = ""


# ── Per-step result types ─────────────────────────────────────────────────────

@dataclass
class SignalGenResult:
    """Result of formula → signal generation step."""
    formula:      str
    miner_uid:    int
    signals:      Optional[dict[str, pd.Series]]
    success:      bool
    error:        Optional[str] = None


@dataclass
class ICStepResult:
    """Result of IC evaluation step for one signal."""
    factor_name:  str
    miner_uid:    int
    mean_ic:      Optional[float]
    ic_ir:        Optional[float]
    n_periods:    int
    passed:       bool
    error:        Optional[str] = None


@dataclass
class LibraryUpdateResult:
    """What changed in the library this epoch."""
    admitted:     list[str]   # factor names newly admitted
    rejected:     list[str]   # factor names rejected (low IC or duplicate)
    retired:      list[str]   # factor names retired by lifecycle manager
    library_size: int
    n_active:     int


@dataclass
class PortfolioStepResult:
    """Portfolio construction result for this epoch."""
    weights:          dict[str, float]
    expected_return:  float
    expected_vol:     float
    sharpe:           float
    method:           str
    n_signals:        int


@dataclass
class AttributionStepResult:
    """Attribution analysis result for this epoch."""
    portfolio_sharpe: float
    top_contributors: list[str]
    drags:            list[str]
    n_drags:          int
    diversification:  float
    signal_mscs:      dict[str, float]   # signal_name → MSC


@dataclass
class StressStepResult:
    """Synthetic market stress test results."""
    n_scenarios:        int
    mean_survival_rate: float
    scenario_summaries: list[dict]


@dataclass
class MinerReward:
    """Reward allocation for one miner this epoch."""
    miner_uid:    int
    miner_hotkey: str
    ic_score:     float     # mean IC of submitted signal (0 if failed)
    msc_score:    float     # marginal Sharpe contribution (0 if not in portfolio)
    final_reward: float     # normalised weight [0, 1]
    rank:         int


# ── Epoch Report ──────────────────────────────────────────────────────────────

@dataclass
class EpochReport:
    """
    Complete results from one research loop epoch.

    Everything that happened in this epoch — every signal evaluated,
    every decision made, every reward allocated — is recorded here.
    """
    epoch:             int
    n_submissions:     int
    elapsed_seconds:   float = 0.0

    # Step results
    signal_gen:        list[SignalGenResult]      = field(default_factory=list)
    ic_results:        list[ICStepResult]         = field(default_factory=list)
    library_update:    Optional[LibraryUpdateResult] = None
    meta_prediction:   Optional[dict]             = None   # top-5 predicted signals
    portfolio:         Optional[PortfolioStepResult] = None
    attribution:       Optional[AttributionStepResult] = None
    stress:            Optional[StressStepResult] = None
    miner_rewards:     list[MinerReward]          = field(default_factory=list)

    # Aggregate stats
    n_signals_passed_ic: int  = 0
    n_signals_admitted:  int  = 0
    n_signals_retired:   int  = 0
    library_size:        int  = 0
    errors:              list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"═══ Epoch {self.epoch} Report ═══",
            f"  Submissions  : {self.n_submissions}",
            f"  Passed IC    : {self.n_signals_passed_ic}",
            f"  Admitted     : {self.n_signals_admitted}",
            f"  Retired      : {self.n_signals_retired}",
            f"  Library size : {self.library_size} active",
            f"  Elapsed      : {self.elapsed_seconds:.2f}s",
        ]
        if self.portfolio:
            lines.append(
                f"  Portfolio    : Sharpe={self.portfolio.sharpe:.3f}  "
                f"Return={self.portfolio.expected_return:.2%}  "
                f"Vol={self.portfolio.expected_vol:.2%}"
            )
        if self.attribution:
            top = ", ".join(self.attribution.top_contributors[:3])
            lines.append(f"  Top signals  : {top}")
        if self.miner_rewards:
            best = max(self.miner_rewards, key=lambda r: r.final_reward)
            lines.append(f"  Best miner   : uid={best.miner_uid}  "
                         f"reward={best.final_reward:.4f}")
        if self.errors:
            lines.append(f"  Errors       : {len(self.errors)}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "epoch":              self.epoch,
            "n_submissions":      self.n_submissions,
            "elapsed_seconds":    round(self.elapsed_seconds, 3),
            "n_signals_passed_ic": self.n_signals_passed_ic,
            "n_signals_admitted": self.n_signals_admitted,
            "n_signals_retired":  self.n_signals_retired,
            "library_size":       self.library_size,
            "portfolio": self.portfolio.__dict__ if self.portfolio else None,
            "attribution": {
                "portfolio_sharpe": self.attribution.portfolio_sharpe,
                "top_contributors": self.attribution.top_contributors,
                "drags":            self.attribution.drags,
                "diversification":  self.attribution.diversification,
            } if self.attribution else None,
            "miner_rewards": [
                {"uid": r.miner_uid, "ic": round(r.ic_score, 4),
                 "msc": round(r.msc_score, 4), "reward": round(r.final_reward, 6)}
                for r in self.miner_rewards
            ],
            "errors": self.errors,
        }


# ── Research Loop ─────────────────────────────────────────────────────────────

class ResearchLoop:
    """
    Self-improving alpha research loop orchestrator.

    Runs the complete pipeline each epoch: formula evaluation → IC scoring
    → library management → meta-model training → portfolio construction
    → attribution → stress testing → reward emission.

    The loop improves over time because:
      - The meta-alpha model accumulates training data
      - The library retires weak signals and admits stronger ones
      - Miners optimise their formulas toward higher IC

    Parameters
    ----------
    prices           : pd.DataFrame of asset closing prices (date × asset)
    library          : AlphaLibrary instance (persists across epochs)
    meta_model       : MetaAlphaModel instance (persists across epochs)
    min_ic_threshold : Minimum mean IC for library admission
    ic_corr_threshold: Maximum allowed signal correlation (orthogonality)
    run_stress       : Whether to run synthetic market stress tests
    verbose          : Print step-by-step progress
    """

    def __init__(
        self,
        prices:            pd.DataFrame,
        library:           AlphaLibrary,
        meta_model:        MetaAlphaModel,
        min_ic_threshold:  float = 0.02,
        ic_corr_threshold: float = 0.90,
        run_stress:        bool  = True,
        verbose:           bool  = True,
    ):
        self.prices    = prices
        self.returns   = prices.pct_change().dropna()
        self.library   = library
        self.meta      = meta_model
        self.min_ic    = min_ic_threshold
        self.verbose   = verbose
        self.run_stress = run_stress

        # Build shared components
        self._feature_store  = FeatureStore(prices)
        self._features       = self._feature_store.build()
        self._formula_engine = FormulaEngine(self._feature_store)
        self._ic_scorer      = ICScorer(min_obs=5, min_ic=min_ic_threshold)
        self._orth_filter    = OrthogonalityFilter(threshold=ic_corr_threshold)
        self._lifecycle_mgr  = AlphaLifecycleManager()
        self._optimizer      = PortfolioOptimizer(method=OptMethod.IC_WEIGHTED, max_weight=0.40)
        self._attribution    = AlphaAttributionEngine()
        self._stress_sim     = SyntheticMarketSimulator(
            assets=list(prices.columns), n_days=60
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def run_epoch(
        self,
        epoch:       int,
        submissions: list[FormulaSubmission],
    ) -> EpochReport:
        """
        Execute one complete research loop epoch.

        Args:
            epoch:       Current epoch number (monotonically increasing).
            submissions: List of FormulaSubmission from miners.

        Returns:
            EpochReport with complete results for this epoch.
        """
        t_start = time.perf_counter()
        report  = EpochReport(epoch=epoch, n_submissions=len(submissions))
        self._log(f"\n{'═'*56}")
        self._log(f"  🔬  RESEARCH LOOP — EPOCH {epoch}")
        self._log(f"{'═'*56}")
        self._log(f"  {len(submissions)} submissions | "
                  f"library={self.library.n_active} active signals")

        # ── Step 1: Formula → Signals ─────────────────────────────────────────
        self._log("\n  Step 1: Generating signals from formulas...")
        gen_results = self._step_generate_signals(submissions)
        report.signal_gen = gen_results
        n_generated = sum(1 for r in gen_results if r.success)
        self._log(f"    {n_generated}/{len(submissions)} formulas produced valid signals")

        # ── Step 2: IC Scoring ────────────────────────────────────────────────
        self._log("\n  Step 2: IC scoring...")
        ic_results = self._step_ic_scoring(gen_results)
        report.ic_results          = ic_results
        report.n_signals_passed_ic = sum(1 for r in ic_results if r.passed)
        self._log(f"    {report.n_signals_passed_ic}/{len(ic_results)} signals passed IC threshold")

        # ── Step 3: Orthogonality filtering + Library admission ───────────────
        self._log("\n  Step 3: Orthogonality filtering + library admission...")
        lib_result = self._step_library_update(
            gen_results, ic_results, epoch
        )
        report.library_update     = lib_result
        report.n_signals_admitted = len(lib_result.admitted)
        report.n_signals_retired  = len(lib_result.retired)
        report.library_size       = lib_result.n_active
        self._log(f"    Admitted={len(lib_result.admitted)} "
                  f"Rejected={len(lib_result.rejected)} "
                  f"Retired={len(lib_result.retired)} "
                  f"Active={lib_result.n_active}")

        # ── Step 4: Meta-alpha model update ───────────────────────────────────
        self._log("\n  Step 4: Meta-alpha model update...")
        report.meta_prediction = self._step_meta_update(ic_results, epoch)
        if report.meta_prediction:
            self._log(f"    Top predicted: {report.meta_prediction.get('top_5', [])}")

        # Proceed to portfolio only if library has signals
        if self.library.n_active == 0:
            self._log("  ⚠️  Empty library — skipping portfolio/attribution steps")
            report.elapsed_seconds = time.perf_counter() - t_start
            report.miner_rewards   = self._compute_rewards(ic_results, {})
            return report

        # ── Step 5: Portfolio construction ────────────────────────────────────
        self._log("\n  Step 5: Portfolio construction...")
        portfolio_result = self._step_build_portfolio()
        report.portfolio = portfolio_result
        if portfolio_result:
            self._log(f"    Sharpe={portfolio_result.sharpe:.3f}  "
                      f"Return={portfolio_result.expected_return:.2%}  "
                      f"Vol={portfolio_result.expected_vol:.2%}  "
                      f"Signals={portfolio_result.n_signals}")

        # ── Step 6: Attribution ───────────────────────────────────────────────
        self._log("\n  Step 6: Alpha attribution...")
        attribution_result = self._step_attribution(portfolio_result)
        report.attribution = attribution_result
        if attribution_result:
            self._log(f"    Top: {attribution_result.top_contributors[:3]}  "
                      f"Drags: {len(attribution_result.drags)}")

        # ── Step 7: Stress testing ────────────────────────────────────────────
        if self.run_stress and portfolio_result and portfolio_result.weights:
            self._log("\n  Step 7: Stress testing...")
            stress_result      = self._step_stress_test(portfolio_result.weights)
            report.stress      = stress_result
            if stress_result:
                self._log(f"    Scenarios={stress_result.n_scenarios}  "
                          f"Avg survival={stress_result.mean_survival_rate:.0%}")

        # ── Step 8: Reward computation ────────────────────────────────────────
        self._log("\n  Step 8: Computing rewards...")
        msc_map = {}
        if attribution_result:
            msc_map = attribution_result.signal_mscs
        report.miner_rewards = self._compute_rewards(ic_results, msc_map)

        report.elapsed_seconds = time.perf_counter() - t_start
        self._log(f"\n{report.summary()}")
        return report

    # ── Step implementations ──────────────────────────────────────────────────

    def _step_generate_signals(
        self,
        submissions: list[FormulaSubmission],
    ) -> list[SignalGenResult]:
        """Step 1: Evaluate each miner's formula to produce signals."""
        results = []
        for sub in submissions:
            try:
                result = self._formula_engine.evaluate(sub.formula)
                results.append(SignalGenResult(
                    formula=sub.formula,
                    miner_uid=sub.miner_uid,
                    signals=result.signals if result.success else None,
                    success=result.success,
                    error=result.error,
                ))
            except Exception as e:
                results.append(SignalGenResult(
                    formula=sub.formula, miner_uid=sub.miner_uid,
                    signals=None, success=False,
                    error=f"{type(e).__name__}: {e}",
                ))
        return results

    def _step_ic_scoring(
        self,
        gen_results: list[SignalGenResult],
    ) -> list[ICStepResult]:
        """Step 2: Score each valid signal using IC analysis."""
        results = []
        for i, gen in enumerate(gen_results):
            if not gen.success or gen.signals is None:
                results.append(ICStepResult(
                    factor_name=f"miner_{gen.miner_uid}_f{i}",
                    miner_uid=gen.miner_uid,
                    mean_ic=None, ic_ir=None, n_periods=0,
                    passed=False, error="Signal generation failed",
                ))
                continue

            name = f"miner_{gen.miner_uid}_formula_{i}"
            try:
                ic_result = self._ic_scorer.score(name, gen.signals, self.returns)
                passed    = (ic_result.success and
                             ic_result.mean_ic is not None and
                             ic_result.mean_ic >= self.min_ic)
                results.append(ICStepResult(
                    factor_name=name,
                    miner_uid=gen.miner_uid,
                    mean_ic=ic_result.mean_ic,
                    ic_ir=ic_result.ic_ir,
                    n_periods=ic_result.n_periods,
                    passed=passed,
                    error=ic_result.error,
                ))
            except Exception as e:
                results.append(ICStepResult(
                    factor_name=name, miner_uid=gen.miner_uid,
                    mean_ic=None, ic_ir=None, n_periods=0,
                    passed=False, error=f"{type(e).__name__}: {e}",
                ))
        return results

    def _step_library_update(
        self,
        gen_results: list[SignalGenResult],
        ic_results:  list[ICStepResult],
        epoch:       int,
    ) -> LibraryUpdateResult:
        """Step 3: Orthogonality filter + admit passing signals + retire weak ones."""
        # Gather passing signals
        passing = {}
        ic_map  = {}
        for gen, ic in zip(gen_results, ic_results):
            if ic.passed and gen.signals:
                passing[ic.factor_name] = gen.signals
                ic_map[ic.factor_name]  = ic.mean_ic or 0.0

        admitted, rejected = [], []

        if passing:
            # Check orthogonality against existing library
            lib_sigs = self.library.all_signals()
            orth     = self._orth_filter.analyse(
                {**lib_sigs, **passing}, ic_scores={**{k: self.library.get_record(k).mean_ic
                for k in lib_sigs if self.library.get_record(k)}, **ic_map}
            )

            for name, signals in passing.items():
                if name in orth.rejected_factors:
                    rejected.append(name)
                    continue

                # Check actual correlation with library
                max_corr, _ = self._orth_filter.correlate_with_library(
                    name, signals, lib_sigs
                )

                # Find the miner_uid for this factor
                miner_uid = next(
                    (ic.miner_uid for ic in ic_results if ic.factor_name == name), 0
                )

                factor = AlphaFactor(
                    name=name, miner_uid=miner_uid, miner_hotkey=f"uid_{miner_uid}",
                    signals=signals, category=AlphaCategory.UNKNOWN,
                )
                evaluation = AlphaEvaluation(
                    factor_name=name, miner_uid=miner_uid,
                    mean_ic=ic_map[name],
                    ic_ir=next((ic.ic_ir for ic in ic_results if ic.factor_name == name), 0.0),
                    max_corr=max_corr,
                    is_duplicate=(max_corr > self._orth_filter.threshold),
                    passes_ic_threshold=True,
                    success=True,
                )
                ok, _ = self.library.add_factor(factor, evaluation, epoch, self.min_ic)
                if ok:
                    admitted.append(name)
                else:
                    rejected.append(name)

        # Lifecycle assessment — retire weak signals
        active_records = [
            self.library.get_record(f.name)
            for f in self.library.all_active_factors()
            if self.library.get_record(f.name)
        ]
        retired = []
        if active_records:
            lifecycle = self._lifecycle_mgr.assess(active_records, epoch)
            for action in lifecycle.actions:
                if action.should_retire:
                    self.library.retire(action.factor_name, epoch)
                    retired.append(action.factor_name)

        return LibraryUpdateResult(
            admitted=admitted,
            rejected=rejected,
            retired=retired,
            library_size=self.library.size,
            n_active=self.library.n_active,
        )

    def _step_meta_update(
        self,
        ic_results: list[ICStepResult],
        epoch:      int,
    ) -> Optional[dict]:
        """Step 4: Feed new IC observations into the meta-alpha model."""
        active_records = [
            self.library.get_record(f.name)
            for f in self.library.all_active_factors()
            if self.library.get_record(f.name)
        ]
        if not active_records:
            return None

        # Add training samples: use current IC as a proxy for "next-period" IC
        # (In production: record IC at epoch t, feed as target at epoch t-1)
        for record in active_records:
            if record.current_ic != 0.0:
                self.meta.add_training_sample(record, record.current_ic)

        report = self.meta.predict_all(active_records)
        return {
            "is_trained":  report.is_trained,
            "n_samples":   report.n_training_samples,
            "top_5":       report.top_signals(5),
            "r_squared":   report.train_r_squared,
        }

    def _step_build_portfolio(self) -> Optional[PortfolioStepResult]:
        """Step 5: Build optimised portfolio from active library signals."""
        active = self.library.all_active_factors()
        if not active:
            return None

        # Use IC scores as signal strength proxy
        signals_dict = {}
        for factor in active:
            record = self.library.get_record(factor.name)
            if record:
                signals_dict[factor.name] = max(record.mean_ic, 0.0)

        if not signals_dict:
            return None

        try:
            # Map signal names → asset allocation using factor signal matrices
            # Use the IC-weighted approach: weight ∝ signal mean IC
            result = self._optimizer.optimize(signals_dict, self._returns_for_library(active))
            if result.success:
                return PortfolioStepResult(
                    weights=result.weights,
                    expected_return=result.expected_return,
                    expected_vol=result.expected_vol,
                    sharpe=result.sharpe_ratio,
                    method=result.method.value,
                    n_signals=len(result.weights),
                )
        except Exception as e:
            pass
        return None

    def _step_attribution(
        self,
        portfolio: Optional[PortfolioStepResult],
    ) -> Optional[AttributionStepResult]:
        """Step 6: Measure per-signal Marginal Sharpe Contribution."""
        if portfolio is None or not portfolio.weights:
            return None

        active = {f.name: f for f in self.library.all_active_factors()}
        if not active:
            return None

        # Build signal return proxy: use each signal's raw values as "returns"
        signal_returns = self._build_signal_returns(active)
        if signal_returns.empty:
            return None

        ic_scores  = {}
        capacities = {}
        for name in active:
            rec = self.library.get_record(name)
            if rec:
                ic_scores[name]  = rec.mean_ic
                capacities[name] = rec.capacity

        try:
            report = self._attribution.attribute(
                signal_returns, portfolio.weights, ic_scores, capacities
            )
            msc_map = {a.signal_name: a.msc for a in report.attributions}
            return AttributionStepResult(
                portfolio_sharpe=report.portfolio_sharpe,
                top_contributors=report.top_contributors,
                drags=report.drags,
                n_drags=len(report.drags),
                diversification=report.diversification_ratio,
                signal_mscs=msc_map,
            )
        except Exception:
            return None

    def _step_stress_test(
        self,
        weights: dict[str, float],
    ) -> Optional[StressStepResult]:
        """Step 7: Stress-test the portfolio across synthetic market scenarios."""
        try:
            from simulation.portfolio_simulator import simulate_portfolio as sp
            from scoring.metrics import max_drawdown as md, total_return as tr

            summaries   = []
            survivals   = []
            scenarios   = [SimModel.JUMP_DIFFUSION, SimModel.CORR_SHOCK,
                           SimModel.REGIME_SWITCH, SimModel.MEAN_REVERT]

            for model in scenarios:
                sim    = self._stress_sim.generate(model)
                # Map signal weights → asset weights (equal share per signal)
                n       = len(self.prices.columns)
                aw      = {a: 1.0 / n for a in self.prices.columns}
                try:
                    pv      = sp(sim.returns, aw)
                    drawdown = md(pv)
                    ret      = tr(pv)
                    survived = drawdown < 0.35
                    survivals.append(survived)
                    summaries.append({
                        "scenario":  model.value,
                        "return":    round(ret, 4),
                        "drawdown":  round(drawdown, 4),
                        "survived":  survived,
                    })
                except Exception:
                    pass

            survival_rate = sum(survivals) / len(survivals) if survivals else 0.0
            return StressStepResult(
                n_scenarios=len(summaries),
                mean_survival_rate=survival_rate,
                scenario_summaries=summaries,
            )
        except Exception:
            return None

    def _compute_rewards(
        self,
        ic_results: list[ICStepResult],
        msc_map:    dict[str, float],
    ) -> list[MinerReward]:
        """Step 8: Compute final miner rewards from IC + MSC."""
        rewards = []
        for ic in ic_results:
            ic_score  = max(ic.mean_ic or 0.0, 0.0)
            msc_score = max(msc_map.get(ic.factor_name, 0.0), 0.0)
            # Combined score: 60% IC contribution, 40% portfolio attribution
            combined  = 0.6 * ic_score + 0.4 * msc_score
            rewards.append((ic.miner_uid, ic_score, msc_score, combined))

        if not rewards:
            return []

        # Normalise to sum to 1.0
        scores = np.array([r[3] for r in rewards])
        total  = scores.sum()
        norm   = scores / total if total > 1e-8 else np.ones(len(scores)) / len(scores)

        # Sort by reward descending
        indexed = sorted(enumerate(rewards), key=lambda x: norm[x[0]], reverse=True)

        result = []
        for rank, (i, (uid, ic, msc, _)) in enumerate(indexed, start=1):
            result.append(MinerReward(
                miner_uid=uid,
                miner_hotkey=f"uid_{uid}",
                ic_score=ic,
                msc_score=msc,
                final_reward=float(norm[i]),
                rank=rank,
            ))
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _returns_for_library(self, active_factors) -> pd.DataFrame:
        """
        Build a returns DataFrame where columns are library signal names.
        Uses mean asset return weighted by signal strength as a proxy.
        """
        rows = {}
        for factor in active_factors:
            sig_matrix = factor.signal_matrix()
            common     = sig_matrix.index.intersection(self.returns.index)
            if len(common) < 5:
                continue
            # Use equally-weighted average signal as return proxy
            avg_signal = sig_matrix.loc[common].mean(axis=1)
            rows[factor.name] = avg_signal

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).dropna()

    def _build_signal_returns(self, active: dict) -> pd.DataFrame:
        """Build signal return proxies for attribution analysis."""
        rows = {}
        for name, factor in active.items():
            try:
                sig_m  = factor.signal_matrix()
                common = sig_m.index.intersection(self.returns.index)
                if len(common) >= 5:
                    rows[name] = sig_m.loc[common].mean(axis=1)
            except Exception:
                pass
        return pd.DataFrame(rows).dropna() if rows else pd.DataFrame()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)
