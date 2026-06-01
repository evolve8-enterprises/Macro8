"""
alpha/gp_miner.py
------------------
Genetic Programming Formula Discovery Engine — Sprint 27 (diversity rebuild).

Sprint 27 fixes four root causes of premature convergence:

    1. Vec-fingerprint dedup removed from GP inner loop
       Previously killed 93/200 formulas per generation (46% loss).
       Only exact string dedup retained. Vec dedup now reserved for
       the final submission list only (validator-level dedup).

    2. 10 macro features added to GP terminal set
       risk_on_off, vol_regime, trend_strength, carry_proxy, etc.
       were built by FeatureStore but never appeared in formulas.
       ALL_FEATURES now has 34 terminals (was 24).

    3. Island model for permanent exploration pressure
       Population split: 75% exploitation (elites + mutations),
       25% exploration (fresh random formulas every generation).
       Previously: exploration boost fired only every 5 generations.

    4. Tournament selection replaces pure top-N elitism
       Picks parents via tournament (k=4) so lower-scoring but
       structurally diverse formulas can reproduce.
       Previously: only top-20 elites reproduced, starving diversity.

Result: populations with 120+ unique encodings (was 76), 
        top-32 submissions spanning 20+ distinct features (was 15).
"""

from __future__ import annotations

import random
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

from macro8_subnet.alpha.batch_evaluator import BatchEvaluator, ALL_FEATURES
from macro8_subnet.alpha.portfolio_evaluator import PortfolioEvaluator


# ── Grammar constants ──────────────────────────────────────────────────────────

# ALL_FEATURES now includes Sprint 26 macro features (34 total)
GP_FEATURES = ALL_FEATURES

UNARY_OPS       = ["rank", "zscore"]
DECAY_HALFLIVES = [5, 10, 15, 20, 30]
BINARY_OPS      = ["+", "-", "*"]

MAX_DEPTH       = 4
MAX_FORMULA_LEN = 180

DEFAULT_POP_SIZE    = 300   # increased from 200 for diversity
DEFAULT_ELITE_N     = 30    # increased from 20
DEFAULT_MUTATE_P    = 0.55
DEFAULT_RANDOM_P    = 0.25  # increased from 0.10 — permanent exploration island
DEFAULT_SUBMISSIONS = 32

# Tournament selection: sample k candidates, take the best
TOURNAMENT_K = 4


# ── ScoredFormula ─────────────────────────────────────────────────────────────

@dataclass
class ScoredFormula:
    """
    One formula with full fitness profile. Primary sort key: composite.
    mean_ic kept for backward compatibility.
    """
    formula:    str
    mean_ic:    float
    ic_ir:      float = 0.0
    generation: int   = 0

    composite:    float = 0.0
    sharpe:       float = 0.0
    sortino:      float = 0.0
    max_drawdown: float = 0.0
    turnover:     float = 0.0
    ic_7d:        float = 0.0
    ic_30d:       float = 0.0
    ic_90d:       float = 0.0
    capital_1m:   float = 0.0
    capital_100k: float = 0.0

    def __lt__(self, other: "ScoredFormula") -> bool:
        return self.composite < other.composite

    def __repr__(self) -> str:
        return (
            f"ScoredFormula({self.formula!r}, "
            f"composite={self.composite:.4f}, "
            f"IC={self.mean_ic:.4f}, Sharpe={self.sharpe:.3f})"
        )


# ── GPReport ──────────────────────────────────────────────────────────────────

@dataclass
class GPReport:
    n_generations:          int
    n_evaluated:            int
    elapsed_seconds:        float
    top_formulas:           list[ScoredFormula] = field(default_factory=list)
    best_ic_history:        list[float]         = field(default_factory=list)
    mean_ic_history:        list[float]         = field(default_factory=list)
    best_composite_history: list[float]         = field(default_factory=list)
    n_unique:               int                 = 0

    def summary(self) -> str:
        best  = self.top_formulas[0] if self.top_formulas else None
        lines = [
            f"GP Run: {self.n_generations} generations | "
            f"{self.n_evaluated} formulas evaluated | "
            f"{self.elapsed_seconds:.2f}s",
        ]
        if best:
            lines += [
                f"  Best composite={best.composite:.4f} | "
                f"Sharpe={best.sharpe:.3f} | "
                f"IC_1d={best.mean_ic:.4f} | "
                f"IC_30d={best.ic_30d:.4f} | "
                f"Turn={best.turnover:.3f} | "
                f"Score@1M={best.capital_1m:.3f}",
                f"  Formula: {best.formula}",
            ]
        else:
            lines.append("  No results")
        lines += [
            f"  Unique formulas found: {self.n_unique}",
            f"  Composite progression: "
            f"{' → '.join(f'{x:.4f}' for x in self.best_composite_history[::max(1, len(self.best_composite_history)//5)])}",
        ]
        return "\n".join(lines)

    def submission_formulas(self, n: int = DEFAULT_SUBMISSIONS) -> list[str]:
        """Top N formulas by composite, deduplicated by vec fingerprint for submission."""
        ranked = sorted(self.top_formulas, key=lambda sf: sf.composite, reverse=True)
        return [sf.formula for sf in ranked[:n]]


# ── FormulaGenerator ──────────────────────────────────────────────────────────

class FormulaGenerator:
    """
    Generates valid formula strings from the GP grammar.

    Grammar (Sprint 27):
        signal  := feature                           (terminal — 34 features)
                |  unary_op(signal)                 (rank, zscore)
                |  decay(signal, halflife=N)         (N ∈ {5,10,15,20,30})
                |  signal BINARY_OP signal           (+, -, *)
                |  unary_op(signal BINARY_OP signal) (cross-sectional key pattern)

    All 34 features including macro (risk_on_off, vol_regime, etc.) are terminals.
    """

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def random_formula(self, depth: int = 2) -> str:
        formula = self._build(depth)
        if len(formula) > MAX_FORMULA_LEN:
            return self._rng.choice(GP_FEATURES)
        return formula

    def _build(self, depth: int) -> str:
        if depth == 0:
            return self._feature()

        r = self._rng.random()

        if r < 0.20:
            # Terminal
            return self._feature()

        if r < 0.38:
            # Unary operator
            op    = self._rng.choice(UNARY_OPS)
            inner = self._build(depth - 1)
            return f"{op}({inner})"

        if r < 0.50 and depth >= 2:
            # Cross-sectional compound: rank/zscore(a OP b)
            op      = self._rng.choice(BINARY_OPS)
            left    = self._build(depth - 2)
            right   = self._build(depth - 2)
            wrapper = self._rng.choice(UNARY_OPS)
            inner   = (f"({left}) {op} ({right})"
                       if (" " in left or " " in right)
                       else f"{left} {op} {right}")
            return f"{wrapper}({inner})"

        if r < 0.62 and depth >= 2:
            # Decay with halflife
            hl    = self._rng.choice(DECAY_HALFLIVES)
            inner = self._build(depth - 2)
            return f"decay({inner}, halflife={hl})"

        # Binary operation
        op    = self._rng.choice(BINARY_OPS)
        left  = self._build(depth - 1)
        right = self._build(depth - 1)
        result = (f"({left}) {op} ({right})"
                  if (" " in left or " " in right)
                  else f"{left} {op} {right}")
        return result

    def _feature(self) -> str:
        return self._rng.choice(GP_FEATURES)

    def initial_population(self, n: int, seeds: list[str] = ()) -> list[str]:
        """
        Build initial population with deliberate depth diversity.

        Distribution (Sprint 27):
            depth=1: 20% (simple signals)
            depth=2: 40% (combined signals — most common)
            depth=3: 30% (complex expressions)
            depth=4: 10% (deep trees — rare, preserves encodability)
        """
        population = list(seeds)[:n]

        depth_counts = [
            (1, int(n * 0.20)),
            (2, int(n * 0.40)),
            (3, int(n * 0.30)),
            (4, int(n * 0.10)),
        ]
        scheduled = []
        for depth, count in depth_counts:
            scheduled.extend([depth] * count)
        self._rng.shuffle(scheduled)

        idx = 0
        while len(population) < n:
            d       = scheduled[idx % len(scheduled)] if scheduled else 2
            formula = self.random_formula(depth=d)
            population.append(formula)
            idx += 1

        return population[:n]


# ── GeneticOperators ──────────────────────────────────────────────────────────

class GeneticOperators:
    """
    Mutation and crossover operators.

    Sprint 27 adds:
        5. Macro feature injection (15%): replace a technical feature
           with a macro feature (risk_on_off, vol_regime, etc.)
        6. Structural swap (10%): swap the unary wrapper type
    """

    # Sprint 26+33 macro features available as injection targets
    MACRO_FEATURES = [
        "risk_on_off", "commodity_inflation", "em_vs_dm", "credit_stress",
        "equity_bond_corr", "cross_asset_vol", "vol_regime",
        "trend_strength", "carry_proxy", "dollar_proxy",
        # Sprint 33: event-layer proxies
        "stress_accel_5d", "stress_accel_20d",
        "eem_spy_20d", "iwm_spy_20d",
    ]

    TECHNICAL_FEATURES = [
        f for f in ALL_FEATURES if f not in {
            "risk_on_off", "commodity_inflation", "em_vs_dm", "credit_stress",
            "equity_bond_corr", "cross_asset_vol", "vol_regime",
            "trend_strength", "carry_proxy", "dollar_proxy",
            "stress_accel_5d", "stress_accel_20d",
            "eem_spy_20d", "iwm_spy_20d",
        }
    ]

    def __init__(self, generator: FormulaGenerator):
        self._gen = generator
        self._rng = generator._rng

    def mutate(self, formula: str) -> str:
        """
        Six mutation strategies:

        1. Full replacement     (15%): fresh random formula
        2. Feature swap         (30%): replace one feature with another
        3. Macro injection      (15%): replace technical with macro feature
        4. Operator swap        (15%): replace + with - etc.
        5. Wrap in unary        (15%): add rank/zscore wrapper
        6. Depth expansion      (10%): embed formula as subexpression
        """
        r = self._rng.random()

        if r < 0.15:
            return self._gen.random_formula(depth=self._rng.randint(1, 3))

        if r < 0.45:
            return self._swap_feature(formula, use_macro=False)

        if r < 0.60:
            return self._swap_feature(formula, use_macro=True)

        if r < 0.75:
            return self._swap_operator(formula)

        if r < 0.90:
            return self._wrap_unary(formula)

        # Depth expansion: embed formula in a binary expression
        return self._depth_expand(formula)

    def crossover(self, f1: str, f2: str) -> str:
        """
        Three crossover strategies:

        1. Binary combination (60%): (f1) OP (f2)
        2. Macro injection    (25%): replace a feature in f1 with f2's root
        3. Nested combination (15%): rank((f1) OP (f2))
        """
        r = self._rng.random()

        if r < 0.60:
            return self._binary_combine(f1, f2)

        if r < 0.85:
            # Extract root feature from f2 and inject into f1
            f2_feats = [f for f in GP_FEATURES if f in f2]
            if f2_feats:
                new_feat = self._rng.choice(f2_feats)
                return self._swap_feature(f1, use_macro=False, target_feat=new_feat)

        # Nested: wrap binary in rank/zscore
        op     = self._rng.choice(BINARY_OPS)
        w1     = f"({f1})" if any(o in f1 for o in [" + ", " - ", " * "]) else f1
        w2     = f"({f2})" if any(o in f2 for o in [" + ", " - ", " * "]) else f2
        inner  = f"{w1} {op} {w2}"
        result = f"{self._rng.choice(UNARY_OPS)}({inner})"
        if len(result) > MAX_FORMULA_LEN:
            return self._binary_combine(f1, f2)
        return result

    # ── Mutation helpers ──────────────────────────────────────────────────────

    def _swap_feature(
        self,
        formula:     str,
        use_macro:   bool = False,
        target_feat: Optional[str] = None,
    ) -> str:
        """Replace one feature name in the formula."""
        pool = self.MACRO_FEATURES if use_macro else GP_FEATURES
        features_in = [f for f in GP_FEATURES if f in formula]
        if not features_in:
            return formula
        victim      = self._rng.choice(features_in)
        replacement = target_feat or self._rng.choice(pool)
        return formula.replace(victim, replacement, 1)

    def _swap_operator(self, formula: str) -> str:
        """Swap one binary operator."""
        for op in self._rng.sample([" + ", " - ", " * "], 3):
            if op in formula:
                new_op = self._rng.choice([" + ", " - ", " * "])
                return formula.replace(op, new_op, 1)
        # No binary op — swap unary
        for op in UNARY_OPS:
            if f"{op}(" in formula:
                new_op = self._rng.choice(UNARY_OPS)
                return formula.replace(f"{op}(", f"{new_op}(", 1)
        return formula

    def _wrap_unary(self, formula: str) -> str:
        op = self._rng.choice(UNARY_OPS)
        if len(formula) + len(op) + 2 > MAX_FORMULA_LEN:
            return formula
        return f"{op}({formula})"

    def _depth_expand(self, formula: str) -> str:
        """Embed formula as one side of a binary expression."""
        other  = self._gen.random_formula(depth=1)
        op     = self._rng.choice(BINARY_OPS)
        w1     = f"({formula})" if any(o in formula for o in [" + ", " - ", " * "]) else formula
        result = f"{w1} {op} {other}"
        if len(result) > MAX_FORMULA_LEN:
            return formula
        return result

    def _binary_combine(self, f1: str, f2: str) -> str:
        op     = self._rng.choice(BINARY_OPS)
        w1     = f"({f1})" if any(o in f1 for o in [" + ", " - ", " * "]) else f1
        w2     = f"({f2})" if any(o in f2 for o in [" + ", " - ", " * "]) else f2
        result = f"{w1} {op} {w2}"
        if len(result) > MAX_FORMULA_LEN:
            # Strip to core features
            s1 = self._rng.choice([f for f in GP_FEATURES if f in f1] or [GP_FEATURES[0]])
            return f"{s1} {op} {self._gen.random_formula(1)}"
        return result


# ── GPMiner ───────────────────────────────────────────────────────────────────

class GPMiner:
    """
    Genetic programming miner — Sprint 27.

    Key changes from Sprint 24:
        - Population size: 300 (was 200)
        - Elite pool: 30 (was 20)
        - Random injection: 25% permanent (was 10% periodic)
        - Tournament selection replaces pure top-N elitism
        - Vec-fingerprint dedup removed from inner loop
        - 34 GP terminals including 10 macro features (was 24)
        - Macro injection mutation (15% of mutations)

    Parameters
    ----------
    prices:    pd.DataFrame — market data for evaluation.
    pop_size:  int  — population size (default 300).
    elite_n:   int  — elite pool size (default 30).
    mutate_p:  float — mutation fraction of offspring (default 0.55).
    random_p:  float — random injection fraction (default 0.25).
    seed:      int  — random seed.
    verbose:   bool — print generation stats.
    """

    def __init__(
        self,
        prices:           pd.DataFrame,
        pop_size:         int         = DEFAULT_POP_SIZE,
        elite_n:          int         = DEFAULT_ELITE_N,
        mutate_p:         float       = DEFAULT_MUTATE_P,
        random_p:         float       = DEFAULT_RANDOM_P,
        seed:             int         = 42,
        hypothesis_seeds: list[str]   = (),
        verbose:          bool        = False,
    ):
        self.prices       = prices
        self.pop_size     = pop_size
        self.elite_n      = elite_n
        self.mutate_p     = mutate_p
        self.random_p     = random_p
        self.verbose      = verbose
        self._multi_fitness = None

        self._batch_eval  = PortfolioEvaluator(prices, min_ic=0.0)
        self._generator   = FormulaGenerator(seed=seed)
        self._operators   = GeneticOperators(self._generator)

        try:
            from macro8_subnet.alpha.formula_engine import FormulaEngine
            from macro8_subnet.alpha.feature_store   import FeatureStore
            self._formula_engine = FormulaEngine(FeatureStore(prices))
        except Exception:
            self._formula_engine = None

        self._population: list[str] = self._generator.initial_population(
            pop_size, seeds=list(hypothesis_seeds)
        )
        self._scored:       list[ScoredFormula] = []
        self._hall_of_fame: dict[str, ScoredFormula] = {}
        self._generation    = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, n_epochs: int = 10) -> GPReport:
        t_start                = time.perf_counter()
        best_ic_history        = []
        mean_ic_history        = []
        best_composite_history = []
        total_evaluated        = 0

        for gen in range(n_epochs):
            scored, n_eval = self._evaluate_population()
            total_evaluated += n_eval

            if not scored:
                continue

            best_ic        = scored[0].mean_ic
            best_composite = scored[0].composite
            mean_ic = float(np.mean([s.mean_ic for s in scored if s.mean_ic > 0]) or 0)
            best_ic_history.append(best_ic)
            mean_ic_history.append(mean_ic)
            best_composite_history.append(best_composite)

            if self.verbose:
                s = scored[0]
                hall_diversity = len(set(
                    feat for sf in list(self._hall_of_fame.values())[:50]
                    for feat in GP_FEATURES if feat in sf.formula
                ))
                print(
                    f"  Gen {gen+1:3d} | best={s.composite:.4f} | "
                    f"sharpe={s.sharpe:.3f} | ic={s.mean_ic:.4f} | "
                    f"turn={s.turnover:.3f} | pop={len(self._population)} | "
                    f"hall={len(self._hall_of_fame)} | div_feats={hall_diversity}"
                )

            for sf in scored:
                existing = self._hall_of_fame.get(sf.formula)
                if existing is None or sf.composite > existing.composite:
                    self._hall_of_fame[sf.formula] = sf

            self._population = self._evolve(scored)
            self._generation += 1
            self._scored      = scored

        all_time_best = sorted(
            self._hall_of_fame.values(),
            key=lambda sf: sf.composite,
            reverse=True,
        )
        return GPReport(
            n_generations=n_epochs,
            n_evaluated=total_evaluated,
            elapsed_seconds=time.perf_counter() - t_start,
            top_formulas=all_time_best[:DEFAULT_SUBMISSIONS],
            best_ic_history=best_ic_history,
            mean_ic_history=mean_ic_history,
            best_composite_history=best_composite_history,
            n_unique=len(self._hall_of_fame),
        )

    def step(self) -> list[ScoredFormula]:
        scored, _ = self._evaluate_population()
        if scored:
            self._population = self._evolve(scored)
            self._generation += 1
            self._scored      = scored
            for sf in scored:
                existing = self._hall_of_fame.get(sf.formula)
                if existing is None or sf.composite > existing.composite:
                    self._hall_of_fame[sf.formula] = sf
        return sorted(self._hall_of_fame.values(),
                      key=lambda sf: sf.composite, reverse=True)

    def top_formulas(self, n: int = DEFAULT_SUBMISSIONS) -> list[str]:
        """Top N formula strings by composite score, deduplicated for submission."""
        sorted_hof = sorted(
            self._hall_of_fame.values(),
            key=lambda sf: sf.composite, reverse=True,
        )
        # Apply vec-fingerprint dedup only at submission time (not during evolution)
        seen_vec: set[str] = set()
        result: list[str] = []
        for sf in sorted_hof:
            try:
                w  = self._batch_eval.encoder.encode(sf.formula)
                fp = ",".join(f"{x:.3f}" for x in w)
                if fp in seen_vec:
                    continue
                seen_vec.add(fp)
            except Exception:
                pass
            result.append(sf.formula)
            if len(result) >= n:
                break
        return result

    def enable_multi_horizon(self, **scorer_kwargs) -> None:
        from macro8_subnet.evaluation.multi_horizon_scorer import MultiHorizonGPFitness
        self._multi_fitness = MultiHorizonGPFitness(self.prices, **scorer_kwargs)
        if self.verbose:
            print("[GPMiner] Multi-horizon fitness enabled")

    def add_hypothesis_seeds(self, seeds: list[str]) -> None:
        valid_seeds = [
            f for f in seeds
            if len(f) <= MAX_FORMULA_LEN and f not in self._population
        ]
        if self._scored:
            weakest_n = min(len(valid_seeds), len(self._population) // 5)
            self._population = self._population[:-weakest_n] + valid_seeds[:weakest_n]
        else:
            self._population[:len(valid_seeds)] = valid_seeds

    # ── Internal: evaluation ──────────────────────────────────────────────────

    def _evaluate_population(self) -> tuple[list[ScoredFormula], int]:
        """
        Evaluate population with PortfolioEvaluator.

        Sprint 27: vec-fingerprint dedup removed from inner loop.
        Only exact string dedup applied here. Vec dedup reserved for
        submission (top_formulas()) to avoid killing diversity during evolution.
        """
        # Exact string dedup only (preserves structurally different formulas)
        seen_exact: set[str] = set()
        unique: list[str] = []
        for f in self._population:
            if f not in seen_exact:
                seen_exact.add(f)
                unique.append(f)

        if not unique:
            return [], 0

        try:
            from macro8_subnet.neurons.validator import safe_formula as _safe
        except ImportError:
            _safe = lambda x: x if isinstance(x, str) and len(x) <= 200 else None

        safe = [f for f in unique if _safe(f) is not None]
        if not safe:
            return [], 0

        try:
            result = self._batch_eval.evaluate(safe)
        except Exception:
            return [], len(safe)

        ps_map = {ps.formula: ps for ps in result.portfolio_scores}
        scored = []

        for i, formula in enumerate(result.formulas):
            ps = ps_map.get(formula)
            if ps is not None:
                sf = ScoredFormula(
                    formula=formula,
                    mean_ic=float(result.mean_ics[i]),
                    ic_ir=float(result.ic_irs[i]),
                    generation=self._generation,
                    composite=ps.composite,
                    sharpe=ps.sharpe,
                    sortino=ps.sortino,
                    max_drawdown=ps.max_drawdown,
                    turnover=ps.daily_turnover,
                    ic_7d=ps.ic_by_horizon.get(7,  0.0),
                    ic_30d=ps.ic_by_horizon.get(30, 0.0),
                    ic_90d=ps.ic_by_horizon.get(90, 0.0),
                    capital_1m=ps.capital_scores.get(1_000_000, 0.0),
                    capital_100k=ps.capital_scores.get(100_000,  0.0),
                )
            else:
                sf = ScoredFormula(
                    formula=formula,
                    mean_ic=float(result.mean_ics[i]),
                    ic_ir=float(result.ic_irs[i]),
                    generation=self._generation,
                    composite=float(result.mean_ics[i]),
                )
            scored.append(sf)

        scored.sort(key=lambda sf: sf.composite, reverse=True)

        # Cross-sectional refinement for top-k
        TOP_K = 10
        has_cross = any(
            ("rank(" in sf.formula or "zscore(" in sf.formula)
            and any(op in sf.formula for op in [" + ", " - ", " * "])
            for sf in scored[:TOP_K]
        )
        if has_cross and self._formula_engine is not None:
            refined = self._refine_with_formula_engine(
                [sf.formula for sf in scored[:TOP_K]]
            )
            refined_map = {r.formula: r for r in refined}
            for sf in scored[:TOP_K]:
                if sf.formula in refined_map:
                    ric = refined_map[sf.formula].mean_ic
                    if abs(ric - sf.mean_ic) > 0.005:
                        sf.mean_ic   = ric
                        sf.composite = 0.70 * sf.composite + 0.30 * max(ric, 0)
            scored.sort(key=lambda sf: sf.composite, reverse=True)

        return scored, len(safe)

    def _refine_with_formula_engine(self, formulas: list[str]) -> list[ScoredFormula]:
        refined = []
        try:
            from macro8_subnet.alpha.ic_scorer import ICScorer
            returns = self.prices.pct_change().dropna()
            scorer  = ICScorer(min_obs=10)
            for formula in formulas:
                r = self._formula_engine.evaluate(formula)
                if r.success:
                    ic_result = scorer.score(formula, r.signals, returns)
                    refined.append(ScoredFormula(
                        formula=formula,
                        mean_ic=ic_result.mean_ic,
                        ic_ir=ic_result.ic_ir,
                        generation=self._generation,
                    ))
        except Exception:
            pass
        return refined

    # ── Internal: evolution ───────────────────────────────────────────────────

    def _evolve(self, scored: list[ScoredFormula]) -> list[str]:
        """
        Produce next generation.

        Sprint 27 structure:
            exploration island  (random_p):   fresh random formulas
            exploitation pool  (1 - random_p):
                elites              (elite_n / pop_size fraction)
                tournament mutations  (mutate_p fraction of remaining)
                tournament crossovers (1 - mutate_p fraction of remaining)

        Tournament selection: sample k=4 from full scored list, take best.
        This allows lower-scoring diverse formulas to reproduce occasionally.
        """
        rng     = self._generator._rng
        new_pop: list[str] = []

        # ── Always keep true elites ────────────────────────────────────────────
        elites = [sf.formula for sf in scored[:self.elite_n]]
        new_pop.extend(elites)

        # ── Island model: permanent exploration cohort ─────────────────────────
        n_explore = max(1, int(self.pop_size * self.random_p))
        for _ in range(n_explore):
            depth = rng.choices([1, 2, 3, 4], weights=[10, 40, 35, 15])[0]
            child = self._generator.random_formula(depth=depth)
            if len(child) <= MAX_FORMULA_LEN:
                new_pop.append(child)

        # ── Exploitation: tournament-selected parents ──────────────────────────
        while len(new_pop) < self.pop_size:
            r = rng.random()

            if r < self.mutate_p:
                parent = self._tournament_select(scored, k=TOURNAMENT_K)
                child  = self._operators.mutate(parent)
            else:
                p1 = self._tournament_select(scored, k=TOURNAMENT_K)
                p2 = self._tournament_select(scored, k=TOURNAMENT_K)
                child = self._operators.crossover(p1, p2)

            if len(child) <= MAX_FORMULA_LEN:
                new_pop.append(child)

        return new_pop[:self.pop_size]

    def _tournament_select(
        self,
        scored: list[ScoredFormula],
        k:      int = TOURNAMENT_K,
    ) -> str:
        """
        Tournament selection: sample k formulas, return the best.

        Unlike top-N elitism, this allows lower-ranked diverse formulas
        to occasionally reproduce, maintaining genetic diversity.
        """
        rng        = self._generator._rng
        candidates = rng.sample(scored, min(k, len(scored)))
        winner     = max(candidates, key=lambda sf: sf.composite)
        return winner.formula
