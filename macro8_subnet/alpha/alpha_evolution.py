"""
alpha/alpha_evolution.py
-------------------------
Evolutionary alpha discovery using genetic programming.

The system maintains a population of alpha formula strings and evolves
them over generations toward higher Information Coefficient (IC).

This is genetic programming applied to quantitative finance — instead
of manually crafting signal formulas, the system discovers them
automatically through guided random search.

Algorithm
---------
    Population: N formula strings
    Fitness:    mean IC over evaluation period
    Selection:  tournament selection
    Mutation:   5 operators (feature_swap, op_insert, op_remove,
                             scale_add, negate)
    Crossover:  subtree blend

Each generation:
    1. Evaluate population fitness (IC scoring)
    2. Tournament select parents (k=3)
    3. Apply mutation (70% probability) or crossover (30%)
    4. Evaluate children
    5. Replace weakest N_REPLACE individuals with best children
    6. Log progress

Convergence:
    The population converges when std(IC) < CONVERGE_THRESHOLD
    or max generations is reached.

Formula grammar
---------------
    Simple:     <feature>
    Unary:      <op>(<simple>)
    Binary:     <simple> +|- <simple>
    Nested:     <op>(<op>(<simple>))

Where:
    <feature> = any FeatureStore feature name
    <op>      = zscore | rank | decay | neutralize | clip | lag | sign | abs

Example evolution trace:
    Gen 0:  momentum_20d                           IC=0.021
    Gen 1:  rank(momentum_20d)                     IC=0.031
    Gen 3:  rank(momentum_20d) - rank(volatility_20d)  IC=0.045
    Gen 7:  decay(rank(momentum_20d) - rank(volatility_20d), halflife=10)  IC=0.052
"""

from __future__ import annotations

import copy
import random
import re
import sys
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

from macro8_subnet.alpha.formula_engine import FormulaEngine
from macro8_subnet.alpha.ic_scorer      import ICScorer


# ── Configuration ─────────────────────────────────────────────────────────────

FEATURES  = [
    "momentum_5d", "momentum_20d", "momentum_60d",
    "volatility_20d", "volatility_60d",
    "zscore_20d", "cross_momentum", "relative_vol",
    "regime_signal", "rsi_14",
]

UNARY_OPS = ["zscore", "rank", "neutralize", "sign", "abs"]
DECAY_OPS = ["decay", "lag", "clip"]    # ops that take extra numeric args
BINARY_OPS = ["+", "-", "*"]


# ── Individual ────────────────────────────────────────────────────────────────

@dataclass
class Individual:
    """One formula in the evolutionary population."""
    formula:    str
    ic:         Optional[float]   = None   # fitness score
    ic_ir:      Optional[float]   = None
    evaluated:  bool              = False
    generation: int               = 0

    @property
    def fitness(self) -> float:
        """Fitness = mean IC, clamped to [0, inf] (positive IC only)."""
        return max(self.ic or 0.0, 0.0)

    def __repr__(self) -> str:
        ic_str = f"IC={self.ic:.4f}" if self.ic is not None else "IC=?"
        return f"Individual({self.formula!r}, {ic_str})"


# ── Evolved Formula Result ────────────────────────────────────────────────────

@dataclass
class EvolvedFormula:
    """Final output — an evolved formula with its performance metrics."""
    formula:      str
    ic:           float
    ic_ir:        float
    generation:   int             # generation when this was best
    ancestry:     list[str]       = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "formula":    self.formula,
            "ic":         round(self.ic,    6),
            "ic_ir":      round(self.ic_ir, 6),
            "generation": self.generation,
        }


@dataclass
class EvolutionReport:
    """Summary of one evolutionary run."""
    n_generations:       int
    population_size:     int
    best_ic:             float
    best_formula:        str
    ic_improvement:      float    # best_ic - initial_best_ic
    top_formulas:        list[EvolvedFormula]
    generation_history:  list[dict]   # per-generation stats
    converged:           bool

    def summary(self) -> str:
        return (
            f"Evolution: {self.n_generations} gen | "
            f"pop={self.population_size} | "
            f"best_IC={self.best_ic:.4f} | "
            f"improvement={self.ic_improvement:+.4f} | "
            f"{'converged' if self.converged else 'max_gen'}"
        )

    def to_dict(self) -> dict:
        return {
            "n_generations":      self.n_generations,
            "population_size":    self.population_size,
            "best_ic":            round(self.best_ic, 6),
            "best_formula":       self.best_formula,
            "ic_improvement":     round(self.ic_improvement, 6),
            "converged":          self.converged,
            "top_formulas":       [f.to_dict() for f in self.top_formulas],
            "generation_history": self.generation_history,
        }


# ── Alpha Evolution Engine ────────────────────────────────────────────────────

class AlphaEvolution:
    """
    Genetic programming engine for alpha formula discovery.

    Maintains a population of formula strings and evolves them
    toward higher IC using mutation and crossover operators.

    Usage
    -----
        evo    = AlphaEvolution(feature_store, population_size=20)
        report = evo.evolve(returns, n_generations=10)
        best   = report.top_formulas[0]
        print(f"Best formula: {best.formula}  IC={best.ic:.4f}")
    """

    def __init__(
        self,
        feature_store,
        population_size:      int   = 20,
        n_top_keep:           int   = 5,    # elite preservation
        tournament_k:         int   = 3,    # tournament size
        mutation_rate:        float = 0.70,
        crossover_rate:       float = 0.30,
        min_ic_threshold:     float = 0.01,
        converge_threshold:   float = 0.001,
        seed:                 int   = 42,
    ):
        self.pop_size          = population_size
        self.n_top             = n_top_keep
        self.tournament_k      = tournament_k
        self.mutation_rate     = mutation_rate
        self.crossover_rate    = crossover_rate
        self.min_ic            = min_ic_threshold
        self.converge_thr      = converge_threshold
        self._rng              = random.Random(seed)
        self._np_rng           = np.random.default_rng(seed)
        self._engine           = FormulaEngine(feature_store)
        self._ic_scorer        = ICScorer(min_obs=5, min_ic=min_ic_threshold)

    # ── Public API ────────────────────────────────────────────────────────────

    def evolve(
        self,
        returns:       pd.DataFrame,
        n_generations: int = 10,
        seed_formulas: list[str] | None = None,
        verbose:       bool = False,
    ) -> EvolutionReport:
        """
        Run the evolutionary algorithm for N generations.

        Args:
            returns:       Daily asset return DataFrame.
            n_generations: Number of evolutionary generations.
            seed_formulas: Optional starting formulas. Random if None.
            verbose:       Print per-generation progress.

        Returns:
            EvolutionReport with top evolved formulas and history.
        """
        # Initialise population
        population = self._init_population(seed_formulas)

        # Evaluate initial fitness
        self._evaluate_population(population, returns)
        population = self._sort_by_fitness(population)
        initial_best_ic = population[0].fitness

        gen_history = []
        converged   = False

        for gen in range(n_generations):
            # ── Elitism: keep top N ────────────────────────────────────────────
            elite    = population[:self.n_top]
            children = []

            # ── Fill rest with offspring ───────────────────────────────────────
            n_needed = self.pop_size - self.n_top
            attempts = 0
            while len(children) < n_needed and attempts < n_needed * 5:
                attempts += 1
                if self._rng.random() < self.crossover_rate and len(population) >= 2:
                    parent1 = self._tournament_select(population)
                    parent2 = self._tournament_select(population)
                    child_formula = self._crossover(parent1.formula, parent2.formula)
                else:
                    parent = self._tournament_select(population)
                    child_formula = self._mutate(parent.formula)

                if child_formula and self._engine.validate_formula(child_formula)[0]:
                    child = Individual(formula=child_formula, generation=gen + 1)
                    children.append(child)

            # ── Evaluate children ──────────────────────────────────────────────
            self._evaluate_population(children, returns)

            # ── Next generation ────────────────────────────────────────────────
            population = self._sort_by_fitness(elite + children)[:self.pop_size]

            # ── Stats ──────────────────────────────────────────────────────────
            ics    = [p.fitness for p in population if p.evaluated]
            best   = population[0]
            gen_stats = {
                "generation":  gen + 1,
                "best_ic":     round(best.fitness, 6),
                "mean_ic":     round(float(np.mean(ics)), 6) if ics else 0.0,
                "std_ic":      round(float(np.std(ics)),  6) if ics else 0.0,
                "best_formula": best.formula,
            }
            gen_history.append(gen_stats)

            if verbose:
                print(f"  Gen {gen+1:3d} | best_IC={best.fitness:.4f} | "
                      f"mean={gen_stats['mean_ic']:.4f} | "
                      f"{best.formula[:50]}")

            # ── Convergence check ──────────────────────────────────────────────
            if len(ics) >= 3 and float(np.std(ics)) < self.converge_thr:
                converged = True
                break

        # ── Compile results ────────────────────────────────────────────────────
        top_formulas = [
            EvolvedFormula(
                formula=p.formula,
                ic=p.ic or 0.0,
                ic_ir=p.ic_ir or 0.0,
                generation=p.generation,
            )
            for p in population[:10]
            if p.evaluated and (p.ic or 0.0) > 0
        ]

        return EvolutionReport(
            n_generations=len(gen_history),
            population_size=self.pop_size,
            best_ic=population[0].fitness,
            best_formula=population[0].formula,
            ic_improvement=population[0].fitness - initial_best_ic,
            top_formulas=top_formulas,
            generation_history=gen_history,
            converged=converged,
        )

    # ── Population initialisation ─────────────────────────────────────────────

    def _init_population(
        self,
        seed_formulas: list[str] | None,
    ) -> list[Individual]:
        """Create initial population from seeds + random formulas."""
        population = []

        if seed_formulas:
            for f in seed_formulas[:self.pop_size]:
                ok, _ = self._engine.validate_formula(f)
                if ok:
                    population.append(Individual(formula=f))

        # Fill remainder with random formulas
        attempts = 0
        while len(population) < self.pop_size and attempts < self.pop_size * 20:
            attempts += 1
            f = self._random_formula()
            ok, _ = self._engine.validate_formula(f)
            if ok:
                population.append(Individual(formula=f))

        return population

    def _random_formula(self, depth: int = 0) -> str:
        """Generate a random valid formula string."""
        max_depth = 2
        roll      = self._rng.random()

        if depth >= max_depth or roll < 0.40:
            # Terminal: raw feature
            return self._rng.choice(FEATURES)
        elif roll < 0.70:
            # Unary operator
            op    = self._rng.choice(UNARY_OPS)
            inner = self._random_formula(depth + 1)
            return f"{op}({inner})"
        elif roll < 0.90:
            # Binary expression
            left  = self._random_formula(depth + 1)
            right = self._random_formula(depth + 1)
            op    = self._rng.choice(["+", "-"])
            return f"{left} {op} {right}"
        else:
            # Decay/lag with numeric arg
            inner = self._random_formula(depth + 1)
            n     = self._rng.choice([5, 10, 20])
            return f"decay({inner}, halflife={n})"

    # ── Fitness evaluation ────────────────────────────────────────────────────

    def _evaluate_population(
        self,
        population: list[Individual],
        returns:    pd.DataFrame,
    ) -> None:
        """Evaluate IC fitness for every unevaluated individual."""
        for ind in population:
            if ind.evaluated:
                continue
            try:
                result  = self._engine.evaluate(ind.formula)
                if not result.success:
                    ind.ic = 0.0; ind.ic_ir = 0.0; ind.evaluated = True
                    continue

                ic_res  = self._ic_scorer.score(
                    f"evo_{ind.formula[:20]}", result.signals, returns
                )
                ind.ic       = ic_res.mean_ic if ic_res.success else 0.0
                ind.ic_ir    = ic_res.ic_ir   if ic_res.success else 0.0
                ind.evaluated = True
            except Exception:
                ind.ic = 0.0; ind.ic_ir = 0.0; ind.evaluated = True

    # ── Selection ─────────────────────────────────────────────────────────────

    def _tournament_select(self, population: list[Individual]) -> Individual:
        """Tournament selection: pick k random, return the best."""
        k         = min(self.tournament_k, len(population))
        candidates = self._rng.sample(population, k)
        return max(candidates, key=lambda x: x.fitness)

    @staticmethod
    def _sort_by_fitness(population: list[Individual]) -> list[Individual]:
        return sorted(population, key=lambda x: x.fitness, reverse=True)

    # ── Mutation operators ────────────────────────────────────────────────────

    def _mutate(self, formula: str) -> str:
        """Apply a randomly selected mutation operator."""
        op = self._rng.choice([
            self._mut_feature_swap,
            self._mut_op_insert,
            self._mut_op_remove,
            self._mut_scale_add,
            self._mut_negate,
        ])
        try:
            result = op(formula)
            return result if result else formula
        except Exception:
            return formula

    def _mut_feature_swap(self, formula: str) -> str:
        """Replace one feature name with another randomly chosen feature."""
        features_in = [f for f in FEATURES if f in formula]
        if not features_in:
            return formula
        old_feat = self._rng.choice(features_in)
        new_feat = self._rng.choice([f for f in FEATURES if f != old_feat])
        # Replace only the first occurrence to avoid breaking operators
        return formula.replace(old_feat, new_feat, 1)

    def _mut_op_insert(self, formula: str) -> str:
        """Wrap the entire formula in a unary operator."""
        op = self._rng.choice(UNARY_OPS)
        return f"{op}({formula})"

    def _mut_op_remove(self, formula: str) -> str:
        """Remove an outer unary operator if present."""
        # Pattern: op(inner) → inner
        match = re.match(r"^([a-z]+)\((.+)\)$", formula.strip())
        if match:
            op, inner = match.group(1), match.group(2)
            if op in UNARY_OPS:
                return inner
        return formula

    def _mut_scale_add(self, formula: str) -> str:
        """Add or subtract a random feature to the formula."""
        new_feat = self._rng.choice(FEATURES)
        op       = self._rng.choice(["+", "-"])
        use_rank = self._rng.random() < 0.5
        feat_str = f"rank({new_feat})" if use_rank else new_feat
        return f"({formula}) {op} {feat_str}"

    def _mut_negate(self, formula: str) -> str:
        """Negate the formula."""
        # Avoid double negation
        if formula.startswith("-"):
            return formula[1:].strip()
        return f"sign(zscore({formula})) * {formula}"

    # ── Crossover operator ────────────────────────────────────────────────────

    def _crossover(self, formula_a: str, formula_b: str) -> str:
        """Blend two parent formulas into one child formula."""
        roll = self._rng.random()

        if roll < 0.5:
            # Additive blend: take part of A and part of B
            return f"({formula_a}) + ({formula_b})"
        elif roll < 0.75:
            # Subtractive: difference of two signals (classic alpha style)
            return f"rank({formula_a}) - rank({formula_b})"
        else:
            # Regime-conditional: use A in strong regimes, B otherwise
            return f"({formula_a}) * regime_signal + ({formula_b})"
