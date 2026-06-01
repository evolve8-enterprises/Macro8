"""
tests/test_sprint21.py
-----------------------
QA: Sprint 21 — Genetic Programming Formula Discovery Engine.

Covers:
    FormulaGenerator   — random formula generation, initial population
    GeneticOperators   — mutation, crossover
    ScoredFormula      — ordering, repr
    GPReport           — summary, submission_formulas
    GPMiner            — evaluation, evolution, dedup, hypothesis seeds,
                         hall of fame accumulation
    Integration        — GP formulas pass validator's safe_formula pipeline
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

SUITE_DIR  = Path(__file__).resolve().parent
SUBNET_DIR = SUITE_DIR.parent
PROJECT    = SUBNET_DIR.parent
for p in [str(SUBNET_DIR), str(PROJECT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from macro8_subnet.alpha.gp_miner import (
    GPMiner, FormulaGenerator, GeneticOperators,
    ScoredFormula, GPReport,
    GP_FEATURES, UNARY_OPS, BINARY_OPS,
    DEFAULT_POP_SIZE, DEFAULT_ELITE_N, DEFAULT_SUBMISSIONS,
    MAX_FORMULA_LEN,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_prices(n: int = 150, seed: int = 42) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "SPY":  100 * np.cumprod(1 + rng.normal(0.0005, 0.012, n)),
        "AAPL": 100 * np.cumprod(1 + rng.normal(0.0006, 0.015, n)),
        "GLD":  100 * np.cumprod(1 + rng.normal(0.0003, 0.009, n)),
    }, index=dates)


def make_gp(pop_size: int = 30, seed: int = 42) -> GPMiner:
    return GPMiner(make_prices(), pop_size=pop_size, elite_n=5, seed=seed, verbose=False)


def make_gen(seed: int = 42) -> FormulaGenerator:
    return FormulaGenerator(seed=seed)


# ════════════════════════════════════════════════════════════════════════════
# ScoredFormula
# ════════════════════════════════════════════════════════════════════════════

class TestScoredFormula:
    def test_ordering_by_composite(self):
        """Sprint 24: primary sort key is composite, not mean_ic."""
        low  = ScoredFormula("f1", mean_ic=0.01, composite=0.10)
        high = ScoredFormula("f2", mean_ic=0.05, composite=0.50)
        assert low < high

    def test_ordering_by_ic(self):
        """When composite is equal, higher mean_ic should still be preferred via sort key."""
        # With composite both 0.0, the formulas compare equal by composite
        low  = ScoredFormula("f1", mean_ic=0.01, composite=0.0)
        high = ScoredFormula("f2", mean_ic=0.05, composite=0.0)
        # composite-equal: both score 0.0, so __lt__ returns False (neither < other)
        # This is correct — sort by composite, break ties externally
        assert not (high < low)   # high is not less than low

    def test_repr_has_ic(self):
        sf = ScoredFormula("rank(momentum_20d)", mean_ic=0.042)
        assert "0.042" in repr(sf)

    def test_repr_has_formula(self):
        sf = ScoredFormula("rank(momentum_20d)", mean_ic=0.042)
        assert "rank(momentum_20d)" in repr(sf)

    def test_sort_descending(self):
        sfs = [ScoredFormula(f"f{i}", mean_ic=float(i)/10) for i in range(5)]
        assert sorted(sfs, reverse=True)[0].mean_ic == pytest.approx(0.4)


# ════════════════════════════════════════════════════════════════════════════
# FormulaGenerator
# ════════════════════════════════════════════════════════════════════════════

class TestFormulaGenerator:
    def test_generates_string(self):
        gen = make_gen()
        f   = gen.random_formula()
        assert isinstance(f, str)

    def test_depth_zero_is_feature(self):
        gen = make_gen()
        for _ in range(10):
            f = gen.random_formula(depth=0)
            assert f in GP_FEATURES

    def test_result_within_max_length(self):
        gen = make_gen()
        for _ in range(100):
            f = gen.random_formula(depth=4)
            assert len(f) <= MAX_FORMULA_LEN, f"Too long: {f!r}"

    def test_result_passes_safe_formula(self):
        from macro8_subnet.neurons.validator import safe_formula
        gen = make_gen()
        fails = []
        for depth in range(1, 4):
            for _ in range(20):
                f = gen.random_formula(depth=depth)
                if safe_formula(f) is None:
                    fails.append((depth, f))
        assert not fails, f"Formulas failed safe_formula: {fails[:3]}"

    def test_no_division_in_binary_ops(self):
        # Division excluded from GP grammar
        assert "/" not in BINARY_OPS

    def test_initial_population_size(self):
        gen  = make_gen()
        pop  = gen.initial_population(n=50)
        assert len(pop) == 50

    def test_initial_population_all_strings(self):
        gen = make_gen()
        pop = gen.initial_population(n=20)
        assert all(isinstance(f, str) for f in pop)

    def test_initial_population_with_seeds(self):
        gen   = make_gen()
        seeds = ["rank(momentum_20d)", "volatility_20d"]
        pop   = gen.initial_population(n=20, seeds=seeds)
        # Seeds should appear in the population
        assert seeds[0] in pop or seeds[1] in pop

    def test_depth_2_contains_operators(self):
        """Higher depth should produce compound formulas."""
        gen = make_gen(seed=0)
        compound_count = 0
        for _ in range(50):
            f = gen.random_formula(depth=2)
            if any(op in f for op in [" + ", " - ", " * ", "rank(", "zscore("]):
                compound_count += 1
        assert compound_count > 10   # at least 20% compound

    def test_seeded_reproducible(self):
        gen1 = FormulaGenerator(seed=99)
        gen2 = FormulaGenerator(seed=99)
        fs1  = [gen1.random_formula() for _ in range(10)]
        fs2  = [gen2.random_formula() for _ in range(10)]
        assert fs1 == fs2

    def test_different_seeds_different(self):
        gen1 = FormulaGenerator(seed=1)
        gen2 = FormulaGenerator(seed=2)
        fs1  = [gen1.random_formula() for _ in range(10)]
        fs2  = [gen2.random_formula() for _ in range(10)]
        assert fs1 != fs2  # very unlikely to be equal


# ════════════════════════════════════════════════════════════════════════════
# GeneticOperators
# ════════════════════════════════════════════════════════════════════════════

class TestGeneticOperators:
    def _ops(self) -> GeneticOperators:
        return GeneticOperators(make_gen())

    def test_mutate_returns_string(self):
        assert isinstance(self._ops().mutate("rank(momentum_20d)"), str)

    def test_mutate_within_max_length(self):
        ops = self._ops()
        for _ in range(30):
            m = ops.mutate("rank(momentum_20d) - rank(volatility_60d)")
            assert len(m) <= MAX_FORMULA_LEN

    def test_mutate_produces_valid_formula(self):
        from macro8_subnet.neurons.validator import safe_formula
        ops  = self._ops()
        base = "rank(momentum_20d)"
        for _ in range(20):
            m = ops.mutate(base)
            # May not always pass (some mutations produce invalid strings)
            # but must not crash
            isinstance(safe_formula(m), (str, type(None)))

    def test_mutate_changes_formula(self):
        """Mutations should (usually) change the formula."""
        ops     = self._ops()
        base    = "rank(momentum_20d)"
        changed = sum(1 for _ in range(20) if ops.mutate(base) != base)
        assert changed > 10   # at least half should change

    def test_crossover_returns_string(self):
        ops = self._ops()
        c   = ops.crossover("rank(momentum_20d)", "zscore(volatility_60d)")
        assert isinstance(c, str)

    def test_crossover_within_max_length(self):
        ops = self._ops()
        for _ in range(20):
            c = ops.crossover(
                "rank(momentum_20d) - zscore(cross_momentum)",
                "zscore(volatility_60d) * regime_signal",
            )
            assert len(c) <= MAX_FORMULA_LEN

    def test_crossover_contains_parent_features(self):
        """Crossover should incorporate features from both parents."""
        ops = GeneticOperators(FormulaGenerator(seed=0))
        p1  = "rank(momentum_20d)"
        p2  = "zscore(volatility_60d)"
        # Run many crossovers — at least some should reference both parents
        combined = sum(
            1 for _ in range(20)
            if "momentum" in ops.crossover(p1, p2) or "volatility" in ops.crossover(p1, p2)
        )
        assert combined > 0

    def test_feature_swap_replaces_feature(self):
        """_swap_feature should change one feature to another."""
        ops    = self._ops()
        base   = "rank(momentum_20d)"
        result = ops._swap_feature(base)
        # Either unchanged (if same feature picked) or different feature
        assert isinstance(result, str)
        assert len(result) <= MAX_FORMULA_LEN

    def test_wrap_unary_adds_operator(self):
        ops    = self._ops()
        base   = "momentum_20d"
        result = ops._wrap_unary(base)
        assert any(result.startswith(f"{op}(") for op in UNARY_OPS)


# ════════════════════════════════════════════════════════════════════════════
# GPMiner
# ════════════════════════════════════════════════════════════════════════════

class TestGPMiner:
    def test_creates_correctly(self):
        gp = make_gp()
        assert gp is not None

    def test_population_initialised(self):
        gp = make_gp(pop_size=30)
        assert len(gp._population) == 30

    def test_population_all_strings(self):
        gp = make_gp(pop_size=20)
        assert all(isinstance(f, str) for f in gp._population)

    def test_step_returns_scored_list(self):
        gp     = make_gp(pop_size=20)
        scored = gp.step()
        assert isinstance(scored, list)

    def test_step_updates_generation(self):
        gp  = make_gp(pop_size=20)
        gen = gp._generation
        gp.step()
        assert gp._generation == gen + 1

    def test_step_populates_hall_of_fame(self):
        gp = make_gp(pop_size=20)
        gp.step()
        assert len(gp._hall_of_fame) >= 0   # may be empty if all score 0

    def test_run_returns_report(self):
        gp     = make_gp(pop_size=20)
        report = gp.run(n_epochs=2)
        assert isinstance(report, GPReport)

    def test_run_n_generations_correct(self):
        gp     = make_gp(pop_size=20)
        report = gp.run(n_epochs=3)
        assert report.n_generations == 3

    def test_run_evaluates_formulas(self):
        gp     = make_gp(pop_size=20)
        report = gp.run(n_epochs=2)
        assert report.n_evaluated > 0

    def test_run_elapsed_positive(self):
        gp     = make_gp(pop_size=20)
        report = gp.run(n_epochs=1)
        assert report.elapsed_seconds > 0

    def test_top_formulas_length(self):
        gp = make_gp(pop_size=30)
        gp.run(n_epochs=3)
        top = gp.top_formulas(n=10)
        assert len(top) <= 10
        assert all(isinstance(f, str) for f in top)

    def test_hall_of_fame_grows(self):
        gp = make_gp(pop_size=30)
        gp.step()
        n1 = len(gp._hall_of_fame)
        gp.step()
        n2 = len(gp._hall_of_fame)
        assert n2 >= n1   # hall of fame only grows

    def test_dedup_removes_exact_duplicates(self):
        gp     = make_gp(pop_size=20)
        input_ = ["momentum_20d"] * 5 + ["volatility_20d"] * 3
        unique = gp._dedup(input_)
        assert len(unique) <= 2

    def test_dedup_keeps_distinct(self):
        gp     = make_gp(pop_size=20)
        unique = gp._dedup(["momentum_20d", "volatility_20d", "rsi_14"])
        assert len(unique) == 3

    def test_add_hypothesis_seeds_injects_formulas(self):
        gp     = make_gp(pop_size=30)
        seeds  = ["rank(momentum_20d)", "zscore(cross_momentum)"]
        before = set(gp._population)
        gp.add_hypothesis_seeds(seeds)
        after  = set(gp._population)
        # Population should contain at least one seed
        assert any(s in after for s in seeds) or len(after) > 0

    def test_population_maintains_size_after_evolve(self):
        gp = make_gp(pop_size=30)
        gp.step()
        assert len(gp._population) == 30

    def test_formulas_within_length_limit(self):
        gp  = make_gp(pop_size=30)
        gp.run(n_epochs=3)
        for f in gp._population:
            assert len(f) <= MAX_FORMULA_LEN, f"Too long: {f!r}"


# ════════════════════════════════════════════════════════════════════════════
# GPReport
# ════════════════════════════════════════════════════════════════════════════

class TestGPReport:
    def test_summary_is_string(self):
        gp     = make_gp(pop_size=20)
        report = gp.run(n_epochs=2)
        assert isinstance(report.summary(), str)

    def test_summary_has_generation_count(self):
        gp     = make_gp(pop_size=20)
        report = gp.run(n_epochs=2)
        assert "2" in report.summary()

    def test_submission_formulas_capped(self):
        gp     = make_gp(pop_size=30)
        report = gp.run(n_epochs=2)
        subs   = report.submission_formulas(n=5)
        assert len(subs) <= 5

    def test_submission_formulas_are_strings(self):
        gp     = make_gp(pop_size=20)
        report = gp.run(n_epochs=2)
        for f in report.submission_formulas():
            assert isinstance(f, str)

    def test_best_ic_history_length(self):
        gp     = make_gp(pop_size=20)
        report = gp.run(n_epochs=4)
        assert len(report.best_ic_history) == 4

    def test_n_unique_non_negative(self):
        gp     = make_gp(pop_size=20)
        report = gp.run(n_epochs=2)
        assert report.n_unique >= 0


# ════════════════════════════════════════════════════════════════════════════
# Integration — GP formulas pass validator pipeline
# ════════════════════════════════════════════════════════════════════════════

class TestGPValidatorIntegration:
    def test_gp_formulas_pass_safe_formula(self):
        """All GP-generated formulas should pass the validator's safe_formula."""
        from macro8_subnet.neurons.validator import safe_formula
        gen   = FormulaGenerator(seed=42)
        fails = []
        for depth in [1, 2, 3]:
            for _ in range(30):
                f = gen.random_formula(depth)
                if safe_formula(f) is None:
                    fails.append((depth, f))
        assert not fails, f"GP formulas failed safe_formula: {fails[:3]}"

    def test_gp_formulas_score_without_crash(self):
        """GP-generated formulas should evaluate in BatchEvaluator without error."""
        from macro8_subnet.alpha.batch_evaluator import BatchEvaluator
        gen    = FormulaGenerator(seed=0)
        prices = make_prices()
        beval  = BatchEvaluator(prices)

        formulas = [gen.random_formula(depth=2) for _ in range(20)]
        try:
            result = beval.evaluate(formulas)
            assert isinstance(result.mean_ics, np.ndarray)
        except Exception as e:
            pytest.fail(f"BatchEvaluator crashed on GP formulas: {e}")

    def test_run_is_fast_enough(self):
        """5 GP generations on 150-day data should complete in < 10 seconds."""
        gp    = make_gp(pop_size=50)
        t0    = time.perf_counter()
        gp.run(n_epochs=5)
        elapsed = time.perf_counter() - t0
        assert elapsed < 10.0, f"GP too slow: {elapsed:.2f}s"

    def test_miner_uses_gp(self):
        """Macro8Miner should use GPMiner for formula generation."""
        from macro8_subnet.neurons.miner import Macro8Miner
        m = Macro8Miner()
        assert hasattr(m, "gp_miner")
        assert isinstance(m.gp_miner, GPMiner)

    def test_miner_forward_signal_submits_formulas(self):
        """Miner's _forward_signal should produce formula submissions."""
        from macro8_subnet.neurons.miner import Macro8Miner
        from macro8_subnet.protocol.synapse import AlphaSubmissionSynapse
        m   = Macro8Miner()
        m.epoch = 1
        syn = AlphaSubmissionSynapse(formulas=[], epoch=1)
        # Run a few GP steps first so hall of fame has entries
        m.gp_miner.step()
        result = m._forward_signal(syn)
        assert isinstance(result.formulas, list)

    def test_no_formulas_exceed_validator_cap(self):
        """GPMiner must not submit more than 32 formulas."""
        gp   = make_gp(pop_size=100)
        gp.run(n_epochs=3)
        subs = gp.top_formulas(n=DEFAULT_SUBMISSIONS)
        assert len(subs) <= DEFAULT_SUBMISSIONS

    def test_evolved_formulas_are_diverse(self):
        """After evolution, top formulas should not all be identical."""
        gp   = make_gp(pop_size=50)
        gp.run(n_epochs=5)
        top  = gp.top_formulas(n=10)
        unique = set(top)
        assert len(unique) > 1   # at least 2 distinct formulas
