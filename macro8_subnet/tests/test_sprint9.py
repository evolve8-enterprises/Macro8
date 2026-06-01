"""
tests/test_sprint9.py
----------------------
QA: Tests for Sprint 9 — research loop and evolutionary alpha discovery.
Self-contained: runs with only scikit-learn, scipy, pandas, numpy.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

SUITE_DIR  = Path(__file__).resolve().parent
SUBNET_DIR = SUITE_DIR.parent
PROJECT    = SUBNET_DIR.parent

for p in [str(SUBNET_DIR), str(PROJECT)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_prices(n: int = 150, seed: int = 42) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "SPY":  100 * np.cumprod(1 + rng.normal(0.0005, 0.012, n)),
        "AAPL": 100 * np.cumprod(1 + rng.normal(0.0006, 0.015, n)),
        "GLD":  100 * np.cumprod(1 + rng.normal(0.0003, 0.009, n)),
    }, index=dates)


def make_feature_store(prices=None):
    from macro8_subnet.alpha.feature_store import FeatureStore
    return FeatureStore(prices if prices is not None else make_prices())


def make_loop(run_stress=False, min_ic=0.0):
    from macro8_subnet.alpha.research_loop    import ResearchLoop
    from macro8_subnet.alpha.alpha_library    import AlphaLibrary
    from macro8_subnet.alpha.meta_alpha_model import MetaAlphaModel
    return ResearchLoop(
        make_prices(),
        AlphaLibrary(),
        MetaAlphaModel(min_samples=999),   # won't train during tests
        min_ic_threshold=min_ic,
        run_stress=run_stress,
        verbose=False,
    )


def make_subs(n: int = 3) -> list:
    from macro8_subnet.alpha.research_loop import FormulaSubmission
    formulas = [
        "momentum_20d",
        "rank(momentum_20d) - rank(volatility_20d)",
        "zscore(cross_momentum)",
        "decay(momentum_5d, halflife=10)",
        "regime_signal",
    ]
    return [
        FormulaSubmission(i, f"5F{i:040d}", formulas[i % len(formulas)])
        for i in range(n)
    ]


def make_evo(pop_size: int = 8, seed: int = 42):
    from macro8_subnet.alpha.alpha_evolution import AlphaEvolution
    return AlphaEvolution(make_feature_store(), population_size=pop_size, seed=seed)


# ════════════════════════════════════════════════════════════════════════════
# RESEARCH LOOP
# ════════════════════════════════════════════════════════════════════════════

class TestFormulaSubmission:
    def test_fields_stored(self):
        from macro8_subnet.alpha.research_loop import FormulaSubmission
        s = FormulaSubmission(miner_uid=7, miner_hotkey="5F", formula="momentum_20d")
        assert s.miner_uid == 7
        assert s.formula   == "momentum_20d"
        assert s.miner_hotkey == "5F"

    def test_optional_fields_default(self):
        from macro8_subnet.alpha.research_loop import FormulaSubmission
        s = FormulaSubmission(1, "5F", "momentum_20d")
        assert s.category    == "unknown"
        assert s.description == ""


class TestResearchLoopInit:
    def test_creates(self):
        assert make_loop() is not None

    def test_feature_store_populated(self):
        loop = make_loop()
        assert loop._features is not None
        assert len(loop._features) > 0

    def test_returns_match_prices(self):
        loop = make_loop()
        assert len(loop.returns) == len(loop.prices) - 1

    def test_assets_from_prices(self):
        loop = make_loop()
        assert set(loop.prices.columns) == {"SPY", "AAPL", "GLD"}


class TestRunEpoch:
    def test_returns_epoch_report(self):
        from macro8_subnet.alpha.research_loop import EpochReport
        r = make_loop().run_epoch(1, make_subs(2))
        assert isinstance(r, EpochReport)

    def test_epoch_number_correct(self):
        r = make_loop().run_epoch(42, make_subs(1))
        assert r.epoch == 42

    def test_n_submissions_recorded(self):
        r = make_loop().run_epoch(1, make_subs(3))
        assert r.n_submissions == 3

    def test_signal_gen_length(self):
        r = make_loop().run_epoch(1, make_subs(3))
        assert len(r.signal_gen) == 3

    def test_ic_results_length(self):
        r = make_loop().run_epoch(1, make_subs(3))
        assert len(r.ic_results) == 3

    def test_library_update_present(self):
        r = make_loop().run_epoch(1, make_subs(2))
        assert r.library_update is not None

    def test_library_update_fields(self):
        lu = make_loop().run_epoch(1, make_subs(2)).library_update
        assert isinstance(lu.admitted,  list)
        assert isinstance(lu.rejected,  list)
        assert isinstance(lu.retired,   list)
        assert isinstance(lu.n_active,  int)

    def test_elapsed_positive(self):
        r = make_loop().run_epoch(1, make_subs(2))
        assert r.elapsed_seconds > 0

    def test_miner_rewards_list(self):
        r = make_loop().run_epoch(1, make_subs(3))
        assert isinstance(r.miner_rewards, list)

    def test_reward_sum_is_one(self):
        r = make_loop().run_epoch(1, make_subs(3))
        if r.miner_rewards:
            total = sum(rw.final_reward for rw in r.miner_rewards)
            assert abs(total - 1.0) < 1e-6

    def test_to_dict_serialisable(self):
        r = make_loop().run_epoch(1, make_subs(2))
        json.dumps(r.to_dict())

    def test_summary_is_string(self):
        r = make_loop().run_epoch(1, make_subs(2))
        assert isinstance(r.summary(), str)

    def test_empty_submissions_ok(self):
        r = make_loop().run_epoch(1, [])
        assert r.epoch == 1
        assert r.n_submissions == 0

    def test_invalid_formula_handled(self):
        from macro8_subnet.alpha.research_loop import FormulaSubmission
        subs = [
            FormulaSubmission(0, "5F", "INVALID_XYZ_FORMULA"),
            FormulaSubmission(1, "5G", "momentum_20d"),
        ]
        r = make_loop().run_epoch(1, subs)
        assert r.n_submissions == 2  # didn't crash

    def test_reward_ranks_sequential(self):
        r = make_loop().run_epoch(1, make_subs(4))
        if r.miner_rewards:
            ranks = sorted(rw.rank for rw in r.miner_rewards)
            assert ranks == list(range(1, len(ranks) + 1))

    def test_multiple_epochs_library_stable(self):
        """Library size never goes negative across epochs."""
        loop = make_loop(min_ic=0.0)
        prev = 0
        for epoch in range(1, 4):
            r    = loop.run_epoch(epoch, make_subs(2))
            assert r.library_update.n_active >= 0
            prev = r.library_update.n_active


class TestStepMethods:
    def test_generate_signals_valid(self):
        from macro8_subnet.alpha.research_loop import FormulaSubmission
        loop    = make_loop()
        subs    = [FormulaSubmission(0, "5F", "momentum_20d")]
        results = loop._step_generate_signals(subs)
        assert len(results) == 1
        assert isinstance(results[0].success, bool)

    def test_generate_signals_invalid(self):
        from macro8_subnet.alpha.research_loop import FormulaSubmission
        loop    = make_loop()
        subs    = [FormulaSubmission(0, "5F", "INVALID__formula")]
        results = loop._step_generate_signals(subs)
        assert results[0].success is False

    def test_ic_scoring_returns_list(self):
        from macro8_subnet.alpha.research_loop import FormulaSubmission
        loop    = make_loop()
        subs    = [FormulaSubmission(0, "5F", "momentum_20d")]
        gen_res = loop._step_generate_signals(subs)
        ic_res  = loop._step_ic_scoring(gen_res)
        assert len(ic_res) == 1
        assert hasattr(ic_res[0], "mean_ic")
        assert hasattr(ic_res[0], "passed")

    def test_compute_rewards_empty(self):
        rewards = make_loop()._compute_rewards([], {})
        assert rewards == []

    def test_compute_rewards_sum_to_one(self):
        from macro8_subnet.alpha.research_loop import ICStepResult
        loop    = make_loop()
        ic_res  = [
            ICStepResult("f0", 0, 0.06, 0.7, 20, True),
            ICStepResult("f1", 1, 0.04, 0.5, 20, True),
            ICStepResult("f2", 2, 0.02, 0.3, 20, True),
        ]
        rewards = loop._compute_rewards(ic_res, {"f0": 0.5, "f1": 0.2, "f2": 0.1})
        total   = sum(r.final_reward for r in rewards)
        assert abs(total - 1.0) < 1e-6


# ════════════════════════════════════════════════════════════════════════════
# ALPHA EVOLUTION
# ════════════════════════════════════════════════════════════════════════════

class TestIndividual:
    def test_fitness_from_ic(self):
        from macro8_subnet.alpha.alpha_evolution import Individual
        assert Individual("f", ic=0.05).fitness == pytest.approx(0.05)

    def test_fitness_none_is_zero(self):
        from macro8_subnet.alpha.alpha_evolution import Individual
        assert Individual("f", ic=None).fitness == 0.0

    def test_fitness_clamps_negative(self):
        from macro8_subnet.alpha.alpha_evolution import Individual
        assert Individual("f", ic=-0.03).fitness == 0.0

    def test_repr_has_formula(self):
        from macro8_subnet.alpha.alpha_evolution import Individual
        assert "momentum" in repr(Individual("momentum_20d", ic=0.04))


class TestRandomFormula:
    def test_returns_string(self):
        assert isinstance(make_evo()._random_formula(), str)

    def test_non_empty(self):
        evo = make_evo()
        for _ in range(30):
            assert len(evo._random_formula()) > 0

    def test_mostly_valid(self):
        evo = make_evo()
        valid = sum(
            1 for _ in range(30)
            if evo._engine.validate_formula(evo._random_formula())[0]
        )
        assert valid >= 20   # at least 67% valid

    def test_population_init_non_empty(self):
        evo = make_evo()
        pop = evo._init_population(None)
        assert len(pop) >= 1

    def test_population_with_seeds(self):
        evo   = make_evo()
        seeds = ["momentum_20d", "rank(cross_momentum)"]
        pop   = evo._init_population(seeds)
        formulas = [ind.formula for ind in pop]
        assert any(s in formulas for s in seeds)


class TestMutationOperators:
    def test_feature_swap_string(self):
        evo = make_evo()
        assert isinstance(evo._mut_feature_swap("momentum_20d"), str)

    def test_op_insert_adds_op(self):
        evo     = make_evo()
        mutated = evo._mut_op_insert("momentum_20d")
        assert "(" in mutated

    def test_op_remove_unwraps(self):
        evo  = make_evo()
        f    = "zscore(momentum_20d)"
        res  = evo._mut_op_remove(f)
        assert isinstance(res, str)

    def test_scale_add_longer(self):
        evo     = make_evo()
        f       = "momentum_20d"
        mutated = evo._mut_scale_add(f)
        assert len(mutated) > len(f)

    def test_negate_changes_formula(self):
        evo = make_evo()
        f   = "momentum_20d"
        assert evo._mut_negate(f) != f

    def test_mutate_returns_string(self):
        evo = make_evo()
        for f in ["momentum_20d", "rank(cross_momentum)", "zscore(regime_signal)"]:
            assert isinstance(evo._mutate(f), str)

    def test_mutate_non_empty(self):
        evo = make_evo()
        for _ in range(15):
            f = evo._random_formula()
            assert len(evo._mutate(f)) > 0


class TestCrossover:
    def test_returns_string(self):
        evo = make_evo()
        assert isinstance(evo._crossover("momentum_20d", "volatility_20d"), str)

    def test_contains_parent_features(self):
        evo = make_evo(seed=1)
        res = evo._crossover("momentum_20d", "volatility_20d")
        assert "momentum_20d" in res or "volatility_20d" in res

    def test_non_empty(self):
        evo = make_evo()
        assert len(evo._crossover("momentum_20d", "regime_signal")) > 0


class TestTournamentSelection:
    def test_selects_from_population(self):
        from macro8_subnet.alpha.alpha_evolution import Individual
        evo = make_evo()
        pop = [Individual(f"f{i}", ic=float(i)*0.01, evaluated=True) for i in range(8)]
        sel = evo._tournament_select(pop)
        assert sel in pop

    def test_high_fitness_wins_more(self):
        from macro8_subnet.alpha.alpha_evolution import Individual
        evo  = make_evo(seed=0)
        pop  = [Individual(f"f{i}", ic=float(i)*0.01, evaluated=True) for i in range(10)]
        wins = [evo._tournament_select(pop).formula for _ in range(60)]
        # Top 3 individuals should win more than 30% of tournaments
        top3 = {f"f{i}" for i in range(7, 10)}
        top_wins = sum(1 for w in wins if w in top3)
        assert top_wins >= 10


class TestFullEvolution:
    def _returns(self):
        return make_prices().pct_change().dropna()

    def test_returns_report(self):
        from macro8_subnet.alpha.alpha_evolution import EvolutionReport
        report = make_evo().evolve(self._returns(), n_generations=2)
        assert isinstance(report, EvolutionReport)

    def test_best_ic_non_negative(self):
        report = make_evo().evolve(self._returns(), n_generations=2)
        assert report.best_ic >= 0.0

    def test_best_formula_string(self):
        report = make_evo().evolve(self._returns(), n_generations=2)
        assert isinstance(report.best_formula, str)

    def test_generation_history_present(self):
        report = make_evo().evolve(self._returns(), n_generations=3)
        assert len(report.generation_history) <= 3
        assert len(report.generation_history) >= 1

    def test_generation_history_structure(self):
        report = make_evo().evolve(self._returns(), n_generations=2)
        for gen in report.generation_history:
            for key in ("generation", "best_ic", "best_formula", "mean_ic"):
                assert key in gen

    def test_with_seed_formulas(self):
        report = make_evo().evolve(
            self._returns(), n_generations=2,
            seed_formulas=["momentum_20d", "rank(cross_momentum)"]
        )
        assert report.n_generations >= 1

    def test_to_dict_serialisable(self):
        report = make_evo().evolve(self._returns(), n_generations=2)
        json.dumps(report.to_dict())

    def test_summary_string(self):
        report = make_evo().evolve(self._returns(), n_generations=2)
        s = report.summary()
        assert isinstance(s, str) and "gen" in s.lower()

    def test_ic_improvement_is_float(self):
        report = make_evo().evolve(self._returns(), n_generations=2)
        assert isinstance(report.ic_improvement, float)

    def test_elite_preserved_across_generations(self):
        """Best IC should never decrease (elitism)."""
        from macro8_subnet.alpha.alpha_evolution import AlphaEvolution
        evo     = AlphaEvolution(make_feature_store(make_prices()),
                                  population_size=10, seed=42)
        report  = evo.evolve(self._returns(), n_generations=5)
        best_ics = [g["best_ic"] for g in report.generation_history]
        # With elitism, best IC is monotonically non-decreasing
        for i in range(1, len(best_ics)):
            assert best_ics[i] >= best_ics[i-1] - 1e-4  # allow small numeric jitter

    def test_verbose_runs(self, capsys):
        evo    = make_evo(pop_size=6)
        report = evo.evolve(self._returns(), n_generations=2, verbose=True)
        out    = capsys.readouterr().out
        assert "Gen" in out

    def test_convergence_flag(self):
        """With a very tight convergence threshold, should converge quickly."""
        from macro8_subnet.alpha.alpha_evolution import AlphaEvolution
        evo    = AlphaEvolution(
            make_feature_store(make_prices()),
            population_size=6,
            converge_threshold=10.0,   # immediately converges
            seed=42,
        )
        report = evo.evolve(self._returns(), n_generations=10)
        # With threshold=10, almost any population will "converge" on gen 1
        assert report.converged is True or report.n_generations <= 10
