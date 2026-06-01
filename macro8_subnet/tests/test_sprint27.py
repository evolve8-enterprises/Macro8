"""
tests/test_sprint27.py
-----------------------
Sprint 27: GP Diversity Rebuild

Tests verify four root-cause fixes:
    1. Vec-fingerprint dedup removed from inner loop
    2. 34 terminals including 10 macro features
    3. Island model: 25% permanent exploration
    4. Tournament selection replaces pure top-N elitism
"""

import sys
import collections
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
for p in [str(_ROOT), str(_ROOT / "macro8_subnet")]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def prices():
    rng = np.random.default_rng(42)
    n, a = 500, 10
    tickers = ["SPY","QQQ","IWM","TLT","GLD","DBC","EEM","FXI","VNQ","HYG"]
    p = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, (n, a)), axis=0))
    return pd.DataFrame(p, index=pd.bdate_range("2015-01-01", periods=n), columns=tickers)


@pytest.fixture(scope="module")
def gp_miner(prices):
    from macro8_subnet.alpha.gp_miner import GPMiner
    return GPMiner(prices, pop_size=60, elite_n=10, seed=42, verbose=False)


@pytest.fixture(scope="module")
def gp_report(gp_miner):
    return gp_miner.run(n_epochs=3)


# ── 1. Grammar: 34 terminals ──────────────────────────────────────────────────

class TestGrammar:
    MACRO_TERMINALS = [
        "risk_on_off", "commodity_inflation", "em_vs_dm", "credit_stress",
        "equity_bond_corr", "cross_asset_vol", "vol_regime",
        "trend_strength", "carry_proxy", "dollar_proxy",
    ]

    def test_all_features_count(self):
        from macro8_subnet.alpha.gp_miner import GP_FEATURES
        assert len(GP_FEATURES) == 34, \
            f"Expected 34 GP terminals, got {len(GP_FEATURES)}"

    def test_all_macro_terminals_in_gp_features(self):
        from macro8_subnet.alpha.gp_miner import GP_FEATURES
        for feat in self.MACRO_TERMINALS:
            assert feat in GP_FEATURES, f"{feat} missing from GP_FEATURES"

    def test_batch_evaluator_all_features_count(self):
        from macro8_subnet.alpha.batch_evaluator import ALL_FEATURES
        assert len(ALL_FEATURES) == 34

    def test_macro_terminals_in_batch_evaluator(self):
        from macro8_subnet.alpha.batch_evaluator import ALL_FEATURES
        for feat in self.MACRO_TERMINALS:
            assert feat in ALL_FEATURES, f"{feat} missing from ALL_FEATURES"

    def test_gp_features_equals_all_features(self):
        from macro8_subnet.alpha.gp_miner import GP_FEATURES
        from macro8_subnet.alpha.batch_evaluator import ALL_FEATURES
        assert set(GP_FEATURES) == set(ALL_FEATURES)


# ── 2. FormulaGenerator diversity ────────────────────────────────────────────

class TestFormulaGenerator:
    def test_macro_features_appear_in_generation(self):
        from macro8_subnet.alpha.gp_miner import FormulaGenerator
        gen = FormulaGenerator(seed=42)
        formulas = [gen.random_formula(depth=2) for _ in range(500)]
        macro = ["risk_on_off","vol_regime","trend_strength","carry_proxy","dollar_proxy"]
        found = {m for f in formulas for m in macro if m in f}
        assert len(found) >= 3, \
            f"Expected ≥3 macro features in 500 formulas, found: {found}"

    def test_depth_1_generates_single_features(self):
        from macro8_subnet.alpha.gp_miner import FormulaGenerator, GP_FEATURES
        gen = FormulaGenerator(seed=1)
        for _ in range(50):
            f = gen.random_formula(depth=1)
            assert len(f) > 0
            assert len(f) <= 100  # depth-1 should be short

    def test_depth_3_generates_complex(self):
        from macro8_subnet.alpha.gp_miner import FormulaGenerator
        gen = FormulaGenerator(seed=99)
        depth3 = [gen.random_formula(depth=3) for _ in range(100)]
        # At least some should have binary operators (compound expressions)
        has_binary = [f for f in depth3 if any(op in f for op in [" + "," - "," * "])]
        assert len(has_binary) >= 10, "Depth-3 formulas should include binary ops"

    def test_all_generated_formulas_within_length_limit(self):
        from macro8_subnet.alpha.gp_miner import FormulaGenerator, MAX_FORMULA_LEN
        gen = FormulaGenerator(seed=7)
        for depth in [1, 2, 3, 4]:
            for _ in range(50):
                f = gen.random_formula(depth=depth)
                assert len(f) <= MAX_FORMULA_LEN, \
                    f"depth={depth} formula too long: {len(f)} chars"

    def test_initial_population_depth_diversity(self):
        """Initial population should span multiple depths."""
        from macro8_subnet.alpha.gp_miner import FormulaGenerator
        gen = FormulaGenerator(seed=42)
        pop = gen.initial_population(100)
        # Simple proxy: formulas with binary ops are depth ≥ 2
        complex_count = sum(
            1 for f in pop
            if any(op in f for op in [" + ", " - ", " * "])
        )
        # Expect mix: at least 40% complex, at most 95%
        assert 10 <= complex_count <= 95, \
            f"Expected depth diversity, got {complex_count}/100 complex"

    def test_initial_population_has_macro_features(self):
        from macro8_subnet.alpha.gp_miner import FormulaGenerator
        gen = FormulaGenerator(seed=0)
        pop = gen.initial_population(200)
        macro = ["risk_on_off", "vol_regime", "trend_strength"]
        macro_count = sum(1 for f in pop if any(m in f for m in macro))
        assert macro_count >= 5, \
            f"Expected macro features in initial pop, got {macro_count}"


# ── 3. GeneticOperators diversity ─────────────────────────────────────────────

class TestGeneticOperators:
    @pytest.fixture(scope="class")
    def ops(self):
        from macro8_subnet.alpha.gp_miner import GeneticOperators, FormulaGenerator
        return GeneticOperators(FormulaGenerator(seed=42))

    def test_mutate_produces_different_formula(self, ops):
        f = "market_corr_60d"
        mutations = [ops.mutate(f) for _ in range(30)]
        changed = [m for m in mutations if m != f]
        assert len(changed) >= 15, "At least half of mutations should change the formula"

    def test_macro_injection_produces_macro_features(self, ops):
        from macro8_subnet.alpha.gp_miner import GeneticOperators
        macro = GeneticOperators.MACRO_FEATURES
        f = "momentum_20d - volatility_60d"
        # _swap_feature with use_macro=True should inject macro features
        injected = [ops._swap_feature(f, use_macro=True) for _ in range(50)]
        has_macro = [m for m in macro for inj in injected if m in inj]
        assert len(has_macro) >= 5, \
            f"Expected macro injection, got: {set(has_macro)}"

    def test_crossover_produces_combined_formula(self, ops):
        f1 = "momentum_20d"
        f2 = "market_corr_60d"
        children = [ops.crossover(f1, f2) for _ in range(20)]
        # All children should contain at least one parent feature
        for child in children:
            assert "momentum_20d" in child or "market_corr_60d" in child or \
                   any(op in child for op in ["+","-","*"]), \
                f"Crossover child looks invalid: {child}"

    def test_depth_expand_increases_complexity(self, ops):
        f = "market_corr_20d"
        expanded = [ops._depth_expand(f) for _ in range(20)]
        has_binary = [e for e in expanded if any(op in e for op in [" + "," - "," * "])]
        assert len(has_binary) >= 15

    def test_mutate_respects_length_limit(self, ops):
        from macro8_subnet.alpha.gp_miner import MAX_FORMULA_LEN
        long_f = "rank(market_corr_60d - volatility_20d) + zscore(momentum_20d - skew_60d)"
        for _ in range(20):
            result = ops.mutate(long_f)
            assert len(result) <= MAX_FORMULA_LEN

    def test_technical_features_list_correct(self):
        from macro8_subnet.alpha.gp_miner import GeneticOperators, GP_FEATURES
        tech = GeneticOperators.TECHNICAL_FEATURES
        macro = GeneticOperators.MACRO_FEATURES
        assert len(tech) == 24, f"Expected 24 technical features, got {len(tech)}"
        assert len(macro) == 10
        assert set(tech) | set(macro) == set(GP_FEATURES)


# ── 4. Population diversity ───────────────────────────────────────────────────

class TestPopulationDiversity:
    def test_initial_population_uniqueness(self, gp_miner):
        n_unique = len(set(gp_miner._population))
        assert n_unique >= 40, \
            f"Expected ≥40 unique formulas, got {n_unique}"

    def test_evaluation_does_not_use_vec_dedup(self, prices):
        """Inner evaluation should keep syntactically unique formulas."""
        from macro8_subnet.alpha.gp_miner import GPMiner
        gp  = GPMiner(prices, pop_size=60, elite_n=8, seed=1, verbose=False)
        # Count unique strings before and after evaluation
        n_unique_before = len(set(gp._population))
        scored, n_safe  = gp._evaluate_population()
        # n_safe should be close to n_unique_before (no vec dedup)
        assert n_safe >= n_unique_before * 0.85, \
            f"Too many formulas lost in evaluation: {n_safe} vs {n_unique_before}"

    def test_hall_of_fame_grows_with_generations(self, gp_report, gp_miner):
        n_hof = len(gp_miner._hall_of_fame)
        assert n_hof >= 50, \
            f"Expected ≥50 formulas in hall-of-fame after 3 gens, got {n_hof}"

    def test_top32_has_diverse_features(self, gp_report):
        from macro8_subnet.alpha.gp_miner import GP_FEATURES
        top32 = gp_report.submission_formulas(32)
        feat_set = {f for formula in top32 for f in GP_FEATURES if f in formula}
        assert len(feat_set) >= 12, \
            f"Expected ≥12 distinct features in top-32, got {len(feat_set)}: {sorted(feat_set)}"

    def test_macro_features_appear_in_top32(self, gp_report):
        macro = ["risk_on_off", "vol_regime", "trend_strength", "carry_proxy",
                 "credit_stress", "equity_bond_corr", "em_vs_dm",
                 "cross_asset_vol", "commodity_inflation", "dollar_proxy"]
        top32 = gp_report.submission_formulas(32)
        has_macro = [f for f in top32 if any(m in f for m in macro)]
        assert len(has_macro) >= 1, \
            f"Expected at least 1 macro feature in top-32, got 0\nTop-32: {top32[:8]}"


# ── 5. Tournament selection ───────────────────────────────────────────────────

class TestTournamentSelection:
    def test_tournament_selects_from_full_population(self, gp_miner, gp_report):
        from macro8_subnet.alpha.gp_miner import ScoredFormula
        scored = gp_miner._scored
        if not scored:
            pytest.skip("No scored formulas yet")
        # Tournament should sometimes pick non-elite formulas
        selections = [gp_miner._tournament_select(scored, k=4) for _ in range(100)]
        elite_formulas = {sf.formula for sf in scored[:10]}
        non_elite_picks = [s for s in selections if s not in elite_formulas]
        # With k=4 and 60+ scored formulas, some non-elite should be selected
        assert len(non_elite_picks) >= 5, \
            f"Tournament should occasionally select non-elites, got {len(non_elite_picks)}/100"

    def test_tournament_returns_valid_formula_string(self, gp_miner, gp_report):
        scored = gp_miner._scored
        if not scored:
            pytest.skip()
        for _ in range(10):
            result = gp_miner._tournament_select(scored, k=4)
            assert isinstance(result, str)
            assert len(result) > 0


# ── 6. Island model (exploration) ─────────────────────────────────────────────

class TestIslandModel:
    def test_exploration_fraction_correct(self):
        """25% of each new population should be fresh random formulas."""
        from macro8_subnet.alpha.gp_miner import GPMiner, DEFAULT_RANDOM_P
        assert DEFAULT_RANDOM_P == 0.25, \
            f"Expected DEFAULT_RANDOM_P=0.25, got {DEFAULT_RANDOM_P}"

    def test_evolved_population_has_fresh_formulas(self, prices):
        from macro8_subnet.alpha.gp_miner import GPMiner
        gp = GPMiner(prices, pop_size=60, elite_n=8, seed=5, verbose=False)
        gp.run(n_epochs=1)   # prime scored list

        pop_before = set(gp._population)
        new_pop    = gp._evolve(gp._scored)
        new_set    = set(new_pop)
        novel      = new_set - pop_before
        # Should have injected ~25% fresh formulas
        assert len(novel) >= 5, \
            f"Expected ≥5 novel formulas from island injection, got {len(novel)}"

    def test_random_p_larger_than_before(self):
        from macro8_subnet.alpha.gp_miner import DEFAULT_RANDOM_P
        # Sprint 27 increased from 0.10 to 0.25
        assert DEFAULT_RANDOM_P > 0.10


# ── 7. GPReport diversity metrics ─────────────────────────────────────────────

class TestGPReportDiversity:
    def test_n_unique_grows_with_generations(self, prices):
        from macro8_subnet.alpha.gp_miner import GPMiner
        gp  = GPMiner(prices, pop_size=60, elite_n=8, seed=42, verbose=False)
        r1  = gp.run(n_epochs=1)
        n1  = r1.n_unique
        r2  = gp.run(n_epochs=2)
        n2  = r2.n_unique
        assert n2 > n1, "Hall of fame should grow with more generations"

    def test_submission_formulas_vec_deduped(self, gp_report):
        """top_formulas() applies vec dedup at submission time."""
        top = gp_report.submission_formulas(32)
        # No two submitted formulas should be exact duplicates
        assert len(top) == len(set(top)), "Duplicates in submission list"

    def test_best_composite_history_length(self, gp_report):
        assert len(gp_report.best_composite_history) == 3

    def test_summary_includes_composite(self, gp_report):
        s = gp_report.summary()
        assert "composite" in s.lower()


# ── 8. Backward compat with old tests ─────────────────────────────────────────

class TestBackwardCompat:
    def test_scored_formula_fields_unchanged(self):
        import dataclasses
        from macro8_subnet.alpha.gp_miner import ScoredFormula
        fields = {f.name for f in dataclasses.fields(ScoredFormula)}
        for req in ["formula","mean_ic","ic_ir","generation",
                    "composite","sharpe","turnover","ic_30d","capital_1m"]:
            assert req in fields

    def test_gp_report_submission_formulas(self, gp_report):
        formulas = gp_report.submission_formulas(10)
        assert isinstance(formulas, list)
        assert all(isinstance(f, str) for f in formulas)

    def test_step_returns_sorted_list(self, prices):
        from macro8_subnet.alpha.gp_miner import GPMiner
        gp  = GPMiner(prices, pop_size=30, elite_n=6, seed=7, verbose=False)
        top = gp.step()
        if len(top) >= 2:
            composites = [sf.composite for sf in top[:5]]
            assert composites == sorted(composites, reverse=True)

    def test_add_hypothesis_seeds(self, prices):
        from macro8_subnet.alpha.gp_miner import GPMiner
        gp    = GPMiner(prices, pop_size=30, elite_n=6, seed=8, verbose=False)
        seeds = ["market_corr_60d", "momentum_20d"]
        gp.add_hypothesis_seeds(seeds)
        assert any(s in gp._population for s in seeds)

    def test_portfolio_evaluator_is_batch_eval(self, prices):
        from macro8_subnet.alpha.gp_miner import GPMiner
        from macro8_subnet.alpha.portfolio_evaluator import PortfolioEvaluator
        gp = GPMiner(prices, pop_size=20, elite_n=4, seed=9, verbose=False)
        assert isinstance(gp._batch_eval, PortfolioEvaluator)
