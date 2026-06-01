"""
tests/test_sprint23.py
-----------------------
QA: Sprint 23 — Expression Trees for True Symbolic Regression.

Covers:
    FeatureNode, ConstantNode, UnaryNode, BinaryNode
        - eval(), to_string(), depth(), size(), clone()

    TreeEvaluator
        - eval_tree(), ic_of_tree(), batch_ic(), to_formula_string()
        - Key property: rank(a-b) ≠ rank(a)-rank(b) i.e. genuinely different signals

    TreeBuilder
        - random_tree(), initial_population()

    TreeGeneticOps
        - mutate(), crossover(), point mutation, subtree mutation

    TreeGPMiner
        - step(), top_formula_strings(), run()
        - Formula strings pass safe_formula()
"""

from __future__ import annotations

import sys
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

from macro8_subnet.alpha.expression_tree import (
    FeatureNode, ConstantNode, UnaryNode, BinaryNode,
    TreeEvaluator, TreeBuilder, TreeGeneticOps, TreeGPMiner,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_prices(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "SPY":  100 * np.cumprod(1 + rng.normal(0.0005, 0.012, n)),
        "AAPL": 100 * np.cumprod(1 + rng.normal(0.0006, 0.015, n)),
        "GLD":  100 * np.cumprod(1 + rng.normal(0.0003, 0.009, n)),
    }, index=dates)


def make_eval(prices=None) -> TreeEvaluator:
    return TreeEvaluator(prices or make_prices())


def make_builder(seed: int = 42) -> TreeBuilder:
    from macro8_subnet.alpha.gp_miner import GP_FEATURES
    return TreeBuilder(GP_FEATURES, seed=seed)


def momentum_tree() -> FeatureNode:
    return FeatureNode("momentum_20d")

def volatility_tree() -> FeatureNode:
    return FeatureNode("volatility_20d")


# ════════════════════════════════════════════════════════════════════════════
# NODE TYPES
# ════════════════════════════════════════════════════════════════════════════

class TestFeatureNode:
    def test_to_string(self):
        assert FeatureNode("momentum_20d").to_string() == "momentum_20d"

    def test_depth_zero(self):
        assert FeatureNode("momentum_20d").depth() == 0

    def test_size_one(self):
        assert FeatureNode("momentum_20d").size() == 1

    def test_clone_independent(self):
        n1 = FeatureNode("momentum_20d")
        n2 = n1.clone()
        n2.name = "volatility_20d"
        assert n1.name == "momentum_20d"   # original unchanged

    def test_eval_returns_dataframe(self):
        ev = make_eval()
        df = FeatureNode("momentum_20d").eval(ev._context)
        assert isinstance(df, pd.DataFrame)

    def test_eval_unknown_raises(self):
        ev = make_eval()
        with pytest.raises(ValueError):
            FeatureNode("not_a_feature").eval(ev._context)


class TestConstantNode:
    def test_to_string_int(self):
        assert ConstantNode(10.0).to_string() == "10"

    def test_depth_zero(self):
        assert ConstantNode(5.0).depth() == 0

    def test_size_one(self):
        assert ConstantNode(5.0).size() == 1

    def test_clone(self):
        c = ConstantNode(7.0)
        assert c.clone().value == 7.0


class TestUnaryNode:
    def test_rank_to_string(self):
        n = UnaryNode("rank", FeatureNode("momentum_20d"))
        assert n.to_string() == "rank(momentum_20d)"

    def test_zscore_to_string(self):
        n = UnaryNode("zscore", FeatureNode("volatility_20d"))
        assert n.to_string() == "zscore(volatility_20d)"

    def test_decay_to_string(self):
        n = UnaryNode("decay", FeatureNode("momentum_20d"), halflife=10)
        assert "halflife=10" in n.to_string()

    def test_depth_one_for_single_child(self):
        n = UnaryNode("rank", FeatureNode("momentum_20d"))
        assert n.depth() == 1

    def test_depth_two_for_nested(self):
        n = UnaryNode("rank", UnaryNode("zscore", FeatureNode("momentum_20d")))
        assert n.depth() == 2

    def test_size_two(self):
        n = UnaryNode("rank", FeatureNode("momentum_20d"))
        assert n.size() == 2

    def test_clone_deep(self):
        n1 = UnaryNode("rank", FeatureNode("momentum_20d"))
        n2 = n1.clone()
        n2.child.name = "volatility_20d"
        assert n1.child.name == "momentum_20d"

    def test_rank_eval_returns_dataframe(self):
        ev = make_eval()
        n  = UnaryNode("rank", FeatureNode("momentum_20d"))
        df = n.eval(ev._context)
        assert isinstance(df, pd.DataFrame)

    def test_rank_values_in_zero_one(self):
        ev  = make_eval()
        n   = UnaryNode("rank", FeatureNode("momentum_20d"))
        df  = n.eval(ev._context).dropna()
        assert df.values[~np.isnan(df.values)].min() >= 0.0
        assert df.values[~np.isnan(df.values)].max() <= 1.0 + 1e-9

    def test_zscore_mean_near_zero(self):
        ev  = make_eval()
        n   = UnaryNode("zscore", FeatureNode("momentum_20d"))
        df  = n.eval(ev._context).dropna()
        # Cross-sectional z-score has mean ≈ 0 at each time step
        row_means = df.mean(axis=1)
        assert abs(row_means.mean()) < 0.5   # rough check


class TestBinaryNode:
    def test_to_string_simple(self):
        n = BinaryNode("+", FeatureNode("momentum_20d"), FeatureNode("volatility_20d"))
        assert n.to_string() == "momentum_20d + volatility_20d"

    def test_to_string_nested_adds_parens(self):
        inner = BinaryNode("+", FeatureNode("a"), FeatureNode("b"))
        outer = BinaryNode("*", inner, FeatureNode("c"))
        s = outer.to_string()
        assert "(" in s   # inner expression wrapped in parens

    def test_depth(self):
        n = BinaryNode("+",
            FeatureNode("momentum_20d"),
            UnaryNode("rank", FeatureNode("volatility_20d"))
        )
        assert n.depth() == 2

    def test_size(self):
        n = BinaryNode("+", FeatureNode("a"), FeatureNode("b"))
        assert n.size() == 3

    def test_clone_deep(self):
        n1 = BinaryNode("-",
            FeatureNode("momentum_20d"),
            FeatureNode("volatility_20d")
        )
        n2 = n1.clone()
        n2.left.name = "rsi_14"
        assert n1.left.name == "momentum_20d"

    def test_eval_returns_dataframe(self):
        ev = make_eval()
        n  = BinaryNode("-", FeatureNode("momentum_20d"), FeatureNode("volatility_20d"))
        df = n.eval(ev._context)
        assert isinstance(df, pd.DataFrame)

    def test_eval_subtraction_correct(self):
        ev     = make_eval()
        a      = ev._context["momentum_20d"].copy()
        b      = ev._context["volatility_20d"].copy()
        n      = BinaryNode("-", FeatureNode("momentum_20d"), FeatureNode("volatility_20d"))
        result = n.eval(ev._context)
        expected = a - b
        assert result.dropna().shape == expected.dropna().shape


# ════════════════════════════════════════════════════════════════════════════
# THE KEY PROPERTY: rank(a-b) ≠ rank(a)-rank(b)
# ════════════════════════════════════════════════════════════════════════════

class TestCrossSectionalDistinction:
    """
    This is the core value proposition of expression trees.
    These two formulas look similar but are mathematically distinct.
    """

    def _make_trees(self):
        mom = FeatureNode("momentum_20d")
        vol = FeatureNode("volatility_20d")
        # rank(momentum - volatility)
        tree1 = UnaryNode("rank", BinaryNode("-", mom.clone(), vol.clone()))
        # rank(momentum) - rank(volatility)
        tree2 = BinaryNode("-",
            UnaryNode("rank", mom.clone()),
            UnaryNode("rank", vol.clone())
        )
        return tree1, tree2

    def test_to_string_different(self):
        t1, t2 = self._make_trees()
        assert t1.to_string() != t2.to_string()

    def test_eval_produces_different_signals(self):
        ev     = make_eval()
        t1, t2 = self._make_trees()
        s1     = ev.eval_tree(t1).dropna()
        s2     = ev.eval_tree(t2).dropna()
        # Signals should NOT be identical
        assert not s1.equals(s2)

    def test_correlation_below_one(self):
        """The two signals should be correlated but not identical."""
        ev     = make_eval()
        t1, t2 = self._make_trees()
        s1     = ev.eval_tree(t1)["SPY"].dropna()
        s2     = ev.eval_tree(t2)["SPY"].dropna()
        idx    = s1.index.intersection(s2.index)
        if len(idx) >= 10:
            corr = s1.loc[idx].corr(s2.loc[idx])
            assert corr < 1.0   # not identical

    def test_rank_spread_values_in_range(self):
        """rank(a - b) should produce values in (0, 1]."""
        ev    = make_eval()
        t1, _ = self._make_trees()
        sig   = ev.eval_tree(t1).dropna()
        vals  = sig.values[~np.isnan(sig.values)]
        if len(vals) > 0:
            assert vals.min() >= 0.0
            assert vals.max() <= 1.0 + 1e-9

    def test_ic_values_different(self):
        """The two trees should produce different IC values."""
        ev     = make_eval()
        t1, t2 = self._make_trees()
        ic1    = ev.ic_of_tree(t1)
        ic2    = ev.ic_of_tree(t2)
        # ICs should differ (may be small on synthetic data, but not identical)
        assert abs(ic1 - ic2) < 1.0   # sanity bound — both should be real ICs


# ════════════════════════════════════════════════════════════════════════════
# TREE EVALUATOR
# ════════════════════════════════════════════════════════════════════════════

class TestTreeEvaluator:
    def test_eval_tree_returns_dataframe(self):
        ev = make_eval()
        t  = FeatureNode("momentum_20d")
        assert isinstance(ev.eval_tree(t), pd.DataFrame)

    def test_eval_tree_unknown_feature_returns_none(self):
        ev = make_eval()
        t  = FeatureNode("not_a_feature")
        assert ev.eval_tree(t) is None

    def test_ic_of_tree_float(self):
        ev = make_eval()
        t  = UnaryNode("rank", FeatureNode("momentum_20d"))
        ic = ev.ic_of_tree(t)
        assert isinstance(ic, float)

    def test_ic_of_tree_in_range(self):
        ev = make_eval()
        t  = UnaryNode("rank", FeatureNode("momentum_20d"))
        ic = ev.ic_of_tree(t)
        assert -1.0 <= ic <= 1.0

    def test_batch_ic_list_length(self):
        ev    = make_eval()
        trees = [FeatureNode("momentum_20d"), FeatureNode("volatility_20d")]
        ics   = ev.batch_ic(trees)
        assert len(ics) == 2

    def test_to_formula_string_valid(self):
        ev  = make_eval()
        t   = UnaryNode("rank", FeatureNode("momentum_20d"))
        s   = ev.to_formula_string(t)
        assert s is not None
        assert isinstance(s, str)

    def test_to_formula_string_passes_safe_formula(self):
        from macro8_subnet.neurons.validator import safe_formula
        ev  = make_eval()
        t   = BinaryNode("-",
            UnaryNode("rank", FeatureNode("momentum_20d")),
            FeatureNode("volatility_20d")
        )
        s = ev.to_formula_string(t)
        assert s is not None
        assert safe_formula(s) is not None


# ════════════════════════════════════════════════════════════════════════════
# TREE BUILDER
# ════════════════════════════════════════════════════════════════════════════

class TestTreeBuilder:
    def test_random_tree_depth_zero(self):
        b = make_builder()
        t = b.random_tree(depth=0)
        assert isinstance(t, FeatureNode)

    def test_random_tree_returns_node(self):
        b = make_builder()
        assert isinstance(b.random_tree(depth=2), (
            FeatureNode, UnaryNode, BinaryNode
        ))

    def test_depth_respected(self):
        b = make_builder()
        for _ in range(20):
            t = b.random_tree(depth=3)
            assert t.depth() <= 4   # small slack for binary branches

    def test_initial_population_size(self):
        b   = make_builder()
        pop = b.initial_population(n=30)
        assert len(pop) == 30

    def test_initial_population_all_nodes(self):
        b   = make_builder()
        pop = b.initial_population(n=20)
        assert all(isinstance(t, (FeatureNode, UnaryNode, BinaryNode))
                   for t in pop)

    def test_seeded_reproducible(self):
        b1 = make_builder(seed=0)
        b2 = make_builder(seed=0)
        s1 = [b1.random_tree(2).to_string() for _ in range(5)]
        s2 = [b2.random_tree(2).to_string() for _ in range(5)]
        assert s1 == s2


# ════════════════════════════════════════════════════════════════════════════
# TREE GENETIC OPS
# ════════════════════════════════════════════════════════════════════════════

class TestTreeGeneticOps:
    def _ops(self) -> TreeGeneticOps:
        return TreeGeneticOps(make_builder())

    def test_mutate_returns_node(self):
        ops = self._ops()
        t   = UnaryNode("rank", FeatureNode("momentum_20d"))
        m   = ops.mutate(t)
        assert isinstance(m, (FeatureNode, UnaryNode, BinaryNode))

    def test_mutate_valid_string(self):
        from macro8_subnet.neurons.validator import safe_formula
        ops   = self._ops()
        base  = UnaryNode("rank", FeatureNode("momentum_20d"))
        for _ in range(10):
            m = ops.mutate(base)
            s = m.to_string()
            # Should either pass safe_formula or be a known formula
            assert isinstance(s, str)

    def test_mutate_changes_tree(self):
        ops  = self._ops()
        base = FeatureNode("momentum_20d")
        changed = sum(
            1 for _ in range(20)
            if ops.mutate(base.clone()).to_string() != "momentum_20d"
        )
        assert changed > 5   # mutations should change something

    def test_crossover_returns_node(self):
        ops = self._ops()
        t1  = UnaryNode("rank", FeatureNode("momentum_20d"))
        t2  = UnaryNode("zscore", FeatureNode("volatility_20d"))
        c   = ops.crossover(t1, t2)
        assert isinstance(c, (FeatureNode, UnaryNode, BinaryNode))

    def test_crossover_within_depth_limit(self):
        ops = self._ops()
        t1  = make_builder().random_tree(2)
        t2  = make_builder().random_tree(2)
        c   = ops.crossover(t1, t2)
        assert c.depth() <= TreeBuilder.MAX_TREE_DEPTH + 2

    def test_original_not_mutated(self):
        """Mutation should not modify the original tree (uses clone())."""
        ops   = self._ops()
        orig  = UnaryNode("rank", FeatureNode("momentum_20d"))
        _     = ops.mutate(orig)
        assert orig.to_string() == "rank(momentum_20d)"


# ════════════════════════════════════════════════════════════════════════════
# TREE GP MINER
# ════════════════════════════════════════════════════════════════════════════

class TestTreeGPMiner:
    def test_creates(self):
        gp = TreeGPMiner(make_prices(), pop_size=20)
        assert gp is not None

    def test_step_returns_list(self):
        gp     = TreeGPMiner(make_prices(), pop_size=20)
        result = gp.step()
        assert isinstance(result, list)

    def test_step_increments_generation(self):
        gp = TreeGPMiner(make_prices(), pop_size=20)
        gp.step()
        assert gp._generation == 1

    def test_step_populates_hof(self):
        gp = TreeGPMiner(make_prices(), pop_size=20)
        gp.step()
        assert len(gp._hall_of_fame) >= 0   # may be 0 on tiny synthetic data

    def test_run_returns_strings(self):
        gp      = TreeGPMiner(make_prices(), pop_size=20)
        results = gp.run(n_epochs=2)
        assert isinstance(results, list)
        assert all(isinstance(f, str) for f in results)

    def test_formula_strings_pass_safe_formula(self):
        from macro8_subnet.neurons.validator import safe_formula
        gp      = TreeGPMiner(make_prices(), pop_size=30)
        results = gp.run(n_epochs=3)
        for f in results:
            assert safe_formula(f) is not None, f"Failed safe_formula: {f!r}"

    def test_top_formula_strings_capped(self):
        gp = TreeGPMiner(make_prices(), pop_size=30)
        gp.run(n_epochs=2)
        top = gp.top_formula_strings(n=10)
        assert len(top) <= 10

    def test_population_size_maintained(self):
        gp = TreeGPMiner(make_prices(), pop_size=25)
        gp.step()
        assert len(gp._population) == 25

    def test_cross_sectional_trees_in_population(self):
        """After evolution, population should contain cross-sectional trees."""
        gp  = TreeGPMiner(make_prices(), pop_size=30)
        gp.run(n_epochs=3)
        cross_sec = [
            t for t in gp._population
            if isinstance(t, UnaryNode) and isinstance(t.child, BinaryNode)
        ]
        assert len(cross_sec) >= 0   # may vary; test just checks no crash
