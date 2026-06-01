"""
tests/test_sprint14.py
-----------------------
QA: Complete self-contained tests for Sprint 14 — MPFG modules.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SUITE_DIR  = Path(__file__).resolve().parent
SUBNET_DIR = SUITE_DIR.parent
PROJECT    = SUBNET_DIR.parent
for p in [str(SUBNET_DIR), str(PROJECT)]:
    if p not in sys.path:
        sys.path.insert(0, p)


def make_prices(n: int = 80, seed: int = 42) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "SPY":  100 * np.cumprod(1 + rng.normal(0.0005, 0.012, n)),
        "AAPL": 100 * np.cumprod(1 + rng.normal(0.0006, 0.015, n)),
        "GLD":  100 * np.cumprod(1 + rng.normal(0.0003, 0.009, n)),
    }, index=dates)


def make_feature_store(prices=None):
    from macro8_subnet.alpha.feature_store import FeatureStore
    return FeatureStore(prices if prices is not None else make_prices())


def make_hyp_lib():
    from macro8_subnet.alpha.hypothesis_engine import (
        HypothesisLibrary, HypothesisCategory, BayesianUpdater
    )
    lib = HypothesisLibrary()
    rec = lib.add("Momentum works", HypothesisCategory.MOMENTUM, 0, 1)
    for _ in range(5):
        BayesianUpdater().update(rec, 0.04)
    return lib


# ════════════════════════════════════════════════════════════════════════════
# BATCH GENERATOR
# ════════════════════════════════════════════════════════════════════════════

from macro8_subnet.alpha.batch_generator import (
    BatchFormulaGenerator, GenerationStrategy, BatchGenerationReport,
)


class TestBatchFormulaGenerator:
    def _gen(self, with_hyp=False):
        hyp = make_hyp_lib() if with_hyp else None
        return BatchFormulaGenerator(hypothesis_library=hyp, seed=42)

    def test_generates_formulas(self):
        assert len(self._gen().generate(20)) > 0

    def test_generates_at_most_n(self):
        assert len(self._gen().generate(10)) <= 10

    def test_all_strings(self):
        for f in self._gen().generate(20):
            assert isinstance(f, str) and len(f) > 0

    def test_no_duplicates_within_batch(self):
        f = self._gen().generate(50)
        assert len(f) == len(set(f))

    def test_no_duplicates_across_calls(self):
        g  = self._gen()
        f1 = set(g.generate(30))
        f2 = set(g.generate(30))
        assert f1.isdisjoint(f2)

    def test_reset_allows_reuse(self):
        g  = self._gen()
        f1 = set(g.generate(30))
        g.reset_seen()
        f2 = set(g.generate(30))
        assert not f1.isdisjoint(f2)

    def test_n_seen_grows(self):
        g = self._gen()
        g.generate(20)
        assert g.n_seen > 0

    def test_template_strategy(self):
        assert len(self._gen().generate(50, GenerationStrategy.TEMPLATE_EXPANSION)) > 0

    def test_random_strategy(self):
        assert len(self._gen().generate(20, GenerationStrategy.RANDOM_EVOLUTION)) > 0

    def test_hypothesis_guided(self):
        assert len(self._gen(with_hyp=True).generate(20, GenerationStrategy.HYPOTHESIS_GUIDED)) > 0

    def test_mixed_strategy(self):
        assert len(self._gen().generate(50, GenerationStrategy.MIXED)) > 0

    def test_generate_with_report(self):
        g        = self._gen()
        f, r     = g.generate_with_report(20)
        assert isinstance(r, BatchGenerationReport)
        assert r.n_generated == len(f)

    def test_report_serialisable(self):
        _, r = self._gen().generate_with_report(10)
        json.dumps(r.to_dict())

    def test_template_has_rank_differences(self):
        f = self._gen().generate(300, GenerationStrategy.TEMPLATE_EXPANSION)
        assert any("rank" in formula and "-" in formula for formula in f)

    def test_formula_length_bounded(self):
        for formula in self._gen().generate(50):
            assert len(formula) <= 200


# ════════════════════════════════════════════════════════════════════════════
# VECTORIZED EVALUATOR
# ════════════════════════════════════════════════════════════════════════════

from macro8_subnet.alpha.vectorized_evaluator import (
    VectorizedEvaluator, VectorizedICResult, is_simple_formula, classify_formula,
)


class TestIsSimpleFormula:
    def test_bare_feature(self):           assert is_simple_formula("momentum_20d")
    def test_single_unary_op(self):        assert is_simple_formula("rank(momentum_20d)")
    def test_binary_is_complex(self):      assert not is_simple_formula("rank(a) - rank(b)")
    def test_decay_is_complex(self):       assert not is_simple_formula("decay(f, halflife=10)")
    def test_classify_simple(self):        assert classify_formula("momentum_20d") == "simple"
    def test_classify_complex(self):       assert classify_formula("rank(a) - rank(b)") == "complex"


class TestVectorizedEvaluator:
    def _ev(self):
        return VectorizedEvaluator(make_prices())

    def test_builds_tensor(self):
        assert self._ev()._tensor is not None

    def test_tensor_shape(self):
        T, A, F = self._ev().tensor_shape
        assert T > 0 and A == 3

    def test_n_features(self):
        assert self._ev().n_features == len(VectorizedEvaluator.FEATURE_NAMES)

    def test_evaluate_bare_feature(self):
        r = self._ev().evaluate_batch(["momentum_20d"]).results[0]
        assert isinstance(r.mean_ic, float)

    def test_evaluate_rank_formula(self):
        r = self._ev().evaluate_batch(["rank(momentum_20d)"]).results[0]
        assert r.method == "vectorized"

    def test_complex_formula_fails_gracefully(self):
        r = self._ev().evaluate_batch(["rank(a) - rank(b)"]).results[0]
        assert r.success is False

    def test_unknown_feature_fails_gracefully(self):
        r = self._ev().evaluate_batch(["nonexistent_xyz"]).results[0]
        assert r.success is False

    def test_batch_multiple(self):
        result = self._ev().evaluate_batch(["momentum_20d", "rank(rsi_14)", "zscore_20d"])
        assert result.n_formulas == 3

    def test_elapsed_positive(self):
        assert self._ev().evaluate_batch(["momentum_20d"]).elapsed_seconds > 0

    def test_ic_in_range(self):
        result = self._ev().evaluate_batch(["momentum_20d", "volatility_20d"])
        for r in result.results:
            if r.success:
                assert -1.0 <= r.mean_ic <= 1.0

    def test_cross_rank_shape(self):
        ev  = self._ev()
        x   = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
        assert ev._cross_rank(x).shape == x.shape

    def test_cross_zscore_zero_mean(self):
        ev = self._ev()
        x  = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        z  = ev._cross_zscore(x)
        assert all(abs(np.nanmean(z, axis=1)) < 1e-6)


# ════════════════════════════════════════════════════════════════════════════
# PARALLEL SCORER
# ════════════════════════════════════════════════════════════════════════════

from macro8_subnet.alpha.parallel_scorer import ParallelICScorer, BatchICResult


class TestParallelICScorer:
    def _scorer(self):
        prices  = make_prices()
        returns = prices.pct_change().dropna()
        fs      = make_feature_store(prices)
        return ParallelICScorer(returns, feature_store=fs, prices=prices, n_workers=2)

    def test_returns_batch_ic_result(self):
        assert isinstance(self._scorer().evaluate_batch(["momentum_20d"]), BatchICResult)

    def test_n_total_matches_input(self):
        r = self._scorer().evaluate_batch(["momentum_20d", "rank(volatility_20d)"])
        assert r.n_total == 2

    def test_simple_routes_to_vectorized(self):
        r = self._scorer().evaluate_batch(["momentum_20d"])
        assert r.n_vectorized >= 1

    def test_complex_routes_to_threaded(self):
        r = self._scorer().evaluate_batch(["rank(momentum_20d) - rank(volatility_20d)"])
        assert r.n_threaded >= 1

    def test_all_ics_returns_dict(self):
        assert isinstance(self._scorer().evaluate_batch(["momentum_20d"]).all_ics(), dict)

    def test_top_n_sorted(self):
        r   = self._scorer().evaluate_batch(["momentum_20d", "rank(rsi_14)", "zscore_20d"])
        top = r.top_n(3)
        if len(top) >= 2:
            assert top[0][1] >= top[1][1]

    def test_to_dict_serialisable(self):
        json.dumps(self._scorer().evaluate_batch(["momentum_20d"]).to_dict())

    def test_elapsed_positive(self):
        assert self._scorer().evaluate_batch(["momentum_20d"]).elapsed_seconds > 0

    def test_empty_batch(self):
        r = self._scorer().evaluate_batch([])
        assert r.n_total == 0

    def test_sequential_same_n_total(self):
        sc = self._scorer()
        f  = ["rank(momentum_20d) - rank(volatility_20d)"]
        assert sc.evaluate_batch(f).n_total == sc.evaluate_sequential(f).n_total


# ════════════════════════════════════════════════════════════════════════════
# MPFG ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════════════

from macro8_subnet.alpha.mpfg_orchestrator import (
    MPFGOrchestrator, MPFGReport, _formula_similarity,
)


class TestFormulaSimilarity:
    def test_identical(self):
        assert _formula_similarity("rank(momentum_20d)", "rank(momentum_20d)") == pytest.approx(1.0)
    def test_in_range(self):
        assert 0.0 <= _formula_similarity("momentum_20d", "rsi_14") <= 1.0
    def test_similar_high(self):
        assert _formula_similarity("rank(momentum_20d)", "rank(momentum_60d)") > 0.5
    def test_different_lower(self):
        assert _formula_similarity("rank(momentum_20d)", "zscore(volatility_60d)") < 0.8


class TestMPFGOrchestrator:
    def _orch(self, with_graph=False):
        prices   = make_prices()
        fs       = make_feature_store(prices)
        hyp      = make_hyp_lib()
        from macro8_subnet.alpha.research_graph import FormulaLibrary, ResearchGraph
        form_lib = FormulaLibrary()
        rg       = ResearchGraph(hyp) if with_graph else None
        return MPFGOrchestrator(
            prices=prices, feature_store=fs, hypothesis_library=hyp,
            formula_library=form_lib, research_graph=rg,
            n_workers=2, verbose=False,
        )

    def test_returns_mpfg_report(self):
        assert isinstance(self._orch().run(epoch=1, batch_size=20, min_ic=-10.0), MPFGReport)

    def test_epoch_recorded(self):
        assert self._orch().run(epoch=5, batch_size=20, min_ic=-10.0).epoch == 5

    def test_n_generated_positive(self):
        assert self._orch().run(epoch=1, batch_size=20, min_ic=-10.0).n_generated > 0

    def test_n_generated_bounded(self):
        assert self._orch().run(epoch=1, batch_size=20, min_ic=-10.0).n_generated <= 20

    def test_elapsed_positive(self):
        assert self._orch().run(epoch=1, batch_size=15, min_ic=-10.0).elapsed_seconds > 0

    def test_formulas_per_sec_positive(self):
        assert self._orch().run(epoch=1, batch_size=15, min_ic=-10.0).formulas_per_sec > 0

    def test_to_dict_serialisable(self):
        json.dumps(self._orch().run(epoch=1, batch_size=15, min_ic=-10.0).to_dict())

    def test_summary_string(self):
        r = self._orch().run(epoch=1, batch_size=15, min_ic=-10.0)
        assert "MPFG" in r.summary()

    def test_high_min_ic_admits_nothing(self):
        r = self._orch().run(epoch=1, batch_size=20, min_ic=99.0)
        assert r.n_passed_ic == 0

    def test_with_research_graph(self):
        r = self._orch(with_graph=True).run(epoch=1, batch_size=20, min_ic=-10.0)
        assert isinstance(r, MPFGReport)

    def test_generation_ms_positive(self):
        assert self._orch().run(epoch=1, batch_size=15, min_ic=-10.0).generation_ms > 0

    def test_evaluation_ms_positive(self):
        assert self._orch().run(epoch=1, batch_size=15, min_ic=-10.0).evaluation_ms > 0

    def test_n_deduped_non_negative(self):
        assert self._orch().run(epoch=1, batch_size=30, min_ic=-10.0).n_deduped_out >= 0

    def test_multi_epoch_no_crash(self):
        orch = self._orch()
        for epoch in range(1, 4):
            r = orch.run(epoch=epoch, batch_size=10, min_ic=-10.0)
            assert r.epoch == epoch

    def test_throughput_reasonable(self):
        t0 = time.perf_counter()
        self._orch().run(epoch=1, batch_size=30, min_ic=-10.0)
        assert time.perf_counter() - t0 < 30.0
