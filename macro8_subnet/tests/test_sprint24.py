"""
tests/test_sprint24.py
-----------------------
Sprint 24: Multi-Horizon Portfolio Evaluator

Tests cover:
    - PortfolioEvaluator: output shape, field completeness, backward compat
    - Multi-horizon IC (1d, 7d, 30d, 90d) computation
    - Portfolio simulation: weights, PnL, Sharpe, drawdown
    - Capital scaling: high-turnover penalty gradient
    - Composite scoring: bounds, ordering, component presence
    - PortfolioResult: leaderboard, ranking views
    - GPMiner: composite as primary fitness, hall-of-fame ordering
    - GPReport: new fields, summary format
    - SignalScorer: portfolio_score integration, scalability component
    - Validator: PortfolioEvaluator wired, import clean
    - Edge cases: empty inputs, unencodable formulas, duplicates
"""

import sys
import time
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
    n_days, n_assets = 500, 10
    tickers = ["SPY","QQQ","IWM","TLT","GLD","DBC","EEM","FXI","VNQ","HYG"]
    log_ret = rng.normal(0.0003, 0.01, (n_days, n_assets))
    prices_arr = 100 * np.exp(np.cumsum(log_ret, axis=0))
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    return pd.DataFrame(prices_arr, index=dates, columns=tickers)


@pytest.fixture(scope="module")
def evaluator(prices):
    from macro8_subnet.alpha.portfolio_evaluator import PortfolioEvaluator
    return PortfolioEvaluator(prices, horizons=[1, 7, 30, 90])


@pytest.fixture(scope="module")
def sample_formulas():
    return [
        "momentum_20d", "reversal_5d", "mean_rev_score",
        "market_corr_20d", "vol_ratio", "skew_20d",
        "rank(momentum_20d) - rank(volatility_20d)", "zscore(momentum_60d)",
    ]


@pytest.fixture(scope="module")
def portfolio_result(evaluator, sample_formulas):
    return evaluator.evaluate(sample_formulas)


# ── 1. Construction ───────────────────────────────────────────────────────────

class TestConstruction:
    def test_import(self):
        from macro8_subnet.alpha.portfolio_evaluator import (
            PortfolioEvaluator, PortfolioResult, PortfolioScore,
            CAPITAL_TIERS, IC_HORIZON_WEIGHTS, COMPOSITE_WEIGHTS,
        )

    def test_is_subclass_of_batch_evaluator(self, evaluator):
        from macro8_subnet.alpha.batch_evaluator import BatchEvaluator
        assert isinstance(evaluator, BatchEvaluator)

    def test_horizons_stored(self, evaluator):
        assert evaluator.horizons == [1, 7, 30, 90]

    def test_forward_returns_precomputed_for_all_horizons(self, evaluator):
        for h in [1, 7, 30, 90]:
            assert h in evaluator._fwd_returns
            assert len(evaluator._fwd_returns[h]) > 0

    def test_forward_returns_shape(self, evaluator, prices):
        for h in [1, 7, 30, 90]:
            fwd = evaluator._fwd_returns[h]
            assert fwd.ndim == 2
            assert fwd.shape[1] == len(prices.columns)


# ── 2. Result shape ───────────────────────────────────────────────────────────

class TestResultShape:
    def test_result_type(self, portfolio_result):
        from macro8_subnet.alpha.portfolio_evaluator import PortfolioResult
        assert isinstance(portfolio_result, PortfolioResult)

    def test_portfolio_scores_length_matches_n_formulas(self, portfolio_result):
        assert len(portfolio_result.portfolio_scores) == portfolio_result.n_formulas

    def test_arrays_length_matches_n_formulas(self, portfolio_result):
        n = portfolio_result.n_formulas
        assert portfolio_result.sharpes.shape[0] == n
        assert portfolio_result.turnovers.shape[0] == n
        assert portfolio_result.composites.shape[0] == n
        assert portfolio_result.mean_ics.shape[0] == n
        assert portfolio_result.ic_irs.shape[0] == n

    def test_at_least_one_formula_evaluated(self, portfolio_result):
        assert portfolio_result.n_formulas > 0


# ── 3. PortfolioScore field completeness ─────────────────────────────────────

class TestPortfolioScoreFields:
    REQUIRED_DICT_FIELDS = [
        "formula", "ic_weighted", "ic_1d", "ic_7d", "ic_30d", "ic_90d",
        "sharpe", "sortino", "max_drawdown", "annualised_ret",
        "daily_turnover", "win_rate",
        "capital_1k", "capital_10k", "capital_100k", "capital_1M",
        "composite",
    ]

    def test_all_dict_fields_present(self, portfolio_result):
        for ps in portfolio_result.portfolio_scores:
            d = ps.to_dict()
            for field in self.REQUIRED_DICT_FIELDS:
                assert field in d, f"Missing '{field}' for {ps.formula}"

    def test_ic_by_horizon_all_four(self, portfolio_result):
        for ps in portfolio_result.portfolio_scores:
            for h in [1, 7, 30, 90]:
                assert h in ps.ic_by_horizon

    def test_capital_scores_all_tiers(self, portfolio_result):
        from macro8_subnet.alpha.portfolio_evaluator import CAPITAL_TIERS
        for ps in portfolio_result.portfolio_scores:
            for cap in CAPITAL_TIERS:
                assert cap in ps.capital_scores

    def test_win_rate_in_unit_interval(self, portfolio_result):
        for ps in portfolio_result.portfolio_scores:
            assert 0.0 <= ps.win_rate <= 1.0

    def test_max_drawdown_non_positive(self, portfolio_result):
        for ps in portfolio_result.portfolio_scores:
            assert ps.max_drawdown <= 0.0

    def test_composite_in_unit_interval(self, portfolio_result):
        for ps in portfolio_result.portfolio_scores:
            assert 0.0 <= ps.composite <= 1.0, \
                f"composite={ps.composite} out of [0,1] for {ps.formula}"

    def test_to_dict_no_nan_inf(self, portfolio_result):
        for ps in portfolio_result.portfolio_scores:
            for k, v in ps.to_dict().items():
                if isinstance(v, float):
                    assert np.isfinite(v), f"Field '{k}'={v} not finite ({ps.formula})"

    def test_one_line_format(self, portfolio_result):
        ps   = portfolio_result.portfolio_scores[0]
        line = ps.one_line(rank=1)
        assert "#1" in line and "Sharpe" in line


# ── 4. Multi-horizon IC ───────────────────────────────────────────────────────

class TestMultiHorizonIC:
    def test_all_ics_finite(self, portfolio_result):
        for ps in portfolio_result.portfolio_scores:
            for h, ic in ps.ic_by_horizon.items():
                assert np.isfinite(ic), f"IC[{h}d]={ic} not finite ({ps.formula})"

    def test_weighted_ic_within_minus_one_plus_one(self, portfolio_result):
        for ps in portfolio_result.portfolio_scores:
            assert -1.0 <= ps.ic_weighted <= 1.0

    def test_ic_1d_matches_mean_ics_array(self, portfolio_result):
        for i, ps in enumerate(portfolio_result.portfolio_scores):
            diff = abs(ps.ic_by_horizon[1] - float(portfolio_result.mean_ics[i]))
            assert diff < 0.001, \
                f"IC[1d] mismatch for {ps.formula}: {diff:.6f}"

    def test_weighted_ic_formula_correct(self, portfolio_result):
        from macro8_subnet.alpha.portfolio_evaluator import IC_HORIZON_WEIGHTS
        for ps in portfolio_result.portfolio_scores:
            expected = sum(IC_HORIZON_WEIGHTS[h] * ps.ic_by_horizon[h]
                          for h in [1, 7, 30, 90])
            assert abs(ps.ic_weighted - expected) < 1e-4

    def test_90d_ic_not_nan(self, portfolio_result):
        for ps in portfolio_result.portfolio_scores:
            assert not np.isnan(ps.ic_by_horizon[90])


# ── 5. Portfolio simulation ───────────────────────────────────────────────────

class TestPortfolioSimulation:
    def test_sharpe_finite(self, portfolio_result):
        for ps in portfolio_result.portfolio_scores:
            assert np.isfinite(ps.sharpe)

    def test_sortino_finite(self, portfolio_result):
        for ps in portfolio_result.portfolio_scores:
            assert np.isfinite(ps.sortino)

    def test_turnover_non_negative(self, portfolio_result):
        for ps in portfolio_result.portfolio_scores:
            assert ps.daily_turnover >= 0.0

    def test_annualised_ret_finite(self, portfolio_result):
        for ps in portfolio_result.portfolio_scores:
            assert np.isfinite(ps.annualised_ret)

    def test_weights_l1_norm_unity(self, prices, evaluator):
        """Portfolio weights should sum to 1 in L1 norm (excluding warmup zeros)."""
        from macro8_subnet.alpha.batch_evaluator import FeatureTensor, FormulaEncoder
        from macro8_subnet.alpha.feature_store import FeatureStore
        from scipy.stats import rankdata

        fs  = FeatureStore(prices)
        ft  = FeatureTensor.from_feature_dict(fs.build())
        enc = FormulaEncoder(ft.feature_names)
        if not enc.can_encode("momentum_20d"):
            pytest.skip("momentum_20d not encodable")

        W      = enc.encode_batch(["momentum_20d"])
        S      = np.einsum("taf,fn->tan", ft.tensor, W, optimize=True)
        T, A, F = S.shape
        T_use  = min(T, len(evaluator.returns)) - 1
        S_use  = S[:T_use + 1]

        S_tfa  = S_use.transpose(0, 2, 1).reshape((T_use + 1) * F, A)
        r_flat = rankdata(S_tfa, axis=1).astype(np.float32)
        ranks  = r_flat.reshape(T_use + 1, F, A).transpose(0, 2, 1)
        ranks -= ranks.mean(axis=1, keepdims=True)
        norms  = np.abs(ranks).sum(axis=1, keepdims=True) + 1e-8
        weights = ranks / norms

        l1 = np.abs(weights).sum(axis=1).squeeze()   # [T]
        # Skip warmup rows where signal is all-zero (before enough history)
        active = l1[l1 > 0.01]
        assert len(active) > 0, "All weights are zero — no active timesteps"
        assert np.allclose(active, 1.0, atol=1e-3)


# ── 6. Capital scaling ────────────────────────────────────────────────────────

class TestCapitalScaling:
    def test_high_turnover_scores_worse_at_1m(self, portfolio_result):
        for ps in portfolio_result.portfolio_scores:
            if ps.daily_turnover > 0.02:
                assert ps.capital_scores[1_000] > ps.capital_scores[1_000_000], \
                    f"{ps.formula}: $1k should beat $1M (turnover={ps.daily_turnover:.4f})"
                return
        pytest.skip("No high-turnover formula in sample")

    def test_capital_score_monotone_for_high_turnover(self, portfolio_result):
        from macro8_subnet.alpha.portfolio_evaluator import CAPITAL_TIERS
        for ps in portfolio_result.portfolio_scores:
            if ps.daily_turnover > 0.03:
                scores = [ps.capital_scores[cap] for cap in sorted(CAPITAL_TIERS)]
                for i in range(len(scores) - 1):
                    assert scores[i] >= scores[i + 1] - 1e-6
                return
        pytest.skip("No very-high-turnover formula in sample")

    def test_zero_turnover_capital_invariant(self):
        from macro8_subnet.alpha.portfolio_evaluator import TURNOVER_CAPITAL_SCALE
        sharpe, turnover = 0.5, 0.0
        for cap in [1_000, 10_000, 100_000, 1_000_000]:
            score = sharpe - turnover * (cap / 1_000_000) * TURNOVER_CAPITAL_SCALE
            assert abs(score - sharpe) < 1e-10

    def test_capital_score_mean_correct(self, portfolio_result):
        from macro8_subnet.alpha.portfolio_evaluator import CAPITAL_TIERS
        for ps in portfolio_result.portfolio_scores:
            expected = np.mean([ps.capital_scores[c] for c in CAPITAL_TIERS])
            assert abs(ps.capital_score_mean - expected) < 1e-4


# ── 7. PortfolioResult ranking views ─────────────────────────────────────────

class TestRankingViews:
    def test_top_by_composite_sorted(self, portfolio_result):
        n   = min(3, portfolio_result.n_formulas)
        top = portfolio_result.top_by_composite(n)
        c   = [s.composite for s in top]
        assert c == sorted(c, reverse=True)

    def test_top_by_sharpe_sorted(self, portfolio_result):
        n   = min(3, portfolio_result.n_formulas)
        top = portfolio_result.top_by_sharpe(n)
        s   = [x.sharpe for x in top]
        assert s == sorted(s, reverse=True)

    def test_top_by_capital_1m_sorted(self, portfolio_result):
        n   = min(3, portfolio_result.n_formulas)
        top = portfolio_result.top_by_capital(1_000_000, n)
        c   = [x.capital_scores[1_000_000] for x in top]
        assert c == sorted(c, reverse=True)

    def test_top_by_ic_30d_sorted(self, portfolio_result):
        n   = min(3, portfolio_result.n_formulas)
        top = portfolio_result.top_by_ic_30d(n)
        i   = [x.ic_by_horizon[30] for x in top]
        assert i == sorted(i, reverse=True)

    def test_leaderboard_string_content(self, portfolio_result):
        lb = portfolio_result.leaderboard(n=5)
        assert "LEADERBOARD" in lb
        assert "ShrpG" in lb or "Sharpe" in lb  # header abbreviation may vary

    def test_best_by_composite_not_none(self, portfolio_result):
        best = portfolio_result.best_by_composite
        assert best is not None

    def test_summary_line_contains_sharpe(self, portfolio_result):
        assert "Sharpe" in portfolio_result.summary_line()


# ── 8. Backward compatibility ─────────────────────────────────────────────────

class TestBackwardCompat:
    def test_mean_ics_is_ndarray(self, portfolio_result):
        assert isinstance(portfolio_result.mean_ics, np.ndarray)

    def test_ic_irs_is_ndarray(self, portfolio_result):
        assert isinstance(portfolio_result.ic_irs, np.ndarray)

    def test_top_n_works(self, portfolio_result):
        top = portfolio_result.top_n(3)
        assert isinstance(top, list)
        assert all("formula" in e and "mean_ic" in e for e in top)

    def test_above_threshold_works(self, portfolio_result):
        result = portfolio_result.above_threshold(0.0)
        assert isinstance(result, list)

    def test_isinstance_batch_evaluation_result(self, portfolio_result):
        from macro8_subnet.alpha.batch_evaluator import BatchEvaluationResult
        assert isinstance(portfolio_result, BatchEvaluationResult)


# ── 9. GPMiner composite fitness ─────────────────────────────────────────────

class TestGPMinerCompositeFitness:
    @pytest.fixture(scope="class")
    def gp_report(self, prices):
        from macro8_subnet.alpha.gp_miner import GPMiner
        gp = GPMiner(prices, pop_size=40, elite_n=8, seed=42, verbose=False)
        return gp.run(n_epochs=3)

    def test_scored_formula_has_new_fields(self):
        import dataclasses
        from macro8_subnet.alpha.gp_miner import ScoredFormula
        fields = {f.name for f in dataclasses.fields(ScoredFormula)}
        for req in ["composite", "sharpe", "turnover", "ic_30d", "capital_1m"]:
            assert req in fields

    def test_top_formulas_sorted_by_composite(self, gp_report):
        composites = [sf.composite for sf in gp_report.top_formulas]
        assert composites == sorted(composites, reverse=True)

    def test_composite_history_length(self, gp_report):
        assert len(gp_report.best_composite_history) == 3

    def test_summary_has_sharpe_and_composite(self, gp_report):
        summary = gp_report.summary()
        assert "Sharpe" in summary
        assert "composite" in summary.lower()

    def test_submission_formulas_returns_strings(self, gp_report):
        formulas = gp_report.submission_formulas(5)
        assert all(isinstance(f, str) for f in formulas)

    def test_step_hall_of_fame_sorted_by_composite(self, prices):
        from macro8_subnet.alpha.gp_miner import GPMiner
        gp  = GPMiner(prices, pop_size=30, elite_n=6, seed=99, verbose=False)
        top = gp.step()
        if len(top) >= 2:
            c = [sf.composite for sf in top[:5]]
            assert c == sorted(c, reverse=True)

    def test_top_formulas_method(self, prices):
        from macro8_subnet.alpha.gp_miner import GPMiner
        gp = GPMiner(prices, pop_size=30, elite_n=6, seed=77, verbose=False)
        gp.run(n_epochs=2)
        tops = gp.top_formulas(5)
        assert isinstance(tops, list)
        assert all(isinstance(f, str) for f in tops)


# ── 10. SignalScorer portfolio integration ────────────────────────────────────

class TestSignalScorerPortfolio:
    @pytest.fixture(scope="class")
    def scorer(self, prices):
        from macro8_subnet.evaluation.signal_scorer import SignalScorer
        from macro8_subnet.alpha.batch_evaluator import BatchEvaluator
        return SignalScorer(BatchEvaluator(prices, min_ic=0.0))

    def test_score_one_without_portfolio_score_succeeds(self, scorer):
        r = scorer._score_one(
            "momentum_20d", "f1", [0.02, 0.025], [],
            (0.022, 1.5), 1, portfolio_score=None,
        )
        assert r.success

    def test_score_one_without_portfolio_no_scalability_component(self, scorer):
        r = scorer._score_one(
            "momentum_20d", "f1", [0.02], [],
            (0.02, 1.0), 1, portfolio_score=None,
        )
        assert "scalability" not in [c.name for c in r.components]

    def test_score_one_with_portfolio_has_scalability(self, scorer, portfolio_result):
        ps = portfolio_result.portfolio_scores[0]
        r  = scorer._score_one(
            ps.formula, "f2", [0.02], [],
            (0.02, 1.0), 1, portfolio_score=ps,
        )
        assert r.success
        assert "scalability" in [c.name for c in r.components]

    def test_scalability_component_in_unit_interval(self, scorer, portfolio_result):
        ps = portfolio_result.portfolio_scores[0]
        r  = scorer._score_one(
            ps.formula, "f3", [0.02], [],
            (0.02, 1.0), 1, portfolio_score=ps,
        )
        for c in r.components:
            if c.name == "scalability":
                assert 0.0 <= c.score   <= 1.0
                assert 0.0 <= c.weighted <= 1.0

    def test_score_batch_with_portfolio_evaluator(self, prices):
        from macro8_subnet.evaluation.signal_scorer import SignalScorer
        from macro8_subnet.alpha.portfolio_evaluator import PortfolioEvaluator
        ev     = PortfolioEvaluator(prices, min_ic=0.0)
        scorer = SignalScorer(ev)
        results = scorer.score_batch(
            ["momentum_20d", "market_corr_20d"],
            ["f1", "f2"],
            {"f1": [0.02], "f2": [0.015]},
            {}, epoch=1,
        )
        assert len(results) == 2
        for r in results:
            assert r.success or r.error


# ── 11. Validator integration ─────────────────────────────────────────────────

class TestValidatorIntegration:
    def test_validator_uses_portfolio_evaluator(self):
        import inspect
        from macro8_subnet.neurons.validator import Macro8Validator
        src = inspect.getsource(Macro8Validator._initialise_engine)
        assert "PortfolioEvaluator" in src

    def test_epoch_scorer_accepts_portfolio_evaluator(self, prices):
        from macro8_subnet.neurons.validator import EpochScorer
        from macro8_subnet.alpha.portfolio_evaluator import PortfolioEvaluator
        from macro8_subnet.alpha.capacity_model import LifecycleEngine
        from macro8_subnet.alpha.orthogonality import OrthogonalityFilter
        ev  = PortfolioEvaluator(prices, min_ic=0.0)
        es  = EpochScorer(ev, LifecycleEngine(), OrthogonalityFilter())
        assert es is not None

    def test_portfolio_result_is_batch_result(self, portfolio_result):
        from macro8_subnet.alpha.batch_evaluator import BatchEvaluationResult
        assert isinstance(portfolio_result, BatchEvaluationResult)


# ── 12. Edge cases ────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_formula_list(self, evaluator):
        result = evaluator.evaluate([])
        assert result.n_formulas == 0
        assert result.portfolio_scores == []

    def test_single_formula(self, evaluator):
        result = evaluator.evaluate(["momentum_20d"])
        if result.n_formulas > 0:
            assert len(result.portfolio_scores) == 1

    def test_unencodable_formula_skipped(self, evaluator):
        result = evaluator.evaluate(["__import__('os')", "momentum_20d"])
        for ps in result.portfolio_scores:
            assert "__import__" not in ps.formula

    def test_duplicate_formulas_each_get_score(self, evaluator):
        """Duplicate inputs each receive a score — dedup is the caller's (GPMiner/validator) job."""
        result = evaluator.evaluate(["momentum_20d"] * 3)
        # All three are evaluated (PortfolioEvaluator doesn't dedup — that's GPMiner's job)
        assert result.n_formulas == 3
        # But they should all have identical scores
        composites = [ps.composite for ps in result.portfolio_scores]
        assert len(set(round(c, 6) for c in composites)) == 1, \
            "Duplicate formulas should produce identical composite scores"

    def test_evaluation_within_30s(self, evaluator, sample_formulas):
        t       = time.perf_counter()
        _       = evaluator.evaluate(sample_formulas)
        elapsed = time.perf_counter() - t
        assert elapsed < 30.0, f"Took {elapsed:.1f}s — exceeds 30s budget"
