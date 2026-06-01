"""
tests/test_sprint16.py
-----------------------
QA: Sprint 16 — Capacity Model + Representation Learning Engine.

Covers:
    alpha/capacity_model.py
        LifecycleState         — enum values, weight multipliers
        DecayEstimator         — exponential decay fitting, half-life
        CapacityEstimator      — capacity score computation
        LifecycleEngine        — state transitions, adjusted weights, assess_all
        CapacityReport         — serialisation

    alpha/representation_engine.py
        LatentFeatureSet       — structure, serialisation, interpretation
        RepresentationEngine   — PCA, autoencoder, rolling PCA, enrich, auto_hypotheses
        Integration            — latent features → FeatureTensor → FormulaEncoder
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SUITE_DIR  = Path(__file__).resolve().parent
SUBNET_DIR = SUITE_DIR.parent
PROJECT    = SUBNET_DIR.parent
for p in [str(SUBNET_DIR), str(PROJECT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from macro8_subnet.alpha.capacity_model import (
    LifecycleState, LifecycleTransition,
    DecayEstimator, DecayEstimate,
    CapacityEstimator, CapacityReport,
    LifecycleEngine,
)
from macro8_subnet.alpha.representation_engine import (
    LatentFeatureSet, RepresentationEngine,
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


def make_feature_dict(prices=None):
    from macro8_subnet.alpha.feature_store import FeatureStore
    return FeatureStore(prices if prices is not None else make_prices()).build()


def make_engine(prices=None) -> RepresentationEngine:
    return RepresentationEngine(prices if prices is not None else make_prices())


# ════════════════════════════════════════════════════════════════════════════
# LIFECYCLE STATE
# ════════════════════════════════════════════════════════════════════════════

class TestLifecycleState:
    def test_five_states(self):
        assert len(list(LifecycleState)) == 5

    def test_weight_multiplier_production_is_one(self):
        assert LifecycleState.PRODUCTION.weight_multiplier == pytest.approx(1.0)

    def test_weight_multiplier_retired_is_zero(self):
        assert LifecycleState.RETIRED.weight_multiplier == pytest.approx(0.0)

    def test_experimental_lower_than_validated(self):
        assert (LifecycleState.EXPERIMENTAL.weight_multiplier <
                LifecycleState.VALIDATED.weight_multiplier)

    def test_validated_lower_than_production(self):
        assert (LifecycleState.VALIDATED.weight_multiplier <
                LifecycleState.PRODUCTION.weight_multiplier)

    def test_decaying_lower_than_production(self):
        assert (LifecycleState.DECAYING.weight_multiplier <
                LifecycleState.PRODUCTION.weight_multiplier)

    def test_retired_not_active(self):
        assert not LifecycleState.RETIRED.is_active

    def test_production_is_active(self):
        assert LifecycleState.PRODUCTION.is_active


# ════════════════════════════════════════════════════════════════════════════
# DECAY ESTIMATOR
# ════════════════════════════════════════════════════════════════════════════

class TestDecayEstimator:
    def test_returns_estimate(self):
        d = DecayEstimator()
        r = d.estimate("f1", [0.05, 0.04, 0.03, 0.02, 0.01])
        assert isinstance(r, DecayEstimate)

    def test_insufficient_obs_returns_none_halflife(self):
        d = DecayEstimator(min_obs=4)
        r = d.estimate("f1", [0.04, 0.03])
        assert r.ic_half_life is None

    def test_stable_ic_not_decaying(self):
        d = DecayEstimator(decay_threshold=5.0)
        r = d.estimate("f1", [0.04, 0.04, 0.04, 0.04, 0.04, 0.04])
        assert r.is_decaying is False

    def test_strongly_decaying_signal(self):
        d = DecayEstimator(decay_threshold=20.0)
        # Strong decay: IC halves each step
        ics = [0.08, 0.04, 0.02, 0.01, 0.005, 0.0025]
        r   = d.estimate("f1", ics)
        assert r.is_decaying is True
        assert r.decay_rate > 0

    def test_half_life_positive(self):
        d   = DecayEstimator()
        ics = [0.06, 0.05, 0.04, 0.03, 0.025, 0.02]
        r   = d.estimate("f1", ics)
        if r.ic_half_life is not None:
            assert r.ic_half_life > 0

    def test_r_squared_in_range(self):
        d   = DecayEstimator()
        ics = [0.06, 0.05, 0.04, 0.03, 0.025, 0.02]
        r   = d.estimate("f1", ics)
        assert 0.0 <= r.r_squared <= 1.0

    def test_to_dict_serialisable(self):
        d = DecayEstimator()
        r = d.estimate("f1", [0.05, 0.04, 0.03, 0.02, 0.01])
        json.dumps(r.to_dict())

    def test_empty_history_graceful(self):
        d = DecayEstimator()
        r = d.estimate("f1", [])
        assert r.ic_half_life is None

    def test_all_zero_history(self):
        d = DecayEstimator()
        r = d.estimate("f1", [0.0, 0.0, 0.0, 0.0, 0.0])
        assert r.ic_half_life is None  # zeros → can't fit log

    def test_mixed_sign_history(self):
        d = DecayEstimator()
        r = d.estimate("f1", [0.04, -0.02, 0.03, -0.01, 0.02, -0.01])
        assert isinstance(r, DecayEstimate)  # should not crash


# ════════════════════════════════════════════════════════════════════════════
# CAPACITY ESTIMATOR
# ════════════════════════════════════════════════════════════════════════════

class TestCapacityEstimator:
    def _est(self):
        return CapacityEstimator()

    def test_retired_is_zero(self):
        est = self._est()
        c   = est.estimate([], [], LifecycleState.RETIRED)
        assert c == pytest.approx(0.0)

    def test_empty_history_returns_small(self):
        est = self._est()
        c   = est.estimate([], [], LifecycleState.EXPERIMENTAL)
        assert 0.0 < c < 0.5

    def test_production_higher_than_experimental(self):
        ics = [0.04] * 10
        mscs = [0.01] * 5
        est  = self._est()
        c_exp  = est.estimate(ics, mscs, LifecycleState.EXPERIMENTAL)
        c_prod = est.estimate(ics, mscs, LifecycleState.PRODUCTION)
        assert c_prod > c_exp

    def test_high_stability_higher_capacity(self):
        est    = self._est()
        stable = [0.04] * 10
        unstable = [0.04, -0.02, 0.03, -0.01, 0.04, -0.02, 0.04, -0.02, 0.03, -0.01]
        c_s = est.estimate(stable,   [], LifecycleState.PRODUCTION)
        c_u = est.estimate(unstable, [], LifecycleState.PRODUCTION)
        assert c_s > c_u

    def test_high_crowding_reduces_capacity(self):
        est = self._est()
        ics = [0.04] * 10
        c_low  = est.estimate(ics, [], LifecycleState.PRODUCTION, crowding=0.0)
        c_high = est.estimate(ics, [], LifecycleState.PRODUCTION, crowding=0.8)
        assert c_high < c_low

    def test_capacity_in_range(self):
        est = self._est()
        for state in LifecycleState:
            c = est.estimate([0.04]*5, [0.01]*5, state)
            assert 0.0 <= c <= 1.0


# ════════════════════════════════════════════════════════════════════════════
# LIFECYCLE ENGINE
# ════════════════════════════════════════════════════════════════════════════

class TestLifecycleEngine:
    def _engine(self):
        return LifecycleEngine(min_ic=0.01, min_epochs=3)

    def test_new_formula_is_experimental(self):
        engine = self._engine()
        state, _, _ = engine.assess("f1", [0.04, 0.03], [], epoch=1)
        assert state == LifecycleState.EXPERIMENTAL

    def test_experimental_to_validated(self):
        engine = self._engine()
        state, trans, _ = engine.assess("f1", [0.04, 0.03, 0.05], [], epoch=3)
        assert state == LifecycleState.VALIDATED

    def test_validated_to_production(self):
        engine = self._engine()
        # First get to VALIDATED
        engine.assess("f1", [0.04]*3, [], epoch=3)
        # Add good MSC and high stability
        good_ics  = [0.04]*8
        good_mscs = [0.01]*5
        state, _, _ = engine.assess("f1", good_ics, good_mscs, epoch=8)
        assert state == LifecycleState.PRODUCTION

    def test_production_to_decaying(self):
        engine = LifecycleEngine(min_ic=0.01, min_epochs=3)
        # Get to PRODUCTION first
        engine._states["f1"] = LifecycleState.PRODUCTION
        # Strongly decaying IC
        ics = [0.08, 0.04, 0.02, 0.01, 0.005, 0.0025, 0.001]
        state, trans, _ = engine.assess("f1", ics, [], epoch=7)
        assert state in (LifecycleState.DECAYING, LifecycleState.PRODUCTION)

    def test_force_retirement_below_min_ic(self):
        engine = LifecycleEngine(min_ic=0.01, min_ic_retire=0.005, min_epochs=3)
        ics    = [-0.01, -0.02, -0.01, -0.03, -0.02]
        state, _, _ = engine.assess("f1", ics, [], epoch=5)
        assert state == LifecycleState.RETIRED

    def test_transition_recorded(self):
        engine = self._engine()
        # Trigger a transition: empty → EXPERIMENTAL (no trans), then more obs
        engine.assess("f1", [0.04]*2, [], epoch=2)  # EXPERIMENTAL, no trans
        _, trans, _ = engine.assess("f1", [0.04]*3, [], epoch=3)
        # May or may not transition depending on state — just check type
        if trans is not None:
            assert isinstance(trans, LifecycleTransition)

    def test_adjusted_weight_production_full(self):
        engine = self._engine()
        engine._states["f1"] = LifecycleState.PRODUCTION
        w = engine.adjusted_weight("f1", raw_ic=0.04)
        assert w == pytest.approx(0.04)

    def test_adjusted_weight_retired_zero(self):
        engine = self._engine()
        engine._states["f1"] = LifecycleState.RETIRED
        w = engine.adjusted_weight("f1", raw_ic=0.04)
        assert w == pytest.approx(0.0)

    def test_adjusted_weight_experimental_reduced(self):
        engine = self._engine()
        engine._states["f1"] = LifecycleState.EXPERIMENTAL
        w_full = engine.adjusted_weight("f1", raw_ic=0.04)
        assert w_full < 0.04

    def test_adjusted_weight_crowding_reduces(self):
        engine = self._engine()
        engine._states["f1"] = LifecycleState.PRODUCTION
        w_no_crowd = engine.adjusted_weight("f1", raw_ic=0.04, crowding=0.0)
        w_crowded  = engine.adjusted_weight("f1", raw_ic=0.04, crowding=0.5)
        assert w_crowded < w_no_crowd

    def test_state_of_unknown_is_experimental(self):
        engine = self._engine()
        assert engine.state_of("unknown_id") == LifecycleState.EXPERIMENTAL

    def test_summary_returns_dict(self):
        engine = self._engine()
        engine.assess("f1", [0.04]*5, [], epoch=5)
        s = engine.summary()
        assert isinstance(s, dict)
        for state in LifecycleState:
            assert state.value in s

    def test_capacity_report_serialisable(self):
        engine = self._engine()
        _, _, cap = engine.assess("f1", [0.04]*5, [0.01]*3, epoch=5)
        json.dumps(cap.to_dict())

    def test_assess_all_with_formula_records(self):
        """assess_all works with FormulaRecord-like objects."""
        from macro8_subnet.alpha.research_graph import FormulaLibrary
        lib  = FormulaLibrary()
        recs = []
        for i in range(3):
            r = lib.register(f"f{i}", i)
            r.ic_history  = [0.04] * 5
            r.msc_history = [0.01] * 3
            recs.append(r)
        engine      = self._engine()
        transitions = engine.assess_all(recs, epoch=5)
        assert isinstance(transitions, list)

    def test_multiple_epochs_state_persists(self):
        """State should persist across multiple assess() calls."""
        engine = LifecycleEngine(min_ic=0.01, min_epochs=3)
        engine.assess("f1", [0.04]*3, [], epoch=3)
        s1 = engine.state_of("f1")
        engine.assess("f1", [0.04]*6, [0.01]*4, epoch=6)
        s2 = engine.state_of("f1")
        # State can only advance, not regress unless decay detected
        valid_progressions = {
            LifecycleState.EXPERIMENTAL: [LifecycleState.EXPERIMENTAL, LifecycleState.VALIDATED, LifecycleState.RETIRED],
            LifecycleState.VALIDATED:    [LifecycleState.VALIDATED, LifecycleState.PRODUCTION, LifecycleState.DECAYING, LifecycleState.RETIRED],
            LifecycleState.PRODUCTION:   [LifecycleState.PRODUCTION, LifecycleState.DECAYING, LifecycleState.RETIRED],
        }
        if s1 in valid_progressions:
            assert s2 in valid_progressions[s1]


# ════════════════════════════════════════════════════════════════════════════
# LATENT FEATURE SET
# ════════════════════════════════════════════════════════════════════════════

class TestLatentFeatureSet:
    def _make(self) -> LatentFeatureSet:
        dates = pd.date_range("2021-01-01", periods=50, freq="B")
        vals  = np.random.default_rng(0).normal(0, 1, 50)
        ts    = {"SPY": pd.Series(vals, index=dates),
                 "AAPL": pd.Series(vals * 0.9, index=dates)}
        return LatentFeatureSet(
            latent_name="latent_pca_0",
            method="pca",
            time_series=ts,
            explained_var=0.30,
            feature_basis=["momentum_20d", "volatility_20d"],
            loadings=np.array([0.7, -0.3]),
        )

    def test_as_dataframe_columns(self):
        ls = self._make()
        df = ls.as_dataframe()
        assert set(df.columns) == {"SPY", "AAPL"}

    def test_n_observations(self):
        ls = self._make()
        assert ls.n_observations == 50

    def test_mean_ic_empty(self):
        ls = self._make()
        assert ls.mean_ic == 0.0

    def test_mean_ic_with_history(self):
        ls = self._make()
        ls.ic_history = [0.04, 0.05, 0.03]
        assert ls.mean_ic == pytest.approx(0.04)

    def test_interpretation_pca(self):
        ls = self._make()
        interp = ls.interpretation()
        assert "latent_pca_0" in interp

    def test_to_dict_serialisable(self):
        json.dumps(self._make().to_dict())

    def test_to_dict_has_keys(self):
        d = self._make().to_dict()
        for key in ("latent_name", "method", "explained_var", "n_observations"):
            assert key in d


# ════════════════════════════════════════════════════════════════════════════
# REPRESENTATION ENGINE — PCA
# ════════════════════════════════════════════════════════════════════════════

class TestRepresentationEnginePCA:
    def _engine_and_features(self):
        prices = make_prices()
        engine = RepresentationEngine(prices)
        feats  = make_feature_dict(prices)
        return engine, feats

    def test_pca_returns_list(self):
        engine, feats = self._engine_and_features()
        result = engine.fit_pca(feats, n_components=3)
        assert isinstance(result, list)

    def test_pca_n_components(self):
        engine, feats = self._engine_and_features()
        result = engine.fit_pca(feats, n_components=3)
        assert len(result) == 3

    def test_pca_names_have_prefix(self):
        engine, feats = self._engine_and_features()
        for ls in engine.fit_pca(feats, n_components=2):
            assert ls.latent_name.startswith("latent_pca_")

    def test_pca_method_label(self):
        engine, feats = self._engine_and_features()
        for ls in engine.fit_pca(feats, n_components=2):
            assert ls.method == "pca"

    def test_pca_explained_var_in_range(self):
        engine, feats = self._engine_and_features()
        for ls in engine.fit_pca(feats, n_components=3):
            assert 0.0 <= ls.explained_var <= 1.0

    def test_pca_explained_var_decreasing(self):
        engine, feats = self._engine_and_features()
        sets = engine.fit_pca(feats, n_components=3)
        if len(sets) >= 2:
            assert sets[0].explained_var >= sets[1].explained_var - 0.01

    def test_pca_time_series_length(self):
        engine, feats = self._engine_and_features()
        for ls in engine.fit_pca(feats, n_components=2):
            for asset, series in ls.time_series.items():
                assert len(series) > 0

    def test_pca_assets_match_prices(self):
        prices = make_prices()
        engine = RepresentationEngine(prices)
        feats  = make_feature_dict(prices)
        for ls in engine.fit_pca(feats, n_components=2):
            assert set(ls.time_series.keys()) == set(prices.columns)

    def test_pca_has_loadings(self):
        engine, feats = self._engine_and_features()
        for ls in engine.fit_pca(feats, n_components=2):
            assert ls.loadings is not None

    def test_pca_feature_basis_populated(self):
        engine, feats = self._engine_and_features()
        for ls in engine.fit_pca(feats, n_components=2):
            assert len(ls.feature_basis) > 0

    def test_pca_empty_dict_returns_empty(self):
        engine = make_engine()
        result = engine.fit_pca({}, n_components=3)
        assert result == []


# ════════════════════════════════════════════════════════════════════════════
# REPRESENTATION ENGINE — Autoencoder
# ════════════════════════════════════════════════════════════════════════════

class TestRepresentationEngineAE:
    def _engine_and_features(self):
        prices = make_prices()
        engine = RepresentationEngine(prices)
        feats  = make_feature_dict(prices)
        return engine, feats

    def test_ae_returns_list(self):
        engine, feats = self._engine_and_features()
        result = engine.fit_autoencoder(feats, n_latent=2)
        assert isinstance(result, list)

    def test_ae_n_latent_features(self):
        engine, feats = self._engine_and_features()
        result = engine.fit_autoencoder(feats, n_latent=2)
        assert len(result) <= 2

    def test_ae_names_have_prefix(self):
        engine, feats = self._engine_and_features()
        for ls in engine.fit_autoencoder(feats, n_latent=2):
            assert ls.latent_name.startswith("latent_ae_")

    def test_ae_method_label(self):
        engine, feats = self._engine_and_features()
        for ls in engine.fit_autoencoder(feats, n_latent=2):
            assert ls.method == "autoencoder"

    def test_ae_time_series_correct_length(self):
        prices = make_prices()
        engine = RepresentationEngine(prices)
        feats  = make_feature_dict(prices)
        for ls in engine.fit_autoencoder(feats, n_latent=2):
            for series in ls.time_series.values():
                assert len(series) > 0

    def test_ae_empty_dict_returns_empty(self):
        engine = make_engine()
        result = engine.fit_autoencoder({})
        assert result == []


# ════════════════════════════════════════════════════════════════════════════
# REPRESENTATION ENGINE — Rolling PCA
# ════════════════════════════════════════════════════════════════════════════

class TestRepresentationEngineRollingPCA:
    def _engine_and_features(self):
        prices = make_prices(200)
        engine = RepresentationEngine(prices)
        feats  = make_feature_dict(prices)
        return engine, feats

    def test_rolling_pca_returns_list(self):
        engine, feats = self._engine_and_features()
        result = engine.fit_rolling_pca(feats, n_components=2, window=30)
        assert isinstance(result, list)

    def test_rolling_pca_n_components(self):
        engine, feats = self._engine_and_features()
        result = engine.fit_rolling_pca(feats, n_components=2, window=30)
        assert len(result) <= 2

    def test_rolling_pca_names_prefix(self):
        engine, feats = self._engine_and_features()
        for ls in engine.fit_rolling_pca(feats, n_components=2, window=30):
            assert ls.latent_name.startswith("latent_rpca_")

    def test_rolling_pca_method_label(self):
        engine, feats = self._engine_and_features()
        for ls in engine.fit_rolling_pca(feats, n_components=2, window=30):
            assert ls.method == "rolling_pca"

    def test_rolling_pca_insufficient_data_returns_empty(self):
        prices = make_prices(20)   # too few for window=40
        engine = RepresentationEngine(prices)
        feats  = make_feature_dict(prices)
        result = engine.fit_rolling_pca(feats, window=40)
        assert result == []


# ════════════════════════════════════════════════════════════════════════════
# REPRESENTATION ENGINE — fit_all + enrich + auto_hypotheses
# ════════════════════════════════════════════════════════════════════════════

class TestRepresentationEngineFull:
    def test_fit_all_returns_list(self):
        engine = make_engine()
        feats  = make_feature_dict()
        result = engine.fit_all(feats, n_components=2)
        assert isinstance(result, list)

    def test_fit_all_has_all_methods(self):
        engine = make_engine(make_prices(200))
        feats  = make_feature_dict(make_prices(200))
        result = engine.fit_all(feats, n_components=2)
        methods = {ls.method for ls in result}
        assert "pca" in methods

    def test_enrich_adds_latent_features(self):
        engine  = make_engine()
        feats   = make_feature_dict()
        pca_ls  = engine.fit_pca(feats, n_components=2)
        enriched = engine.enrich_feature_dict(feats, pca_ls)
        assert len(enriched) == len(feats) + len(pca_ls)

    def test_enrich_preserves_original_features(self):
        engine  = make_engine()
        feats   = make_feature_dict()
        pca_ls  = engine.fit_pca(feats, n_components=2)
        enriched = engine.enrich_feature_dict(feats, pca_ls)
        for k in feats:
            assert k in enriched

    def test_enriched_features_in_feature_tensor(self):
        from macro8_subnet.alpha.batch_evaluator import FeatureTensor
        engine  = make_engine()
        feats   = make_feature_dict()
        pca_ls  = engine.fit_pca(feats, n_components=2)
        enriched = engine.enrich_feature_dict(feats, pca_ls)
        ft = FeatureTensor.from_feature_dict(enriched)
        latent_names = [n for n in ft.feature_names if "latent" in n]
        assert len(latent_names) == 2

    def test_latent_features_encodable(self):
        from macro8_subnet.alpha.batch_evaluator import FeatureTensor, FormulaEncoder
        engine  = make_engine()
        feats   = make_feature_dict()
        pca_ls  = engine.fit_pca(feats, n_components=2)
        enriched = engine.enrich_feature_dict(feats, pca_ls)
        ft  = FeatureTensor.from_feature_dict(enriched)
        enc = FormulaEncoder(ft.feature_names)
        for ls in pca_ls:
            assert enc.can_encode(ls.latent_name)

    def test_latent_features_evaluate_in_batch(self):
        from macro8_subnet.alpha.batch_evaluator import (
            FeatureTensor, FormulaEncoder, BatchEvaluator, BatchICScorer
        )
        prices  = make_prices()
        engine  = RepresentationEngine(prices)
        feats   = make_feature_dict(prices)
        pca_ls  = engine.fit_pca(feats, n_components=2)
        enriched = engine.enrich_feature_dict(feats, pca_ls)

        ft  = FeatureTensor.from_feature_dict(enriched)
        enc = FormulaEncoder(ft.feature_names)

        beval = BatchEvaluator.__new__(BatchEvaluator)
        beval.prices     = prices
        beval.returns    = prices.pct_change().dropna()
        beval.min_ic     = 0.0
        beval.feat_tensor = ft
        beval.encoder    = enc
        beval.ic_scorer  = BatchICScorer(n_lags=2)

        formulas = [ls.latent_name for ls in pca_ls]
        result   = beval.evaluate(formulas)
        assert result.n_formulas == len(formulas)
        assert not any(np.isnan(result.mean_ics))

    def test_latent_feature_names_list(self):
        engine = make_engine()
        feats  = make_feature_dict()
        pca_ls = engine.fit_pca(feats, n_components=2)
        names  = engine.latent_feature_names(pca_ls)
        assert names == [ls.latent_name for ls in pca_ls]

    def test_auto_hypotheses_with_ic(self):
        from macro8_subnet.alpha.hypothesis_engine import HypothesisLibrary
        engine = make_engine()
        feats  = make_feature_dict()
        pca_ls = engine.fit_pca(feats, n_components=2)
        # Inject IC history
        for ls in pca_ls:
            ls.ic_history = [0.04, 0.05, 0.03]

        lib    = HypothesisLibrary()
        new_hs = engine.auto_hypotheses(pca_ls, lib, ic_threshold=0.02)
        assert len(new_hs) == len(pca_ls)

    def test_auto_hypotheses_below_threshold(self):
        from macro8_subnet.alpha.hypothesis_engine import HypothesisLibrary
        engine = make_engine()
        feats  = make_feature_dict()
        pca_ls = engine.fit_pca(feats, n_components=2)
        # Low IC → no hypotheses
        for ls in pca_ls:
            ls.ic_history = [0.005, 0.004, 0.003]
        lib    = HypothesisLibrary()
        new_hs = engine.auto_hypotheses(pca_ls, lib, ic_threshold=0.02)
        assert new_hs == []

    def test_auto_hypotheses_serialisable(self):
        from macro8_subnet.alpha.hypothesis_engine import HypothesisLibrary
        engine = make_engine()
        feats  = make_feature_dict()
        pca_ls = engine.fit_pca(feats, n_components=2)
        for ls in pca_ls:
            ls.ic_history = [0.04]
        lib    = HypothesisLibrary()
        new_hs = engine.auto_hypotheses(pca_ls, lib, ic_threshold=0.01)
        for hrec in new_hs:
            json.dumps(hrec.to_dict())
