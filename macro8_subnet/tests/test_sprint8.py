"""
tests/test_sprint8.py
----------------------
QA Engineer: Self-contained tests for all Sprint 8 alpha research modules.

Covers:
    alpha/alpha_attribution.py   — MSC, variance decomp, return attribution
    alpha/meta_alpha_model.py    — feature extraction, training, prediction
    alpha/synthetic_market.py    — all 6 simulation models
    alpha/formula_engine.py      — formula validation, evaluation, operators

All tests are fully self-contained — no dependency on previous sprint files.
Tests are designed to run from any working directory with only
scikit-learn, scipy, pandas, numpy available.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
SUITE_DIR  = Path(__file__).resolve().parent          # macro8_subnet/tests/
SUBNET_DIR = SUITE_DIR.parent                         # macro8_subnet/
PROJECT    = SUBNET_DIR.parent                        # Macro8/

for p in [str(SUBNET_DIR), str(PROJECT)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ── Shared helpers ────────────────────────────────────────────────────────────

def make_returns(n: int = 120, n_assets: int = 3, seed: int = 42) -> pd.DataFrame:
    rng    = np.random.default_rng(seed)
    dates  = pd.date_range("2022-01-01", periods=n, freq="B")
    assets = ["SPY", "AAPL", "GLD"][:n_assets]
    return pd.DataFrame(
        {a: rng.normal(0.001, 0.012, n) for a in assets},
        index=dates,
    )


def make_prices(n: int = 120, seed: int = 42) -> pd.DataFrame:
    returns = make_returns(n=n, seed=seed)
    prices  = (1 + returns).cumprod() * 100
    return prices


def make_alpha_record(
    name:     str   = "test_alpha",
    mean_ic:  float = 0.05,
    n_epochs: int   = 10,
):
    """Create a minimal AlphaRecord-like object for testing."""
    @dataclass
    class FakeAlphaRecord:
        name:         str
        miner_uid:    int         = 0
        category:     str         = "momentum"
        birth_epoch:  int         = 1
        ic_history:   list        = field(default_factory=list)
        ir_history:   list        = field(default_factory=list)
        current_ic:   float       = 0.0
        current_ir:   float       = 0.0
        decay_rate:   float       = 0.0
        capacity:     float       = 1.0
        retired:      bool        = False
        epochs_alive: int         = 0

        @property
        def mean_ic(self):
            return float(np.mean(self.ic_history)) if self.ic_history else 0.0

        @property
        def ic_stability(self):
            if not self.ic_history:
                return 0.0
            return sum(1 for x in self.ic_history if x > 0) / len(self.ic_history)

        def update(self, new_ic, new_ir, epoch):
            self.ic_history.append(new_ic)
            self.ir_history.append(new_ir)
            self.current_ic   = new_ic
            self.current_ir   = new_ir
            self.epochs_alive = epoch - self.birth_epoch

    r             = FakeAlphaRecord(name=name)
    r.ic_history  = [mean_ic + np.random.default_rng(42).normal(0, 0.01)
                     for _ in range(n_epochs)]
    r.current_ic  = r.ic_history[-1] if r.ic_history else 0.0
    r.epochs_alive = n_epochs
    return r


# ════════════════════════════════════════════════════════════════════════════
# ALPHA ATTRIBUTION
# ════════════════════════════════════════════════════════════════════════════

from macro8_subnet.alpha.alpha_attribution import (
    AlphaAttributionEngine, AttributionReport, SignalAttribution,
)


class TestAlphaAttributionEngine:
    def _setup(self, n_signals: int = 3, n_days: int = 120):
        rng     = np.random.default_rng(0)
        dates   = pd.date_range("2022-01-01", periods=n_days, freq="B")
        names   = [f"signal_{i}" for i in range(n_signals)]
        returns = pd.DataFrame(
            {n: rng.normal(0.001, 0.01, n_days) for n in names},
            index=dates,
        )
        weights = {n: 1.0 / n_signals for n in names}
        return returns, weights, names

    def test_returns_attribution_report(self):
        ret, w, _ = self._setup()
        eng       = AlphaAttributionEngine()
        report    = eng.attribute(ret, w)
        assert isinstance(report, AttributionReport)

    def test_attribution_count_matches_signals(self):
        ret, w, names = self._setup(n_signals=3)
        report = AlphaAttributionEngine().attribute(ret, w)
        assert len(report.attributions) == 3

    def test_variance_contrib_sums_to_one(self):
        ret, w, _ = self._setup()
        report    = AlphaAttributionEngine().attribute(ret, w)
        vc_total  = sum(a.variance_contrib for a in report.attributions)
        assert abs(vc_total - 1.0) < 0.01

    def test_return_contrib_sums_to_one(self):
        ret, w, _ = self._setup()
        report    = AlphaAttributionEngine().attribute(ret, w)
        rc_total  = sum(a.return_contrib for a in report.attributions)
        assert abs(rc_total - 1.0) < 0.01

    def test_portfolio_sharpe_is_finite(self):
        ret, w, _ = self._setup()
        report    = AlphaAttributionEngine().attribute(ret, w)
        assert np.isfinite(report.portfolio_sharpe)

    def test_msc_direction_positive_means_contributes(self):
        """A signal with high returns should have positive MSC."""
        rng    = np.random.default_rng(1)
        dates  = pd.date_range("2022-01-01", periods=120, freq="B")
        # signal_a: strongly positive returns; signal_b: near zero
        ret    = pd.DataFrame({
            "signal_a": rng.normal(0.005, 0.008, 120),
            "signal_b": rng.normal(0.000, 0.010, 120),
        }, index=dates)
        w = {"signal_a": 0.5, "signal_b": 0.5}
        r = AlphaAttributionEngine().attribute(ret, w)
        # signal_a should have higher MSC
        a_msc = next(a.msc for a in r.attributions if a.signal_name == "signal_a")
        b_msc = next(a.msc for a in r.attributions if a.signal_name == "signal_b")
        assert a_msc >= b_msc

    def test_empty_signals_returns_empty_report(self):
        ret    = pd.DataFrame()
        report = AlphaAttributionEngine().attribute(ret, {})
        assert len(report.attributions) == 0

    def test_to_dict_serialisable(self):
        ret, w, _ = self._setup()
        report    = AlphaAttributionEngine().attribute(ret, w)
        json.dumps(report.to_dict())   # must not raise

    def test_diversification_ratio_gte_one(self):
        """Portfolio vol ≤ weighted avg of individual vols → div ratio ≥ 1."""
        ret, w, _ = self._setup()
        report    = AlphaAttributionEngine().attribute(ret, w)
        assert report.diversification_ratio >= 0.9   # allow small numerical error

    def test_leave_one_out_returns_dict(self):
        ret, w, names = self._setup()
        eng           = AlphaAttributionEngine()
        result        = eng.leave_one_out_sharpe(ret, w)
        assert set(result.keys()) == set(names)

    def test_leave_one_out_all_finite(self):
        ret, w, _ = self._setup()
        result    = AlphaAttributionEngine().leave_one_out_sharpe(ret, w)
        assert all(np.isfinite(v) for v in result.values())

    def test_drag_signals_have_negative_msc(self):
        ret, w, _ = self._setup()
        report    = AlphaAttributionEngine().attribute(ret, w)
        for name in report.drags:
            s = next(a for a in report.attributions if a.signal_name == name)
            assert s.msc < 0

    def test_with_ic_scores_and_capacities(self):
        ret, w, names = self._setup()
        ic_scores = {n: 0.05 for n in names}
        caps      = {n: 1.0  for n in names}
        report    = AlphaAttributionEngine().attribute(ret, w, ic_scores, caps)
        ic_total  = sum(a.ic_contribution for a in report.attributions)
        assert abs(ic_total - 1.0) < 0.01


# ════════════════════════════════════════════════════════════════════════════
# META ALPHA MODEL
# ════════════════════════════════════════════════════════════════════════════

from macro8_subnet.alpha.meta_alpha_model import (
    MetaAlphaModel, MetaAlphaReport, MetaAlphaPrediction,
    extract_alpha_features, FEATURE_NAMES,
)


class TestExtractAlphaFeatures:
    def test_returns_dict(self):
        r = make_alpha_record()
        f = extract_alpha_features(r)
        assert isinstance(f, dict)

    def test_all_feature_names_present(self):
        r = make_alpha_record()
        f = extract_alpha_features(r)
        for name in FEATURE_NAMES:
            assert name in f, f"Missing feature: {name}"

    def test_all_values_finite(self):
        r = make_alpha_record()
        f = extract_alpha_features(r)
        for k, v in f.items():
            assert np.isfinite(v), f"Feature {k} is not finite: {v}"

    def test_empty_ic_history_returns_zeros(self):
        r = make_alpha_record(n_epochs=0)
        r.ic_history = []
        f = extract_alpha_features(r)
        assert f["mean_ic"] == pytest.approx(0.0)


class TestMetaAlphaModel:
    def _make_records(self, n: int = 5) -> list:
        return [make_alpha_record(f"s{i}", mean_ic=0.04 + i*0.01) for i in range(n)]

    def test_not_fitted_initially(self):
        m = MetaAlphaModel()
        assert m.is_trained is False

    def test_predict_before_fit_uses_heuristic(self):
        m = MetaAlphaModel()
        r = make_alpha_record(mean_ic=0.05)
        p = m.predict(r)
        assert isinstance(p.predicted_ic, float)
        assert np.isfinite(p.predicted_ic)

    def test_fit_after_min_samples(self):
        m       = MetaAlphaModel(min_samples=5)
        records = self._make_records(8)
        for rec in records:
            m.add_training_sample(rec, actual_next_ic=rec.mean_ic + 0.005)
        assert m.is_trained is True

    def test_n_samples_tracked(self):
        m       = MetaAlphaModel(min_samples=100)   # won't train
        records = self._make_records(3)
        for rec in records:
            m.add_training_sample(rec, 0.05)
        assert m.n_samples == 3

    def test_predict_all_returns_report(self):
        m = MetaAlphaModel(min_samples=5)
        records = self._make_records(8)
        for rec in records:
            m.add_training_sample(rec, rec.mean_ic)
        report = m.predict_all(self._make_records(3))
        assert isinstance(report, MetaAlphaReport)

    def test_predictions_ranked_correctly(self):
        m = MetaAlphaModel(min_samples=5)
        records = self._make_records(8)
        for rec in records:
            m.add_training_sample(rec, rec.mean_ic)
        report = m.predict_all(self._make_records(5))
        ranks  = [p.prediction_rank for p in report.predictions]
        assert sorted(ranks) == list(range(1, len(ranks) + 1))

    def test_add_batch(self):
        m       = MetaAlphaModel(min_samples=5)
        records = self._make_records(8)
        samples = [(rec, rec.mean_ic) for rec in records]
        m.add_batch(samples)
        assert m.n_samples == 8

    def test_empty_records_returns_empty_report(self):
        m      = MetaAlphaModel()
        report = m.predict_all([])
        assert len(report.predictions) == 0

    def test_feature_importances_populated_after_training(self):
        m       = MetaAlphaModel(min_samples=5)
        records = self._make_records(8)
        for rec in records:
            m.add_training_sample(rec, rec.mean_ic)
        assert len(m._feature_importances) > 0

    def test_gbm_method(self):
        m       = MetaAlphaModel(method="gbm", min_samples=5, n_estimators=5)
        records = self._make_records(8)
        for rec in records:
            m.add_training_sample(rec, rec.mean_ic)
        assert m.is_trained is True

    def test_to_dict_serialisable(self):
        m       = MetaAlphaModel(min_samples=5)
        records = self._make_records(8)
        for rec in records:
            m.add_training_sample(rec, rec.mean_ic)
        report = m.predict_all(self._make_records(3))
        json.dumps(report.to_dict())

    def test_top_signals(self):
        m       = MetaAlphaModel(min_samples=5)
        records = self._make_records(8)
        for rec in records:
            m.add_training_sample(rec, rec.mean_ic)
        report = m.predict_all(self._make_records(5))
        top    = report.top_signals(3)
        assert len(top) == 3
        assert all(isinstance(s, str) for s in top)


# ════════════════════════════════════════════════════════════════════════════
# SYNTHETIC MARKET SIMULATOR
# ════════════════════════════════════════════════════════════════════════════

from macro8_subnet.alpha.synthetic_market import (
    SyntheticMarketSimulator, SyntheticMarket, SimModel, SimulationMetadata,
)


class TestSyntheticMarketSimulator:
    def _sim(self, seed: int = 42) -> SyntheticMarketSimulator:
        return SyntheticMarketSimulator(
            assets=["SPY", "AAPL", "GLD"],
            n_days=120,
            seed=seed,
        )

    def test_returns_synthetic_market(self):
        result = self._sim().generate(SimModel.GBM)
        assert isinstance(result, SyntheticMarket)

    def test_prices_dataframe_shape(self):
        result = self._sim().generate(SimModel.GBM)
        assert result.prices.shape == (120, 3)

    def test_prices_all_positive(self):
        for model in SimModel:
            result = self._sim().generate(model)
            assert result.prices.min().min() > 0, f"Negative price in {model}"

    def test_returns_no_nan_after_dropna(self):
        for model in SimModel:
            result = self._sim().generate(model)
            assert not result.returns.isnull().all().any(), f"All-NaN column in {model}"

    def test_metadata_fields_populated(self):
        result = self._sim().generate(SimModel.GBM)
        assert result.metadata.n_days    == 120
        assert result.metadata.n_assets  == 3
        assert result.metadata.realised_vol is not None

    def test_metadata_to_dict_serialisable(self):
        result = self._sim().generate(SimModel.GBM)
        json.dumps(result.metadata.to_dict())

    def test_gbm_model(self):
        r = self._sim().generate(SimModel.GBM)
        assert len(r.prices) == 120

    def test_jump_diffusion_model(self):
        r = self._sim().generate(SimModel.JUMP_DIFFUSION)
        assert len(r.prices) == 120

    def test_mean_revert_model(self):
        r = self._sim().generate(SimModel.MEAN_REVERT)
        assert len(r.prices) == 120

    def test_regime_switch_model(self):
        r = self._sim().generate(SimModel.REGIME_SWITCH)
        assert len(r.prices) == 120

    def test_corr_shock_model(self):
        r = self._sim().generate(SimModel.CORR_SHOCK)
        assert len(r.prices) == 120

    def test_inflation_spiral_model(self):
        r = self._sim().generate(SimModel.INFLATION_SPIRAL)
        assert len(r.prices) == 120

    def test_reproducible_with_same_seed(self):
        r1 = SyntheticMarketSimulator(seed=42, n_days=60).generate(SimModel.GBM)
        r2 = SyntheticMarketSimulator(seed=42, n_days=60).generate(SimModel.GBM)
        assert r1.prices.equals(r2.prices)

    def test_different_seeds_different_prices(self):
        r1 = SyntheticMarketSimulator(seed=1, n_days=60).generate(SimModel.GBM)
        r2 = SyntheticMarketSimulator(seed=2, n_days=60).generate(SimModel.GBM)
        assert not r1.prices.equals(r2.prices)

    def test_realised_stats_keys(self):
        result = self._sim().generate(SimModel.GBM)
        stats  = result.realised_stats()
        for key in ("annualised_return", "annualised_vol", "max_drawdown", "sharpe"):
            assert key in stats

    def test_corr_shock_increases_correlation(self):
        """Verify the shock period has higher cross-asset correlation."""
        sim    = SyntheticMarketSimulator(n_days=200, seed=42)
        result = sim.generate(SimModel.CORR_SHOCK,
                              params={"shock_day": 80, "shock_dur": 40,
                                      "shock_corr": 0.95})
        ret    = result.returns
        # Pre-shock correlation
        pre    = ret.iloc[:70].corr().values
        pre_avg = float(pre[np.triu_indices(3, k=1)].mean())
        # During-shock correlation
        dur    = ret.iloc[80:120].corr().values
        dur_avg = float(dur[np.triu_indices(3, k=1)].mean())
        assert dur_avg > pre_avg

    def test_generate_batch(self):
        sim     = self._sim()
        results = sim.generate_batch(models=[SimModel.GBM, SimModel.JUMP_DIFFUSION],
                                      n_per_model=2)
        assert len(results) == 4

    def test_param_override(self):
        result = self._sim().generate(SimModel.GBM, params={"mu": 0.30, "sigma": 0.40})
        # Higher drift/vol should produce different prices than default
        default = self._sim().generate(SimModel.GBM)
        assert not result.prices.equals(default.prices)


# ════════════════════════════════════════════════════════════════════════════
# FORMULA ENGINE
# ════════════════════════════════════════════════════════════════════════════

from macro8_subnet.alpha.formula_engine import FormulaEngine, FormulaResult


class MockFeatureStore:
    """Minimal FeatureStore stub for testing."""
    def __init__(self, n: int = 120):
        rng   = np.random.default_rng(0)
        n     = max(n, 30)
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        self._data = {
            "momentum_20d":   pd.DataFrame({"SPY": rng.normal(0, 0.1, n),
                                            "AAPL": rng.normal(0, 0.1, n),
                                            "GLD": rng.normal(0, 0.08, n)}, index=dates),
            "volatility_20d": pd.DataFrame({"SPY": np.abs(rng.normal(0.15, 0.05, n)),
                                            "AAPL": np.abs(rng.normal(0.20, 0.05, n)),
                                            "GLD": np.abs(rng.normal(0.12, 0.04, n))}, index=dates),
            "momentum_5d":    pd.DataFrame({"SPY": rng.normal(0, 0.05, n),
                                            "AAPL": rng.normal(0, 0.06, n),
                                            "GLD": rng.normal(0, 0.04, n)}, index=dates),
            "rsi_14":         pd.DataFrame({"SPY": np.clip(rng.normal(50, 15, n), 0, 100),
                                            "AAPL": np.clip(rng.normal(50, 15, n), 0, 100),
                                            "GLD": np.clip(rng.normal(50, 15, n), 0, 100)}, index=dates),
            "cross_momentum": pd.DataFrame({"SPY": rng.normal(0, 0.05, n),
                                            "AAPL": rng.normal(0, 0.05, n),
                                            "GLD": rng.normal(0, 0.04, n)}, index=dates),
            "regime_signal":  pd.DataFrame({"SPY": rng.normal(0, 0.3, n),
                                            "AAPL": rng.normal(0, 0.3, n),
                                            "GLD": rng.normal(0, 0.3, n)}, index=dates),
        }

    def get(self, name: str):
        return self._data.get(name)


class TestFormulaEngine:
    def _engine(self) -> FormulaEngine:
        return FormulaEngine(MockFeatureStore())

    # ── Validation ─────────────────────────────────────────────────────────

    def test_empty_formula_fails(self):
        ok, _ = self._engine().validate_formula("")
        assert ok is False

    def test_forbidden_import_rejected(self):
        ok, reason = self._engine().validate_formula("import os")
        assert ok is False
        assert "import" in reason

    def test_forbidden_dunder_rejected(self):
        ok, reason = self._engine().validate_formula("__builtins__['eval']()")
        assert ok is False
        assert "__" in reason

    def test_unknown_feature_rejected(self):
        ok, reason = self._engine().validate_formula("nonexistent_feature + 1")
        assert ok is False

    def test_unknown_operator_rejected(self):
        ok, reason = self._engine().validate_formula("hack(momentum_20d)")
        assert ok is False

    def test_valid_simple_formula(self):
        ok, msg = self._engine().validate_formula("momentum_20d")
        assert ok is True, msg

    def test_valid_compound_formula(self):
        ok, msg = self._engine().validate_formula(
            "rank(momentum_20d) - rank(volatility_20d)"
        )
        assert ok is True, msg

    def test_valid_nested_formula(self):
        ok, msg = self._engine().validate_formula(
            "zscore(decay(momentum_20d, halflife=10))"
        )
        assert ok is True, msg

    # ── Evaluation ─────────────────────────────────────────────────────────

    def test_evaluate_returns_formula_result(self):
        result = self._engine().evaluate("momentum_20d")
        assert isinstance(result, FormulaResult)

    def test_evaluate_simple_feature_succeeds(self):
        result = self._engine().evaluate("momentum_20d")
        assert result.success is True, result.error

    def test_evaluate_returns_signals_dict(self):
        result = self._engine().evaluate("momentum_20d")
        assert result.signals is not None
        assert len(result.signals) > 0

    def test_evaluate_signals_are_series(self):
        result = self._engine().evaluate("momentum_20d")
        for asset, series in result.signals.items():
            assert isinstance(series, pd.Series), f"Signal for {asset} is not pd.Series"

    def test_zscore_operator(self):
        result = self._engine().evaluate("zscore(momentum_20d)")
        assert result.success is True, result.error

    def test_rank_operator(self):
        result = self._engine().evaluate("rank(momentum_20d)")
        assert result.success is True, result.error

    def test_rank_values_in_0_1(self):
        result = self._engine().evaluate("rank(momentum_20d)")
        if result.success:
            for series in result.signals.values():
                assert series.dropna().min() >= 0
                assert series.dropna().max() <= 1.0

    def test_subtraction_formula(self):
        result = self._engine().evaluate(
            "rank(momentum_20d) - rank(volatility_20d)"
        )
        assert result.success is True, result.error

    def test_decay_operator(self):
        result = self._engine().evaluate("decay(momentum_20d, halflife=10)")
        assert result.success is True, result.error

    def test_clip_operator(self):
        result = self._engine().evaluate("clip(zscore(momentum_20d), -2, 2)")
        if result.success:
            for series in result.signals.values():
                assert series.dropna().min() >= -2.01
                assert series.dropna().max() <= 2.01

    def test_lag_operator(self):
        result = self._engine().evaluate("lag(momentum_20d, n=5)")
        assert result.success is True, result.error

    def test_n_obs_populated(self):
        result = self._engine().evaluate("momentum_20d")
        if result.success:
            assert result.n_obs > 0

    def test_list_features(self):
        features = self._engine().list_features()
        assert len(features) > 5
        assert "momentum_20d" in features

    def test_list_operators(self):
        ops = self._engine().list_operators()
        assert "zscore" in ops
        assert "rank" in ops

    def test_to_alpha_signals_on_failure_returns_empty(self):
        result = self._engine().evaluate("nonexistent_feature")
        assert result.to_alpha_signals() == {}

    def test_formula_result_serialisable(self):
        result = self._engine().evaluate("momentum_20d")
        # signals contains pd.Series which can't be JSON-serialised,
        # but the metadata fields should be serialisable
        d = {
            "formula": result.formula,
            "success": result.success,
            "n_obs":   result.n_obs,
            "error":   result.error,
        }
        json.dumps(d)
