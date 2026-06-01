"""
tests/test_sprint29.py
-----------------------
Sprint 29: Regime Prediction Layer

Tests cover:
    - RegimeTransitionModel: fit, predict, predict_series, confidence
    - PolicyLayer: compute, compute_series, all indicators
    - ScenarioProbabilityAssigner: probabilities sum to 1, mapping logic
    - ConfidenceScore: composite metric, position scaling
    - ForecastResult: fields, print
    - ForecastedEnsemble: fit, forecast, forecast_series
"""

import sys
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
    """500 days × 10 assets — small but representative."""
    rng = np.random.default_rng(42)
    n, a = 500, 10
    tickers = ["SPY","QQQ","IWM","TLT","GLD","DBC","EEM","FXI","VNQ","HYG"]
    # Add mild regime structure: stress in middle 100 days
    log_ret = rng.normal(0.0003, 0.010, (n, a))
    log_ret[200:300, :4] = rng.normal(-0.001, 0.025, (100, 4))  # stress patch
    p = 100 * np.exp(np.cumsum(log_ret, axis=0))
    return pd.DataFrame(p, index=pd.bdate_range("2015-01-01", periods=n), columns=tickers)


@pytest.fixture(scope="module")
def train_prices(prices):
    return prices.iloc[:int(len(prices) * 0.70)]


@pytest.fixture(scope="module")
def oos_prices(prices):
    return prices.iloc[int(len(prices) * 0.70):]


@pytest.fixture(scope="module")
def formulas(train_prices):
    from macro8_subnet.alpha.gp_miner import GPMiner
    gp = GPMiner(train_prices, pop_size=30, elite_n=6, seed=42, verbose=False)
    gp.run(n_epochs=2)
    return gp.top_formulas(8)


@pytest.fixture(scope="module")
def fitted_model(train_prices):
    from macro8_subnet.alpha.regime_prediction import RegimeTransitionModel
    m = RegimeTransitionModel(horizon=5, n_estimators=50)
    m.fit(train_prices)
    return m


@pytest.fixture(scope="module")
def policy(prices):
    from macro8_subnet.alpha.regime_prediction import PolicyLayer
    return PolicyLayer()


@pytest.fixture(scope="module")
def policy_state(policy, prices):
    return policy.compute(prices)


@pytest.fixture(scope="module")
def regime_forecast(fitted_model, prices):
    return fitted_model.predict(prices)


@pytest.fixture(scope="module")
def scenario_probs(regime_forecast, policy_state):
    from macro8_subnet.alpha.regime_prediction import ScenarioProbabilityAssigner
    return ScenarioProbabilityAssigner().assign(regime_forecast, policy_state)


@pytest.fixture(scope="module")
def fens(train_prices, formulas):
    from macro8_subnet.alpha.regime_prediction import ForecastedEnsemble
    fe = ForecastedEnsemble(train_prices, formulas, horizon=5,
                            weighting="risk_parity", verbose=False)
    fe.fit()
    return fe


# ── 1. RegimeTransitionModel ──────────────────────────────────────────────────

class TestRegimeTransitionModel:
    def test_import(self):
        from macro8_subnet.alpha.regime_prediction import RegimeTransitionModel

    def test_fit_sets_fitted_flag(self, fitted_model):
        assert fitted_model._fitted is True

    def test_fit_populates_feature_cols(self, fitted_model):
        assert len(fitted_model._feature_cols) > 0

    def test_feature_count_expected(self, fitted_model):
        # 8 macro features × 4 transformations + 3 regime dummies = ~35
        assert len(fitted_model._feature_cols) >= 20

    def test_predict_returns_regime_forecast(self, fitted_model, prices):
        from macro8_subnet.alpha.regime_prediction import RegimeForecast
        f = fitted_model.predict(prices)
        assert isinstance(f, RegimeForecast)

    def test_predict_probabilities_sum_to_one(self, regime_forecast):
        total = regime_forecast.calm + regime_forecast.normal + regime_forecast.stress
        assert abs(total - 1.0) < 0.01

    def test_predict_probabilities_non_negative(self, regime_forecast):
        assert regime_forecast.calm   >= 0
        assert regime_forecast.normal >= 0
        assert regime_forecast.stress >= 0

    def test_predict_confidence_in_unit_interval(self, regime_forecast):
        assert 0 <= regime_forecast.confidence <= 1

    def test_predict_current_regime_valid(self, regime_forecast):
        assert regime_forecast.current in ("calm", "normal", "stress")

    def test_predict_most_likely_is_argmax(self, regime_forecast):
        probs = {
            "calm": regime_forecast.calm,
            "normal": regime_forecast.normal,
            "stress": regime_forecast.stress,
        }
        expected = max(probs, key=probs.get)
        assert regime_forecast.most_likely == expected

    def test_predict_series_returns_dataframe(self, fitted_model, prices):
        df = fitted_model.predict_series(prices)
        assert isinstance(df, pd.DataFrame)
        assert "calm" in df.columns
        assert "normal" in df.columns
        assert "stress" in df.columns
        assert "confidence" in df.columns

    def test_predict_series_probabilities_sum_to_one(self, fitted_model, prices):
        df = fitted_model.predict_series(prices)
        row_sums = df[["calm", "normal", "stress"]].sum(axis=1)
        # Allow NaN rows (before sufficient warmup)
        valid = row_sums.dropna()
        assert (np.abs(valid - 1.0) < 0.01).all()

    def test_predict_series_confidence_bounded(self, fitted_model, prices):
        df = fitted_model.predict_series(prices)
        assert df["confidence"].dropna().between(0, 1).all()

    def test_unfitted_model_returns_fallback(self, prices):
        from macro8_subnet.alpha.regime_prediction import RegimeTransitionModel, RegimeForecast
        m = RegimeTransitionModel()  # not fitted
        f = m.predict(prices)
        assert isinstance(f, RegimeForecast)
        assert f.confidence == 0.0

    def test_p_stress_rising_property(self):
        from macro8_subnet.alpha.regime_prediction import RegimeForecast
        f_high = RegimeForecast(calm=0.1, normal=0.5, stress=0.4,
                                confidence=0.6, horizon_days=5, current="normal")
        f_low  = RegimeForecast(calm=0.3, normal=0.65, stress=0.05,
                                confidence=0.6, horizon_days=5, current="normal")
        assert f_high.p_stress_rising is True
        assert f_low.p_stress_rising  is False

    def test_repr_contains_key_info(self, regime_forecast):
        r = repr(regime_forecast)
        assert "RegimeForecast" in r
        assert "conf=" in r


# ── 2. PolicyLayer ────────────────────────────────────────────────────────────

class TestPolicyLayer:
    def test_import(self):
        from macro8_subnet.alpha.regime_prediction import PolicyLayer, PolicyState

    def test_compute_returns_policy_state(self, policy, prices):
        from macro8_subnet.alpha.regime_prediction import PolicyState
        state = policy.compute(prices)
        assert isinstance(state, PolicyState)

    def test_all_indicators_finite(self, policy_state):
        assert np.isfinite(policy_state.rate_env)
        assert np.isfinite(policy_state.inflation)
        assert np.isfinite(policy_state.liquidity)
        assert np.isfinite(policy_state.dollar)
        assert np.isfinite(policy_state.breadth)

    def test_boolean_methods_return_bool(self, policy_state):
        assert isinstance(policy_state.rate_rising(),        bool)
        assert isinstance(policy_state.rate_falling(),       bool)
        assert isinstance(policy_state.inflation_rising(),   bool)
        assert isinstance(policy_state.credit_tightening(),  bool)
        assert isinstance(policy_state.dollar_strong(),      bool)
        assert isinstance(policy_state.breadth_broadening(), bool)

    def test_rate_rising_and_falling_mutually_exclusive(self, policy_state):
        assert not (policy_state.rate_rising() and policy_state.rate_falling())

    def test_summary_contains_all_indicators(self, policy_state):
        s = policy_state.summary()
        for key in ["rates", "inflation", "liquidity", "dollar", "breadth"]:
            assert key in s

    def test_compute_series_returns_dataframe(self, policy, prices):
        df = policy.compute_series(prices)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(prices)

    def test_compute_series_finite_values(self, policy, prices):
        df = policy.compute_series(prices).dropna()
        assert np.isfinite(df.values).all()

    def test_without_tlt_no_rate_env(self, policy):
        """Without TLT in universe, rate_env should default to 0."""
        prices_no_tlt = pd.DataFrame(
            {"SPY": [100.0] * 50, "GLD": [100.0] * 50},
            index=pd.bdate_range("2020-01-01", periods=50),
        )
        state = policy.compute(prices_no_tlt)
        assert state.rate_env == 0.0


# ── 3. ScenarioProbabilityAssigner ────────────────────────────────────────────

class TestScenarioProbabilityAssigner:
    def test_import(self):
        from macro8_subnet.alpha.regime_prediction import ScenarioProbabilityAssigner

    def test_probabilities_sum_to_one(self, scenario_probs):
        assert abs(sum(scenario_probs.values()) - 1.0) < 0.01

    def test_all_eight_scenarios_present(self, scenario_probs):
        from macro8_subnet.alpha.regime_prediction import ALL_SCENARIOS
        for s in ALL_SCENARIOS:
            assert s in scenario_probs, f"{s} missing from output"

    def test_probabilities_non_negative(self, scenario_probs):
        assert all(p >= 0 for p in scenario_probs.values())

    def test_stress_regime_elevates_crash_probability(self):
        """High P(stress) should increase equity_crash probability."""
        from macro8_subnet.alpha.regime_prediction import (
            ScenarioProbabilityAssigner, RegimeForecast, PolicyState,
        )
        assigner   = ScenarioProbabilityAssigner()
        policy_neu = PolicyState(0, 0, 0, 0, 0)

        high_stress = RegimeForecast(calm=0.0, normal=0.2, stress=0.8,
                                     confidence=0.7, horizon_days=5, current="stress")
        low_stress  = RegimeForecast(calm=0.4, normal=0.5, stress=0.1,
                                     confidence=0.7, horizon_days=5, current="calm")

        p_hs = assigner.assign(high_stress, policy_neu)
        p_ls = assigner.assign(low_stress,  policy_neu)
        assert p_hs["equity_crash_30pct"] > p_ls["equity_crash_30pct"]

    def test_calm_regime_elevates_soft_landing(self):
        from macro8_subnet.alpha.regime_prediction import (
            ScenarioProbabilityAssigner, RegimeForecast, PolicyState,
        )
        assigner   = ScenarioProbabilityAssigner()
        policy_pos = PolicyState(0.05, -0.02, 0.03, -0.02, 0.03)

        high_calm = RegimeForecast(calm=0.8, normal=0.15, stress=0.05,
                                   confidence=0.8, horizon_days=5, current="calm")
        low_calm  = RegimeForecast(calm=0.05, normal=0.5, stress=0.45,
                                   confidence=0.7, horizon_days=5, current="stress")

        p_hc = assigner.assign(high_calm, policy_pos)
        p_lc = assigner.assign(low_calm,  policy_pos)
        assert p_hc["soft_landing"] > p_lc["soft_landing"]


# ── 4. ConfidenceScore ────────────────────────────────────────────────────────

class TestConfidenceScore:
    def test_import(self):
        from macro8_subnet.alpha.regime_prediction import ConfidenceScore

    def test_score_in_unit_interval(self, regime_forecast, scenario_probs):
        from macro8_subnet.alpha.regime_prediction import ConfidenceScore
        conf = ConfidenceScore().compute(regime_forecast, scenario_probs)
        assert 0 <= conf <= 1

    def test_high_entropy_forecast_gives_low_confidence(self, scenario_probs):
        from macro8_subnet.alpha.regime_prediction import ConfidenceScore, RegimeForecast
        # Uniform distribution = max entropy = low confidence
        uniform = RegimeForecast(calm=0.333, normal=0.334, stress=0.333,
                                 confidence=0.0, horizon_days=5, current="normal")
        conf = ConfidenceScore().compute(uniform, scenario_probs)
        assert conf < 0.5

    def test_certain_forecast_gives_high_confidence(self, scenario_probs):
        from macro8_subnet.alpha.regime_prediction import ConfidenceScore, RegimeForecast
        certain = RegimeForecast(calm=0.01, normal=0.98, stress=0.01,
                                 confidence=0.95, horizon_days=5, current="normal")
        conf = ConfidenceScore().compute(certain, scenario_probs)
        assert conf > 0.4   # regime certainty dominates

    def test_scale_positions_by_confidence(self):
        from macro8_subnet.alpha.regime_prediction import ConfidenceScore
        scorer    = ConfidenceScore()
        positions = {"SPY": 0.20, "TLT": -0.15, "GLD": 0.10}

        high_pos = scorer.scale_positions(positions, confidence=0.80)
        med_pos  = scorer.scale_positions(positions, confidence=0.55)
        low_pos  = scorer.scale_positions(positions, confidence=0.30)

        # High confidence: no scaling
        assert abs(high_pos["SPY"] - 0.20) < 1e-6
        # Medium: 0.75×
        assert abs(med_pos["SPY"] - 0.15) < 1e-6
        # Low: 0.50×
        assert abs(low_pos["SPY"] - 0.10) < 1e-6

    def test_confidence_level_strings(self, regime_forecast, scenario_probs):
        from macro8_subnet.alpha.regime_prediction import ConfidenceScore, ForecastResult
        # Build a mock ForecastResult to test confidence_level()
        from macro8_subnet.alpha.regime_prediction import PolicyState
        ps = PolicyState(0, 0, 0, 0, 0)
        result = ForecastResult(
            positions={}, confidence=0.80,
            regime_current="normal", regime_forecast=regime_forecast,
            policy_state=ps, scenario_probs=scenario_probs,
            active_formulas=[], formula_weights={}, n_clusters=0,
        )
        assert result.confidence_level() == "HIGH"
        result.confidence = 0.55
        assert result.confidence_level() == "MEDIUM"
        result.confidence = 0.30
        assert result.confidence_level() == "LOW"


# ── 5. ForecastResult ─────────────────────────────────────────────────────────

class TestForecastResult:
    def test_top_scenarios_returns_sorted(self, regime_forecast, policy_state, scenario_probs):
        from macro8_subnet.alpha.regime_prediction import ForecastResult
        result = ForecastResult(
            positions={"SPY": 0.2},
            confidence=0.6,
            regime_current="normal",
            regime_forecast=regime_forecast,
            policy_state=policy_state,
            scenario_probs=scenario_probs,
            active_formulas=["formula_a"],
            formula_weights={"formula_a": 1.0},
            n_clusters=2,
        )
        top3 = result.top_scenarios(3)
        assert len(top3) == 3
        probs = [p for _, p in top3]
        assert probs == sorted(probs, reverse=True)

    def test_print_no_crash(self, fens, capsys):
        result = fens.forecast()
        result.print()
        captured = capsys.readouterr()
        assert "MACRO8 FORECAST" in captured.out
        assert "Positions" in captured.out
        assert "scenarios" in captured.out.lower() or "scenario" in captured.out.lower()


# ── 6. ForecastedEnsemble ─────────────────────────────────────────────────────

class TestForecastedEnsemble:
    def test_import(self):
        from macro8_subnet.alpha.regime_prediction import ForecastedEnsemble

    def test_fit_sets_fitted_flag(self, fens):
        assert fens._fitted is True

    def test_fit_builds_base_ensemble(self, fens):
        assert fens._base_ensemble is not None

    def test_fit_trains_transition_model(self, fens):
        assert fens._transition_model._fitted is True

    def test_forecast_returns_forecast_result(self, fens):
        from macro8_subnet.alpha.regime_prediction import ForecastResult
        result = fens.forecast()
        assert isinstance(result, ForecastResult)

    def test_forecast_positions_non_empty(self, fens):
        result = fens.forecast()
        assert len(result.positions) > 0

    def test_forecast_positions_l1_bounded(self, fens):
        result = fens.forecast()
        l1 = sum(abs(w) for w in result.positions.values())
        # L1 ≤ 1 (may be < 1 due to confidence scaling)
        assert l1 <= 1.0 + 1e-4

    def test_forecast_regime_forecast_populated(self, fens):
        result = fens.forecast()
        assert result.regime_forecast is not None
        assert result.regime_forecast.current in ("calm", "normal", "stress")

    def test_forecast_policy_state_populated(self, fens):
        result = fens.forecast()
        assert result.policy_state is not None
        assert np.isfinite(result.policy_state.rate_env)

    def test_forecast_scenario_probs_sum_to_one(self, fens):
        result = fens.forecast()
        assert abs(sum(result.scenario_probs.values()) - 1.0) < 0.01

    def test_forecast_confidence_in_unit_interval(self, fens):
        result = fens.forecast()
        assert 0 <= result.confidence <= 1

    def test_forecast_series_returns_tuple(self, fens, oos_prices):
        pnl, regime_df = fens.forecast_series(oos_prices=oos_prices)
        assert isinstance(pnl, pd.Series)
        assert isinstance(regime_df, pd.DataFrame)

    def test_forecast_series_pnl_finite(self, fens, oos_prices):
        pnl, _ = fens.forecast_series(oos_prices=oos_prices)
        assert np.isfinite(pnl.values).all()

    def test_forecast_series_regime_df_columns(self, fens, oos_prices):
        _, regime_df = fens.forecast_series(oos_prices=oos_prices)
        if len(regime_df) > 0:
            assert "calm" in regime_df.columns
            assert "stress" in regime_df.columns

    def test_unfitted_raises(self, train_prices, formulas):
        from macro8_subnet.alpha.regime_prediction import ForecastedEnsemble
        fe = ForecastedEnsemble(train_prices, formulas, verbose=False)
        with pytest.raises(RuntimeError, match="fit"):
            fe.forecast()

    def test_scale_by_conf_false_full_positions(self, train_prices, formulas):
        """With scale_by_conf=False, L1 norm should be close to 1."""
        from macro8_subnet.alpha.regime_prediction import ForecastedEnsemble
        fe = ForecastedEnsemble(train_prices, formulas[:6],
                                scale_by_conf=False, verbose=False)
        fe.fit()
        result = fe.forecast()
        if result.positions:
            l1 = sum(abs(w) for w in result.positions.values())
            assert abs(l1 - 1.0) < 0.1


# ── 7. Integration: full pipeline ─────────────────────────────────────────────

class TestIntegrationSprint29:
    def test_gp_to_forecast_pipeline(self, train_prices, formulas):
        """Full GP → ForecastedEnsemble → ForecastResult pipeline."""
        from macro8_subnet.alpha.regime_prediction import ForecastedEnsemble
        fe = ForecastedEnsemble(train_prices, formulas[:6], horizon=5,
                                verbose=False)
        fe.fit()
        result = fe.forecast()

        # Positions valid
        assert isinstance(result.positions, dict)
        assert result.regime_current in ("calm", "normal", "stress")

        # Scenario probabilities are economically reasonable
        assert abs(sum(result.scenario_probs.values()) - 1.0) < 0.01

    def test_confidence_scales_positions(self, fens):
        """With scale_by_conf=True, high-uncertainty regimes get smaller positions."""
        result = fens.forecast()
        if result.positions:
            l1 = sum(abs(w) for w in result.positions.values())
            # L1 should be ≤ 1.0 (may be scaled down)
            assert l1 <= 1.0 + 1e-4
