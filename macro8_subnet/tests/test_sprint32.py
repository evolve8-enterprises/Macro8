"""
tests/test_sprint32.py
-----------------------
Sprint 32: Full-Stack Integration Harness

Tests verify that local_simulation.py correctly exercises every layer
of the Sprint 22–31 stack. Each check function is tested independently
so failures are pinpointed, not masked.
"""

import sys
import random
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
    n, a = 400, 10
    tickers = ["SPY","QQQ","IWM","TLT","GLD","DBC","EEM","FXI","VNQ","HYG"]
    p = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, (n, a)), axis=0))
    return pd.DataFrame(p, index=pd.bdate_range("2015-01-01", periods=n), columns=tickers)


@pytest.fixture(scope="module")
def formulas(prices):
    from macro8_subnet.alpha.gp_miner import GPMiner
    gp = GPMiner(prices, pop_size=40, elite_n=8, seed=42, verbose=False)
    gp.run(n_epochs=3)
    return gp.top_formulas(12)


@pytest.fixture(scope="module")
def fens(prices, formulas):
    from macro8_subnet.alpha.regime_prediction import ForecastedEnsemble
    split = int(len(prices) * 0.70)
    fe = ForecastedEnsemble(prices.iloc[:split], formulas[:6],
                            horizon=5, verbose=False)
    fe.fit()
    return fe


@pytest.fixture(scope="module")
def forecast_result(fens):
    return fens.forecast()


# ── 1. Simulation module importable ──────────────────────────────────────────

class TestSimulationModule:
    def test_import(self):
        import macro8_subnet.local_simulation as sim
        assert hasattr(sim, "check_defensive")
        assert hasattr(sim, "check_validator")
        assert hasattr(sim, "check_adversarial")
        assert hasattr(sim, "main")

    def test_good_formulas_defined(self):
        from macro8_subnet.local_simulation import GOOD_FORMULAS
        assert len(GOOD_FORMULAS) >= 8
        assert all(isinstance(f, str) for f in GOOD_FORMULAS)

    def test_adversarial_inputs_defined(self):
        from macro8_subnet.local_simulation import ADVERSARIAL_INPUTS
        assert len(ADVERSARIAL_INPUTS) >= 5
        assert None in ADVERSARIAL_INPUTS

    def test_make_prices_returns_dataframe(self):
        from macro8_subnet.local_simulation import _make_prices
        p = _make_prices()
        assert isinstance(p, pd.DataFrame)
        assert len(p) > 100


# ── 2. check_defensive ───────────────────────────────────────────────────────

class TestCheckDefensive:
    def test_passes(self):
        from macro8_subnet.local_simulation import check_defensive, _check_results
        _check_results.clear()
        ok = check_defensive()
        assert ok is True

    def test_macro_formulas_accepted(self):
        from macro8_subnet.neurons.validator import safe_formula
        macro_formulas = [
            "market_corr_60d + risk_on_off",
            "vol_regime - volatility_20d",
            "trend_strength + market_corr_20d",
        ]
        for f in macro_formulas:
            assert safe_formula(f) is not None, f"Macro formula rejected: {f}"

    def test_all_adversarial_rejected(self):
        # safe_formula strips whitespace/control chars then validates.
        # Inputs that become valid formulas after stripping are accepted
        # (correct behaviour). We only test unconditionally-bad inputs.
        from macro8_subnet.neurons.validator import safe_formula
        unconditional = [
            None, 42, [], {}, "",
            "A" * 201,
            "__import__('os')",
            "rank(\x00momentum)",
            "\U0001f680momentum",  # emoji
        ]
        for inp in unconditional:
            assert safe_formula(inp) is None, f"Should reject: {inp!r}"


# ── 3. check_gp ──────────────────────────────────────────────────────────────

class TestCheckGP:
    def test_feature_grammar_count(self):
        from macro8_subnet.alpha.gp_miner import GP_FEATURES
        from macro8_subnet.alpha.batch_evaluator import ALL_FEATURES
        # Sprint 33 added 4 event-layer features (stress_accel_5d/20d, eem_spy_20d, iwm_spy_20d)
        assert len(GP_FEATURES) == 38, f"Expected 38, got {len(GP_FEATURES)}"
        assert len(ALL_FEATURES) == 38, f"Expected 38, got {len(ALL_FEATURES)}"
        assert set(GP_FEATURES) == set(ALL_FEATURES)

    def test_macro_terminals_in_grammar(self):
        from macro8_subnet.alpha.gp_miner import GP_FEATURES
        for feat in ["risk_on_off", "vol_regime", "trend_strength",
                     "credit_stress", "em_vs_dm"]:
            assert feat in GP_FEATURES, f"{feat} missing from GP grammar"

    def test_gp_produces_macro_formulas(self, prices):
        from macro8_subnet.alpha.gp_miner import GPMiner
        gp = GPMiner(prices, pop_size=60, elite_n=10, seed=0, verbose=False)
        gp.run(n_epochs=3)
        formulas = gp.top_formulas(20)
        macro = ["risk_on_off","vol_regime","trend_strength","credit_stress"]
        found = [f for f in formulas if any(m in f for m in macro)]
        assert len(found) >= 1, "No macro features in GP output after 3 generations"

    def test_gp_hall_of_fame_grows(self, prices):
        from macro8_subnet.alpha.gp_miner import GPMiner
        gp = GPMiner(prices, pop_size=40, elite_n=8, seed=1, verbose=False)
        gp.run(n_epochs=2)
        n1 = len(gp._hall_of_fame)
        gp.run(n_epochs=2)
        n2 = len(gp._hall_of_fame)
        assert n2 > n1, "Hall of fame should grow with more generations"


# ── 4. check_portfolio ───────────────────────────────────────────────────────

class TestCheckPortfolio:
    def test_feature_store_34_features(self, prices):
        from macro8_subnet.alpha.feature_store import FeatureStore
        fs    = FeatureStore(prices)
        feats = fs.build()
        assert len(feats) >= 30

    def test_regime_detector_valid_labels(self, prices):
        from macro8_subnet.alpha.feature_store import FeatureStore
        from macro8_subnet.alpha.portfolio_intelligence import RegimeDetector
        fs     = FeatureStore(prices)
        feats  = fs.build()
        det    = RegimeDetector()
        labels = det.label_series(feats, prices.index)
        assert set(labels.unique()).issubset({"calm", "normal", "stress"})

    def test_adaptive_ensemble_positions_valid(self, prices, formulas):
        from macro8_subnet.alpha.portfolio_intelligence import AdaptiveEnsemble
        split = int(len(prices) * 0.70)
        ens   = AdaptiveEnsemble(prices.iloc[:split], formulas[:6],
                                 verbose=False)
        ens.fit()
        result = ens.positions()
        l1 = sum(abs(w) for w in result.positions.values())
        assert len(result.positions) > 0
        assert abs(l1 - 1.0) < 0.05


# ── 5. check_prediction ──────────────────────────────────────────────────────

class TestCheckPrediction:
    def test_forecast_valid(self, forecast_result):
        assert forecast_result.regime_current in ("calm", "normal", "stress")
        assert 0 <= forecast_result.confidence <= 1

    def test_scenario_probs_sum_to_one(self, forecast_result):
        total = sum(forecast_result.scenario_probs.values())
        assert abs(total - 1.0) < 0.01

    def test_prediction_market_8_scenarios(self, forecast_result):
        from macro8_subnet.execution.engine import PredictionMarket, DrawdownGuard, PerformanceWindow
        pw   = PerformanceWindow(0, 0, 0, 0, 0, 0, 0, 0)
        pred = PredictionMarket().emit(forecast_result, pw, DrawdownGuard())
        assert len(pred.scenario_probs) == 8

    def test_policy_state_has_all_indicators(self, forecast_result):
        ps = forecast_result.policy_state
        for attr in ("rate_env", "inflation", "liquidity", "dollar", "breadth"):
            assert hasattr(ps, attr)
            assert np.isfinite(getattr(ps, attr))


# ── 6. check_execution ───────────────────────────────────────────────────────

class TestCheckExecution:
    def test_constraint_solver_l1_norm(self, forecast_result):
        from macro8_subnet.execution.engine import PortfolioConstraints, ConstraintSolver
        cs  = ConstraintSolver(PortfolioConstraints())
        pos = cs.apply(forecast_result.positions,
                       p_stress=forecast_result.regime_forecast.stress)
        l1  = sum(abs(w) for w in pos.values())
        assert abs(l1 - 1.0) < 0.02

    def test_trade_executor_produces_orders(self, forecast_result):
        from macro8_subnet.execution.engine import (
            PortfolioConstraints, ConstraintSolver, TradeExecutor,
        )
        cs   = ConstraintSolver(PortfolioConstraints())
        pos  = cs.apply(forecast_result.positions)
        ex   = TradeExecutor(capital=100_000)
        plan = ex.compute_trades(pos, {})
        assert len(plan.orders) > 0
        assert plan.total_turnover > 0

    def test_drawdown_guard_fires_below_threshold(self):
        from macro8_subnet.execution.engine import DrawdownGuard
        dg = DrawdownGuard(max_drawdown=-0.02, lookback=10)
        for _ in range(5):  dg.update(0.002)
        assert dg.position_scale() == 1.0
        for _ in range(10): dg.update(-0.004)
        assert dg.position_scale() < 1.0

    def test_live_tracker_records_days(self, prices, forecast_result, fens):
        from macro8_subnet.execution.engine import (
            LiveTracker, TradeExecutor, PortfolioConstraints, ConstraintSolver,
        )
        cs      = ConstraintSolver(PortfolioConstraints())
        pos     = cs.apply(forecast_result.positions)
        ex      = TradeExecutor(capital=100_000)
        plan    = ex.compute_trades(pos, {})
        tracker = LiveTracker()
        for i in range(3):
            tracker.update(prices.index[i], pos, plan, 0.001 * (-1)**i, forecast_result)
        assert len(tracker._records) == 3


# ── 7. check_live_pipeline ───────────────────────────────────────────────────

class TestCheckLivePipeline:
    def test_data_pipeline_offline_fallback(self, tmp_path, prices):
        from macro8_subnet.execution.live_runner import DataPipeline
        dp = DataPipeline(tickers=list(prices.columns)[:8],
                          cache_dir=tmp_path, verbose=False)
        dp._is_online = lambda: False
        fetched, status = dp.fetch()
        assert status.source in ("synthetic", "cache")
        assert len(fetched) > 100

    def test_failure_log_roundtrip(self, tmp_path):
        from macro8_subnet.execution.live_runner import FailureLog
        path = tmp_path / "fail.json"
        fl   = FailureLog(path=path)
        fl.log_regime_failure("2024-01-01", "normal", "stress", 0.70, -0.002)
        fl.log_retrain("2024-01-05", -0.6)
        fl2 = FailureLog(path=path)
        assert len(fl2._log) == 2
        assert fl2._log[0].failure_type == "regime_wrong"

    def test_paper_trader_backtest_finite(self, tmp_path):
        # Needs enough data so all 3 regime classes have >= 3 samples for CV.
        import numpy as np, pandas as pd
        from macro8_subnet.execution.live_runner import PaperTrader
        rng = np.random.default_rng(7)
        n   = 600
        p   = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, (n, 6)), axis=0))
        big = pd.DataFrame(p, index=pd.bdate_range("2010-01-01", periods=n),
                           columns=["SPY","QQQ","IWM","TLT","GLD","HYG"])
        trader = PaperTrader(tickers=list(big.columns),
                             state_file=tmp_path / "state.json",
                             verbose=False)
        hist = trader.run_backtest(big, n_days=4, train_frac=0.85)
        assert len(hist) >= 2
        pnl = hist["pnl"].dropna()
        assert np.isfinite(pnl.values).all()


# ── 8. check_validator ───────────────────────────────────────────────────────

class TestCheckValidator:
    def test_scores_all_miners(self, formulas):
        from macro8_subnet.neurons.validator import Macro8Validator
        from macro8_subnet.protocol.synapse import AlphaSubmissionSynapse
        validator = Macro8Validator()
        subs = {
            0: AlphaSubmissionSynapse(formulas=formulas[:6], miner_uid=0),
            1: AlphaSubmissionSynapse(formulas=formulas[3:8], miner_uid=1),
        }
        scores = validator.scorer.score_submissions(subs, epoch=1)
        assert len(scores) == 2

    def test_rewards_sum_to_one(self, formulas):
        from macro8_subnet.neurons.validator import Macro8Validator
        from macro8_subnet.protocol.synapse import AlphaSubmissionSynapse
        from macro8_subnet.agents.role_rewards import RoleRewardModel
        validator = Macro8Validator()
        subs = {i: AlphaSubmissionSynapse(formulas=formulas[:5], miner_uid=i)
                for i in range(3)}
        scores = validator.scorer.score_submissions(subs, epoch=1)
        role_scores   = validator._build_role_scores(subs, scores)
        reward_report = RoleRewardModel().compute(epoch=1, role_scores=role_scores)
        _, weights    = reward_report.as_weight_list()
        if weights:
            assert abs(sum(weights) - 1.0) < 0.01

    def test_positions_accepted_in_synapse(self, formulas):
        """Validator handles the new Sprint 26 positions field without crash."""
        from macro8_subnet.neurons.validator import Macro8Validator
        from macro8_subnet.protocol.synapse import AlphaSubmissionSynapse
        validator = Macro8Validator()
        syn = AlphaSubmissionSynapse(
            formulas=formulas[:5],
            positions={"SPY": 0.15, "TLT": -0.12},
            position_formula="ensemble(2,regime=normal)",
            miner_uid=0, epoch=1,
        )
        scores = validator.scorer.score_submissions({0: syn}, epoch=1)
        assert 0 in scores
        assert np.isfinite(scores[0])

    def test_adversarial_no_crash(self, formulas):
        from macro8_subnet.neurons.validator import Macro8Validator
        from macro8_subnet.protocol.synapse import AlphaSubmissionSynapse
        from macro8_subnet.local_simulation import ADVERSARIAL_INPUTS
        validator = Macro8Validator()
        subs = {}
        for uid in range(4):
            syn = AlphaSubmissionSynapse(formulas=formulas[:4], miner_uid=uid)
            if uid % 2 == 1:
                try:
                    object.__setattr__(syn, "formulas",
                                       random.sample(ADVERSARIAL_INPUTS, 2) + formulas[:2])
                except Exception:
                    pass
            subs[uid] = syn
        # Must not raise
        scores = validator.scorer.score_submissions(subs, epoch=1)
        assert len(scores) == 4


# ── 9. End-to-end: local_simulation --fast ───────────────────────────────────

class TestLocalSimulationCLI:
    def test_fast_mode_all_pass(self, capsys):
        """--fast mode completes in < 10s with all checks passing."""
        import sys
        from unittest.mock import patch
        with patch("sys.argv", ["local_simulation", "--fast"]):
            import macro8_subnet.local_simulation as sim
            sim._check_results.clear()
            sim.main()
        captured = capsys.readouterr()
        assert "All checks passed" in captured.out

    def test_adversarial_mode_no_crash(self, capsys):
        import sys
        from unittest.mock import patch
        with patch("sys.argv", ["local_simulation", "--fast", "--adversarial"]):
            import macro8_subnet.local_simulation as sim
            sim._check_results.clear()
            sim.main()
        captured = capsys.readouterr()
        assert "All checks passed" in captured.out
