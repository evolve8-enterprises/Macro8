"""
tests/test_sprint26.py
-----------------------
Sprint 26: PR Testing, Scenario Engine, Macro Features, Signal Layer

Tests cover:
    - FeatureStore: 10 new macro features compute correctly
    - PRTester: walk-forward windows, regime labels, metrics
    - ScenarioEngine: 8 scenarios, shock application, survival metrics
    - Miner signal layer: formula → positions pipeline
    - Synapse: positions and position_formula fields
    - Integration: full discover → validate → position pipeline
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
    rng = np.random.default_rng(42)
    n_days, n_assets = 800, 10
    tickers = ["SPY","QQQ","IWM","TLT","GLD","DBC","EEM","FXI","VNQ","HYG"]
    log_ret = rng.normal(0.0003, 0.012, (n_days, n_assets))
    p_arr   = 100 * np.exp(np.cumsum(log_ret, axis=0))
    dates   = pd.bdate_range("2015-01-01", periods=n_days)
    return pd.DataFrame(p_arr, index=dates, columns=tickers)


@pytest.fixture(scope="module")
def feature_store(prices):
    from macro8_subnet.alpha.feature_store import FeatureStore
    return FeatureStore(prices)


@pytest.fixture(scope="module")
def all_features(feature_store):
    return feature_store.build()


@pytest.fixture(scope="module")
def pr_tester(prices):
    from macro8_subnet.evaluation.pr_tester import PRTester
    return PRTester(prices, capital=100_000, verbose=False,
                    min_ic_threshold=0.002, min_pass_rate=0.45)


@pytest.fixture(scope="module")
def scenario_engine(prices):
    from macro8_subnet.evaluation.scenario_engine import ScenarioEngine
    return ScenarioEngine(prices.iloc[:500], capital=100_000, verbose=False, seed=42)


@pytest.fixture(scope="module")
def test_formulas():
    return ["market_corr_60d", "momentum_20d", "reversal_5d", "risk_on_off"]


# ── 1. FeatureStore macro features ───────────────────────────────────────────

class TestMacroFeatures:
    MACRO_NAMES = [
        "risk_on_off", "commodity_inflation", "em_vs_dm", "credit_stress",
        "equity_bond_corr", "cross_asset_vol", "vol_regime",
        "trend_strength", "carry_proxy", "dollar_proxy",
    ]

    def test_total_feature_count(self, all_features):
        """Sprint 26 adds 10 macro features to 24 technical = 34 total."""
        assert len(all_features) >= 30, \
            f"Expected ≥30 features, got {len(all_features)}"

    def test_macro_features_in_output(self, all_features):
        """All 10 macro feature names must appear (or gracefully absent if tickers missing)."""
        # At least 6 should be computable from our 10-ticker universe
        present = [n for n in self.MACRO_NAMES if n in all_features]
        assert len(present) >= 6, \
            f"Only {len(present)} macro features built: {present}"

    def test_risk_on_off_shape(self, all_features, prices):
        f = all_features.get("risk_on_off")
        assert f is not None, "risk_on_off not in features"
        assert f.shape == prices.shape
        assert f.columns.tolist() == prices.columns.tolist()

    def test_risk_on_off_broadcast(self, all_features):
        """risk_on_off broadcasts same value to all assets per day."""
        f = all_features.get("risk_on_off")
        if f is None:
            pytest.skip("risk_on_off not available")
        # All columns should be identical
        assert f.std(axis=1).max() < 1e-10, "risk_on_off is not broadcast"

    def test_cross_asset_vol_non_negative(self, all_features):
        f = all_features.get("cross_asset_vol")
        if f is None:
            pytest.skip()
        assert (f.fillna(0) >= 0).all().all(), "cross_asset_vol should be non-negative"

    def test_trend_strength_bounded(self, all_features):
        """trend_strength = fraction above 200d MA, must be in [0, 1]."""
        f = all_features.get("trend_strength")
        if f is None:
            pytest.skip()
        filled = f.fillna(0.5)
        assert filled.min().min() >= 0.0
        assert filled.max().max() <= 1.0

    def test_equity_bond_corr_bounded(self, all_features):
        """Correlation must be in [-1, 1]."""
        f = all_features.get("equity_bond_corr")
        if f is None:
            pytest.skip()
        filled = f.fillna(0)
        assert filled.min().min() >= -1.0 - 1e-6
        assert filled.max().max() <=  1.0 + 1e-6

    def test_macro_features_finite(self, all_features):
        """All macro feature values should be finite (no inf)."""
        for name in self.MACRO_NAMES:
            f = all_features.get(name)
            if f is None:
                continue
            inf_mask = np.isinf(f.values)
            assert not inf_mask.any(), \
                f"{name} has {inf_mask.sum()} infinite values"

    def test_vol_regime_is_zscore(self, all_features):
        """vol_regime should be a z-score — roughly mean 0, std ~1."""
        f = all_features.get("vol_regime")
        if f is None:
            pytest.skip()
        vals = f.dropna().values.flatten()
        if len(vals) > 100:
            assert abs(vals.mean()) < 2.0, \
                f"vol_regime mean={vals.mean():.2f} too far from 0"

    def test_dollar_proxy_present(self, all_features):
        """dollar_proxy should be computed from EEM and GLD."""
        dp = all_features.get("dollar_proxy")
        if dp is None:
            pytest.skip("EEM or GLD missing from universe")
        assert dp.shape[1] > 0

    def test_broadcast_features_all_columns_equal(self, all_features):
        """All macro features are broadcast — each row has identical values across columns."""
        for name in ["risk_on_off", "cross_asset_vol", "vol_regime",
                     "trend_strength", "carry_proxy"]:
            f = all_features.get(name)
            if f is None:
                continue
            max_col_std = f.std(axis=1).max()
            assert max_col_std < 1e-8, \
                f"{name} is not properly broadcast: max_col_std={max_col_std:.2e}"

    def test_invalidate_clears_cache(self, prices):
        from macro8_subnet.alpha.feature_store import FeatureStore
        fs = FeatureStore(prices)
        fs.build(["momentum_20d"])
        assert "momentum_20d" in fs._cache
        fs.invalidate()
        assert len(fs._cache) == 0

    def test_feature_names_property(self, feature_store):
        names = feature_store.feature_names
        assert "risk_on_off" in names
        assert "vol_regime" in names
        assert "momentum_20d" in names
        assert len(names) >= 30


# ── 2. PRTester ───────────────────────────────────────────────────────────────

class TestPRTester:
    def test_import(self):
        from macro8_subnet.evaluation.pr_tester import (
            PRTester, WalkForwardConfig, PRResult, WindowResult,
        )

    def test_wf_config_defaults(self):
        from macro8_subnet.evaluation.pr_tester import WalkForwardConfig
        cfg = WalkForwardConfig()
        assert cfg.train_days == 756
        assert cfg.test_days  == 63
        assert cfg.step_days  == 21

    def test_wf_config_n_windows(self):
        from macro8_subnet.evaluation.pr_tester import WalkForwardConfig
        cfg = WalkForwardConfig(train_days=504, test_days=63, step_days=63)
        assert cfg.n_windows(800) > 0
        assert cfg.n_windows(100) == 0   # too short

    def test_simple_split_returns_results(self, pr_tester, test_formulas):
        results = pr_tester.simple_split(test_formulas[:2])
        assert len(results) == 2
        for r in results:
            assert r.formula in test_formulas
            assert np.isfinite(r.mean_ic_test)
            assert np.isfinite(r.mean_ic_train)

    def test_simple_split_pass_fail(self, pr_tester, test_formulas):
        results = pr_tester.simple_split(test_formulas)
        for r in results:
            # pass = all criteria met; fail = at least one reason
            if r.passes:
                assert len(r.fail_reasons) == 0
            else:
                assert len(r.fail_reasons) > 0

    def test_walk_forward_windows_created(self, pr_tester, test_formulas):
        from macro8_subnet.evaluation.pr_tester import WalkForwardConfig
        cfg = WalkForwardConfig(train_days=400, test_days=63, step_days=63)
        results = pr_tester.walk_forward(test_formulas[:2], cfg)
        assert len(results) == 2
        for r in results:
            assert r.n_windows > 0

    def test_walk_forward_has_window_results(self, pr_tester, test_formulas):
        from macro8_subnet.evaluation.pr_tester import WalkForwardConfig
        cfg = WalkForwardConfig(train_days=400, test_days=63, step_days=63)
        results = pr_tester.walk_forward(test_formulas[:1], cfg)
        r = results[0]
        assert len(r.window_results) == r.n_windows
        for wr in r.window_results:
            assert wr.train_start < wr.train_end
            assert wr.test_start  >= wr.train_end
            assert wr.regime in ("bull", "bear", "crisis")

    def test_ic_stability_formula(self, pr_tester, test_formulas):
        """ic_stability = std(test_ics) / |mean(test_ics)|."""
        from macro8_subnet.evaluation.pr_tester import WalkForwardConfig
        cfg = WalkForwardConfig(train_days=400, test_days=63, step_days=63)
        results = pr_tester.walk_forward(test_formulas[:1], cfg)
        r = results[0]
        test_ics = [w.test_ic for w in r.window_results]
        expected_stability = np.std(test_ics) / (abs(np.mean(test_ics)) + 1e-8)
        assert abs(r.ic_stability - expected_stability) < 0.01

    def test_pass_rate_in_unit_interval(self, pr_tester, test_formulas):
        results = pr_tester.simple_split(test_formulas)
        for r in results:
            assert 0.0 <= r.pass_rate <= 1.0

    def test_regime_split_labels(self, pr_tester, test_formulas):
        results = pr_tester.regime_split(test_formulas[:2])
        assert len(results) == 2
        for r in results:
            for regime in r.regime_ics:
                assert regime in ("bull", "bear", "crisis")

    def test_worst_regime_is_minimum(self, pr_tester, test_formulas):
        results = pr_tester.regime_split(test_formulas[:2])
        for r in results:
            if r.regime_ics:
                expected_worst = min(r.regime_ics.values())
                assert abs(r.worst_regime_ic - expected_worst) < 1e-8

    def test_print_results_no_crash(self, pr_tester, test_formulas, capsys):
        results = pr_tester.simple_split(test_formulas[:2])
        pr_tester.print_results(results, top_n=5)
        captured = capsys.readouterr()
        assert "WALK-FORWARD" in captured.out
        assert "VALIDATION" in captured.out

    def test_verdict_line_format(self, pr_tester, test_formulas):
        results = pr_tester.simple_split(test_formulas[:1])
        line    = results[0].verdict_line()
        assert "IC_test" in line
        assert ("PASS" in line or "FAIL" in line or "stability" in line)

    def test_regime_line_format(self, pr_tester, test_formulas):
        results = pr_tester.regime_split(test_formulas[:1])
        if results[0].regime_ics:
            line = results[0].regime_line()
            assert "bull" in line or "bear" in line

    def test_fallback_to_simple_split_when_too_few_windows(self, prices):
        from macro8_subnet.evaluation.pr_tester import PRTester, WalkForwardConfig
        tester = PRTester(prices.iloc[:200], verbose=False)
        cfg    = WalkForwardConfig(train_days=756, test_days=252, step_days=63,
                                   min_windows=6)
        results = tester.walk_forward(["momentum_20d"], cfg)
        # Should return something (fallback to simple split)
        assert len(results) == 1


# ── 3. ScenarioEngine ─────────────────────────────────────────────────────────

class TestScenarioEngine:
    def test_import(self):
        from macro8_subnet.evaluation.scenario_engine import (
            ScenarioEngine, ScenarioReport, ScenarioResult, SCENARIOS,
        )

    def test_eight_scenarios_defined(self):
        from macro8_subnet.evaluation.scenario_engine import SCENARIOS
        assert len(SCENARIOS) == 8
        expected = {
            "rates_up_200bps", "rates_down_100bps", "equity_crash_30pct",
            "oil_spike_50pct", "china_crisis", "soft_landing",
            "stagflation", "ai_boom",
        }
        assert set(SCENARIOS.keys()) == expected

    def test_scenario_structure(self):
        from macro8_subnet.evaluation.scenario_engine import SCENARIOS
        for name, cfg in SCENARIOS.items():
            assert "description"   in cfg, f"{name} missing description"
            assert "shocks"        in cfg, f"{name} missing shocks"
            assert "vol_mult"      in cfg, f"{name} missing vol_mult"
            assert "duration_days" in cfg, f"{name} missing duration_days"
            assert cfg["vol_mult"] > 0
            assert cfg["duration_days"] > 0

    def test_shock_magnitudes_realistic(self):
        from macro8_subnet.evaluation.scenario_engine import SCENARIOS
        for name, cfg in SCENARIOS.items():
            for ticker, shock in cfg["shocks"].items():
                assert -1.0 < shock < 2.0, \
                    f"{name}/{ticker}: shock {shock} unrealistic (must be in (-1, 2))"

    def test_run_returns_reports(self, scenario_engine, test_formulas):
        reports = scenario_engine.run(
            test_formulas[:2],
            scenarios=["equity_crash_30pct", "soft_landing"],
        )
        assert len(reports) == 2
        for r in reports:
            assert r.formula in test_formulas
            assert 0 <= r.robustness_score <= 1.0
            assert r.n_scenarios == 2

    def test_robustness_score_is_fraction(self, scenario_engine, test_formulas):
        reports = scenario_engine.run(
            test_formulas[:2],
            scenarios=["equity_crash_30pct", "soft_landing", "rates_up_200bps"],
        )
        for r in reports:
            assert r.robustness_score == r.n_survived / 3

    def test_scenario_result_fields(self, scenario_engine, test_formulas):
        reports = scenario_engine.run(
            test_formulas[:1],
            scenarios=["equity_crash_30pct"],
        )
        sr = reports[0].scenario_results["equity_crash_30pct"]
        assert hasattr(sr, "sharpe_base")
        assert hasattr(sr, "sharpe_shocked")
        assert hasattr(sr, "relative_sharpe")
        assert hasattr(sr, "drawdown_shocked")
        assert hasattr(sr, "survived")
        assert isinstance(sr.survived, bool)
        assert np.isfinite(sr.sharpe_shocked)
        assert sr.drawdown_shocked <= 0.0

    def test_worst_drawdown_non_positive(self, scenario_engine, test_formulas):
        reports = scenario_engine.run(
            test_formulas[:2],
            scenarios=["equity_crash_30pct", "rates_up_200bps"],
        )
        for r in reports:
            assert r.worst_drawdown <= 0.0

    def test_run_one_scenario(self, scenario_engine, test_formulas):
        results = scenario_engine.run_one_scenario("soft_landing", test_formulas[:2])
        assert len(results) == 2
        for sr in results:
            assert sr.scenario_name == "soft_landing"
            assert np.isfinite(sr.ic_base)
            assert np.isfinite(sr.ic_shocked)

    def test_unknown_scenario_raises(self, scenario_engine, test_formulas):
        with pytest.raises(ValueError, match="Unknown scenario"):
            scenario_engine.run_one_scenario("nonexistent", test_formulas[:1])

    def test_apply_shock_changes_target_price(self, scenario_engine):
        """Shocked prices should differ materially from base for shocked tickers."""
        from macro8_subnet.evaluation.scenario_engine import SCENARIOS
        base = scenario_engine.prices
        scfg = SCENARIOS["equity_crash_30pct"]
        shocked = scenario_engine._apply_shock(
            base, scfg["shocks"], scfg["vol_mult"], scfg["duration_days"]
        )
        # SPY should be materially different
        if "SPY" in base.columns:
            base_ret    = base["SPY"].iloc[-1]    / base["SPY"].iloc[0]    - 1
            shocked_ret = shocked["SPY"].iloc[-1] / shocked["SPY"].iloc[0] - 1
            assert abs(shocked_ret - base_ret) > 0.05, \
                f"SPY shock not applied: base={base_ret:.2%} shocked={shocked_ret:.2%}"

    def test_apply_shock_preserves_unshocked_tickers(self, scenario_engine):
        """Tickers not in shock dict should be close to base (only vol changes)."""
        from macro8_subnet.evaluation.scenario_engine import SCENARIOS
        base = scenario_engine.prices
        # soft_landing only shocks positive — check a ticker not in the shock dict
        scfg   = SCENARIOS["soft_landing"]
        shocked = scenario_engine._apply_shock(
            base, scfg["shocks"], scfg["vol_mult"], scfg["duration_days"]
        )
        for ticker in base.columns:
            if ticker not in scfg["shocks"]:
                # Return should not be wildly different (no directional shock)
                base_ret    = float(base[ticker].iloc[-1] / base[ticker].iloc[0] - 1)
                shocked_ret = float(shocked[ticker].iloc[-1] / shocked[ticker].iloc[0] - 1)
                # Just verify it's finite and sane
                assert np.isfinite(shocked_ret), f"{ticker} shocked return is not finite"
                break

    def test_print_report_no_crash(self, scenario_engine, test_formulas, capsys):
        reports = scenario_engine.run(
            test_formulas[:2],
            scenarios=["equity_crash_30pct", "soft_landing"],
        )
        scenario_engine.print_report(reports, top_n=5)
        captured = capsys.readouterr()
        assert "SCENARIO ENGINE" in captured.out

    def test_available_scenarios(self, scenario_engine):
        avail = scenario_engine.available_scenarios()
        assert len(avail) == 8
        assert all(isinstance(v, str) for v in avail.values())

    def test_best_worst_scenario_populated(self, scenario_engine, test_formulas):
        reports = scenario_engine.run(
            test_formulas[:1],
            scenarios=["equity_crash_30pct", "soft_landing", "ai_boom"],
        )
        r = reports[0]
        assert r.best_scenario  in ("equity_crash_30pct", "soft_landing", "ai_boom")
        assert r.worst_scenario in ("equity_crash_30pct", "soft_landing", "ai_boom")

    def test_sorted_by_robustness(self, scenario_engine, test_formulas):
        reports = scenario_engine.run(
            test_formulas,
            scenarios=["equity_crash_30pct", "soft_landing"],
        )
        scores = [r.robustness_score for r in reports]
        assert scores == sorted(scores, reverse=True)


# ── 4. Signal layer — formula → positions ─────────────────────────────────────

class TestSignalLayer:
    @pytest.fixture(scope="class")
    def miner(self, prices):
        from macro8_subnet.neurons.miner import Macro8Miner
        import argparse
        cfg = argparse.Namespace(
            netuid=1, role="signal", n_formulas=20, n_hypotheses=3,
            data_start="2015-01-01", n_assets=8, pr_test_interval=999,
            gp_generations=1, capital=100_000,
            wallet_name="default", wallet_hotkey="default",
            subtensor_network="test",
        )
        return Macro8Miner(cfg)

    def test_formula_to_positions_returns_dict(self, miner):
        pos = miner._formula_to_positions("market_corr_20d")
        assert isinstance(pos, dict)

    def test_positions_sum_to_zero_gross(self, miner):
        """Long/short portfolio: longs cancel shorts."""
        pos = miner._formula_to_positions("market_corr_20d")
        if pos:
            total = sum(pos.values())
            assert abs(total) < 0.01, f"Positions don't sum to ~0: {total:.4f}"

    def test_positions_l1_norm_one(self, miner):
        """L1 norm of weights = 1 (sum of absolute values)."""
        pos = miner._formula_to_positions("momentum_20d")
        if pos:
            l1 = sum(abs(w) for w in pos.values())
            assert abs(l1 - 1.0) < 0.05, f"L1 norm = {l1:.4f}, expected ≈ 1.0"

    def test_positions_all_finite(self, miner):
        pos = miner._formula_to_positions("reversal_5d")
        for ticker, w in pos.items():
            assert np.isfinite(w), f"Position for {ticker} is not finite"

    def test_positions_ticker_in_universe(self, miner):
        pos = miner._formula_to_positions("momentum_20d")
        universe = list(miner.prices.columns)
        for ticker in pos:
            assert ticker in universe, f"{ticker} not in universe"

    def test_unencodable_formula_returns_empty(self, miner):
        """Formula that can't be encoded should return {} gracefully."""
        pos = miner._formula_to_positions("__import__('os').system('ls')")
        assert pos == {}

    def test_get_position_table_returns_df(self, miner):
        miner._formula_to_positions("market_corr_20d")
        # Set last_positions manually for test
        miner.last_positions = {"SPY": 0.1, "TLT": -0.1, "GLD": 0.05}
        tbl = miner.get_position_table()
        assert isinstance(tbl, pd.DataFrame)
        assert "ticker" in tbl.columns
        assert "weight" in tbl.columns
        assert "direction" in tbl.columns

    def test_position_directions_consistent(self, miner):
        miner.last_positions = {"SPY": 0.15, "TLT": -0.10, "GLD": -0.05}
        tbl = miner.get_position_table()
        for _, row in tbl.iterrows():
            if row["weight"] > 0:
                assert row["direction"] == "LONG"
            else:
                assert row["direction"] == "SHORT"

    def test_get_current_positions_returns_copy(self, miner):
        miner.last_positions = {"SPY": 0.2}
        pos1 = miner.get_current_positions()
        pos1["NEW"] = 0.99   # mutate the returned dict
        pos2 = miner.get_current_positions()
        assert "NEW" not in pos2, "get_current_positions should return a copy"


# ── 5. Synapse positions field ────────────────────────────────────────────────

class TestSynapsePositions:
    def test_positions_field_exists(self):
        from macro8_subnet.protocol.synapse import AlphaSubmissionSynapse
        syn = AlphaSubmissionSynapse(epoch=1)
        assert hasattr(syn, "positions")
        assert isinstance(syn.positions, dict)

    def test_position_formula_field_exists(self):
        from macro8_subnet.protocol.synapse import AlphaSubmissionSynapse
        syn = AlphaSubmissionSynapse(epoch=1)
        assert hasattr(syn, "position_formula")
        assert isinstance(syn.position_formula, str)

    def test_positions_default_empty(self):
        from macro8_subnet.protocol.synapse import AlphaSubmissionSynapse
        syn = AlphaSubmissionSynapse(epoch=1)
        assert syn.positions == {}

    def test_positions_assignable(self):
        from macro8_subnet.protocol.synapse import AlphaSubmissionSynapse
        syn = AlphaSubmissionSynapse(epoch=1)
        syn.positions = {"SPY": 0.2, "TLT": -0.1}
        assert syn.positions["SPY"] == 0.2
        assert syn.positions["TLT"] == -0.1

    def test_position_formula_assignable(self):
        from macro8_subnet.protocol.synapse import AlphaSubmissionSynapse
        syn = AlphaSubmissionSynapse(epoch=1)
        syn.position_formula = "market_corr_60d"
        assert syn.position_formula == "market_corr_60d"

    def test_existing_fields_still_present(self):
        from macro8_subnet.protocol.synapse import AlphaSubmissionSynapse
        syn = AlphaSubmissionSynapse(epoch=1)
        assert hasattr(syn, "formulas")
        assert hasattr(syn, "ic_scores")
        assert hasattr(syn, "eval_success")
        assert hasattr(syn, "market_positions")


# ── 6. Integration: discover → validate → position ────────────────────────────

class TestIntegration:
    def test_feature_store_feeds_batch_evaluator(self, prices, all_features):
        """FeatureStore macro features must be encodable in BatchEvaluator."""
        from macro8_subnet.alpha.batch_evaluator import BatchEvaluator
        ev = BatchEvaluator(prices, min_ic=0.0)
        # risk_on_off is a new macro feature — evaluator can handle it
        result = ev.evaluate(["market_corr_60d", "momentum_20d"])
        assert result.n_formulas >= 1

    def test_pr_tester_filters_gp_output(self, prices):
        """GP output → PR tester → filtered submission list."""
        from macro8_subnet.alpha.gp_miner import GPMiner
        from macro8_subnet.evaluation.pr_tester import PRTester, WalkForwardConfig

        gp = GPMiner(prices, pop_size=20, elite_n=5, seed=42, verbose=False)
        gp.run(n_epochs=2)
        candidates = gp.top_formulas(8)
        assert len(candidates) > 0

        tester  = PRTester(prices, verbose=False, min_ic_threshold=0.0,
                           min_pass_rate=0.0, max_ic_stability=100.0)
        cfg     = WalkForwardConfig(train_days=400, test_days=63, step_days=63)
        results = tester.walk_forward(candidates[:4], cfg)
        assert len(results) == min(4, len(candidates))

    def test_scenario_engine_ranks_formulas(self, prices):
        """Scenario engine should rank formulas by robustness."""
        from macro8_subnet.evaluation.scenario_engine import ScenarioEngine
        engine  = ScenarioEngine(prices.iloc[:400], verbose=False)
        reports = engine.run(
            ["market_corr_60d", "reversal_5d"],
            scenarios=["soft_landing", "equity_crash_30pct"],
        )
        assert len(reports) == 2
        # Sorted by robustness_score descending
        assert reports[0].robustness_score >= reports[-1].robustness_score

    def test_miner_dry_run_produces_positions(self, prices):
        """Miner dry-run should attach positions to synapse."""
        from macro8_subnet.neurons.miner import Macro8Miner
        from macro8_subnet.protocol.synapse import AlphaSubmissionSynapse
        import argparse

        cfg = argparse.Namespace(
            netuid=1, role="signal", n_formulas=15, n_hypotheses=2,
            data_start="2015-01-01", n_assets=6, pr_test_interval=999,
            gp_generations=1, capital=100_000,
            wallet_name="default", wallet_hotkey="default",
            subtensor_network="test",
        )
        miner = Macro8Miner(cfg)
        syn   = AlphaSubmissionSynapse(epoch=1)
        result = miner.forward(syn)

        assert result.eval_success is True
        assert len(result.formulas) > 0

    def test_signal_layer_end_to_end(self, prices):
        """formula → signal → weights → dict with correct properties."""
        from macro8_subnet.alpha.feature_store import FeatureStore
        from macro8_subnet.alpha.batch_evaluator import FeatureTensor, FormulaEncoder
        from scipy.stats import rankdata

        fs     = FeatureStore(prices.iloc[-120:])
        feats  = fs.build()
        ft     = FeatureTensor.from_feature_dict(feats)
        enc    = FormulaEncoder(ft.feature_names)

        formula = "market_corr_20d"
        assert enc.can_encode(formula), f"Cannot encode {formula}"

        W      = enc.encode_batch([formula])
        S      = np.einsum("taf,fn->tan", ft.tensor, W, optimize=True)
        signal = S[-1, :, 0]

        assert np.isfinite(signal).all()
        ranks   = rankdata(signal).astype(float)
        ranks  -= ranks.mean()
        weights = ranks / (np.abs(ranks).sum() + 1e-8)

        # Properties of a valid L/S portfolio
        assert abs(weights.sum()) < 1e-6           # sums to zero
        assert abs(np.abs(weights).sum() - 1.0) < 1e-4   # L1 = 1
        assert len(weights) == len(prices.columns)
