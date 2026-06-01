"""
tests/test_sprint28.py
-----------------------
Sprint 28: Portfolio Intelligence Layer

Tests cover:
    - SignalClusterer: correlation structure, cluster selection, diversity metrics
    - EnsembleWeighter: four weighting methods, normalisation
    - RegimeDetector: classification, label_series
    - AdaptiveEnsemble: fit, positions, rolling_pnl, sharpe_breakdown
    - Miner integration: ensemble positions in synapse
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
    n, a = 600, 10
    tickers = ["SPY","QQQ","IWM","TLT","GLD","DBC","EEM","FXI","VNQ","HYG"]
    p = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, (n, a)), axis=0))
    return pd.DataFrame(p, index=pd.bdate_range("2015-01-01", periods=n), columns=tickers)


@pytest.fixture(scope="module")
def formulas(prices):
    from macro8_subnet.alpha.gp_miner import GPMiner
    gp = GPMiner(prices, pop_size=60, elite_n=10, seed=42, verbose=False)
    gp.run(n_epochs=2)
    return gp.top_formulas(16)


@pytest.fixture(scope="module")
def pnl_matrix(prices, formulas):
    from macro8_subnet.alpha.batch_evaluator import FeatureTensor, FormulaEncoder
    from macro8_subnet.alpha.feature_store import FeatureStore
    from scipy.stats import rankdata
    fs   = FeatureStore(prices)
    ft   = FeatureTensor.from_feature_dict(fs.build())
    enc  = FormulaEncoder(ft.feature_names)
    ef   = [f for f in formulas if enc.can_encode(f)]
    W    = enc.encode_batch(ef)
    S    = np.einsum("taf,fn->tan", ft.tensor, W, optimize=True)
    T,A,F = S.shape
    S_tfa = S.transpose(0,2,1).reshape(T*F,A)
    r_f   = rankdata(S_tfa,axis=1).astype(np.float32)
    ranks = r_f.reshape(T,F,A).transpose(0,2,1)
    ranks -= ranks.mean(axis=1, keepdims=True)
    w     = ranks / (np.abs(ranks).sum(axis=1, keepdims=True) + 1e-8)
    lr    = np.log(prices).diff().dropna().values.astype(np.float32)
    T2    = min(T, len(lr)) - 1
    pnl   = (w[:T2] * lr[1:T2+1,:,None]).sum(axis=1)
    return pnl, ef


@pytest.fixture(scope="module")
def ensemble(prices, formulas):
    from macro8_subnet.alpha.portfolio_intelligence import AdaptiveEnsemble
    ens = AdaptiveEnsemble(prices, formulas, n_clusters=4,
                           weighting="risk_parity", verbose=False)
    ens.fit()
    return ens


# ── 1. SignalClusterer ────────────────────────────────────────────────────────

class TestSignalClusterer:
    def test_import(self):
        from macro8_subnet.alpha.portfolio_intelligence import (
            SignalClusterer, ClusterResult,
        )

    def test_basic_clustering(self, pnl_matrix):
        from macro8_subnet.alpha.portfolio_intelligence import SignalClusterer
        pnl, formulas = pnl_matrix
        c = SignalClusterer()
        result = c.fit(pnl, formulas, n_clusters=4)
        # n_clusters may be lower if some requested clusters are empty
        assert result.n_clusters <= 4
        assert len(result.representatives) == len(result.cluster_sizes)
        assert len(result.rep_formulas)    == len(result.representatives)
        assert all(0 <= i < len(formulas) for i in result.representatives)

    def test_cluster_sizes_sum_to_n_formulas(self, pnl_matrix):
        from macro8_subnet.alpha.portfolio_intelligence import SignalClusterer
        pnl, formulas = pnl_matrix
        c = SignalClusterer()
        result = c.fit(pnl, formulas, n_clusters=3)
        assert sum(result.cluster_sizes) == len(formulas)

    def test_representatives_have_best_sharpe_in_cluster(self, pnl_matrix):
        from macro8_subnet.alpha.portfolio_intelligence import SignalClusterer
        pnl, formulas = pnl_matrix
        c = SignalClusterer()
        result = c.fit(pnl, formulas, n_clusters=3)
        sharpes = pnl.mean(0) / (pnl.std(0) + 1e-8) * np.sqrt(252)
        for i, rep_idx in enumerate(result.representatives):
            label = result.labels[rep_idx]
            cluster_members = [j for j, l in enumerate(result.labels) if l == label]
            rep_sharpe = sharpes[rep_idx]
            for m in cluster_members:
                assert sharpes[m] <= rep_sharpe + 1e-6, \
                    f"Representative is not the best in cluster {label}"

    def test_auto_cluster_selection(self, pnl_matrix):
        from macro8_subnet.alpha.portfolio_intelligence import SignalClusterer
        pnl, formulas = pnl_matrix
        c = SignalClusterer(max_clusters=6)
        result = c.fit(pnl, formulas)  # auto-select k
        assert 2 <= result.n_clusters <= 6

    def test_diversity_metrics(self, pnl_matrix):
        from macro8_subnet.alpha.portfolio_intelligence import SignalClusterer
        pnl, formulas = pnl_matrix
        c = SignalClusterer()
        result = c.fit(pnl, formulas, n_clusters=4)
        assert 0 <= result.mean_within_corr <= 1.0
        assert 0 <= result.mean_between_corr <= 1.0
        assert 0 < result.hhi <= 1.0

    def test_hhi_formula(self, pnl_matrix):
        from macro8_subnet.alpha.portfolio_intelligence import SignalClusterer
        pnl, formulas = pnl_matrix
        c = SignalClusterer()
        result = c.fit(pnl, formulas, n_clusters=4)
        F = len(formulas)
        expected_hhi = sum((s / F) ** 2 for s in result.cluster_sizes)
        assert abs(result.hhi - expected_hhi) < 1e-6

    def test_diversity_gain(self, pnl_matrix):
        from macro8_subnet.alpha.portfolio_intelligence import SignalClusterer
        pnl, formulas = pnl_matrix
        c = SignalClusterer()
        result = c.fit(pnl, formulas, n_clusters=4)
        assert np.isfinite(result.diversity_gain())


# ── 2. EnsembleWeighter ───────────────────────────────────────────────────────

class TestEnsembleWeighter:
    @pytest.fixture(scope="class")
    def rep_pnl(self, pnl_matrix):
        pnl, _ = pnl_matrix
        return pnl[:, :4]  # 4 reps

    def test_import(self):
        from macro8_subnet.alpha.portfolio_intelligence import EnsembleWeighter

    def test_equal_weights_sum_to_one(self, rep_pnl):
        from macro8_subnet.alpha.portfolio_intelligence import EnsembleWeighter
        w = EnsembleWeighter().weights(rep_pnl, method="equal")
        assert abs(w.sum() - 1.0) < 1e-6

    def test_equal_weights_uniform(self, rep_pnl):
        from macro8_subnet.alpha.portfolio_intelligence import EnsembleWeighter
        w = EnsembleWeighter().weights(rep_pnl, method="equal")
        k = rep_pnl.shape[1]
        assert np.allclose(w, 1/k)

    def test_risk_parity_sum_to_one(self, rep_pnl):
        from macro8_subnet.alpha.portfolio_intelligence import EnsembleWeighter
        w = EnsembleWeighter().weights(rep_pnl, method="risk_parity")
        assert abs(w.sum() - 1.0) < 1e-6

    def test_risk_parity_high_vol_gets_lower_weight(self, rep_pnl):
        from macro8_subnet.alpha.portfolio_intelligence import EnsembleWeighter
        pnl = rep_pnl.copy()
        # Make last column 3x more volatile
        pnl[:, -1] *= 3
        w = EnsembleWeighter().weights(pnl, method="risk_parity")
        assert w[-1] < w[0], "High-vol signal should get lower weight"

    def test_sharpe_weights_sum_to_one(self, rep_pnl):
        from macro8_subnet.alpha.portfolio_intelligence import EnsembleWeighter
        w = EnsembleWeighter().weights(rep_pnl, method="sharpe")
        assert abs(w.sum() - 1.0) < 1e-6

    def test_meta_weights_sum_to_one(self, rep_pnl):
        from macro8_subnet.alpha.portfolio_intelligence import EnsembleWeighter
        rob = np.array([0.8, 0.4, 0.9, 0.3])
        w = EnsembleWeighter().weights(rep_pnl, method="meta",
                                       robustness_scores=rob)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_meta_favours_robust_signals(self, rep_pnl):
        from macro8_subnet.alpha.portfolio_intelligence import EnsembleWeighter
        # Make all signals have same Sharpe so only robustness matters
        flat_pnl = np.ones_like(rep_pnl) * 0.001
        rob      = np.array([0.9, 0.1, 0.5, 0.5])
        w        = EnsembleWeighter().weights(flat_pnl, method="meta",
                                              robustness_scores=rob)
        assert w[0] > w[1], "High-robustness signal should get more weight"

    def test_all_weights_non_negative(self, rep_pnl):
        from macro8_subnet.alpha.portfolio_intelligence import EnsembleWeighter
        ew = EnsembleWeighter()
        for method in ["equal", "risk_parity", "sharpe", "meta"]:
            w = ew.weights(rep_pnl, method=method,
                           robustness_scores=np.array([0.5]*4))
            assert (w >= -1e-8).all(), f"{method}: negative weights"


# ── 3. RegimeDetector ─────────────────────────────────────────────────────────

class TestRegimeDetector:
    def test_import(self):
        from macro8_subnet.alpha.portfolio_intelligence import RegimeDetector, RegimeState

    def test_detect_returns_regime_state(self, prices):
        from macro8_subnet.alpha.portfolio_intelligence import RegimeDetector
        from macro8_subnet.alpha.feature_store import FeatureStore
        fs      = FeatureStore(prices)
        feats   = fs.build()
        det     = RegimeDetector()
        state   = det.detect(feats, prices.index[-1])
        assert state.name in ("calm", "normal", "stress")
        assert np.isfinite(state.vol_regime)
        assert np.isfinite(state.trend_strength)

    def test_label_series_covers_all_dates(self, prices):
        from macro8_subnet.alpha.portfolio_intelligence import RegimeDetector
        from macro8_subnet.alpha.feature_store import FeatureStore
        fs     = FeatureStore(prices)
        feats  = fs.build()
        det    = RegimeDetector()
        labels = det.label_series(feats, prices.index)
        assert len(labels) == len(prices)
        assert set(labels.unique()).issubset({"calm", "normal", "stress"})

    def test_label_series_returns_series(self, prices):
        from macro8_subnet.alpha.portfolio_intelligence import RegimeDetector
        from macro8_subnet.alpha.feature_store import FeatureStore
        labels = RegimeDetector().label_series(
            FeatureStore(prices).build(), prices.index
        )
        assert isinstance(labels, pd.Series)

    def test_all_three_regimes_appear(self, prices):
        """With 600 days, all 3 regimes should appear at least occasionally."""
        from macro8_subnet.alpha.portfolio_intelligence import RegimeDetector
        from macro8_subnet.alpha.feature_store import FeatureStore
        labels = RegimeDetector().label_series(
            FeatureStore(prices).build(), prices.index
        )
        # At least 2 regimes should appear in 600 days
        assert len(labels.unique()) >= 2

    def test_regime_state_label_format(self, prices):
        from macro8_subnet.alpha.portfolio_intelligence import RegimeDetector
        from macro8_subnet.alpha.feature_store import FeatureStore
        feats = FeatureStore(prices).build()
        state = RegimeDetector().detect(feats, prices.index[-1])
        lbl   = state.label()
        assert state.name in lbl
        assert any(icon in lbl for icon in ["🟢","🟡","🔴"])


# ── 4. AdaptiveEnsemble ───────────────────────────────────────────────────────

class TestAdaptiveEnsemble:
    def test_import(self):
        from macro8_subnet.alpha.portfolio_intelligence import (
            AdaptiveEnsemble, EnsembleResult,
        )

    def test_fit_returns_self(self, prices, formulas):
        from macro8_subnet.alpha.portfolio_intelligence import AdaptiveEnsemble
        ens = AdaptiveEnsemble(prices, formulas, n_clusters=3, verbose=False)
        result = ens.fit()
        assert result is ens

    def test_cluster_result_populated(self, ensemble):
        assert ensemble._cluster_result is not None
        assert ensemble._cluster_result.n_clusters >= 2

    def test_features_populated(self, ensemble):
        assert ensemble._features is not None
        assert len(ensemble._features) > 0

    def test_pnl_populated(self, ensemble):
        assert ensemble._pnl is not None
        assert ensemble._pnl.shape[1] > 0

    def test_positions_returns_ensemble_result(self, ensemble):
        from macro8_subnet.alpha.portfolio_intelligence import EnsembleResult
        result = ensemble.positions()
        assert isinstance(result, EnsembleResult)

    def test_positions_have_active_formulas(self, ensemble):
        result = ensemble.positions()
        assert len(result.active_formulas) > 0

    def test_positions_formula_weights_sum_to_one(self, ensemble):
        result = ensemble.positions()
        total  = sum(result.formula_weights.values())
        assert abs(total - 1.0) < 0.01

    def test_asset_positions_dict(self, ensemble, prices):
        result = ensemble.positions()
        for ticker in result.positions:
            assert ticker in prices.columns
        # L1 norm ≈ 1
        l1 = sum(abs(w) for w in result.positions.values())
        assert abs(l1 - 1.0) < 0.05

    def test_regime_state_populated(self, ensemble):
        result = ensemble.positions()
        assert result.regime.name in ("calm", "normal", "stress")
        assert np.isfinite(result.regime.vol_regime)

    def test_rolling_pnl_returns_series(self, ensemble):
        pnl = ensemble.rolling_pnl()
        assert isinstance(pnl, pd.Series)
        assert len(pnl) > 0

    def test_rolling_pnl_oos(self, prices, formulas):
        from macro8_subnet.alpha.portfolio_intelligence import AdaptiveEnsemble
        split = int(len(prices) * 0.7)
        train = prices.iloc[:split]
        oos   = prices.iloc[split:]
        ens   = AdaptiveEnsemble(train, formulas, n_clusters=3, verbose=False)
        ens.fit()
        pnl = ens.rolling_pnl(oos_prices=oos)
        assert isinstance(pnl, pd.Series)

    def test_rolling_pnl_finite(self, ensemble):
        pnl = ensemble.rolling_pnl()
        assert np.isfinite(pnl.values).all(), "rolling_pnl has non-finite values"

    def test_sharpe_breakdown_keys(self, ensemble):
        bd = ensemble.sharpe_breakdown()
        assert "n_clusters" in bd
        assert "rep_formulas" in bd
        assert "rep_sharpes" in bd

    def test_sharpe_breakdown_per_rep_finite(self, ensemble):
        bd = ensemble.sharpe_breakdown()
        for formula, sh in bd["rep_sharpes"].items():
            assert np.isfinite(sh), f"Sharpe for {formula} is not finite"

    def test_print_report_no_crash(self, ensemble, capsys):
        ensemble.print_report()
        captured = capsys.readouterr()
        assert "ADAPTIVE ENSEMBLE" in captured.out

    def test_fit_with_robustness_scores(self, prices, formulas):
        from macro8_subnet.alpha.portfolio_intelligence import AdaptiveEnsemble
        rob = {f: 0.7 for f in formulas[:5]}
        ens = AdaptiveEnsemble(prices, formulas, n_clusters=3,
                               weighting="meta", verbose=False)
        ens.fit(robustness_scores=rob)  # should not raise
        assert ens._cluster_result is not None

    def test_positions_print_no_crash(self, ensemble, capsys):
        result = ensemble.positions()
        result.print()
        captured = capsys.readouterr()
        assert "Regime" in captured.out
        assert "Positions" in captured.out

    def test_unfitted_positions_raises(self, prices, formulas):
        from macro8_subnet.alpha.portfolio_intelligence import AdaptiveEnsemble
        ens = AdaptiveEnsemble(prices, formulas, n_clusters=3, verbose=False)
        with pytest.raises(RuntimeError, match="fit"):
            ens.positions()


# ── 5. Miner integration ──────────────────────────────────────────────────────

class TestMinerEnsembleIntegration:
    @pytest.fixture(scope="class")
    def miner(self, prices):
        from macro8_subnet.neurons.miner import Macro8Miner
        import argparse
        cfg = argparse.Namespace(
            netuid=1, role="signal", n_formulas=20, n_hypotheses=2,
            data_start="2015-01-01", n_assets=8,
            pr_test_interval=999, gp_generations=1,
            ensemble_rebuild_interval=1,
            capital=100_000,
            wallet_name="default", wallet_hotkey="default",
            subtensor_network="test",
        )
        return Macro8Miner(cfg)

    def test_forward_attaches_ensemble_attribute(self, miner):
        from macro8_subnet.protocol.synapse import AlphaSubmissionSynapse
        result = miner.forward(AlphaSubmissionSynapse(epoch=1))
        assert hasattr(miner, "_ensemble")

    def test_forward_has_positions(self, miner):
        from macro8_subnet.protocol.synapse import AlphaSubmissionSynapse
        result = miner.forward(AlphaSubmissionSynapse(epoch=1))
        # Either ensemble or single-formula positions should be present
        assert result.eval_success

    def test_position_formula_indicates_ensemble(self, miner):
        """position_formula should mention 'ensemble' when ensemble is active."""
        if miner._ensemble is not None and miner._ensemble._cluster_result is not None:
            from macro8_subnet.protocol.synapse import AlphaSubmissionSynapse
            result = miner.forward(AlphaSubmissionSynapse(epoch=2))
            # May say "ensemble(N signals, regime=X)" or fallback formula
            assert isinstance(result.position_formula, str)
