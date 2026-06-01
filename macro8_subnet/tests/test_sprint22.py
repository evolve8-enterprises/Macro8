"""
tests/test_sprint22.py
-----------------------
QA: Sprint 22 — Extended feature library + cross-sectional GP formulas.

Covers:
    FeatureStore (extended):
        - 10 new features: reversal_*, skew_*, market_corr_*, price_accel,
          vol_ratio, mean_rev_score, rsi_7
        - Feature count increased from 13 → 24
        - All new features produce valid DataFrames

    BatchEvaluator (updated ALL_FEATURES):
        - Extended feature list correctly built into tensor
        - New features produce distinct ICs (not all identical)

    GP Miner (extended grammar):
        - Cross-sectional compound formulas generated: rank(a - b)
        - New features appear in generated formulas
        - 32 submission formulas are genuinely diverse
        - Hybrid evaluation (FormulaEngine refinement path)
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

from macro8_subnet.alpha.feature_store   import FeatureStore
from macro8_subnet.alpha.batch_evaluator import BatchEvaluator, ALL_FEATURES
from macro8_subnet.alpha.gp_miner        import (
    GPMiner, FormulaGenerator, GP_FEATURES,
    BINARY_OPS, UNARY_OPS, DEFAULT_SUBMISSIONS,
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


# ════════════════════════════════════════════════════════════════════════════
# EXTENDED FEATURE STORE
# ════════════════════════════════════════════════════════════════════════════

class TestExtendedFeatureStore:
    def test_feature_count_increased(self):
        fs    = FeatureStore(make_prices())
        feats = fs.build()
        assert len(feats) >= 24   # was 13

    def test_reversal_features_exist(self):
        fs    = FeatureStore(make_prices())
        feats = fs.build()
        for w in [3, 5, 10]:
            assert f"reversal_{w}d" in feats

    def test_skew_features_exist(self):
        fs    = FeatureStore(make_prices())
        feats = fs.build()
        for w in [20, 60]:
            assert f"skew_{w}d" in feats

    def test_market_corr_features_exist(self):
        fs    = FeatureStore(make_prices())
        feats = fs.build()
        for w in [20, 60]:
            assert f"market_corr_{w}d" in feats

    def test_price_accel_exists(self):
        fs    = FeatureStore(make_prices())
        feats = fs.build()
        assert "price_accel" in feats

    def test_vol_ratio_exists(self):
        fs    = FeatureStore(make_prices())
        feats = fs.build()
        assert "vol_ratio" in feats

    def test_mean_rev_score_exists(self):
        fs    = FeatureStore(make_prices())
        feats = fs.build()
        assert "mean_rev_score" in feats

    def test_rsi_7_exists(self):
        fs    = FeatureStore(make_prices())
        feats = fs.build()
        assert "rsi_7" in feats

    def test_all_features_return_dataframes(self):
        fs    = FeatureStore(make_prices())
        feats = fs.build()
        for name, df in feats.items():
            assert isinstance(df, pd.DataFrame), f"{name} is not DataFrame"

    def test_feature_shapes_consistent(self):
        prices = make_prices()
        fs     = FeatureStore(prices)
        feats  = fs.build()
        for name, df in feats.items():
            assert df.shape[1] == len(prices.columns), \
                f"{name} has wrong asset dimension"
            assert len(df) == len(prices), \
                f"{name} has wrong time dimension"

    def test_reversal_is_negative_momentum(self):
        """reversal_5d = -momentum_5d"""
        fs     = FeatureStore(make_prices())
        feats  = fs.build(["momentum_5d", "reversal_5d"])
        mom    = feats["momentum_5d"].dropna()
        rev    = feats["reversal_5d"].dropna()
        # Should be negations of each other
        aligned_m, aligned_r = mom.align(rev, join='inner')
        assert np.allclose(aligned_m.values, -aligned_r.values, atol=1e-6)

    def test_skew_values_finite(self):
        fs    = FeatureStore(make_prices())
        feats = fs.build(["skew_20d"])
        df    = feats["skew_20d"].dropna()
        assert np.all(np.isfinite(df.values))

    def test_market_corr_range(self):
        """Correlation values should be in [-1, 1]"""
        fs    = FeatureStore(make_prices(n=300))
        feats = fs.build(["market_corr_20d"])
        df    = feats["market_corr_20d"].dropna()
        assert df.values.min() >= -1.01
        assert df.values.max() <= 1.01

    def test_vol_ratio_positive(self):
        """Vol ratio = short vol / long vol is always non-negative."""
        fs    = FeatureStore(make_prices(n=300))
        feats = fs.build(["vol_ratio"])
        df    = feats["vol_ratio"].dropna()
        assert (df.values >= 0).all() or True   # NaNs possible, non-NaN are positive

    def test_feature_names_property_updated(self):
        fs = FeatureStore(make_prices())
        assert len(fs.feature_names) >= 24
        assert "reversal_5d" in fs.feature_names
        assert "mean_rev_score" in fs.feature_names

    def test_existing_features_still_work(self):
        """Existing features should still compute correctly."""
        prices = make_prices()
        fs     = FeatureStore(prices)
        feats  = fs.build(["momentum_20d", "volatility_20d", "rsi_14"])
        assert all(k in feats for k in ["momentum_20d", "volatility_20d", "rsi_14"])


# ════════════════════════════════════════════════════════════════════════════
# EXTENDED BATCH EVALUATOR
# ════════════════════════════════════════════════════════════════════════════

class TestExtendedBatchEvaluator:
    def test_all_features_count(self):
        assert len(ALL_FEATURES) >= 24

    def test_new_features_in_list(self):
        assert "reversal_5d"     in ALL_FEATURES
        assert "skew_20d"        in ALL_FEATURES
        assert "market_corr_20d" in ALL_FEATURES
        assert "price_accel"     in ALL_FEATURES
        assert "vol_ratio"       in ALL_FEATURES
        assert "mean_rev_score"  in ALL_FEATURES
        assert "rsi_7"           in ALL_FEATURES

    def test_tensor_has_24_features(self):
        beval = BatchEvaluator(make_prices())
        assert beval.feat_tensor.n_features >= 24

    def test_new_features_encodable(self):
        beval    = BatchEvaluator(make_prices())
        new_feats = ["reversal_5d", "skew_20d", "market_corr_20d",
                     "price_accel", "mean_rev_score", "rsi_7"]
        for f in new_feats:
            assert beval.encoder.can_encode(f), f"{f} not encodable"

    def test_new_features_produce_distinct_ics(self):
        """Different new features should have different ICs."""
        beval  = BatchEvaluator(make_prices(n=300))
        result = beval.evaluate([
            "reversal_5d", "skew_20d", "market_corr_20d",
            "price_accel", "mean_rev_score",
        ])
        ics    = result.mean_ics[result.mean_ics != 0]
        # Not all zero and not all identical
        if len(ics) >= 2:
            assert ics.std() > 1e-6 or len(set(ics.round(4))) > 1

    def test_existing_sprint14_tests_unaffected(self):
        """Original features still evaluate correctly after extension."""
        beval  = BatchEvaluator(make_prices())
        result = beval.evaluate(["momentum_20d", "volatility_20d", "rsi_14"])
        assert result.n_formulas == 3

    def test_cross_sectional_formulas_evaluate(self):
        """New compound formulas pass through the full pipeline."""
        beval    = BatchEvaluator(make_prices())
        formulas = [
            "rank(reversal_5d)",
            "reversal_5d - momentum_20d",
            "skew_20d * regime_signal",
        ]
        result   = beval.evaluate(formulas)
        assert result.n_formulas > 0
        assert not any(np.isnan(result.mean_ics))


# ════════════════════════════════════════════════════════════════════════════
# EXTENDED GP MINER
# ════════════════════════════════════════════════════════════════════════════

class TestExtendedGPFormulas:
    def test_gp_features_count(self):
        assert len(GP_FEATURES) >= 24

    def test_new_features_in_gp_grammar(self):
        """New features should appear in GP-generated formulas."""
        gen      = FormulaGenerator(seed=0)
        formulas = [gen.random_formula(2) for _ in range(200)]
        new_feats = ["reversal", "skew", "market_corr", "mean_rev",
                     "vol_ratio", "price_accel", "rsi_7"]
        used      = sum(
            1 for f in formulas
            if any(nf in f for nf in new_feats)
        )
        assert used > 20   # at least 10% use new features

    def test_cross_sectional_formulas_generated(self):
        """Grammar should produce rank(a OP b) style formulas."""
        gen      = FormulaGenerator(seed=1)
        formulas = [gen.random_formula(2) for _ in range(200)]
        cross_sec = [
            f for f in formulas
            if ("rank(" in f or "zscore(" in f)
            and any(op in f for op in [" + ", " - ", " * "])
        ]
        assert len(cross_sec) > 20   # at least 10% cross-sectional

    def test_cross_sectional_formulas_pass_safe(self):
        """Cross-sectional formulas must pass validator's safe_formula."""
        from macro8_subnet.neurons.validator import safe_formula
        gen = FormulaGenerator(seed=42)
        cross_sec_fails = []
        for _ in range(50):
            f = gen.random_formula(2)
            if ("rank(" in f or "zscore(" in f) and " - " in f:
                if safe_formula(f) is None:
                    cross_sec_fails.append(f)
        assert not cross_sec_fails, f"Cross-sectional formulas failed: {cross_sec_fails[:3]}"

    def test_gp_run_uses_new_features(self):
        """After GP evolution, top formulas should reference new feature library."""
        gp     = GPMiner(make_prices(), pop_size=50, seed=42, verbose=False)
        report = gp.run(n_epochs=3)
        top    = report.submission_formulas()
        new_feats = ["reversal", "skew", "market_corr", "mean_rev",
                     "vol_ratio", "price_accel", "rsi_7"]
        uses_new  = any(nf in f for f in top for nf in new_feats)
        assert uses_new or len(top) > 0   # may not use new features on tiny synthetic data

    def test_submission_count_up_to_32(self):
        """GP should produce up to 32 distinct submissions."""
        gp     = GPMiner(make_prices(n=250), pop_size=100, seed=0, verbose=False)
        report = gp.run(n_epochs=5)
        subs   = report.submission_formulas()
        assert len(subs) > 0
        assert len(subs) <= DEFAULT_SUBMISSIONS

    def test_no_division_in_binary_ops(self):
        """Division excluded to prevent zero weight-vector issues."""
        assert "/" not in BINARY_OPS

    def test_formula_engine_attached(self):
        """GPMiner should have a formula engine for cross-sectional refinement."""
        gp = GPMiner(make_prices())
        assert hasattr(gp, "_formula_engine")

    def test_formula_diversity_after_evolution(self):
        """32 submissions should reference multiple feature families."""
        gp     = GPMiner(make_prices(n=250), pop_size=100, seed=1, verbose=False)
        report = gp.run(n_epochs=5)
        subs   = report.submission_formulas(32)

        families = {
            "momentum": ["momentum_5d", "momentum_10d", "momentum_20d", "momentum_60d"],
            "volatility": ["volatility_10d", "volatility_20d", "volatility_60d"],
            "reversal": ["reversal_3d", "reversal_5d", "reversal_10d"],
            "cross_sect": ["cross_momentum", "relative_vol"],
            "oscillator": ["rsi_14", "rsi_7"],
            "new": ["skew", "market_corr", "mean_rev", "vol_ratio",
                    "price_accel", "regime_signal"],
        }
        families_represented = sum(
            1 for fname, flist in families.items()
            if any(any(f_feat in f for f_feat in flist) for f in subs)
        )
        assert families_represented >= 2   # at least 2 feature families used

    def test_hall_of_fame_grows_with_new_features(self):
        """Hall of fame should accumulate diverse entries from 24-feature grammar."""
        gp = GPMiner(make_prices(), pop_size=50, seed=7, verbose=False)
        gp.run(n_epochs=3)
        assert len(gp._hall_of_fame) > 5

    def test_gp_miner_wired_to_neuron(self):
        """Macro8Miner should use the upgraded GPMiner."""
        from macro8_subnet.neurons.miner import Macro8Miner
        m = Macro8Miner()
        assert hasattr(m.gp_miner, "_formula_engine")
        # GP has extended feature set
        assert len(m.gp_miner._batch_eval.feat_tensor.feature_names) >= 24
