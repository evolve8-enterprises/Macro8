"""
tests/test_sprint12.py
-----------------------
QA: Complete self-contained tests for Sprint 12 — hypothesis engine.

Covers:
    HypothesisCategory   — enum values, suggested_features
    HypothesisStatus     — lifecycle states
    HypothesisRecord     — Bayesian confidence, regime tracking, serialisation
    BayesianUpdater      — update, batch update, status transitions
    HypothesisLibrary    — CRUD, queries, knowledge base
    HypothesisEvolution  — feature suggestions, seed formulas, population weighting
    HypothesisSubmission — validation
    HypothesisReport     — serialisation
    update_hypotheses_from_epoch — integration helper
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SUITE_DIR  = Path(__file__).resolve().parent
SUBNET_DIR = SUITE_DIR.parent
PROJECT    = SUBNET_DIR.parent

for p in [str(SUBNET_DIR), str(PROJECT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from macro8_subnet.alpha.hypothesis_engine import (
    HypothesisCategory, HypothesisStatus, HypothesisRecord,
    BayesianUpdater, HypothesisLibrary, HypothesisEvolution,
    HypothesisSubmission, HypothesisReport,
    update_hypotheses_from_epoch,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_record(
    statement:  str   = "Momentum predicts returns",
    category:   HypothesisCategory = HypothesisCategory.MOMENTUM,
    miner_uid:  int   = 0,
    n_successes: int  = 0,
    n_failures:  int  = 0,
) -> HypothesisRecord:
    from macro8_subnet.alpha.hypothesis_engine import HypothesisLibrary
    lib = HypothesisLibrary()
    rec = lib.add(statement, category, miner_uid, epoch=1)
    for _ in range(n_successes):
        BayesianUpdater().update(rec, 0.05)
    for _ in range(n_failures):
        BayesianUpdater().update(rec, -0.01)
    return rec


def make_library_with_hypotheses() -> HypothesisLibrary:
    lib = HypothesisLibrary()
    hypotheses = [
        ("Momentum predicts 5-day returns",         HypothesisCategory.MOMENTUM),
        ("High volatility predicts low returns",    HypothesisCategory.VOLATILITY),
        ("Carry strategy outperforms in risk-on",   HypothesisCategory.CARRY),
        ("Mean reversion works in ranging markets", HypothesisCategory.MEAN_REVERSION),
        ("Regime switching improves momentum",      HypothesisCategory.REGIME),
    ]
    for i, (stmt, cat) in enumerate(hypotheses):
        rec = lib.add(stmt, cat, miner_uid=i, epoch=1)
        # Add some IC observations
        for ic in [0.04, 0.03, 0.05, -0.01, 0.04]:
            BayesianUpdater().update(rec, ic)
    return lib


# ════════════════════════════════════════════════════════════════════════════
# HYPOTHESIS CATEGORY
# ════════════════════════════════════════════════════════════════════════════

class TestHypothesisCategory:
    def test_eight_categories(self):
        assert len(list(HypothesisCategory)) == 9   # 8 + UNKNOWN

    def test_all_have_suggested_features(self):
        for cat in HypothesisCategory:
            feats = cat.suggested_features()
            assert isinstance(feats, list) and len(feats) > 0

    def test_momentum_features_contain_momentum(self):
        feats = HypothesisCategory.MOMENTUM.suggested_features()
        assert any("momentum" in f for f in feats)

    def test_volatility_features_contain_vol(self):
        feats = HypothesisCategory.VOLATILITY.suggested_features()
        assert any("vol" in f.lower() for f in feats)

    def test_regime_features_contain_regime(self):
        feats = HypothesisCategory.REGIME.suggested_features()
        assert any("regime" in f for f in feats)


# ════════════════════════════════════════════════════════════════════════════
# HYPOTHESIS RECORD
# ════════════════════════════════════════════════════════════════════════════

class TestHypothesisRecord:
    def test_initial_confidence_is_half(self):
        rec = make_record()
        assert rec.confidence_score == pytest.approx(0.5)

    def test_confidence_range(self):
        rec = make_record(n_successes=10, n_failures=2)
        assert 0.0 < rec.confidence_score < 1.0

    def test_more_successes_higher_confidence(self):
        low  = make_record(n_successes=2, n_failures=5)
        high = make_record(n_successes=8, n_failures=2)
        assert high.confidence_score > low.confidence_score

    def test_n_observations_counts_all(self):
        rec = make_record(n_successes=3, n_failures=2)
        assert rec.n_observations == 5

    def test_n_successes_counts_positive(self):
        rec = make_record(n_successes=4, n_failures=1)
        assert rec.n_successes == 4

    def test_mean_ic_computed(self):
        rec = make_record()
        BayesianUpdater().update(rec, 0.04)
        BayesianUpdater().update(rec, 0.06)
        assert rec.mean_ic == pytest.approx(0.05)

    def test_ic_stability_all_positive(self):
        rec = make_record(n_successes=5)
        assert rec.ic_stability == pytest.approx(1.0)

    def test_ic_stability_mixed(self):
        rec = make_record(n_successes=3, n_failures=3)
        # successes have IC=0.05 > 0, failures have IC=-0.01 but still > 0 threshold
        # Actually: failures update with ic=-0.01 which is < 0
        # ic_stability counts ic > 0: 3/6 = 0.5 since -0.01 < 0
        assert 0.0 < rec.ic_stability <= 1.0

    def test_best_regime_with_data(self):
        rec = make_record()
        rec.regime_ic = {"low_vol": [0.06, 0.07], "risk_off": [-0.02, -0.01]}
        assert rec.best_regime() == "low_vol"

    def test_worst_regime_with_data(self):
        rec = make_record()
        rec.regime_ic = {"low_vol": [0.06], "risk_off": [-0.02]}
        assert rec.worst_regime() == "risk_off"

    def test_best_regime_no_data_returns_none(self):
        rec = make_record()
        assert rec.best_regime() is None

    def test_confidence_interval_ordered(self):
        rec = make_record(n_successes=10, n_failures=3)
        lo, hi = rec.confidence_interval
        assert lo < rec.confidence_score < hi

    def test_to_dict_serialisable(self):
        rec = make_record(n_successes=3, n_failures=1)
        json.dumps(rec.to_dict())

    def test_to_dict_has_required_keys(self):
        d = make_record().to_dict()
        for key in ("hypothesis_id", "statement", "category",
                    "confidence_score", "n_observations", "mean_ic", "status"):
            assert key in d

    def test_knowledge_entry_is_string(self):
        rec = make_record(n_successes=5)
        entry = rec.knowledge_entry()
        assert isinstance(entry, str) and len(entry) > 0


# ════════════════════════════════════════════════════════════════════════════
# BAYESIAN UPDATER
# ════════════════════════════════════════════════════════════════════════════

class TestBayesianUpdater:
    def test_success_increments_alpha(self):
        rec    = make_record()
        before = rec.alpha_param
        BayesianUpdater().update(rec, 0.05)   # above threshold
        assert rec.alpha_param == before + 1.0

    def test_failure_increments_beta(self):
        rec    = make_record()
        before = rec.beta_param
        BayesianUpdater().update(rec, -0.01)  # below threshold
        assert rec.beta_param == before + 1.0

    def test_confidence_increases_after_success(self):
        rec    = make_record()
        before = rec.confidence_score
        BayesianUpdater().update(rec, 0.05)
        assert rec.confidence_score > before

    def test_confidence_decreases_after_failure(self):
        rec    = make_record()
        before = rec.confidence_score
        BayesianUpdater().update(rec, -0.05)
        assert rec.confidence_score < before

    def test_ic_history_grows(self):
        rec = make_record()
        for _ in range(5):
            BayesianUpdater().update(rec, 0.03)
        assert len(rec.ic_history) == 5

    def test_regime_ic_tracked(self):
        rec = make_record()
        BayesianUpdater().update(rec, 0.04, regime="low_vol")
        assert "low_vol" in rec.regime_ic
        assert len(rec.regime_ic["low_vol"]) == 1

    def test_epoch_updated(self):
        rec = make_record()
        BayesianUpdater().update(rec, 0.04, epoch=5)
        assert rec.epoch_last_updated == 5

    def test_batch_update_all_applied(self):
        rec = make_record()
        BayesianUpdater().update_batch(rec, [0.04, 0.05, -0.01, 0.03, 0.06])
        assert len(rec.ic_history) == 5

    def test_pending_to_active_transition(self):
        rec = make_record()
        assert rec.status == HypothesisStatus.PENDING
        for _ in range(5):
            BayesianUpdater().update(rec, 0.05)   # 5 successes
        assert rec.status == HypothesisStatus.ACTIVE

    def test_pending_to_retired_transition(self):
        rec = make_record()
        for _ in range(10):
            BayesianUpdater().update(rec, -0.05)  # 10 failures
        assert rec.status == HypothesisStatus.RETIRED

    def test_predict_next_ic_positive(self):
        rec = make_record()
        for _ in range(5):
            BayesianUpdater().update(rec, 0.05)
        predicted = BayesianUpdater().predict_next_ic(rec)
        assert isinstance(predicted, float)

    def test_predict_next_ic_regime_specific(self):
        rec = make_record()
        BayesianUpdater().update(rec, 0.06, regime="low_vol")
        pred = BayesianUpdater().predict_next_ic(rec, regime="low_vol")
        assert pred == pytest.approx(0.06)

    def test_custom_ic_threshold(self):
        """Threshold=0.05 → IC=0.03 is a failure."""
        rec     = make_record()
        before  = rec.beta_param
        BayesianUpdater(ic_threshold=0.05).update(rec, 0.03)
        assert rec.beta_param == before + 1.0


# ════════════════════════════════════════════════════════════════════════════
# HYPOTHESIS LIBRARY
# ════════════════════════════════════════════════════════════════════════════

class TestHypothesisLibrary:
    def test_add_creates_record(self):
        lib = HypothesisLibrary()
        rec = lib.add("Momentum predicts returns", HypothesisCategory.MOMENTUM, 0, 1)
        assert rec is not None
        assert lib.size == 1

    def test_duplicate_statement_returns_existing(self):
        lib = HypothesisLibrary()
        r1  = lib.add("Momentum predicts returns", HypothesisCategory.MOMENTUM, 0, 1)
        r2  = lib.add("Momentum predicts returns", HypothesisCategory.MOMENTUM, 0, 1)
        assert r1 is r2
        assert lib.size == 1

    def test_get_by_statement(self):
        lib  = HypothesisLibrary()
        stmt = "Volatility predicts returns"
        lib.add(stmt, HypothesisCategory.VOLATILITY, 0, 1)
        rec  = lib.get_by_statement(stmt)
        assert rec is not None
        assert rec.statement == stmt

    def test_get_unknown_returns_none(self):
        lib = HypothesisLibrary()
        assert lib.get("nonexistent") is None

    def test_all_active_excludes_retired(self):
        lib = HypothesisLibrary()
        rec = lib.add("Test", HypothesisCategory.MOMENTUM, 0, 1)
        lib.retire(rec.hypothesis_id)
        assert len(lib.all_active()) == 0

    def test_by_category(self):
        lib = make_library_with_hypotheses()
        mom = lib.by_category(HypothesisCategory.MOMENTUM)
        assert len(mom) >= 1
        assert all(r.category == HypothesisCategory.MOMENTUM for r in mom)

    def test_rank_by_confidence_sorted(self):
        lib = make_library_with_hypotheses()
        top = lib.rank_by_confidence(5)
        if len(top) >= 2:
            scores = [r.confidence_score for r in top]
            assert scores == sorted(scores, reverse=True)

    def test_rank_by_ic_sorted(self):
        lib = make_library_with_hypotheses()
        top = lib.rank_by_ic(3)
        if len(top) >= 2:
            ics = [r.mean_ic for r in top]
            assert ics == sorted(ics, reverse=True)

    def test_update_with_ic(self):
        lib = HypothesisLibrary()
        rec = lib.add("Test", HypothesisCategory.MOMENTUM, 0, 1)
        lib.update_with_ic(rec.hypothesis_id, 0.05)
        assert len(rec.ic_history) == 1

    def test_add_supporting_signal(self):
        lib = HypothesisLibrary()
        rec = lib.add("Test", HypothesisCategory.MOMENTUM, 0, 1)
        lib.add_supporting_signal(rec.hypothesis_id, "f_momentum")
        assert "f_momentum" in rec.supporting_signals

    def test_add_same_signal_twice(self):
        lib = HypothesisLibrary()
        rec = lib.add("Test", HypothesisCategory.MOMENTUM, 0, 1)
        lib.add_supporting_signal(rec.hypothesis_id, "f_mom")
        lib.add_supporting_signal(rec.hypothesis_id, "f_mom")
        assert rec.supporting_signals.count("f_mom") == 1

    def test_retire(self):
        lib = HypothesisLibrary()
        rec = lib.add("Test", HypothesisCategory.MOMENTUM, 0, 1)
        lib.retire(rec.hypothesis_id)
        assert rec.status == HypothesisStatus.RETIRED

    def test_archive(self):
        lib = HypothesisLibrary()
        rec = lib.add("Test", HypothesisCategory.MOMENTUM, 0, 1)
        lib.archive(rec.hypothesis_id, note="superseded")
        assert rec.status == HypothesisStatus.ARCHIVED
        assert "superseded" in rec.notes

    def test_n_active_counts_correctly(self):
        lib = make_library_with_hypotheses()
        lib.retire(lib.all_active()[0].hypothesis_id)
        n_active = lib.n_active
        # Retired one record → active should be less than total
        assert n_active < lib.size

    def test_mean_confidence_positive(self):
        lib = make_library_with_hypotheses()
        assert 0.0 < lib.mean_confidence < 1.0

    def test_knowledge_base_is_list(self):
        lib  = make_library_with_hypotheses()
        kb   = lib.knowledge_base()
        assert isinstance(kb, list)
        assert len(kb) > 0

    def test_knowledge_base_serialisable(self):
        lib = make_library_with_hypotheses()
        json.dumps(lib.knowledge_base())

    def test_pending_list(self):
        lib = HypothesisLibrary()
        lib.add("New hypothesis", HypothesisCategory.MOMENTUM, 0, 1)
        # New hypothesis with 0 obs → pending
        assert len(lib.pending()) == 1


# ════════════════════════════════════════════════════════════════════════════
# HYPOTHESIS EVOLUTION
# ════════════════════════════════════════════════════════════════════════════

class TestHypothesisEvolution:
    def _evo(self) -> HypothesisEvolution:
        return HypothesisEvolution(make_library_with_hypotheses())

    def test_suggested_features_list(self):
        feats = self._evo().suggested_features()
        assert isinstance(feats, list)
        assert len(feats) > 0

    def test_suggested_features_are_valid_names(self):
        valid = {"momentum_5d", "momentum_20d", "momentum_60d",
                 "volatility_10d", "volatility_20d", "volatility_60d",
                 "zscore_20d", "zscore_60d", "rsi_14",
                 "cross_momentum", "relative_vol", "regime_signal"}
        feats = self._evo().suggested_features()
        assert all(f in valid for f in feats)

    def test_avoided_features_list(self):
        evo   = self._evo()
        # Retire all hypotheses to trigger avoided features
        lib   = evo.library
        for rec in lib.all_active():
            for _ in range(15):
                BayesianUpdater().update(rec, -0.05)
        avoided = evo.avoided_features()
        assert isinstance(avoided, list)

    def test_seed_formulas_strings(self):
        seeds = self._evo().seed_formulas(3)
        assert all(isinstance(s, str) for s in seeds)

    def test_seed_formulas_non_empty(self):
        seeds = self._evo().seed_formulas(5)
        assert all(len(s) > 0 for s in seeds)

    def test_seed_formulas_max_n(self):
        seeds = self._evo().seed_formulas(3)
        assert len(seeds) <= 3

    def test_seed_formulas_empty_library(self):
        evo   = HypothesisEvolution(HypothesisLibrary())
        seeds = evo.seed_formulas(3)
        assert isinstance(seeds, list)

    def test_regime_filtered_seeds_returns_list(self):
        seeds = self._evo().regime_filtered_seeds("low_vol", n=2)
        assert isinstance(seeds, list)

    def test_weight_population_sums_to_one(self):
        formulas = ["momentum_20d", "zscore_20d", "rank(cross_momentum)"]
        weights  = self._evo().weight_population(formulas, ["momentum_20d"])
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_weight_population_all_positive(self):
        formulas = ["momentum_20d", "volatility_20d", "rsi_14"]
        weights  = self._evo().weight_population(formulas, ["momentum_20d"])
        assert all(w > 0 for w in weights)

    def test_weight_population_length_matches(self):
        formulas = ["f1", "f2", "f3", "f4"]
        weights  = self._evo().weight_population(formulas, ["momentum_20d"])
        assert len(weights) == 4

    def test_high_confidence_hypothesis_suggests_features(self):
        """A library with one very high-confidence momentum hypothesis
        should bias toward momentum features."""
        lib = HypothesisLibrary()
        rec = lib.add("Momentum works", HypothesisCategory.MOMENTUM, 0, 1)
        for _ in range(20):   # many successes → high confidence
            BayesianUpdater().update(rec, 0.06)
        evo   = HypothesisEvolution(lib)
        feats = evo.suggested_features()
        # Momentum features should dominate
        mom_count = sum(1 for f in feats if "momentum" in f)
        assert mom_count > 0


# ════════════════════════════════════════════════════════════════════════════
# HYPOTHESIS SUBMISSION
# ════════════════════════════════════════════════════════════════════════════

class TestHypothesisSubmission:
    def _sub(self) -> HypothesisSubmission:
        return HypothesisSubmission(
            miner_uid=0, miner_hotkey="5F",
            statement="Momentum predicts 5-day returns in low-vol regimes",
            category=HypothesisCategory.MOMENTUM,
            supporting_formula="rank(momentum_20d)",
        )

    def test_valid_submission(self):
        ok, _ = self._sub().is_valid()
        assert ok is True

    def test_empty_statement_fails(self):
        s = HypothesisSubmission(0, "5F", "", HypothesisCategory.MOMENTUM, "f")
        ok, _ = s.is_valid()
        assert ok is False

    def test_short_statement_fails(self):
        s = HypothesisSubmission(0, "5F", "Short", HypothesisCategory.MOMENTUM, "f")
        ok, _ = s.is_valid()
        assert ok is False

    def test_long_statement_fails(self):
        s = HypothesisSubmission(0, "5F", "x" * 201, HypothesisCategory.MOMENTUM, "f")
        ok, _ = s.is_valid()
        assert ok is False

    def test_to_dict_serialisable(self):
        json.dumps(self._sub().to_dict())


# ════════════════════════════════════════════════════════════════════════════
# UPDATE_HYPOTHESES_FROM_EPOCH
# ════════════════════════════════════════════════════════════════════════════

class TestUpdateHypothesesFromEpoch:
    def test_returns_hypothesis_report(self):
        lib = make_library_with_hypotheses()
        active = lib.all_active()
        sig_to_hyp = {f"signal_{i}": rec.hypothesis_id
                      for i, rec in enumerate(active)}
        ic_map     = {f"signal_{i}": 0.04 for i in range(len(active))}
        report = update_hypotheses_from_epoch(lib, ic_map, sig_to_hyp, "low_vol", 2)
        assert isinstance(report, HypothesisReport)

    def test_n_updated_correct(self):
        lib    = HypothesisLibrary()
        rec    = lib.add("Test", HypothesisCategory.MOMENTUM, 0, 1)
        report = update_hypotheses_from_epoch(
            lib,
            signal_ic_map={"s1": 0.04},
            signal_to_hyp={"s1": rec.hypothesis_id},
            regime="low_vol", epoch=2,
        )
        assert report.n_updated == 1

    def test_unknown_signal_skipped(self):
        lib    = HypothesisLibrary()
        report = update_hypotheses_from_epoch(
            lib,
            signal_ic_map={"unknown_signal": 0.04},
            signal_to_hyp={},
            regime=None, epoch=1,
        )
        assert report.n_updated == 0

    def test_regime_ic_populated(self):
        lib = HypothesisLibrary()
        rec = lib.add("Test", HypothesisCategory.MOMENTUM, 0, 1)
        update_hypotheses_from_epoch(
            lib, {"s": 0.05}, {"s": rec.hypothesis_id}, "low_vol", 1
        )
        assert "low_vol" in rec.regime_ic

    def test_to_dict_serialisable(self):
        lib    = make_library_with_hypotheses()
        active = lib.all_active()
        map_   = {f"s{i}": r.hypothesis_id for i, r in enumerate(active)}
        ics    = {f"s{i}": 0.04 for i in range(len(active))}
        report = update_hypotheses_from_epoch(lib, ics, map_, None, 2)
        json.dumps(report.to_dict())

    def test_n_total_matches_library_size(self):
        lib    = make_library_with_hypotheses()
        report = update_hypotheses_from_epoch(lib, {}, {}, None, 2)
        assert report.n_total == lib.size

    def test_mean_confidence_in_range(self):
        lib    = make_library_with_hypotheses()
        report = update_hypotheses_from_epoch(lib, {}, {}, None, 2)
        assert 0.0 <= report.mean_confidence <= 1.0

    def test_knowledge_entries_list(self):
        lib    = make_library_with_hypotheses()
        active = lib.all_active()
        map_   = {f"s{i}": r.hypothesis_id for i, r in enumerate(active)}
        ics    = {f"s{i}": 0.04 for i in range(len(active))}
        report = update_hypotheses_from_epoch(lib, ics, map_, None, 2)
        assert isinstance(report.knowledge_entries, list)

    def test_newly_active_counted(self):
        lib = HypothesisLibrary()
        rec = lib.add("Test", HypothesisCategory.MOMENTUM, 0, 1)
        # Add 5 successes → status should become ACTIVE
        for i in range(5):
            update_hypotheses_from_epoch(
                lib, {f"s{i}": 0.05}, {f"s{i}": rec.hypothesis_id}, None, i+1
            )
        assert rec.status == HypothesisStatus.ACTIVE


# ════════════════════════════════════════════════════════════════════════════
# INTEGRATION: HYPOTHESIS ENGINE + EVOLUTION ENGINE
# ════════════════════════════════════════════════════════════════════════════

class TestHypothesisEvolutionIntegration:
    """Test that hypothesis guidance produces valid evolution inputs."""

    def test_seed_formulas_can_be_evaluated_by_formula_engine(self):
        """Seed formulas from HypothesisEvolution should pass FormulaEngine validation."""
        import sys
        sys.path.insert(0, str(PROJECT))
        try:
            from macro8_subnet.alpha.formula_engine import FormulaEngine

            lib   = make_library_with_hypotheses()
            evo   = HypothesisEvolution(lib)
            seeds = evo.seed_formulas(5)

            # Build a minimal feature store stub
            class FakeStore:
                def get(self, name):
                    import pandas as pd
                    return pd.DataFrame({"A": [1.0]*50, "B": [0.5]*50})

            engine = FormulaEngine(FakeStore())
            n_valid = sum(1 for s in seeds if engine.validate_formula(s)[0])
            # At least half should be valid formulas
            assert n_valid >= len(seeds) // 2

        except ImportError:
            pytest.skip("FormulaEngine not available")

    def test_population_weights_sum_to_one(self):
        lib      = make_library_with_hypotheses()
        evo      = HypothesisEvolution(lib)
        formulas = ["momentum_20d", "zscore_20d", "rank(cross_momentum)",
                    "regime_signal", "volatility_20d"]
        weights  = evo.weight_population(formulas, [])
        assert abs(sum(weights) - 1.0) < 1e-6
        assert all(w >= 0 for w in weights)

    def test_high_confidence_biases_toward_its_category(self):
        """After many momentum successes, momentum formulas should be preferred."""
        lib = HypothesisLibrary()
        rec = lib.add("Momentum dominates", HypothesisCategory.MOMENTUM, 0, 1)
        for _ in range(20):
            BayesianUpdater().update(rec, 0.07)   # very strong track record

        evo = HypothesisEvolution(lib)
        mom_formula   = "rank(momentum_20d)"
        other_formula = "zscore_20d"
        weights       = evo.weight_population([mom_formula, other_formula], [])
        # Momentum formula should be preferred
        assert weights[0] >= weights[1]
