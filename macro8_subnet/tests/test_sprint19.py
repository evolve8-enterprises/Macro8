"""
tests/test_sprint19.py
-----------------------
QA: Sprint 19 — Multi-component signal scoring engine.

Covers:
    evaluation/signal_scorer.py
        ScoreComponent         — structure, serialisation
        SignalScoreResult      — breakdown, serialisation
        SignalScorer           — all five components, EMA smoothing
        normalise_rewards      — softmax normalisation

    Integration
        Real alpha vs noise discrimination
        Consistent > unstable at same mean IC
        Stable decay (non-decaying) > fast-decaying
        EMA smoothing reduces weight volatility
"""

from __future__ import annotations

import json
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

from macro8_subnet.evaluation.signal_scorer import (
    SignalScorer, SignalScoreResult, ScoreComponent,
    normalise_rewards, DEFAULT_WEIGHTS,
    IC_REFERENCE, STABILITY_REFERENCE, HALF_LIFE_FULL,
)
from macro8_subnet.alpha.capacity_model import LifecycleState
from macro8_subnet.alpha.batch_evaluator import BatchEvaluator


# ── Shared fixtures ───────────────────────────────────────────────────────────

def make_prices(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "SPY":  100 * np.cumprod(1 + rng.normal(0.0005, 0.012, n)),
        "AAPL": 100 * np.cumprod(1 + rng.normal(0.0006, 0.015, n)),
        "GLD":  100 * np.cumprod(1 + rng.normal(0.0003, 0.009, n)),
    }, index=dates)


def make_scorer(prices=None) -> SignalScorer:
    beval = BatchEvaluator(prices or make_prices())
    return SignalScorer(beval)


def make_result(
    composite: float = 0.5,
    lifecycle: LifecycleState = LifecycleState.PRODUCTION,
) -> SignalScoreResult:
    mult = lifecycle.weight_multiplier
    return SignalScoreResult(
        formula_id="test_id",
        formula_string="rank(momentum_20d)",
        composite_score=composite,
        lifecycle_state=lifecycle,
        lifecycle_mult=mult,
        final_reward=composite * mult,
        components=[
            ScoreComponent("ic",        0.04, 0.55, 0.40, 0.22),
            ScoreComponent("stability", 2.00, 0.86, 0.20, 0.17),
            ScoreComponent("decay",     25.0, 0.83, 0.15, 0.12),
            ScoreComponent("novelty",   0.20, 0.80, 0.15, 0.12),
            ScoreComponent("capacity",  0.65, 0.65, 0.10, 0.065),
        ],
    )


# ════════════════════════════════════════════════════════════════════════════
# ScoreComponent
# ════════════════════════════════════════════════════════════════════════════

class TestScoreComponent:
    def test_to_dict_has_keys(self):
        c = ScoreComponent("ic", 0.04, 0.55, 0.40, 0.22)
        d = c.to_dict()
        for key in ("name", "raw", "score", "weight", "weighted"):
            assert key in d

    def test_to_dict_serialisable(self):
        c = ScoreComponent("stability", 1.5, 0.78, 0.20, 0.156)
        json.dumps(c.to_dict())

    def test_weighted_is_score_times_weight(self):
        score, weight = 0.7, 0.40
        c = ScoreComponent("ic", 0.05, score, weight, score * weight)
        assert c.weighted == pytest.approx(score * weight)


# ════════════════════════════════════════════════════════════════════════════
# SignalScoreResult
# ════════════════════════════════════════════════════════════════════════════

class TestSignalScoreResult:
    def test_breakdown_is_string(self):
        r = make_result()
        assert isinstance(r.breakdown(), str)

    def test_breakdown_contains_formula(self):
        r = make_result()
        assert "momentum" in r.breakdown()

    def test_to_dict_has_keys(self):
        d = make_result().to_dict()
        for key in ("formula_id", "composite_score", "lifecycle",
                    "final_reward", "components"):
            assert key in d

    def test_to_dict_serialisable(self):
        json.dumps(make_result().to_dict())

    def test_final_reward_is_composite_times_mult(self):
        r = make_result(composite=0.6, lifecycle=LifecycleState.PRODUCTION)
        assert r.final_reward == pytest.approx(0.6 * 1.0, abs=0.001)

    def test_retired_lifecycle_gives_zero_reward(self):
        r = make_result(composite=0.8, lifecycle=LifecycleState.RETIRED)
        assert r.final_reward == pytest.approx(0.0)

    def test_experimental_lifecycle_reduces_reward(self):
        r_exp  = make_result(composite=0.5, lifecycle=LifecycleState.EXPERIMENTAL)
        r_prod = make_result(composite=0.5, lifecycle=LifecycleState.PRODUCTION)
        assert r_exp.final_reward < r_prod.final_reward


# ════════════════════════════════════════════════════════════════════════════
# SignalScorer — component tests
# ════════════════════════════════════════════════════════════════════════════

class TestICComponent:
    def test_zero_ic_zero_score(self):
        scorer = make_scorer()
        c = scorer._ic_score(0.0)
        assert c.score == pytest.approx(0.0)

    def test_negative_ic_zero_score(self):
        scorer = make_scorer()
        c = scorer._ic_score(-0.03)
        assert c.score == pytest.approx(0.0)

    def test_reference_ic_gives_nonzero_score(self):
        scorer = make_scorer()
        c = scorer._ic_score(IC_REFERENCE)
        assert c.score > 0.5   # soft saturation, ~0.63 at reference

    def test_high_ic_approaches_one(self):
        scorer = make_scorer()
        c = scorer._ic_score(IC_REFERENCE * 5)
        assert c.score > 0.99

    def test_ic_score_monotone(self):
        scorer = make_scorer()
        ics    = [0.01, 0.02, 0.03, 0.05, 0.08]
        scores = [scorer._ic_score(ic).score for ic in ics]
        assert all(scores[i] <= scores[i+1] for i in range(len(scores)-1))

    def test_weight_correct(self):
        scorer = make_scorer()
        c = scorer._ic_score(0.04)
        assert c.weight == pytest.approx(DEFAULT_WEIGHTS["ic"])


class TestStabilityComponent:
    def test_zero_ir_zero_score(self):
        scorer = make_scorer()
        c = scorer._stability_score(0.0)
        assert c.score == pytest.approx(0.0)

    def test_reference_ir_gives_nonzero(self):
        scorer = make_scorer()
        c = scorer._stability_score(STABILITY_REFERENCE)
        assert c.score > 0.5

    def test_high_ir_high_score(self):
        scorer = make_scorer()
        c = scorer._stability_score(5.0)
        assert c.score > 0.99

    def test_negative_ir_zero_score(self):
        scorer = make_scorer()
        c = scorer._stability_score(-0.5)
        assert c.score == pytest.approx(0.0)

    def test_weight_correct(self):
        scorer = make_scorer()
        c = scorer._stability_score(1.0)
        assert c.weight == pytest.approx(DEFAULT_WEIGHTS["stability"])


class TestDecayComponent:
    def test_few_observations_neutral(self):
        scorer = make_scorer()
        c = scorer._decay_score([0.04, 0.03], "f1")
        assert c.score == pytest.approx(0.5)   # neutral with < 4 obs

    def test_stable_ic_full_score(self):
        scorer = make_scorer()
        stable = [0.04, 0.04, 0.04, 0.04, 0.04]
        c = scorer._decay_score(stable, "f2")
        assert c.score == pytest.approx(1.0)

    def test_fast_decaying_low_score(self):
        scorer = make_scorer()
        # Rapidly halving IC
        decaying = [0.08, 0.04, 0.02, 0.01, 0.005, 0.0025]
        c = scorer._decay_score(decaying, "f3")
        assert c.score < 0.5

    def test_slow_decaying_high_score(self):
        scorer = make_scorer()
        # Very gentle decline: IC drops only ~2% per period
        # With this series, decay rate λ is tiny → half-life >> 30 epochs
        slow = [0.050 - i * 0.0003 for i in range(20)]   # 20 obs, slow decline
        c = scorer._decay_score(slow, "f4")
        # Half-life should be very long → score well above 0.5
        assert c.score > 0.7

    def test_weight_correct(self):
        scorer = make_scorer()
        c = scorer._decay_score([0.04]*5, "f5")
        assert c.weight == pytest.approx(DEFAULT_WEIGHTS["decay"])


class TestNoveltyComponent:
    def test_no_library_full_novelty(self):
        scorer = make_scorer()
        c = scorer._novelty_score("rank(momentum_20d)", None)
        assert c.score == pytest.approx(1.0)

    def test_empty_library_full_novelty(self):
        scorer = make_scorer()
        scorer._library_signals = {}
        c = scorer._novelty_score("rank(momentum_20d)", None)
        assert c.score == pytest.approx(1.0)

    def test_score_in_range(self):
        scorer = make_scorer()
        c = scorer._novelty_score("rank(momentum_20d)", None)
        assert 0.0 <= c.score <= 1.0

    def test_weight_correct(self):
        scorer = make_scorer()
        c = scorer._novelty_score("formula", None)
        assert c.weight == pytest.approx(DEFAULT_WEIGHTS["novelty"])


class TestCapacityComponent:
    def test_empty_history_baseline(self):
        scorer = make_scorer()
        c = scorer._capacity_score([], [], LifecycleState.EXPERIMENTAL)
        assert 0.0 <= c.score <= 1.0

    def test_production_higher_than_experimental(self):
        scorer = make_scorer()
        ics = [0.04] * 10
        c_exp  = scorer._capacity_score(ics, [], LifecycleState.EXPERIMENTAL)
        c_prod = scorer._capacity_score(ics, [], LifecycleState.PRODUCTION)
        assert c_prod.score >= c_exp.score

    def test_score_in_range(self):
        scorer = make_scorer()
        c = scorer._capacity_score([0.04]*8, [0.01]*4, LifecycleState.PRODUCTION)
        assert 0.0 <= c.score <= 1.0

    def test_weight_correct(self):
        scorer = make_scorer()
        c = scorer._capacity_score([], [], LifecycleState.PRODUCTION)
        assert c.weight == pytest.approx(DEFAULT_WEIGHTS["capacity"])


# ════════════════════════════════════════════════════════════════════════════
# SignalScorer — score_simple integration
# ════════════════════════════════════════════════════════════════════════════

class TestScoreSimple:
    def test_no_history_returns_zero_reward(self):
        scorer = make_scorer()
        r = scorer.score_simple("rank(momentum_20d)", "f1", [])
        assert r.final_reward == pytest.approx(0.0)
        assert r.success is False

    def test_with_history_returns_nonzero(self):
        scorer = make_scorer()
        r = scorer.score_simple("rank(momentum_20d)", "f1",
                                 [0.04]*10, epoch=3)
        assert r.final_reward >= 0.0
        assert r.success is True

    def test_five_components_present(self):
        scorer = make_scorer()
        r = scorer.score_simple("rank(momentum_20d)", "f1",
                                 [0.04]*8, epoch=2)
        if r.success:
            assert len(r.components) == 5

    def test_composite_in_range(self):
        scorer = make_scorer()
        r = scorer.score_simple("volatility_20d", "f2", [0.03]*8, epoch=2)
        if r.success:
            assert 0.0 <= r.composite_score <= 1.0

    def test_final_reward_in_range(self):
        scorer = make_scorer()
        r = scorer.score_simple("cross_momentum", "f3", [0.04]*8, epoch=2)
        if r.success:
            assert 0.0 <= r.final_reward <= 1.0

    def test_result_serialisable(self):
        scorer = make_scorer()
        r = scorer.score_simple("rank(momentum_20d)", "f1",
                                 [0.04]*8, epoch=2)
        json.dumps(r.to_dict())

    def test_breakdown_string(self):
        scorer = make_scorer()
        r = scorer.score_simple("rank(momentum_20d)", "f1",
                                 [0.04]*8, epoch=2)
        assert isinstance(r.breakdown(), str)


# ════════════════════════════════════════════════════════════════════════════
# EMA Smoothing
# ════════════════════════════════════════════════════════════════════════════

class TestEMASmoothing:
    def test_ema_starts_at_first_reward(self):
        scorer  = make_scorer()
        reward  = 0.5
        smoothed = scorer.update_ema("f1", reward)
        assert smoothed == pytest.approx(reward)   # no history → first value

    def test_ema_smooths_spikes(self):
        scorer = make_scorer()
        # Build up history
        for _ in range(10):
            scorer.update_ema("f1", 0.3)
        # Sudden spike — EMA should dampen it
        after_spike = scorer.update_ema("f1", 1.0)
        assert after_spike < 1.0   # smoothed below spike

    def test_ema_increases_with_high_rewards(self):
        scorer = make_scorer()
        scorer.update_ema("f1", 0.1)
        w1 = scorer.get_ema_weight("f1")
        scorer.update_ema("f1", 0.8)
        w2 = scorer.get_ema_weight("f1")
        assert w2 > w1

    def test_ema_decreases_with_low_rewards(self):
        scorer = make_scorer()
        scorer.update_ema("f1", 0.8)
        w1 = scorer.get_ema_weight("f1")
        scorer.update_ema("f1", 0.1)
        w2 = scorer.get_ema_weight("f1")
        assert w2 < w1

    def test_unknown_formula_returns_zero(self):
        scorer = make_scorer()
        assert scorer.get_ema_weight("unknown") == 0.0

    def test_separate_formula_ids_independent(self):
        scorer = make_scorer()
        scorer.update_ema("f1", 0.8)
        scorer.update_ema("f2", 0.2)
        assert scorer.get_ema_weight("f1") > scorer.get_ema_weight("f2")


# ════════════════════════════════════════════════════════════════════════════
# normalise_rewards
# ════════════════════════════════════════════════════════════════════════════

class TestNormaliseRewards:
    def test_empty_returns_empty(self):
        assert normalise_rewards({}) == {}

    def test_sums_to_one(self):
        scores  = {"f1": 0.5, "f2": 0.3, "f3": 0.1}
        weights = normalise_rewards(scores)
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_all_non_negative(self):
        weights = normalise_rewards({"f1": 0.5, "f2": 0.0, "f3": 0.2})
        assert all(w >= 0 for w in weights.values())

    def test_all_zero_equal_distribution(self):
        weights = normalise_rewards({"f1": 0.0, "f2": 0.0, "f3": 0.0})
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        # Should be equal
        vals = list(weights.values())
        assert all(abs(v - vals[0]) < 1e-6 for v in vals)

    def test_higher_score_higher_weight(self):
        weights = normalise_rewards({"f_best": 0.8, "f_worst": 0.1})
        assert weights["f_best"] > weights["f_worst"]

    def test_winner_takes_more_with_high_temperature(self):
        scores   = {"f1": 0.5, "f2": 0.1}
        low_temp = normalise_rewards(scores, temperature=0.5)
        high_temp = normalise_rewards(scores, temperature=5.0)
        # High temperature → bigger gap between winner and loser
        gap_low  = low_temp["f1"]  - low_temp["f2"]
        gap_high = high_temp["f1"] - high_temp["f2"]
        assert gap_high > gap_low

    def test_preserves_all_keys(self):
        scores  = {"f1": 0.3, "f2": 0.5, "f3": 0.1, "f4": 0.4}
        weights = normalise_rewards(scores)
        assert set(weights.keys()) == set(scores.keys())


# ════════════════════════════════════════════════════════════════════════════
# Discrimination tests — does scoring filter noise from real alpha?
# ════════════════════════════════════════════════════════════════════════════

class TestNoiseVsAlphaDiscrimination:
    """Verify the scoring engine correctly ranks alpha over noise."""

    def test_consistent_beats_unstable_same_mean(self):
        """
        Real alpha: moderate IC, very stable.
        Noise: same mean IC, highly variable.
        Consistent should score higher.
        """
        scorer = make_scorer()

        consistent_ic = [0.04] * 20        # mean=0.04, std≈0
        noisy_ic      = ([0.08, -0.02, 0.09, -0.03] * 5)  # mean≈0.03, high std

        r_consistent = scorer.score_simple("rank(momentum_20d)", "f_con",
                                            consistent_ic, epoch=5)
        r_noisy      = scorer.score_simple("volatility_20d", "f_noisy",
                                            noisy_ic, epoch=5)

        if r_consistent.success and r_noisy.success:
            assert r_consistent.final_reward > r_noisy.final_reward

    def test_stable_decay_beats_fast_decay(self):
        """
        Slow-decaying signal should score higher than fast-decaying.
        """
        scorer = make_scorer()
        slow_ic  = [0.05, 0.048, 0.046, 0.044, 0.042, 0.040, 0.038, 0.036]
        fast_ic  = [0.08, 0.04, 0.02, 0.01, 0.005, 0.0025, 0.001]

        r_slow = scorer.score_simple("rank(momentum_20d)", "f_slow", slow_ic, epoch=5)
        r_fast = scorer.score_simple("cross_momentum",     "f_fast", fast_ic, epoch=5)

        if r_slow.success and r_fast.success:
            # Decay component should favour slow-decaying
            slow_decay = next((c.score for c in r_slow.components if c.name=="decay"), 0)
            fast_decay = next((c.score for c in r_fast.components if c.name=="decay"), 0)
            assert slow_decay >= fast_decay

    def test_production_lifecycle_beats_experimental(self):
        """PRODUCTION lifecycle multiplier (1.0) > EXPERIMENTAL (0.3)."""
        assert (LifecycleState.PRODUCTION.weight_multiplier >
                LifecycleState.EXPERIMENTAL.weight_multiplier)

    def test_retired_lifecycle_zero_reward(self):
        """RETIRED signals get zero weight regardless of IC."""
        scorer = make_scorer()
        # Force lifecycle to RETIRED
        scorer.lifecycle._states["f_retired"] = LifecycleState.RETIRED

        # Even with great IC history, retired signal should get zero
        r = scorer.score_simple("rank(momentum_20d)", "f_retired",
                                 [0.08]*20, epoch=10)
        # Final reward is composite × lifecycle_mult(RETIRED=0.0) = 0
        assert r.final_reward == pytest.approx(0.0, abs=0.01)

    def test_component_weights_sum_to_one(self):
        total = sum(DEFAULT_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-6

    def test_all_components_have_positive_weight(self):
        for name, weight in DEFAULT_WEIGHTS.items():
            assert weight > 0, f"Component {name} has zero weight"
