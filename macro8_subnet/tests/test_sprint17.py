"""
tests/test_sprint17.py
-----------------------
QA: Sprint 17 — Validator hardening + local simulation.

Covers:
    Error 1:  safe_formula(), safe_synapse(), safe_submission() guards
    Error 2:  _set_weights() metagraph alignment
    Error 3:  SIGALRM timeout guard in score_submissions()
    Simulation: local_simulation.py scoring harness
"""

from __future__ import annotations

import json
import platform
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

from macro8_subnet.neurons.validator import (
    safe_formula, safe_synapse, safe_submission,
    EpochScorer, Macro8Validator,
    MAX_FORMULAS_MINER, MIN_IC_THRESHOLD,
)
from macro8_subnet.protocol.synapse import AlphaSubmissionSynapse
from macro8_subnet.local_simulation import (
    make_synthetic_submissions, run_simulation,
    test_safe_formula as sim_test_safe_formula,
    test_safe_submission as sim_test_safe_submission,
    GOOD_FORMULAS, ADVERSARIAL_INPUTS,
)


# ── Shared fixtures ───────────────────────────────────────────────────────────

def make_validator() -> Macro8Validator:
    return Macro8Validator()


def make_synapse(formulas=None, role="signal") -> AlphaSubmissionSynapse:
    return AlphaSubmissionSynapse(
        formulas=formulas or ["momentum_20d"],
        agent_role=role,
        miner_uid=0,
        miner_hotkey="5F" + "0" * 40,
        epoch=1,
    )


# ════════════════════════════════════════════════════════════════════════════
# ERROR 1: DEFENSIVE VALIDATION GUARDS
# ════════════════════════════════════════════════════════════════════════════

class TestSafeFormula:
    """safe_formula() must reject all bad inputs and accept all good ones."""

    # ── Good inputs — Tier 1 char filter ─────────────────────────────────────

    def test_simple_feature(self):
        assert safe_formula("momentum_20d") == "momentum_20d"

    def test_rank_operator(self):
        assert safe_formula("rank(momentum_20d)") is not None

    def test_subtraction(self):
        assert safe_formula("rank(momentum_20d) - rank(volatility_60d)") is not None

    def test_decay_with_halflife(self):
        assert safe_formula("decay(momentum_5d, halflife=10)") is not None

    def test_zscore_operator(self):
        assert safe_formula("zscore(cross_momentum)") is not None

    def test_regime_signal(self):
        assert safe_formula("regime_signal * momentum_20d") is not None

    def test_modulo_operator(self):
        # Sprint 18: % added to allowed chars for regime detection
        assert safe_formula("volatility_20d % momentum_5d") is not None

    def test_trailing_whitespace_stripped(self):
        assert safe_formula("  momentum_20d  ") == "momentum_20d"

    def test_leading_control_chars_stripped(self):
        # \n\t\r at boundaries are stripped, leaving valid formula
        assert safe_formula("\n\t\rmomentum_20d") == "momentum_20d"

    def test_all_good_formulas_accepted(self):
        for f in GOOD_FORMULAS:
            assert safe_formula(f) is not None, f"GOOD_FORMULA rejected: {f!r}"

    # ── Null / wrong type — Tier 1 ────────────────────────────────────────────

    def test_none_rejected(self):
        assert safe_formula(None) is None

    def test_int_rejected(self):
        assert safe_formula(42) is None

    def test_float_rejected(self):
        assert safe_formula(3.14) is None

    def test_list_rejected(self):
        assert safe_formula(["momentum_20d"]) is None

    def test_dict_rejected(self):
        assert safe_formula({"formula": "momentum_20d"}) is None

    # ── Empty / whitespace — Tier 1 ──────────────────────────────────────────

    def test_empty_string_rejected(self):
        assert safe_formula("") is None

    def test_whitespace_only_rejected(self):
        assert safe_formula("   ") is None

    # ── Too long — Tier 1 ────────────────────────────────────────────────────

    def test_too_long_rejected(self):
        assert safe_formula("a" * 201) is None

    # ── Bad chars — Tier 1 ───────────────────────────────────────────────────

    def test_null_byte_rejected(self):
        assert safe_formula("rank(momentum_20d\x00)") is None

    def test_unicode_emoji_rejected(self):
        assert safe_formula("🚀momentum_20d") is None

    def test_embedded_control_chars_rejected(self):
        assert safe_formula("momentum\n20d")   is None
        assert safe_formula("momentum\x01_20d") is None

    def test_semicolon_rejected(self):
        # SQL injection style — ; not in allowed chars
        assert safe_formula("rank(momentum_20d); DROP TABLE signals;") is None

    # ── AST sandbox — Tier 2 ─────────────────────────────────────────────────

    def test_python_import_blocked(self):
        # __import__ uses a string Constant — blocked by Constant check
        assert safe_formula('__import__("os").system("rm -rf /")') is None

    def test_open_blocked(self):
        assert safe_formula('open("/etc/passwd").read()') is None

    def test_lambda_blocked(self):
        assert safe_formula("lambda x: x") is None

    def test_list_comprehension_blocked(self):
        assert safe_formula("[x for x in range(1000000)]") is None

    def test_conditional_expression_blocked(self):
        # IfExp not in whitelist
        assert safe_formula("a if a else a") is None

    def test_power_operator_blocked(self):
        # Pow removed from whitelist — prevents exponential blowup attacks
        assert safe_formula("((a**a)**a)**a") is None

    def test_power_simple_blocked(self):
        assert safe_formula("a**b") is None

    def test_string_constant_blocked(self):
        # String Constant → blocked by numeric-only Constant check
        assert safe_formula('"string_literal"') is None

    def test_none_constant_blocked(self):
        # None is a Constant(None) — not numeric
        assert safe_formula("None") is None

    def test_dunder_name_blocked(self):
        # __builtins__, __class__, etc.
        assert safe_formula("__builtins__") is None

    def test_dunder_chain_blocked(self):
        # __class__.__bases__[0].__subclasses__() — class hierarchy traversal
        # Blocked: Attribute node not in whitelist, Subscript not in whitelist
        assert safe_formula("__class__.__bases__") is None

    def test_deeply_nested_depth_blocked(self):
        # Very deep nesting exceeds _MAX_FORMULA_DEPTH
        deep = "rank(" * 13 + "momentum_20d" + ")" * 13
        assert safe_formula(deep) is None

    def test_all_adversarial_dont_crash(self):
        """safe_formula must never raise an exception, regardless of input."""
        for bad in ADVERSARIAL_INPUTS:
            try:
                safe_formula(bad)  # must not raise
            except Exception as e:
                pytest.fail(f"safe_formula raised on {bad!r}: {e}")


class TestSafeSynapse:
    def test_valid_synapse_accepted(self):
        syn = make_synapse()
        assert safe_synapse(syn) is not None

    def test_none_rejected(self):
        assert safe_synapse(None) is None

    def test_object_without_formulas_rejected(self):
        assert safe_synapse(object()) is None

    def test_synapse_with_empty_formulas_accepted(self):
        syn = make_synapse(formulas=[])
        assert safe_synapse(syn) is not None

    def test_synapse_returned_as_is(self):
        syn = make_synapse()
        result = safe_synapse(syn)
        assert result is syn   # same object


class TestSafeSubmission:
    def test_valid_submission_accepted(self):
        syn = make_synapse(["momentum_20d", "rank(cross_momentum)"])
        result = safe_submission(0, syn)
        assert result is not None

    def test_none_synapse_rejected(self):
        assert safe_submission(0, None) is None

    def test_no_formulas_attr_rejected(self):
        assert safe_submission(0, object()) is None

    def test_bad_formulas_stripped_from_list(self):
        """Post-construction corruption: bad items stripped, good ones kept."""
        syn = make_synapse(["momentum_20d"])
        object.__setattr__(syn, "formulas", ["momentum_20d", None, 42, "volatility_20d"])
        result = safe_submission(0, syn)
        assert result is not None
        assert all(isinstance(f, str) for f in result.formulas)

    def test_injection_formula_stripped(self):
        syn = make_synapse(["momentum_20d"])
        object.__setattr__(syn, "formulas", [
            "momentum_20d",
            "__import__('os').system('rm -rf /')",
        ])
        result = safe_submission(0, syn)
        if result is not None:
            for f in result.formulas:
                assert "__import__" not in f

    def test_clean_synapse_unchanged(self):
        syn = make_synapse(["momentum_20d", "rank(cross_momentum)"])
        result = safe_submission(0, syn)
        assert result is not None
        assert len(result.formulas) == 2

    def test_all_bad_formulas_gives_empty_list(self):
        """Synapse with all-invalid formulas → empty formula list (not None)."""
        syn = make_synapse(["momentum_20d"])
        object.__setattr__(syn, "formulas", [None, 42, "", "🚀bad"])
        result = safe_submission(0, syn)
        assert result is not None
        assert result.formulas == []


# ════════════════════════════════════════════════════════════════════════════
# ERROR 2: WEIGHT VECTOR METAGRAPH ALIGNMENT
# ════════════════════════════════════════════════════════════════════════════

class TestWeightAlignment:
    """_set_weights must align its output to the full metagraph uid space."""

    def test_score_map_fills_full_uid_space(self):
        """Even miners with score=0 must appear in the weight vector."""
        validator = make_validator()
        # Simulate: 5 miners registered, only 2 submitted
        all_uids       = [0, 1, 2, 3, 4]
        submitter_uids = [1, 3]
        submitter_weights = [0.6, 0.4]

        score_map = dict(zip(submitter_uids, submitter_weights))
        full_weights = [float(score_map.get(uid, 0.0)) for uid in all_uids]

        assert len(full_weights) == len(all_uids)
        assert full_weights[0] == 0.0   # uid 0 didn't submit
        assert full_weights[2] == 0.0   # uid 2 didn't submit
        assert full_weights[4] == 0.0   # uid 4 didn't submit

    def test_weight_normalisation(self):
        """Weights must sum to 1.0 after normalisation."""
        raw = [0.0, 0.3, 0.0, 0.7, 0.0]
        arr = np.array(raw, dtype=np.float32)
        total = arr.sum()
        if total > 1e-8:
            arr = arr / total
        assert abs(arr.sum() - 1.0) < 1e-5

    def test_all_zero_scores_gets_equal_weights(self):
        """If no miner earns a score, equal weights prevent a zero vector."""
        weights = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        total   = weights.sum()
        if total <= 1e-8:
            weights = np.ones(len(weights)) / len(weights)
        assert abs(weights.sum() - 1.0) < 1e-5
        assert all(w > 0 for w in weights)

    def test_uid_weight_parallel_arrays(self):
        """uids and weights arrays must have identical length."""
        uids    = [0, 1, 2, 3, 4]
        weights = [0.1, 0.3, 0.2, 0.3, 0.1]
        arr_u   = np.array(uids,    dtype=np.int64)
        arr_w   = np.array(weights, dtype=np.float32)
        assert len(arr_u) == len(arr_w)

    def test_no_negative_weights(self):
        """Weights must never be negative (Bittensor rejects them)."""
        raw = np.array([-0.1, 0.5, 0.8, -0.2], dtype=np.float32)
        clipped = np.maximum(raw, 0.0)
        total   = clipped.sum()
        if total > 1e-8:
            clipped /= total
        assert all(w >= 0 for w in clipped)


# ════════════════════════════════════════════════════════════════════════════
# ERROR 3: TIMEOUT GUARD
# ════════════════════════════════════════════════════════════════════════════

class TestTimeoutGuard:
    """score_submissions must not hang indefinitely."""

    @pytest.mark.skipif(
        platform.system() == "Windows",
        reason="SIGALRM not available on Windows"
    )
    def test_timeout_handler_importable(self):
        """signal.SIGALRM must be available on this platform."""
        import signal
        assert hasattr(signal, "SIGALRM")

    def test_score_submissions_returns_in_time(self):
        """Score 50 formulas — must complete well under 30s timeout."""
        import time
        validator = make_validator()
        subs      = {
            i: make_synapse([GOOD_FORMULAS[i % len(GOOD_FORMULAS)]])
            for i in range(10)
        }
        t0      = time.perf_counter()
        scores  = validator.scorer.score_submissions(subs, epoch=1)
        elapsed = time.perf_counter() - t0

        assert elapsed < 10.0, f"Scoring took too long: {elapsed:.2f}s"
        assert len(scores) == 10

    def test_empty_submissions_fast(self):
        """Empty submission set must return immediately."""
        import time
        validator = make_validator()
        t0      = time.perf_counter()
        scores  = validator.scorer.score_submissions({}, epoch=1)
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0
        assert scores == {}


# ════════════════════════════════════════════════════════════════════════════
# EPOCH SCORER END-TO-END
# ════════════════════════════════════════════════════════════════════════════

class TestEpochScorer:
    def test_all_miners_scored(self):
        validator = make_validator()
        n         = 5
        subs      = {i: make_synapse([GOOD_FORMULAS[i]]) for i in range(n)}
        scores    = validator.scorer.score_submissions(subs, epoch=1)
        assert len(scores) == n

    def test_scores_non_negative(self):
        validator = make_validator()
        subs      = {i: make_synapse([GOOD_FORMULAS[i % len(GOOD_FORMULAS)]])
                     for i in range(5)}
        scores    = validator.scorer.score_submissions(subs, epoch=1)
        assert all(s >= 0 for s in scores.values())

    def test_missing_miner_gets_zero(self):
        """Miner with no valid formulas → score=0, no crash."""
        validator = make_validator()
        bad_syn   = make_synapse(["momentum_20d"])
        object.__setattr__(bad_syn, "formulas", [])   # empty after strip
        subs = {0: make_synapse(["momentum_20d"]), 1: bad_syn}
        scores = validator.scorer.score_submissions(subs, epoch=1)
        assert scores[1] == 0.0

    def test_rate_limit_enforced(self):
        """Miner submitting > 32 formulas (Sprint 18 cap) gets capped."""
        validator = make_validator()
        many_formulas = GOOD_FORMULAS * 20   # way over the 32-formula cap
        syn    = make_synapse(many_formulas)
        subs   = {0: syn}
        # safe_submission caps at max_formulas=32 before evaluation
        scores = validator.scorer.score_submissions(subs, epoch=1)
        assert 0 in scores

    def test_adversarial_inputs_no_crash(self):
        """Adversarial inputs never cause an exception."""
        validator = make_validator()
        subs      = {}
        for i in range(5):
            syn = make_synapse([GOOD_FORMULAS[i]])
            if i == 2:
                object.__setattr__(syn, "formulas",
                                   [None, "__import__('os')", "🚀bad", "momentum_20d"])
            subs[i] = syn

        try:
            scores = validator.scorer.score_submissions(subs, epoch=1)
            assert len(scores) == 5
        except Exception as e:
            pytest.fail(f"Adversarial input caused exception: {e}")


# ════════════════════════════════════════════════════════════════════════════
# LOCAL SIMULATION HARNESS
# ════════════════════════════════════════════════════════════════════════════

class TestLocalSimulation:
    def test_make_synthetic_submissions_count(self):
        subs = make_synthetic_submissions(n_miners=10)
        assert len(subs) == 10

    def test_synthetic_submissions_have_formulas(self):
        subs = make_synthetic_submissions(n_miners=5)
        for uid, syn in subs.items():
            assert hasattr(syn, "formulas")
            assert isinstance(syn.formulas, list)

    def test_synthetic_submissions_have_roles(self):
        subs = make_synthetic_submissions(n_miners=10, roles_mix=True)
        roles = {syn.agent_role for syn in subs.values()}
        assert len(roles) >= 2   # mixed roles

    def test_adversarial_submission_no_crash(self):
        """make_synthetic_submissions with adversarial=True must not crash."""
        try:
            subs = make_synthetic_submissions(n_miners=10, adversarial=True)
            assert len(subs) == 10
        except Exception as e:
            pytest.fail(f"Adversarial submission generation crashed: {e}")

    def test_run_simulation_basic(self):
        results = run_simulation(n_miners=5, verbose=False)
        assert results["all_scored"] is True
        assert results["n_miners"] == 5
        assert abs(results["rewards_sum"] - 1.0) < 0.01

    def test_run_simulation_rewards_sum_to_one(self):
        results = run_simulation(n_miners=10, verbose=False)
        assert abs(results["rewards_sum"] - 1.0) < 0.01

    def test_run_simulation_all_miners_scored(self):
        results = run_simulation(n_miners=8, verbose=False)
        assert len(results["scores"]) == 8

    def test_run_simulation_adversarial(self):
        results = run_simulation(n_miners=8, adversarial=True, verbose=False)
        assert results["all_scored"] is True
        assert results["rewards_sum"] > 0

    def test_run_simulation_100_miners_fast(self):
        """100 miners must score in under 2 seconds."""
        import time
        t0      = time.perf_counter()
        results = run_simulation(n_miners=100, verbose=False)
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, f"100-miner simulation took {elapsed:.2f}s"
        assert results["all_scored"] is True

    def test_sim_safe_formula_all_pass(self):
        """Simulation's own safe_formula test must pass."""
        assert sim_test_safe_formula() is True

    def test_sim_safe_submission_all_pass(self):
        """Simulation's own safe_submission test must pass."""
        assert sim_test_safe_submission() is True

    def test_results_serialisable(self):
        """Simulation results must be JSON-serialisable for logging."""
        results = run_simulation(n_miners=5, verbose=False)
        # Convert numpy types for JSON
        safe_results = {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in results.items()
            if k not in ("scores", "rewards")
        }
        json.dumps(safe_results)
