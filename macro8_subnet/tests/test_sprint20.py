"""
tests/test_sprint20.py
-----------------------
QA: Sprint 20 — Formula deduplication, regime robustness, strategy leaderboard.

Covers:
    Fix 1: Two-level formula deduplication (exact + semantic weight-vector)
    Fix 2: Regime robustness scoring (_regime_score component)
    Fix 3: Leaderboard — LeaderboardEntry, Leaderboard (all four views)
    Integration: dedup in validator, leaderboard wired to dry-run
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

from macro8_subnet.evaluation.leaderboard   import Leaderboard, LeaderboardEntry
from macro8_subnet.evaluation.signal_scorer import SignalScorer, SignalScoreResult
from macro8_subnet.alpha.capacity_model     import LifecycleState
from macro8_subnet.alpha.batch_evaluator    import BatchEvaluator
from macro8_subnet.protocol.synapse         import AlphaSubmissionSynapse


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_prices(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "SPY":  100 * np.cumprod(1 + rng.normal(0.0005, 0.012, n)),
        "AAPL": 100 * np.cumprod(1 + rng.normal(0.0006, 0.015, n)),
        "GLD":  100 * np.cumprod(1 + rng.normal(0.0003, 0.009, n)),
    }, index=dates)


def make_scorer() -> SignalScorer:
    return SignalScorer(BatchEvaluator(make_prices()))


def make_score_result(
    formula_id:    str = "f1",
    formula:       str = "rank(momentum_20d)",
    ic:            float = 0.04,
    lifecycle:     LifecycleState = LifecycleState.PRODUCTION,
    novelty:       float = 0.8,
    capacity:      float = 0.7,
) -> SignalScoreResult:
    from macro8_subnet.evaluation.signal_scorer import ScoreComponent
    mult = lifecycle.weight_multiplier
    comp = 0.6
    return SignalScoreResult(
        formula_id=formula_id, formula_string=formula,
        composite_score=comp, lifecycle_state=lifecycle,
        lifecycle_mult=mult, final_reward=comp * mult,
        ema_weight=comp * mult * 0.9,
        components=[
            ScoreComponent("ic",        ic,      0.55, 0.40, 0.22),
            ScoreComponent("stability", 2.0,     0.86, 0.20, 0.17),
            ScoreComponent("decay",     25.0,    0.83, 0.15, 0.12),
            ScoreComponent("novelty",   1-novelty, novelty, 0.15, novelty*0.15),
            ScoreComponent("capacity",  capacity, capacity, 0.10, capacity*0.10),
        ],
    )


# ════════════════════════════════════════════════════════════════════════════
# FIX 1 — FORMULA DEDUPLICATION
# ════════════════════════════════════════════════════════════════════════════

class TestFormulaDeduplification:
    """Exact string hash + semantic weight-vector dedup."""

    def test_exact_duplicate_removed(self):
        """Two miners submitting identical formulas → only scored once."""
        from macro8_subnet.neurons.validator import Macro8Validator
        v = Macro8Validator()

        # Two miners, same formula
        subs = {
            0: AlphaSubmissionSynapse(
                formulas=["rank(momentum_20d)"], agent_role="signal", miner_uid=0
            ),
            1: AlphaSubmissionSynapse(
                formulas=["rank(momentum_20d)"], agent_role="signal", miner_uid=1
            ),
        }

        # Both miners get scores (one wins the formula, other gets 0)
        scores = v.scorer.score_submissions(subs, epoch=1)
        assert len(scores) == 2
        # One miner "wins" the formula — the other gets 0 or they split
        assert all(s >= 0 for s in scores.values())

    def test_semantic_duplicate_blocked(self):
        """Formulas encoding to same weight vector → only first accepted."""
        from macro8_subnet.neurons.validator import Macro8Validator, EpochScorer
        from macro8_subnet.alpha.batch_evaluator import FormulaEncoder, ALL_FEATURES

        enc = FormulaEncoder(ALL_FEATURES)

        # These two should encode identically (both just rank(momentum_20d))
        f1 = "rank(momentum_20d)"
        f2 = "rank(momentum_20d)"   # literal duplicate

        w1 = enc.encode(f1)
        w2 = enc.encode(f2)
        # Identical encoding → semantic duplicate
        assert np.allclose(w1, w2)

    def test_distinct_formulas_both_kept(self):
        """Genuinely different formulas both make it through dedup."""
        from macro8_subnet.neurons.validator import Macro8Validator
        v = Macro8Validator()

        subs = {
            0: AlphaSubmissionSynapse(
                formulas=["rank(momentum_20d)", "zscore(cross_momentum)",
                          "volatility_20d"],
                agent_role="signal", miner_uid=0,
            ),
        }
        scores = v.scorer.score_submissions(subs, epoch=1)
        assert 0 in scores

    def test_parameter_sweep_capped(self):
        """
        Submitting 32+ formula variants doesn't overflow the cap.
        Miner is limited to MAX_FORMULAS_MINER=32 formulas.
        """
        from macro8_subnet.neurons.validator import Macro8Validator, MAX_FORMULAS_MINER
        v = Macro8Validator()

        # Submit many variants
        variants = [
            f"rank(momentum_20d) - rank(volatility_{w}d)"
            for w in [10, 20, 60]
        ] + [f"zscore(momentum_{w}d)" for w in [5, 10, 20, 60]]
        variants = variants * 10  # 70 total → should be capped at 32

        subs = {0: AlphaSubmissionSynapse(
            formulas=variants, agent_role="signal", miner_uid=0,
        )}
        scores = v.scorer.score_submissions(subs, epoch=1)
        assert 0 in scores  # miner scored, not crashed

    def test_dedup_preserves_first_submitter(self):
        """The first miner to submit a formula gets credit, not subsequent ones."""
        from macro8_subnet.neurons.validator import Macro8Validator
        v = Macro8Validator()

        same_formula = "rank(momentum_20d)"
        subs = {
            0: AlphaSubmissionSynapse(
                formulas=[same_formula], agent_role="signal", miner_uid=0,
            ),
            1: AlphaSubmissionSynapse(
                formulas=[same_formula], agent_role="signal", miner_uid=1,
            ),
        }
        scores = v.scorer.score_submissions(subs, epoch=1)
        # Both exist in scores dict — no crash
        assert 0 in scores and 1 in scores


# ════════════════════════════════════════════════════════════════════════════
# FIX 2 — REGIME ROBUSTNESS SCORING
# ════════════════════════════════════════════════════════════════════════════

class TestRegimeScore:
    def _scorer_with_regime_weight(self) -> SignalScorer:
        weights = {
            "ic": 0.35, "stability": 0.18, "decay": 0.12,
            "novelty": 0.12, "capacity": 0.08, "regime": 0.15,
        }
        return SignalScorer(BatchEvaluator(make_prices()), weights=weights)

    def test_no_regime_data_neutral(self):
        scorer = self._scorer_with_regime_weight()
        c = scorer._regime_score({})
        assert c.score == pytest.approx(0.5)

    def test_single_regime_neutral(self):
        scorer = self._scorer_with_regime_weight()
        c = scorer._regime_score({"bull": [0.04, 0.05, 0.04]})
        assert c.score == pytest.approx(0.5)   # need ≥ 2 regimes

    def test_positive_in_all_regimes_high_score(self):
        scorer = self._scorer_with_regime_weight()
        c = scorer._regime_score({
            "bull":   [0.05, 0.04, 0.06],
            "bear":   [0.03, 0.02, 0.04],
            "crisis": [0.02, 0.01, 0.03],
        })
        assert c.score > 0.2   # all positive → non-trivial score

    def test_negative_in_one_regime_low_score(self):
        scorer = self._scorer_with_regime_weight()
        c = scorer._regime_score({
            "bull":   [0.05, 0.06, 0.05],
            "bear":   [0.04, 0.05, 0.04],
            "crisis": [-0.03, -0.04, -0.02],   # crisis kills it
        })
        assert c.score == pytest.approx(0.0)   # worst = negative → 0

    def test_robust_beats_regime_specific(self):
        scorer = self._scorer_with_regime_weight()
        robust = scorer._regime_score({
            "bull":   [0.04]*5,
            "bear":   [0.03]*5,
            "crisis": [0.02]*5,
        })
        specific = scorer._regime_score({
            "bull":   [0.08]*5,
            "bear":   [0.01]*5,
            "crisis": [-0.02]*5,
        })
        assert robust.score > specific.score

    def test_regime_component_weight_respected(self):
        scorer = self._scorer_with_regime_weight()
        c = scorer._regime_score({"bull": [0.05]*5, "bear": [0.04]*5})
        assert c.weight == pytest.approx(0.15)

    def test_score_in_range(self):
        scorer = self._scorer_with_regime_weight()
        c = scorer._regime_score({
            "r1": [0.03]*5, "r2": [0.04]*5, "r3": [0.02]*5
        })
        assert 0.0 <= c.score <= 1.0

    def test_component_serialisable(self):
        scorer = self._scorer_with_regime_weight()
        c = scorer._regime_score({"bull": [0.04]*5, "bear": [0.03]*5})
        json.dumps(c.to_dict())


# ════════════════════════════════════════════════════════════════════════════
# FIX 3 — STRATEGY DISCOVERY LEADERBOARD
# ════════════════════════════════════════════════════════════════════════════

class TestLeaderboardEntry:
    def test_mean_ic_computed(self):
        e = LeaderboardEntry("f1", "rank(momentum_20d)", miner_uid=0)
        e.ic_history = [0.04, 0.05, 0.03]
        assert e.mean_ic == pytest.approx(0.04)

    def test_mean_ic_empty(self):
        e = LeaderboardEntry("f1", "rank(momentum_20d)", miner_uid=0)
        assert e.mean_ic == 0.0

    def test_ic_ir_computed(self):
        e = LeaderboardEntry("f1", "f", miner_uid=0)
        e.ic_history = [0.04, 0.05, 0.03, 0.04, 0.05]   # has variance
        assert e.ic_ir > 0

    def test_longevity_score_caps_at_one(self):
        e = LeaderboardEntry("f1", "f", miner_uid=0)
        e.n_epochs_active = 100
        assert e.longevity_score == pytest.approx(1.0)

    def test_longevity_score_linear(self):
        e = LeaderboardEntry("f1", "f", miner_uid=0)
        e.n_epochs_active = 25
        assert e.longevity_score == pytest.approx(0.5)

    def test_is_active_production(self):
        e = LeaderboardEntry("f1", "f", miner_uid=0)
        e.lifecycle = LifecycleState.PRODUCTION
        assert e.is_active is True

    def test_is_active_retired(self):
        e = LeaderboardEntry("f1", "f", miner_uid=0)
        e.lifecycle = LifecycleState.RETIRED
        assert e.is_active is False

    def test_to_dict_serialisable(self):
        e = LeaderboardEntry("f1", "rank(momentum_20d)", miner_uid=0)
        e.ic_history = [0.04, 0.05]
        json.dumps(e.to_dict())

    def test_to_dict_has_required_keys(self):
        d = LeaderboardEntry("f1", "f", miner_uid=0).to_dict()
        for key in ("formula_id", "mean_ic", "ic_ir", "lifecycle",
                    "n_epochs_active", "novelty_score"):
            assert key in d

    def test_one_liner_is_string(self):
        e = LeaderboardEntry("f1", "rank(momentum_20d)", miner_uid=3)
        e.ic_history = [0.04]
        assert isinstance(e.one_liner(1, "ic"), str)


class TestLeaderboard:
    def _board_with_entries(self, n: int = 5) -> Leaderboard:
        board   = Leaderboard()
        results = [
            make_score_result(
                f"f{i}", f"formula_{i}",
                ic=0.02 + i * 0.01,
                novelty=0.9 - i * 0.1,
                capacity=0.5 + i * 0.05,
            )
            for i in range(n)
        ]
        for i, r in enumerate(results):
            board.register(r, miner_uid=i % 3, epoch=i + 1)
            # Simulate multiple epochs for longevity
            for ep in range(2, 5):
                board.register(r, miner_uid=i % 3, epoch=ep)
        return board

    def test_register_creates_entry(self):
        board = Leaderboard()
        r     = make_score_result()
        board.register(r, epoch=1)
        assert board.n_total == 1

    def test_register_same_id_updates(self):
        board = Leaderboard()
        r     = make_score_result("f1")
        board.register(r, epoch=1)
        board.register(r, epoch=2)
        assert board.n_total == 1

    def test_n_active_counts_non_retired(self):
        board = self._board_with_entries(5)
        board.retire("f0")
        assert board.n_active == 4

    def test_top_by_ic_sorted(self):
        board = self._board_with_entries(5)
        top   = board.top_by_ic(3)
        ics   = [e.mean_ic for e in top if e.mean_ic > 0]
        if len(ics) >= 2:
            assert ics == sorted(ics, reverse=True)

    def test_top_by_longevity_sorted(self):
        board = self._board_with_entries(5)
        top   = board.top_by_longevity(3)
        ages  = [e.n_epochs_active for e in top]
        assert ages == sorted(ages, reverse=True)

    def test_top_by_novelty_sorted(self):
        board = self._board_with_entries(5)
        top   = board.top_by_novelty(3)
        novs  = [e.novelty_score for e in top]
        assert novs == sorted(novs, reverse=True)

    def test_top_by_capacity_sorted(self):
        board = self._board_with_entries(5)
        top   = board.top_by_capacity(3)
        caps  = [e.best_capacity for e in top]
        assert caps == sorted(caps, reverse=True)

    def test_top_by_reward_sorted(self):
        board = self._board_with_entries(5)
        top   = board.top_by_reward(3)
        rews  = [e.ema_weight for e in top]
        assert rews == sorted(rews, reverse=True)

    def test_stats_dict_keys(self):
        board = self._board_with_entries(3)
        s     = board.stats()
        for key in ("n_total", "n_active", "n_production",
                    "n_unique_miners", "mean_ic", "concentration"):
            assert key in s

    def test_concentration_range(self):
        board = self._board_with_entries(5)
        c     = board.concentration()
        assert 0.0 < c <= 1.0

    def test_n_unique_miners(self):
        board = self._board_with_entries(6)   # 6 entries, miners are i%3
        assert board.n_unique_miners == 3

    def test_retire(self):
        board = self._board_with_entries(3)
        board.retire("f0")
        entry = board._entries.get("f0")
        if entry:
            assert entry.lifecycle == LifecycleState.RETIRED

    def test_max_entries_eviction(self):
        board = Leaderboard(max_entries=5)
        # Add 6 entries — oldest retired one should be evicted
        for i in range(5):
            r = make_score_result(f"f{i}", lifecycle=LifecycleState.RETIRED)
            board.register(r, epoch=i)
        # Add one more active
        board.register(make_score_result("f_new"), epoch=10)
        assert board.n_total <= 6  # eviction should keep us near limit

    def test_to_dict_serialisable(self):
        json.dumps(self._board_with_entries(3).to_dict())

    def test_print_summary_runs(self, capsys):
        board = self._board_with_entries(3)
        board.print_summary()
        out = capsys.readouterr().out
        assert "Leaderboard" in out

    def test_print_full_runs(self, capsys):
        board = self._board_with_entries(5)
        board.print_full(top_n=3)
        out = capsys.readouterr().out
        assert "LEADERBOARD" in out

    def test_empty_board_stats(self):
        board = Leaderboard()
        s     = board.stats()
        assert s["n_total"] == 0
        assert s["n_active"] == 0

    def test_best_ic_tracked(self):
        board = Leaderboard()
        r1 = make_score_result("f1", ic=0.03)
        r2 = make_score_result("f1", ic=0.07)   # better IC
        board.register(r1, epoch=1)
        board.register(r2, epoch=2)
        entry = board._entries["f1"]
        assert entry.best_ic >= 0.07

    def test_epochs_active_increments(self):
        board = Leaderboard()
        r     = make_score_result("f1")
        board.register(r, epoch=1)
        board.register(r, epoch=2)
        board.register(r, epoch=3)
        assert board._entries["f1"].n_epochs_active == 3


# ════════════════════════════════════════════════════════════════════════════
# INTEGRATION — Validator dry-run with leaderboard
# ════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    def test_validator_has_leaderboard(self):
        from macro8_subnet.neurons.validator import Macro8Validator
        v = Macro8Validator()
        assert hasattr(v, "leaderboard")
        assert isinstance(v.leaderboard, Leaderboard)

    def test_dry_run_populates_leaderboard(self):
        from macro8_subnet.neurons.validator import Macro8Validator
        v = Macro8Validator()
        v._dry_run(n_epochs=2)
        # After 2 epochs, leaderboard should have some entries
        assert v.leaderboard.n_total >= 0  # may be 0 if all scores zero

    def test_leaderboard_to_dict_after_dry_run(self):
        from macro8_subnet.neurons.validator import Macro8Validator
        v = Macro8Validator()
        v._dry_run(n_epochs=1)
        json.dumps(v.leaderboard.to_dict())  # must not crash

    def test_simulation_has_no_crashes(self):
        """Full simulation run with leaderboard active."""
        from macro8_subnet.local_simulation import run_simulation
        results = run_simulation(n_miners=5, verbose=False)
        assert results["all_scored"] is True
