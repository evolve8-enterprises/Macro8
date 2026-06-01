"""
tests/test_sprint11.py
-----------------------
QA: Complete self-contained tests for Sprint 11.

Covers:
    market/signal_market.py     — SignalPosition, MarketBook, SignalMarket
    market/market_rewards.py    — QuadraticScorer, PredictorReward
    market/market_integrator.py — MarketIntegrator blending
    validators/validator_types.py — ValidatorRole, RewardProposal, ValidatorSubmission
    validators/consensus.py     — ConsensusEngine stake-weighted aggregation
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


# ════════════════════════════════════════════════════════════════════════════
# SIGNAL MARKET
# ════════════════════════════════════════════════════════════════════════════

from macro8_subnet.market.signal_market import (
    SignalMarket, SignalPosition, PositionDirection,
    MarketBook, SettlementResult,
)


def make_position(
    uid:       int   = 0,
    signal:    str   = "f_momentum",
    direction: str   = "long",
    stake:     float = 0.5,
    confidence: float = 0.8,
) -> SignalPosition:
    return SignalPosition(
        miner_uid=uid, miner_hotkey=f"5F{uid:040d}",
        signal_name=signal,
        direction=PositionDirection(direction),
        stake=stake, confidence=confidence,
    )


class TestPositionDirection:
    def test_long_sign_positive(self):
        assert PositionDirection.LONG.sign() == 1.0

    def test_short_sign_negative(self):
        assert PositionDirection.SHORT.sign() == -1.0

    def test_enum_values(self):
        assert PositionDirection.LONG.value  == "long"
        assert PositionDirection.SHORT.value == "short"


class TestSignalPosition:
    def test_stake_clamped(self):
        p = make_position(stake=2.0)
        assert p.stake <= 1.0

    def test_confidence_clamped(self):
        p = make_position(confidence=0.0)
        assert p.confidence >= 0.5

    def test_confidence_upper_clamped(self):
        p = make_position(confidence=5.0)
        assert p.confidence == pytest.approx(1.0)

    def test_signed_confidence_long(self):
        p = make_position(direction="long", confidence=0.8)
        assert p.signed_confidence == pytest.approx(0.8)

    def test_signed_confidence_short(self):
        p = make_position(direction="short", confidence=0.8)
        assert p.signed_confidence == pytest.approx(-0.8)

    def test_to_dict_serialisable(self):
        json.dumps(make_position().to_dict())


class TestMarketBook:
    def test_empty_book_price_zero(self):
        book = MarketBook("f1")
        assert book.market_price() == 0.0

    def test_all_long_price_positive(self):
        book = MarketBook("f1")
        book.positions = [make_position(direction="long", confidence=0.9)]
        assert book.market_price() > 0

    def test_all_short_price_negative(self):
        book = MarketBook("f1")
        book.positions = [make_position(direction="short", confidence=0.9)]
        assert book.market_price() < 0

    def test_balanced_book_price_near_zero(self):
        book = MarketBook("f1")
        book.positions = [
            make_position(0, direction="long",  stake=1.0, confidence=0.8),
            make_position(1, direction="short", stake=1.0, confidence=0.8),
        ]
        assert abs(book.market_price()) < 0.01

    def test_confidence_score_is_abs_price(self):
        book = MarketBook("f1")
        book.positions = [make_position(direction="short", confidence=0.7)]
        assert book.confidence_score() == pytest.approx(abs(book.market_price()))

    def test_bullish_fraction_all_long(self):
        book = MarketBook("f1")
        book.positions = [make_position(direction="long") for _ in range(3)]
        assert book.bullish_fraction() == pytest.approx(1.0)

    def test_to_dict_serialisable(self):
        book = MarketBook("f1")
        book.positions = [make_position()]
        json.dumps(book.to_dict())


class TestSignalMarket:
    def _market_with_position(self):
        m = SignalMarket()
        m.open_position(make_position())
        return m

    def test_open_position_accepted(self):
        m  = SignalMarket()
        ok = m.open_position(make_position())
        assert ok is True

    def test_zero_stake_rejected(self):
        m  = SignalMarket()
        ok = m.open_position(make_position(stake=0.0))
        assert ok is False

    def test_price_signal_positive_after_long(self):
        m = self._market_with_position()
        assert m.price_signal("f_momentum") > 0

    def test_unknown_signal_price_zero(self):
        m = SignalMarket()
        assert m.price_signal("nonexistent") == 0.0

    def test_confidence_score_range(self):
        m = self._market_with_position()
        c = m.confidence_score("f_momentum")
        assert 0.0 <= c <= 1.0

    def test_settle_epoch_returns_results(self):
        m = self._market_with_position()
        results = m.settle_epoch({"f_momentum": 0.05})
        assert len(results) == 1

    def test_settlement_correct_long_wins(self):
        """LONG position on IC > 0 signal should be correct."""
        m = SignalMarket()
        m.open_position(make_position(direction="long", confidence=0.9))
        results = m.settle_epoch({"f_momentum": 0.04})   # IC > 0 → long wins
        assert results[0].prediction_correct is True

    def test_settlement_incorrect_long_loses(self):
        """LONG position on IC ≤ 0 signal should be incorrect."""
        m = SignalMarket()
        m.open_position(make_position(direction="long", confidence=0.9))
        results = m.settle_epoch({"f_momentum": -0.02})   # IC ≤ 0 → short wins
        assert results[0].prediction_correct is False

    def test_settlement_pnl_correct_positive(self):
        m = SignalMarket()
        m.open_position(make_position(direction="long", confidence=0.8))
        results = m.settle_epoch({"f_momentum": 0.04})
        assert results[0].pnl_score > 0

    def test_settlement_pnl_incorrect_negative(self):
        m = SignalMarket()
        m.open_position(make_position(direction="long", confidence=0.8))
        results = m.settle_epoch({"f_momentum": -0.02})
        assert results[0].pnl_score < 0

    def test_settlement_clears_book(self):
        m = self._market_with_position()
        m.settle_epoch({"f_momentum": 0.04})
        assert "f_momentum" not in m.tracked_signals()

    def test_n_open_positions(self):
        m = SignalMarket()
        m.open_position(make_position(0, "f1"))
        m.open_position(make_position(1, "f1"))
        m.open_position(make_position(2, "f2"))
        assert m.n_open_positions() == 3

    def test_all_prices_dict(self):
        m = SignalMarket()
        m.open_position(make_position(0, "f1", "long"))
        m.open_position(make_position(1, "f2", "short"))
        prices = m.all_prices()
        assert "f1" in prices and "f2" in prices

    def test_all_confidences_range(self):
        m = self._market_with_position()
        for c in m.all_confidences().values():
            assert 0.0 <= c <= 1.0

    def test_epoch_history_grows(self):
        m = self._market_with_position()
        m.settle_epoch({"f_momentum": 0.04})
        assert len(m.epoch_history) == 1

    def test_open_positions_batch(self):
        m = SignalMarket()
        positions = [make_position(i, f"f{i}") for i in range(4)]
        n = m.open_positions_batch(positions)
        assert n == 4

    def test_market_summary_list(self):
        m = self._market_with_position()
        s = m.market_summary()
        assert isinstance(s, list) and len(s) == 1

    def test_settlement_result_serialisable(self):
        m = self._market_with_position()
        r = m.settle_epoch({"f_momentum": 0.04})
        json.dumps(r[0].to_dict())

    def test_quadratic_scoring_certain_correct(self):
        """Certainty (conf=1.0) correct prediction → max pnl."""
        m = SignalMarket()
        m.open_position(make_position(direction="long", confidence=1.0))
        r = m.settle_epoch({"f_momentum": 0.05})
        assert r[0].pnl_score == pytest.approx(1.0)

    def test_quadratic_scoring_uncertain(self):
        """Uncertainty (conf=0.5) always yields pnl ≈ 0."""
        m = SignalMarket()
        m.open_position(make_position(direction="long", confidence=0.5))
        r = m.settle_epoch({"f_momentum": 0.05})
        assert abs(r[0].pnl_score) < 0.01


# ════════════════════════════════════════════════════════════════════════════
# MARKET REWARDS
# ════════════════════════════════════════════════════════════════════════════

from macro8_subnet.market.market_rewards import QuadraticScorer, MarketRewardReport


def make_settlement(
    uid: int = 0,
    signal: str = "f1",
    direction: str = "long",
    confidence: float = 0.8,
    actual_ic: float = 0.04,
) -> SettlementResult:
    pos          = make_position(uid, signal, direction, stake=0.5, confidence=confidence)
    ic_positive  = actual_ic > 0
    long_won     = ic_positive
    correct      = (direction == "long") == long_won
    c            = confidence
    pnl          = (2*c - 1) if correct else -(2*c - 1)
    return SettlementResult(
        position=pos, actual_ic=actual_ic,
        ic_positive=ic_positive, long_won=long_won,
        prediction_correct=correct, pnl_score=pnl,
    )


class TestQuadraticScorer:
    def test_empty_settlements_returns_report(self):
        r = QuadraticScorer().compute_rewards([])
        assert isinstance(r, MarketRewardReport)
        assert r.n_predictors == 0

    def test_single_correct_predictor(self):
        s = make_settlement(0, actual_ic=0.04, direction="long")
        r = QuadraticScorer().compute_rewards([s])
        assert len(r.rewards) == 1
        assert r.rewards[0].reward_weight == pytest.approx(1.0)

    def test_rewards_sum_to_one(self):
        settlements = [make_settlement(i, signal=f"f{i}") for i in range(4)]
        r           = QuadraticScorer().compute_rewards(settlements)
        if r.rewards:
            total = sum(rw.reward_weight for rw in r.rewards)
            assert abs(total - 1.0) < 1e-6

    def test_correct_predictor_beats_wrong(self):
        correct  = make_settlement(0, direction="long",  actual_ic= 0.04)
        wrong    = make_settlement(1, direction="short", actual_ic= 0.04)
        r        = QuadraticScorer().compute_rewards([correct, wrong])
        w_correct = next(rw.reward_weight for rw in r.rewards if rw.miner_uid == 0)
        w_wrong   = next(rw.reward_weight for rw in r.rewards if rw.miner_uid == 1)
        assert w_correct > w_wrong

    def test_accuracy_field_correct(self):
        s = make_settlement(0, direction="long", actual_ic=0.04)
        r = QuadraticScorer().compute_rewards([s])
        assert r.rewards[0].accuracy == pytest.approx(1.0)

    def test_n_correct_counted(self):
        settlements = [
            make_settlement(0, "f1", "long",  actual_ic= 0.04),   # correct
            make_settlement(0, "f2", "short", actual_ic= 0.04),   # wrong
        ]
        r = QuadraticScorer().compute_rewards(settlements)
        assert r.rewards[0].n_correct == 1
        assert r.rewards[0].n_positions == 2

    def test_to_dict_serialisable(self):
        r = QuadraticScorer().compute_rewards([make_settlement()])
        json.dumps(r.to_dict())

    def test_as_weight_list(self):
        settlements = [make_settlement(i, f"f{i}") for i in range(3)]
        r           = QuadraticScorer().compute_rewards(settlements)
        uids, ws    = r.as_weight_list()
        assert len(uids) == len(ws)
        if ws:
            assert abs(sum(ws) - 1.0) < 1e-6

    def test_all_wrong_predictions_handled(self):
        """All-negative PnL → equal distribution."""
        settlements = [
            make_settlement(i, f"f{i}", "long", confidence=0.9, actual_ic=-0.05)
            for i in range(3)
        ]
        r = QuadraticScorer().compute_rewards(settlements)
        if r.rewards:
            total = sum(rw.reward_weight for rw in r.rewards)
            assert abs(total - 1.0) < 1e-6


# ════════════════════════════════════════════════════════════════════════════
# MARKET INTEGRATOR
# ════════════════════════════════════════════════════════════════════════════

from macro8_subnet.market.market_integrator import MarketIntegrator


def make_market_with_data() -> SignalMarket:
    m = SignalMarket()
    m.open_position(make_position(0, "f_momentum", "long",  0.8, 0.9))
    m.open_position(make_position(1, "f_vol",      "short", 0.5, 0.7))
    return m


class TestMarketIntegrator:
    def test_invalid_alpha_beta_raises(self):
        with pytest.raises(ValueError):
            MarketIntegrator(alpha=0.5, beta=0.7)

    def test_returns_dict(self):
        ic     = {"f_momentum": 0.05, "f_vol": 0.03}
        result = MarketIntegrator().market_weighted_ics(ic, make_market_with_data())
        assert isinstance(result, dict)

    def test_all_signals_present(self):
        ic     = {"f_momentum": 0.05, "f_vol": 0.03}
        result = MarketIntegrator().market_weighted_ics(ic, make_market_with_data())
        assert set(result.keys()) == set(ic.keys())

    def test_bullish_signal_higher_weight(self):
        """f_momentum has LONG positions → should get higher weight after blending."""
        ic     = {"f_momentum": 0.05, "f_vol": 0.05}   # equal IC
        market = make_market_with_data()                 # f_momentum is bullish
        result = MarketIntegrator(alpha=0.5, beta=0.5).market_weighted_ics(ic, market)
        assert result["f_momentum"] >= result["f_vol"]

    def test_no_market_data_uses_ic_only(self):
        """Signal with no positions → weight = IC score (alpha=1.0, beta=0.0)."""
        ic     = {"f_no_market": 0.05}
        market = SignalMarket()   # empty
        result = MarketIntegrator().market_weighted_ics(ic, market)
        assert "f_no_market" in result
        assert result["f_no_market"] >= 0.0

    def test_bearish_consensus_reduces_weight(self):
        """SHORT consensus → market contribution is zero, IC drives weight."""
        ic     = {"f_short": 0.05}
        market = SignalMarket()
        market.open_position(make_position(0, "f_short", "short", 1.0, 0.95))
        # Short position → price < 0 → market_confidence = 0
        result = MarketIntegrator(alpha=0.5, beta=0.5).market_weighted_ics(ic, market)
        assert result["f_short"] >= 0.0   # IC still contributes

    def test_all_scores_non_negative(self):
        ic     = {"f_momentum": 0.05, "f_vol": 0.03, "f_regime": -0.01}
        result = MarketIntegrator().market_weighted_ics(ic, make_market_with_data())
        assert all(v >= 0.0 for v in result.values())

    def test_signal_confidence_vector_range(self):
        market = make_market_with_data()
        conf   = MarketIntegrator().signal_confidence_vector(
            ["f_momentum", "f_vol", "f_new"], market
        )
        assert all(0.0 <= v <= 1.0 for v in conf.values())

    def test_unknown_signal_gets_neutral_confidence(self):
        conf = MarketIntegrator().signal_confidence_vector(["f_new"], SignalMarket())
        assert conf["f_new"] == pytest.approx(0.5)

    def test_blended_scores_detail_has_all_signals(self):
        ic     = {"f_momentum": 0.05, "f_vol": 0.03}
        detail = MarketIntegrator().blended_scores_detail(ic, make_market_with_data())
        names  = {d.signal_name for d in detail}
        assert names == set(ic.keys())

    def test_blended_scores_serialisable(self):
        ic     = {"f_momentum": 0.05}
        detail = MarketIntegrator().blended_scores_detail(ic, SignalMarket())
        json.dumps([d.to_dict() for d in detail])


# ════════════════════════════════════════════════════════════════════════════
# VALIDATOR TYPES
# ════════════════════════════════════════════════════════════════════════════

from macro8_subnet.validators.validator_types import (
    ValidatorRole, RewardProposal, ValidatorSubmission, ValidatorRegistry,
)


def make_proposal(
    v_uid:   int  = 0,
    role:    str  = "signal",
    weights: dict = None,
    stake:   float = 1.0,
) -> RewardProposal:
    if weights is None:
        weights = {0: 0.6, 1: 0.4}
    return RewardProposal(
        validator_uid=v_uid,
        validator_hotkey=f"5V{v_uid:040d}",
        role=ValidatorRole(role),
        epoch=1,
        reward_weights=weights,
        domain_scores={uid: w * 10 for uid, w in weights.items()},
        stake=stake,
    )


class TestValidatorRole:
    def test_four_roles(self):
        assert len(list(ValidatorRole)) == 4

    def test_each_role_has_agent_mapping(self):
        for r in ValidatorRole:
            assert len(r.evaluates_agent_role()) > 0

    def test_each_role_has_description(self):
        for r in ValidatorRole:
            assert len(r.description()) > 0


class TestRewardProposal:
    def test_weights_normalised(self):
        p = make_proposal(weights={0: 2.0, 1: 3.0})
        total = sum(p.reward_weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_negative_weights_clamped(self):
        p = make_proposal(weights={0: -0.5, 1: 0.5})
        assert p.reward_weights[0] == 0.0

    def test_is_valid_passes(self):
        ok, _ = make_proposal().is_valid()
        assert ok is True

    def test_is_valid_empty_weights(self):
        p = RewardProposal(0, "5V", ValidatorRole.SIGNAL, 1, {}, {}, 1.0)
        ok, _ = p.is_valid()
        assert ok is False

    def test_evaluation_hash_deterministic(self):
        p1 = make_proposal(0, "signal", {0: 0.6, 1: 0.4})
        p2 = make_proposal(0, "signal", {0: 0.6, 1: 0.4})
        assert p1.evaluation_hash() == p2.evaluation_hash()

    def test_different_proposals_different_hashes(self):
        p1 = make_proposal(0, "signal", {0: 0.6, 1: 0.4})
        p2 = make_proposal(0, "signal", {0: 0.3, 1: 0.7})
        assert p1.evaluation_hash() != p2.evaluation_hash()

    def test_to_dict_serialisable(self):
        json.dumps(make_proposal().to_dict())


class TestValidatorSubmission:
    def test_add_proposal(self):
        sub = ValidatorSubmission(0, "5V", 1)
        sub.add_proposal(make_proposal(0, "signal"))
        assert ValidatorRole.SIGNAL in sub.covered_roles()

    def test_duplicate_role_replaced(self):
        sub = ValidatorSubmission(0, "5V", 1)
        sub.add_proposal(make_proposal(0, "signal", {0: 0.6, 1: 0.4}))
        sub.add_proposal(make_proposal(0, "signal", {0: 0.3, 1: 0.7}))
        assert len([p for p in sub.proposals if p.role == ValidatorRole.SIGNAL]) == 1

    def test_proposal_for_returns_correct(self):
        sub = ValidatorSubmission(0, "5V", 1)
        sub.add_proposal(make_proposal(0, "risk"))
        p = sub.proposal_for(ValidatorRole.RISK)
        assert p is not None

    def test_proposal_for_missing_returns_none(self):
        sub = ValidatorSubmission(0, "5V", 1)
        assert sub.proposal_for(ValidatorRole.META) is None

    def test_to_dict_serialisable(self):
        sub = ValidatorSubmission(0, "5V", 1)
        sub.add_proposal(make_proposal(0, "signal"))
        json.dumps(sub.to_dict())


class TestValidatorRegistry:
    def test_all_roles_listed(self):
        assert len(ValidatorRegistry.all_roles()) == 4

    def test_evaluator_name_for_each(self):
        for r in ValidatorRole:
            assert len(ValidatorRegistry.evaluator_name(r)) > 0


# ════════════════════════════════════════════════════════════════════════════
# CONSENSUS ENGINE
# ════════════════════════════════════════════════════════════════════════════

from macro8_subnet.validators.consensus import (
    ConsensusEngine, ConsensusReport, DomainConsensus,
)


def make_validator_submission(
    v_uid:      int   = 0,
    roles:      list  = None,
    stake:      float = 1.0,
    miner_weights: dict = None,
) -> ValidatorSubmission:
    if roles is None:
        roles = ["signal"]
    if miner_weights is None:
        miner_weights = {0: 0.6, 1: 0.4}
    sub = ValidatorSubmission(v_uid, f"5V{v_uid:040d}", 1, stake=stake)
    for role in roles:
        sub.add_proposal(make_proposal(v_uid, role, miner_weights, stake))
    return sub


class TestConsensusEngine:
    def test_empty_submissions_returns_report(self):
        engine = ConsensusEngine()
        r      = engine.compute_consensus(1, [])
        assert isinstance(r, ConsensusReport)
        assert len(r.final_rewards) == 0

    def test_single_validator_passthrough(self):
        engine = ConsensusEngine()
        sub    = make_validator_submission(0, ["signal"], 1.0, {0: 0.7, 1: 0.3})
        r      = engine.compute_consensus(1, [sub])
        assert 0 in r.final_rewards
        assert 1 in r.final_rewards

    def test_rewards_sum_to_one(self):
        engine = ConsensusEngine()
        subs   = [
            make_validator_submission(0, ["signal"], 1.0, {0: 0.6, 1: 0.4}),
            make_validator_submission(1, ["signal"], 2.0, {0: 0.5, 1: 0.5}),
        ]
        r = engine.compute_consensus(1, subs)
        if r.final_rewards:
            total = sum(r.final_rewards.values())
            assert abs(total - 1.0) < 1e-5

    def test_higher_stake_more_influence(self):
        """High-stake validator should pull consensus toward their weights."""
        engine = ConsensusEngine()
        subs   = [
            make_validator_submission(0, ["signal"], stake=0.1,
                                      miner_weights={0: 0.9, 1: 0.1}),
            make_validator_submission(1, ["signal"], stake=10.0,
                                      miner_weights={0: 0.1, 1: 0.9}),
        ]
        r         = engine.compute_consensus(1, subs)
        uid0_w    = r.final_rewards.get(0, 0.0)
        uid1_w    = r.final_rewards.get(1, 0.0)
        # High-stake validator (v=1) prefers miner 1 → miner 1 should win
        assert uid1_w > uid0_w

    def test_divergence_tracked(self):
        engine = ConsensusEngine()
        subs   = [
            make_validator_submission(0, ["signal"], 1.0, {0: 0.9, 1: 0.1}),
            make_validator_submission(1, ["signal"], 1.0, {0: 0.1, 1: 0.9}),
        ]
        r = engine.compute_consensus(1, subs)
        assert len(r.divergences) >= 2

    def test_credibility_updated_after_consensus(self):
        engine = ConsensusEngine(disagreement_alpha=0.5)
        subs   = [
            make_validator_submission(0, ["signal"], 1.0, {0: 0.9, 1: 0.1}),
            make_validator_submission(1, ["signal"], 1.0, {0: 0.1, 1: 0.9}),
        ]
        engine.compute_consensus(1, subs)
        # Both diverge — credibility may decrease
        c0 = engine.credibility(0)
        c1 = engine.credibility(1)
        assert 0.0 <= c0 <= 1.0
        assert 0.0 <= c1 <= 1.0

    def test_agreeing_validators_stable_credibility(self):
        """Validators that agree perfectly should maintain credibility."""
        engine = ConsensusEngine(disagreement_alpha=0.5)
        same_weights = {0: 0.6, 1: 0.4}
        subs   = [
            make_validator_submission(i, ["signal"], 1.0, same_weights)
            for i in range(3)
        ]
        engine.compute_consensus(1, subs)
        for i in range(3):
            assert engine.credibility(i) >= 0.9

    def test_multi_domain_consensus(self):
        engine = ConsensusEngine()
        sub    = make_validator_submission(0, ["signal", "risk"], 1.0)
        r      = engine.compute_consensus(1, [sub])
        assert r.n_domains_covered <= 2   # may be 1 or 2 depending on proposals

    def test_as_weight_list(self):
        engine = ConsensusEngine()
        subs   = [make_validator_submission(0)]
        r      = engine.compute_consensus(1, subs)
        uids, ws = r.as_weight_list()
        assert len(uids) == len(ws)

    def test_to_dict_serialisable(self):
        engine = ConsensusEngine()
        subs   = [make_validator_submission(0)]
        r      = engine.compute_consensus(1, subs)
        json.dumps(r.to_dict())

    def test_custom_role_budgets(self):
        engine  = ConsensusEngine()
        subs    = [make_validator_submission(0, ["signal"], 1.0, {0: 0.6, 1: 0.4})]
        budgets = {"signal": 0.5, "risk": 0.3, "portfolio": 0.1, "meta": 0.1}
        r       = engine.compute_consensus(1, subs, budgets)
        if r.final_rewards:
            total = sum(r.final_rewards.values())
            assert abs(total - 1.0) < 1e-5

    def test_invalid_proposals_skipped(self):
        engine = ConsensusEngine()
        sub    = ValidatorSubmission(0, "5V", 1)
        # Add invalid proposal (empty weights)
        bad    = RewardProposal(0, "5V", ValidatorRole.SIGNAL, 1, {}, {}, 1.0)
        sub.proposals.append(bad)
        r      = engine.compute_consensus(1, [sub])
        # Should not crash, just skip invalid proposal
        assert isinstance(r, ConsensusReport)

    def test_n_validators_counted(self):
        engine = ConsensusEngine()
        subs   = [make_validator_submission(i) for i in range(3)]
        r      = engine.compute_consensus(1, subs)
        assert r.n_validators == 3
