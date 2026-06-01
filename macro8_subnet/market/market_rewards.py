"""
market/market_rewards.py
-------------------------
Reward computation for prediction market participants.

Uses the Quadratic Scoring Rule (QSR), a proper scoring rule
that makes truthful reporting the dominant strategy.

Proper scoring rules
---------------------
A scoring rule is *proper* if the optimal strategy for a rational
agent is to report their true beliefs. This prevents miners from
gaining by misreporting their confidence.

Quadratic Scoring Rule:
    For a binary event (IC > 0 or not):
    If outcome = 1 (event occurred):
        score = 2 * p - p^2
    If outcome = 0 (event did not occur):
        score = 1 - p^2
    where p = stated probability [0, 1]

    Expected score is maximised when p = true probability.

Simplified form used here (aligned with SignalMarket pnl_score):
    correct:   pnl = +(2 * confidence - 1) ∈ [0, 1]
    incorrect: pnl = -(2 * confidence - 1) ∈ [-1, 0]

    A miner who says confidence=1.0 and is wrong gets pnl = -1.0
    A miner who says confidence=0.5 (uncertain) gets pnl ≈ 0

Reward aggregation
-------------------
Each miner's reward = normalised sum of pnl_scores across all positions.
Negative total pnl → zero reward (no punishment in TAO, just no reward).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_SUBNET = Path(__file__).resolve().parent.parent
_ROOT   = _SUBNET.parent
for p in [str(_SUBNET), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from macro8_subnet.market.signal_market import SettlementResult


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class PredictorReward:
    """Aggregated market reward for one predictor miner."""
    miner_uid:         int
    miner_hotkey:      str
    n_positions:       int
    n_correct:         int
    accuracy:          float        # fraction of correct predictions
    total_pnl:         float        # sum of quadratic scoring rule scores
    reward_weight:     float        # normalised TAO weight [0, 1]
    rank:              int

    def to_dict(self) -> dict:
        return {
            "miner_uid":     self.miner_uid,
            "n_positions":   self.n_positions,
            "n_correct":     self.n_correct,
            "accuracy":      round(self.accuracy,      4),
            "total_pnl":     round(self.total_pnl,     6),
            "reward_weight": round(self.reward_weight, 6),
            "rank":          self.rank,
        }


@dataclass
class MarketRewardReport:
    """Complete reward report for all predictors in one settlement."""
    epoch:           int
    n_predictors:    int
    n_positions:     int
    rewards:         list[PredictorReward]
    total_weight:    float   # should be 1.0

    def as_weight_list(self) -> tuple[list[int], list[float]]:
        return (
            [r.miner_uid     for r in self.rewards],
            [r.reward_weight for r in self.rewards],
        )

    def to_dict(self) -> dict:
        return {
            "epoch":        self.epoch,
            "n_predictors": self.n_predictors,
            "n_positions":  self.n_positions,
            "total_weight": round(self.total_weight, 6),
            "rewards":      [r.to_dict() for r in self.rewards],
        }


# ── Quadratic Scoring Rule ────────────────────────────────────────────────────

class QuadraticScorer:
    """
    Computes proper-scoring-rule rewards for market participants.

    Takes settlement results (from SignalMarket.settle_epoch()) and
    produces normalised reward weights for each predictor miner.
    """

    def __init__(
        self,
        temperature:    float = 1.0,
        min_reward:     float = 0.0,   # minimum reward (non-negative)
    ):
        """
        Args:
            temperature: Softmax sharpness for reward normalisation.
            min_reward:  Floor on individual rewards (negative PnL → 0).
        """
        self.temperature = temperature
        self.min_reward  = min_reward

    def compute_rewards(
        self,
        settlements: list[SettlementResult],
        epoch:       int = 0,
    ) -> MarketRewardReport:
        """
        Compute predictor rewards from settlement results.

        Args:
            settlements: List of SettlementResult from SignalMarket.settle_epoch().
            epoch:       Current epoch number.

        Returns:
            MarketRewardReport with per-miner rewards summing to 1.0.
        """
        if not settlements:
            return MarketRewardReport(
                epoch=epoch, n_predictors=0, n_positions=0,
                rewards=[], total_weight=0.0,
            )

        # Aggregate by miner
        miner_data: dict[int, dict] = {}
        for s in settlements:
            uid = s.position.miner_uid
            if uid not in miner_data:
                miner_data[uid] = {
                    "hotkey":     s.position.miner_hotkey,
                    "pnl_scores": [],
                    "correct":    [],
                }
            miner_data[uid]["pnl_scores"].append(s.pnl_score)
            miner_data[uid]["correct"].append(s.prediction_correct)

        # Compute per-miner totals
        uids         = []
        total_pnls   = []
        accuracies   = []
        n_positions  = []
        n_corrects   = []

        for uid, data in miner_data.items():
            pnl       = float(sum(data["pnl_scores"]))
            acc       = float(sum(data["correct"]) / len(data["correct"]))
            uids.append(uid)
            total_pnls.append(pnl)
            accuracies.append(acc)
            n_positions.append(len(data["pnl_scores"]))
            n_corrects.append(sum(data["correct"]))

        # Convert PnL → non-negative reward signal
        raw = np.maximum(np.array(total_pnls), self.min_reward)

        # Softmax normalisation
        total = raw.sum()
        if total < 1e-8:
            weights = np.ones(len(raw)) / max(len(raw), 1)
        else:
            scaled   = raw * self.temperature
            shifted  = scaled - scaled.max()
            exp_vals = np.exp(np.clip(shifted, -500, 500))
            weights  = exp_vals / exp_vals.sum()

        # Build result sorted by reward descending
        order   = np.argsort(weights)[::-1]
        rewards = []
        for rank, i in enumerate(order, start=1):
            uid = uids[i]
            rewards.append(PredictorReward(
                miner_uid=uid,
                miner_hotkey=miner_data[uid]["hotkey"],
                n_positions=n_positions[i],
                n_correct=n_corrects[i],
                accuracy=accuracies[i],
                total_pnl=total_pnls[i],
                reward_weight=float(weights[i]),
                rank=rank,
            ))

        return MarketRewardReport(
            epoch=epoch,
            n_predictors=len(rewards),
            n_positions=len(settlements),
            rewards=rewards,
            total_weight=float(weights.sum()),
        )
