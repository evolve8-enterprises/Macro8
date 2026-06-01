"""
agents/meta_miner_agent.py
---------------------------
Evaluates MetaMiner submissions.

MetaMiners predict which library signals will have the highest IC in
the next epoch. They are rewarded for accurate IC predictions.

What MetaMiners submit
-----------------------
    ic_predictions : dict[str, float] — {signal_name: predicted_ic}
                     Predictions of next-period IC for each library signal.

How submissions are evaluated
------------------------------
The validator:
  1. Collects MetaMiner IC predictions from epoch t
  2. At epoch t+1, observes actual ICs for all library signals
  3. Computes rank correlation between predicted and actual IC
  4. Reward = max(rank_correlation, 0) — only positive prediction accuracy rewarded

This incentivises MetaMiners to build sophisticated models of signal
quality, exactly mirroring how proprietary meta-learning systems work
at top quant firms.

MetaMiners feed their predictions into the evolution engine — signals
predicted to have high future IC get more exploration in the formula
mutation step.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_SUBNET = Path(__file__).resolve().parent.parent
_ROOT   = _SUBNET.parent
for p in [str(_SUBNET), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from macro8_subnet.agents.agent_roles import AgentSubmission, AgentRole


@dataclass
class MetaEvalResult:
    """Evaluation result for one MetaMiner submission."""
    miner_uid:          int
    n_predictions:      int
    n_matched:          int              # predictions that matched library signals
    rank_correlation:   Optional[float]  # Spearman rank corr(predicted, actual)
    prediction_accuracy: Optional[float] # fraction of direction calls correct
    reward_score:        float
    success:             bool
    error:               Optional[str]  = None

    def to_dict(self) -> dict:
        return {
            "miner_uid":           self.miner_uid,
            "n_predictions":       self.n_predictions,
            "n_matched":           self.n_matched,
            "rank_correlation":    round(self.rank_correlation,   4) if self.rank_correlation    else None,
            "prediction_accuracy": round(self.prediction_accuracy, 4) if self.prediction_accuracy else None,
            "reward_score":        round(self.reward_score, 6),
            "success":             self.success,
        }


class MetaMinerEvaluator:
    """
    Evaluates MetaMiner IC prediction submissions.

    Stores predictions from epoch t and scores them at epoch t+1 when
    actual ICs are observed. Between epochs, predictions from the
    best-performing MetaMiner guide the evolution engine.
    """

    def __init__(self, min_matched: int = 2):
        """
        Args:
            min_matched: Minimum number of matched predictions required.
        """
        self.min_matched = min_matched
        # Pending predictions: {miner_uid: {signal_name: predicted_ic}}
        self._pending: dict[int, dict[str, float]] = {}

    # ── Deferred scoring (two-epoch cycle) ───────────────────────────────────

    def store_predictions(
        self,
        submission: AgentSubmission,
    ) -> None:
        """
        Store predictions from epoch t for scoring at epoch t+1.

        Args:
            submission: MetaMiner AgentSubmission.
        """
        assert submission.role == AgentRole.META
        preds = submission.payload.get("ic_predictions", {})
        if preds:
            self._pending[submission.miner_uid] = dict(preds)

    def score_pending(
        self,
        actual_ics: dict[str, float],
    ) -> list[MetaEvalResult]:
        """
        Score all pending predictions against observed actual ICs.

        Call this at the start of each epoch with the ICs observed
        in the previous epoch.

        Args:
            actual_ics: {signal_name: mean_ic} — realised ICs this epoch.

        Returns:
            List of MetaEvalResult, one per miner with pending predictions.
        """
        results = []
        for uid, predictions in self._pending.items():
            result = self._score_one(uid, predictions, actual_ics)
            results.append(result)
        self._pending.clear()
        return results

    # ── Immediate scoring (same-epoch, for testing/live validation) ──────────

    def evaluate(
        self,
        submission:  AgentSubmission,
        actual_ics:  dict[str, float],
    ) -> MetaEvalResult:
        """
        Immediately evaluate a META submission against known actual ICs.

        Args:
            submission:  MetaMiner AgentSubmission.
            actual_ics:  {signal_name: mean_ic} — ground truth ICs.
        """
        assert submission.role == AgentRole.META
        predictions = submission.payload.get("ic_predictions", {})
        if not predictions:
            return self._failed(submission, "Empty ic_predictions")
        return self._score_one(submission.miner_uid, predictions, actual_ics)

    def evaluate_batch(
        self,
        submissions: list[AgentSubmission],
        actual_ics:  dict[str, float],
    ) -> list[MetaEvalResult]:
        return [self.evaluate(s, actual_ics) for s in submissions]

    def best_predictions(
        self,
        results: list[MetaEvalResult],
        submissions: list[AgentSubmission],
    ) -> Optional[dict[str, float]]:
        """
        Return IC predictions from the highest-accuracy MetaMiner.
        Used by the evolution engine to guide formula mutation.
        """
        valid = [
            (r, s) for r, s in zip(results, submissions)
            if r.success and r.rank_correlation is not None
        ]
        if not valid:
            return None
        best_result, best_sub = max(valid, key=lambda x: x[0].rank_correlation or 0.0)
        return best_sub.payload.get("ic_predictions", {})

    # ── Internal ──────────────────────────────────────────────────────────────

    def _score_one(
        self,
        miner_uid:   int,
        predictions: dict[str, float],
        actual_ics:  dict[str, float],
    ) -> MetaEvalResult:
        """Score one miner's predictions against actual ICs."""
        from scipy.stats import spearmanr

        common_keys = [k for k in predictions if k in actual_ics]
        n_matched   = len(common_keys)

        if n_matched < self.min_matched:
            return MetaEvalResult(
                miner_uid=miner_uid,
                n_predictions=len(predictions),
                n_matched=n_matched,
                rank_correlation=None,
                prediction_accuracy=None,
                reward_score=0.0,
                success=False,
                error=f"Only {n_matched} matched predictions (min {self.min_matched})",
            )

        pred_vals   = np.array([predictions[k]  for k in common_keys])
        actual_vals = np.array([actual_ics[k]   for k in common_keys])

        # Rank correlation
        try:
            rho, _ = spearmanr(pred_vals, actual_vals)
            rho    = 0.0 if np.isnan(rho) else float(rho)
        except Exception:
            rho    = 0.0

        # Directional accuracy (same sign as actual?)
        directions_correct = np.sign(pred_vals) == np.sign(actual_vals)
        dir_accuracy       = float(directions_correct.mean())

        # Reward: max(rank_correlation, 0) — no reward for wrong-way predictions
        reward = float(max(rho, 0.0))

        return MetaEvalResult(
            miner_uid=miner_uid,
            n_predictions=len(predictions),
            n_matched=n_matched,
            rank_correlation=rho,
            prediction_accuracy=dir_accuracy,
            reward_score=reward,
            success=True,
        )

    @staticmethod
    def _failed(sub: AgentSubmission, reason: str) -> MetaEvalResult:
        return MetaEvalResult(
            miner_uid=sub.miner_uid,
            n_predictions=len(sub.payload.get("ic_predictions", {})),
            n_matched=0,
            rank_correlation=None, prediction_accuracy=None,
            reward_score=0.0, success=False, error=reason,
        )
