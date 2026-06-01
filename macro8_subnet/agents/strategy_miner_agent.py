"""
agents/strategy_miner_agent.py
-------------------------------
Evaluates StrategyMiner submissions.

StrategyMiners compose signal combinations — they don't discover new
signals, they decide how to weight the existing library signals into
a tradable strategy.

What StrategyMiners submit
----------------------------
    signal_weights : dict[str, float] — {library_signal_name: weight}
                     Weights for combining library signals. Must sum to 1.0.

How submissions are evaluated
------------------------------
The validator:
  1. Loads the current alpha library
  2. Combines signals using submitted weights into a composite signal
  3. Scores the composite signal using IC scoring
  4. Reward = composite IC score (how predictive is this combination?)

This creates competition between StrategyMiners to find the best
signal combination, complementing SignalMiners who find the best
individual signals.

A StrategyMiner who finds that "70% momentum + 30% regime" outperforms
"50% momentum + 50% vol" earns higher rewards.
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
from macro8_subnet.alpha.ic_scorer    import ICScorer


@dataclass
class StrategyEvalResult:
    """Evaluation result for one StrategyMiner submission."""
    miner_uid:       int
    signal_weights:  dict[str, float]
    composite_ic:    Optional[float]   # IC of the combined signal
    composite_ir:    Optional[float]
    n_signals_used:  int
    reward_score:    float
    success:         bool
    error:           Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "miner_uid":       self.miner_uid,
            "signal_weights":  {k: round(v, 4) for k, v in self.signal_weights.items()},
            "composite_ic":    round(self.composite_ic, 6)  if self.composite_ic  else None,
            "composite_ir":    round(self.composite_ir, 6)  if self.composite_ir  else None,
            "n_signals_used":  self.n_signals_used,
            "reward_score":    round(self.reward_score, 6),
            "success":         self.success,
        }


class StrategyMinerEvaluator:
    """
    Evaluates StrategyMiner signal combination submissions.

    Combines library signals using submitted weights and measures
    the IC of the resulting composite signal.
    """

    def __init__(
        self,
        library_signals: dict[str, dict[str, pd.Series]],
        returns:         pd.DataFrame,
        risk_free:       float = 0.0,
    ):
        """
        Args:
            library_signals : {factor_name: {asset: series}} from AlphaLibrary.all_signals()
            returns         : Daily asset return DataFrame.
            risk_free       : Annual risk-free rate.
        """
        self.library  = library_signals
        self.returns  = returns
        self.risk_free = risk_free
        self._scorer  = ICScorer(min_obs=5, min_ic=0.0)

    def evaluate(self, submission: AgentSubmission) -> StrategyEvalResult:
        """Evaluate one StrategyMiner submission."""
        assert submission.role == AgentRole.STRATEGY

        sw = submission.payload.get("signal_weights", {})
        if not sw:
            return self._failed(submission, "Empty signal_weights")

        # Filter to signals that exist in the library
        valid_sw = {k: v for k, v in sw.items() if k in self.library and v > 0}
        if not valid_sw:
            return self._failed(submission, "No submitted signals found in library")

        # Renormalise
        total    = sum(valid_sw.values())
        norm_sw  = {k: v / total for k, v in valid_sw.items()}

        try:
            # Build composite signal: weighted average per asset across signals
            composite = self._combine_signals(norm_sw)
            if not composite:
                return self._failed(submission, "Could not build composite signal")

            ic_res = self._scorer.score(
                f"strategy_uid{submission.miner_uid}", composite, self.returns
            )

            ic    = ic_res.mean_ic if ic_res.success else 0.0
            ir    = ic_res.ic_ir   if ic_res.success else 0.0
            score = max(ic or 0.0, 0.0)   # reward ∝ positive IC

            return StrategyEvalResult(
                miner_uid=submission.miner_uid,
                signal_weights=norm_sw,
                composite_ic=ic,
                composite_ir=ir,
                n_signals_used=len(valid_sw),
                reward_score=score,
                success=True,
            )

        except Exception as exc:
            return self._failed(submission, f"{type(exc).__name__}: {exc}")

    def evaluate_batch(
        self, submissions: list[AgentSubmission]
    ) -> list[StrategyEvalResult]:
        return [self.evaluate(s) for s in submissions]

    def _combine_signals(
        self,
        weights: dict[str, float],
    ) -> dict[str, pd.Series]:
        """Weighted average of library signals per asset."""
        # Collect all assets across selected signals
        all_assets = sorted({
            asset
            for name in weights
            for asset in self.library[name]
        })

        composite = {}
        for asset in all_assets:
            weighted_sum = None
            for name, w in weights.items():
                if asset not in self.library[name]:
                    continue
                sig = self.library[name][asset]
                if weighted_sum is None:
                    weighted_sum = sig * w
                else:
                    common = weighted_sum.index.intersection(sig.index)
                    if len(common) > 0:
                        weighted_sum = weighted_sum.loc[common] + sig.loc[common] * w
            if weighted_sum is not None and len(weighted_sum) >= 5:
                composite[asset] = weighted_sum
        return composite

    @staticmethod
    def _failed(sub: AgentSubmission, reason: str) -> StrategyEvalResult:
        return StrategyEvalResult(
            miner_uid=sub.miner_uid,
            signal_weights=sub.payload.get("signal_weights", {}),
            composite_ic=None, composite_ir=None,
            n_signals_used=0,
            reward_score=0.0, success=False, error=reason,
        )
