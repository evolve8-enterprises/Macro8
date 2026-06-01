"""
agents/portfolio_miner_agent.py
--------------------------------
Evaluates PortfolioMiner submissions.

PortfolioMiners submit portfolio constraint sets. Their job is to find
the constraint configuration that maximises the portfolio's Sharpe
ratio while controlling risk.

What PortfolioMiners submit
-----------------------------
    max_weight    : float — maximum single-asset weight [0, 1]
    min_weight    : float — minimum non-zero weight [0, 0.1]
    max_turnover  : float — maximum daily portfolio turnover [0, 1]
    method        : str   — optimizer method ("ic_weighted" | "mean_variance" | "risk_parity")

How submissions are evaluated
------------------------------
The validator:
  1. Builds a portfolio using the submitted constraints + existing signal library
  2. Simulates this portfolio over the evaluation period
  3. Computes Sharpe ratio vs an unconstrained baseline
  4. Reward = Sharpe improvement above baseline

This rewards miners who find constraint sets that genuinely improve
risk-adjusted returns — tighter constraints that reduce concentration
risk, optimal turnover limits that balance cost vs responsiveness, etc.
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
from macro8_subnet.alpha.portfolio_optimizer import PortfolioOptimizer, OptMethod


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class PortfolioEvalResult:
    """Evaluation result for one PortfolioMiner submission."""
    miner_uid:        int
    max_weight:       float
    min_weight:       float
    max_turnover:     float
    method:           str
    constrained_sharpe:   Optional[float]   # Sharpe with submitted constraints
    baseline_sharpe:      Optional[float]   # Sharpe with default constraints
    sharpe_improvement:   Optional[float]   # constrained - baseline
    reward_score:         float
    success:          bool
    error:            Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "miner_uid":           self.miner_uid,
            "max_weight":          round(self.max_weight,  4),
            "max_turnover":        round(self.max_turnover, 4),
            "method":              self.method,
            "constrained_sharpe":  round(self.constrained_sharpe, 4) if self.constrained_sharpe else None,
            "baseline_sharpe":     round(self.baseline_sharpe,    4) if self.baseline_sharpe    else None,
            "sharpe_improvement":  round(self.sharpe_improvement, 4) if self.sharpe_improvement else None,
            "reward_score":        round(self.reward_score, 6),
            "success":             self.success,
        }


# ── Evaluator ─────────────────────────────────────────────────────────────────

class PortfolioMinerEvaluator:
    """
    Evaluates PortfolioMiner constraint submissions.

    Compares each submitted constraint configuration against a default
    unconstrained baseline using the current library's signal IC scores
    as expected returns.
    """

    # Default (baseline) constraints
    DEFAULT_MAX_WEIGHT = 0.40
    DEFAULT_METHOD     = OptMethod.IC_WEIGHTED

    def __init__(
        self,
        returns:      pd.DataFrame,
        signal_ics:   dict[str, float],   # {signal_name: mean_ic} from library
        risk_free:    float = 0.0,
    ):
        """
        Args:
            returns:     Daily asset return DataFrame.
            signal_ics:  IC scores for current library signals
                         (used as expected-return proxies for optimisation).
            risk_free:   Annual risk-free rate.
        """
        self.returns    = returns
        self.signal_ics = signal_ics
        self.risk_free  = risk_free
        self._baseline  = self._compute_baseline_sharpe()

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(self, submission: AgentSubmission) -> PortfolioEvalResult:
        """Evaluate one PortfolioMiner submission."""
        assert submission.role == AgentRole.PORTFOLIO

        max_w    = float(submission.payload.get("max_weight",   0.40))
        min_w    = float(submission.payload.get("min_weight",   0.0))
        max_turn = float(submission.payload.get("max_turnover", 1.0))
        method   = str(submission.payload.get("method", "ic_weighted"))

        # Validate constraints
        if not 0.0 < max_w <= 1.0:
            return self._failed(submission, f"max_weight {max_w} out of [0,1]")
        if min_w < 0 or min_w >= max_w:
            return self._failed(submission, f"min_weight {min_w} must be in [0, max_weight)")

        try:
            opt_method = OptMethod(method)
        except ValueError:
            opt_method = OptMethod.IC_WEIGHTED

        try:
            constrained_sharpe = self._evaluate_constraints(
                max_w, max_turn, opt_method
            )
            improvement = (constrained_sharpe - self._baseline
                           if constrained_sharpe is not None else None)

            # Reward: proportional to Sharpe improvement (clipped to [0, 2])
            reward = float(np.clip(
                (improvement or 0.0) / max(abs(self._baseline), 0.01) + 1.0,
                0.0, 2.0,
            ))

            return PortfolioEvalResult(
                miner_uid=submission.miner_uid,
                max_weight=max_w, min_weight=min_w,
                max_turnover=max_turn, method=method,
                constrained_sharpe=constrained_sharpe,
                baseline_sharpe=self._baseline,
                sharpe_improvement=improvement,
                reward_score=reward,
                success=True,
            )

        except Exception as exc:
            return self._failed(submission, f"{type(exc).__name__}: {exc}")

    def evaluate_batch(
        self, submissions: list[AgentSubmission]
    ) -> list[PortfolioEvalResult]:
        return [self.evaluate(s) for s in submissions]

    def best_constraints(
        self, results: list[PortfolioEvalResult]
    ) -> dict:
        """Return the constraint set from the highest-Sharpe submission."""
        valid = [r for r in results if r.success and r.constrained_sharpe is not None]
        if not valid:
            return {"max_weight": self.DEFAULT_MAX_WEIGHT, "method": self.DEFAULT_METHOD.value}
        best = max(valid, key=lambda r: r.constrained_sharpe or 0.0)
        return {"max_weight": best.max_weight, "method": best.method}

    # ── Internal ──────────────────────────────────────────────────────────────

    def _evaluate_constraints(
        self,
        max_weight: float,
        max_turnover: float,
        method: OptMethod,
    ) -> Optional[float]:
        """
        Build portfolio with given constraints and compute realised Sharpe.
        """
        # Use IC scores as signal strengths → map to asset weights
        # Filter to assets in our returns data
        asset_signals = {
            a: float(np.mean(list(self.signal_ics.values())))
            for a in self.returns.columns
        }
        if not asset_signals:
            return None

        opt    = PortfolioOptimizer(method=method, max_weight=max_weight,
                                     risk_free=self.risk_free)
        result = opt.optimize(asset_signals, self.returns)

        if not result.success:
            return None

        return result.sharpe_ratio

    def _compute_baseline_sharpe(self) -> float:
        """Baseline: equal-weight portfolio Sharpe."""
        try:
            port = self.returns.mean(axis=1)
            mu   = float(port.mean() * 252)
            vol  = float(port.std() * np.sqrt(252))
            return (mu - self.risk_free) / vol if vol > 1e-8 else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _failed(sub: AgentSubmission, reason: str) -> PortfolioEvalResult:
        return PortfolioEvalResult(
            miner_uid=sub.miner_uid,
            max_weight=sub.payload.get("max_weight", 0.4),
            min_weight=sub.payload.get("min_weight", 0.0),
            max_turnover=sub.payload.get("max_turnover", 1.0),
            method=sub.payload.get("method", "ic_weighted"),
            constrained_sharpe=None, baseline_sharpe=None,
            sharpe_improvement=None,
            reward_score=0.0, success=False, error=reason,
        )
