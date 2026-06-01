"""
agents/risk_miner_agent.py
---------------------------
Evaluates RiskMiner submissions.

RiskMiners submit covariance model parameters. Their job is to model
the covariance structure of the asset universe — better covariance
estimates lead to better portfolio construction.

What RiskMiners submit
-----------------------
    shrinkage   : float [0,1]  — Ledoit-Wolf shrinkage intensity
    n_factors   : int          — number of statistical risk factors
    model_type  : str          — "ledoit_wolf" | "factor" | "diagonal"

How submissions are evaluated
------------------------------
The validator:
  1. Estimates the covariance matrix using the submitted parameters
  2. Builds an equal-weight portfolio
  3. Measures how well the submitted covariance predicts next-period
     realised portfolio variance (out-of-sample test)

Scoring metric:
    covariance_accuracy = 1 - |predicted_vol - realised_vol| / realised_vol

Higher accuracy = better risk model = higher reward.

This directly improves portfolio construction quality because better
covariance estimates allow the portfolio optimizer to take more
accurate risk positions.
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


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class CovarianceEstimate:
    """A fitted covariance matrix with metadata."""
    matrix:       np.ndarray     # n_assets × n_assets
    asset_names:  list[str]
    model_type:   str
    shrinkage:    float
    n_factors:    int

    def portfolio_vol(self, weights: np.ndarray) -> float:
        """Expected annualised portfolio volatility."""
        var = float(weights @ self.matrix * 252 @ weights)
        return float(np.sqrt(max(var, 1e-12)))

    def to_dict(self) -> dict:
        return {
            "model_type": self.model_type,
            "shrinkage":  round(self.shrinkage, 4),
            "n_factors":  self.n_factors,
            "n_assets":   len(self.asset_names),
        }


@dataclass
class RiskEvalResult:
    """Evaluation result for one RiskMiner submission."""
    miner_uid:          int
    model_type:         str
    shrinkage:          float
    predicted_vol:      Optional[float]   # annualised portfolio vol (predicted)
    realised_vol:       Optional[float]   # actual next-period vol
    covariance_accuracy: Optional[float]  # 1 - |pred-real|/real, ∈ [0,1]
    reward_score:       float             # normalised reward signal
    success:            bool
    error:              Optional[str]     = None

    def to_dict(self) -> dict:
        return {
            "miner_uid":           self.miner_uid,
            "model_type":          self.model_type,
            "shrinkage":           round(self.shrinkage, 4),
            "predicted_vol":       round(self.predicted_vol, 6) if self.predicted_vol else None,
            "realised_vol":        round(self.realised_vol, 6)  if self.realised_vol  else None,
            "covariance_accuracy": round(self.covariance_accuracy, 4) if self.covariance_accuracy else None,
            "reward_score":        round(self.reward_score, 6),
            "success":             self.success,
        }


# ── Evaluator ─────────────────────────────────────────────────────────────────

class RiskMinerEvaluator:
    """
    Evaluates RiskMiner covariance model submissions.

    Uses a train/test split: estimate covariance on the first 80%
    of returns data, then compare predicted portfolio vol against
    realised vol on the remaining 20%.
    """

    def __init__(
        self,
        returns:        pd.DataFrame,
        train_fraction: float = 0.80,
    ):
        """
        Args:
            returns:        Daily asset return DataFrame.
            train_fraction: Fraction of data used for covariance estimation.
        """
        self.returns        = returns
        split               = int(len(returns) * train_fraction)
        self.train_returns  = returns.iloc[:split]
        self.test_returns   = returns.iloc[split:]
        self.assets         = list(returns.columns)
        self.n_assets       = len(self.assets)
        self._baseline_accuracy = self._compute_baseline()

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(self, submission: AgentSubmission) -> RiskEvalResult:
        """
        Evaluate one RiskMiner submission.

        Args:
            submission: AgentSubmission with role=RISK.

        Returns:
            RiskEvalResult with covariance accuracy and reward score.
        """
        assert submission.role == AgentRole.RISK

        shrinkage  = float(submission.payload.get("shrinkage",  0.5))
        n_factors  = int(submission.payload.get("n_factors",    3))
        model_type = str(submission.payload.get("model_type",   "ledoit_wolf"))

        try:
            # Estimate covariance on training data
            cov_est = self._estimate_covariance(
                self.train_returns, shrinkage, n_factors, model_type
            )

            # Equal-weight portfolio for evaluation
            w = np.ones(self.n_assets) / self.n_assets

            # Predicted portfolio volatility
            predicted_vol = cov_est.portfolio_vol(w)

            # Realised portfolio volatility on test data
            port_test  = self.test_returns.mean(axis=1)
            realised_vol = float(port_test.std() * np.sqrt(252))

            # Accuracy: 1 - relative error, clipped to [0, 1]
            if realised_vol > 1e-8:
                rel_error = abs(predicted_vol - realised_vol) / realised_vol
                accuracy  = float(max(1.0 - rel_error, 0.0))
            else:
                accuracy  = 0.0

            # Reward: accuracy relative to baseline (sample covariance)
            reward = float(np.clip(accuracy / max(self._baseline_accuracy, 0.01), 0.0, 2.0))

            return RiskEvalResult(
                miner_uid=submission.miner_uid,
                model_type=model_type,
                shrinkage=shrinkage,
                predicted_vol=predicted_vol,
                realised_vol=realised_vol,
                covariance_accuracy=accuracy,
                reward_score=reward,
                success=True,
            )

        except Exception as exc:
            return RiskEvalResult(
                miner_uid=submission.miner_uid,
                model_type=model_type,
                shrinkage=shrinkage,
                predicted_vol=None, realised_vol=None,
                covariance_accuracy=None,
                reward_score=0.0, success=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    def evaluate_batch(
        self, submissions: list[AgentSubmission]
    ) -> list[RiskEvalResult]:
        """Evaluate multiple submissions. Failures don't affect others."""
        return [self.evaluate(s) for s in submissions]

    def best_covariance(
        self, results: list[RiskEvalResult]
    ) -> Optional[CovarianceEstimate]:
        """
        Return the covariance estimate from the best submission.
        Used by MultiAgentLoop to pass to the portfolio optimizer.
        """
        valid = [r for r in results if r.success]
        if not valid:
            return None
        best = max(valid, key=lambda r: r.covariance_accuracy or 0.0)
        return self._estimate_covariance(
            self.returns,
            best.shrinkage, 3, best.model_type
        )

    # ── Covariance models ─────────────────────────────────────────────────────

    def _estimate_covariance(
        self,
        data:       pd.DataFrame,
        shrinkage:  float,
        n_factors:  int,
        model_type: str,
    ) -> CovarianceEstimate:
        """Fit a covariance matrix using the specified model."""
        X     = data.values
        n, p  = X.shape

        if model_type == "diagonal":
            # Diagonal only — ignores correlations
            vols = np.var(X, axis=0)
            cov  = np.diag(vols)

        elif model_type == "factor":
            # Factor model: Σ = B Φ B' + D
            cov = self._factor_covariance(X, n_factors)

        else:
            # Ledoit-Wolf analytical shrinkage (default)
            cov = self._ledoit_wolf(X, shrinkage)

        return CovarianceEstimate(
            matrix=cov, asset_names=list(data.columns),
            model_type=model_type, shrinkage=shrinkage, n_factors=n_factors,
        )

    @staticmethod
    def _ledoit_wolf(X: np.ndarray, alpha: float) -> np.ndarray:
        """
        Ledoit-Wolf shrinkage: blend sample covariance with scaled identity.
        Σ_lw = (1-α) * Σ_sample + α * μ_var * I
        """
        S    = np.cov(X.T)
        mu   = float(np.trace(S) / S.shape[0])
        F    = mu * np.eye(S.shape[0])   # shrinkage target
        return (1.0 - alpha) * S + alpha * F

    @staticmethod
    def _factor_covariance(X: np.ndarray, n_factors: int) -> np.ndarray:
        """PCA-based factor covariance model."""
        from numpy.linalg import svd
        _, _, Vt = svd(X - X.mean(axis=0), full_matrices=False)
        B        = Vt[:n_factors].T   # loadings
        F_cov    = np.cov((X @ B).T)  # factor covariance
        specific = np.diag(np.maximum(np.var(X, axis=0) -
                                       np.diag(B @ F_cov @ B.T), 1e-8))
        return B @ F_cov @ B.T + specific

    def _compute_baseline(self) -> float:
        """Accuracy of the naive sample covariance on the test set."""
        try:
            S    = np.cov(self.train_returns.values.T)
            w    = np.ones(self.n_assets) / self.n_assets
            var  = float(w @ S * 252 @ w)
            pred = float(np.sqrt(max(var, 1e-12)))
            real = float(self.test_returns.mean(axis=1).std() * np.sqrt(252))
            if real > 1e-8:
                return float(max(1.0 - abs(pred - real) / real, 0.0))
        except Exception:
            pass
        return 0.5   # default baseline
