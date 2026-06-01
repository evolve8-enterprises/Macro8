"""
agents/agent_roles.py
----------------------
Defines the five agent roles in the Macro8 multi-agent network and
the submission schema that all role-specific agents share.

The five roles
--------------
    SIGNAL      Submits alpha formula strings for IC evaluation
    STRATEGY    Submits signal combination weights for backtest evaluation
    RISK        Submits covariance model parameters for vol prediction
    PORTFOLIO   Submits portfolio constraint sets for Sharpe improvement
    META        Submits IC predictions for prediction accuracy scoring

Every submission carries a role tag so the MultiAgentLoop can route
it to the correct evaluator without isinstance checks.

AgentSubmission extends FormulaSubmission with:
    role        : AgentRole enum value
    payload     : dict — role-specific parameters beyond the formula
                  SIGNAL:    {} (formula carries all information)
                  STRATEGY:  {"signal_weights": {"f1": 0.4, "f2": 0.6}}
                  RISK:      {"shrinkage": 0.5, "n_factors": 3}
                  PORTFOLIO: {"max_weight": 0.4, "max_turnover": 0.3}
                  META:      {"ic_predictions": {"f1": 0.05, "f2": 0.03}}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ── Role enum ─────────────────────────────────────────────────────────────────

class AgentRole(str, Enum):
    SIGNAL    = "signal"      # alpha formula miner
    STRATEGY  = "strategy"    # signal combination miner
    RISK      = "risk"        # covariance / vol model miner
    PORTFOLIO = "portfolio"   # portfolio constraint miner
    META      = "meta"        # IC prediction miner

    def reward_budget(self) -> float:
        """Fraction of total epoch rewards allocated to this role."""
        budgets = {
            AgentRole.SIGNAL:    0.30,
            AgentRole.STRATEGY:  0.25,
            AgentRole.RISK:      0.20,
            AgentRole.PORTFOLIO: 0.15,
            AgentRole.META:      0.10,
        }
        return budgets[self]

    def description(self) -> str:
        descs = {
            AgentRole.SIGNAL:    "Discovers predictive alpha signals via formula submission",
            AgentRole.STRATEGY:  "Combines library signals into tradable strategies",
            AgentRole.RISK:      "Models covariance structure and volatility forecasts",
            AgentRole.PORTFOLIO: "Optimises constraint sets for better risk-adjusted returns",
            AgentRole.META:      "Predicts which signals will perform best next period",
        }
        return descs[self]


# ── Submission schema ─────────────────────────────────────────────────────────

@dataclass
class AgentSubmission:
    """
    A miner submission with an explicit role tag.

    All five agent types use this single schema. The `role` field
    determines which evaluator processes the submission. The `payload`
    dict carries role-specific parameters.

    Attributes
    ----------
    miner_uid      : int — unique miner identifier
    miner_hotkey   : str — ss58 address
    role           : AgentRole — which pipeline stage this miner serves
    formula        : str — alpha formula (used by SIGNAL role;
                           empty string for other roles)
    payload        : dict — role-specific parameters (see module docstring)
    category       : str — optional signal category label
    description    : str — human-readable description of this submission
    """
    miner_uid:    int
    miner_hotkey: str
    role:         AgentRole
    formula:      str              = ""
    payload:      dict[str, Any]   = field(default_factory=dict)
    category:     str              = "unknown"
    description:  str              = ""

    # ── Convenience constructors ──────────────────────────────────────────────

    @classmethod
    def signal(
        cls,
        miner_uid:    int,
        miner_hotkey: str,
        formula:      str,
        category:     str = "unknown",
    ) -> "AgentSubmission":
        """Create a SIGNAL submission."""
        return cls(miner_uid=miner_uid, miner_hotkey=miner_hotkey,
                   role=AgentRole.SIGNAL, formula=formula, category=category)

    @classmethod
    def strategy(
        cls,
        miner_uid:      int,
        miner_hotkey:   str,
        signal_weights: dict[str, float],
    ) -> "AgentSubmission":
        """Create a STRATEGY submission."""
        return cls(miner_uid=miner_uid, miner_hotkey=miner_hotkey,
                   role=AgentRole.STRATEGY,
                   payload={"signal_weights": signal_weights})

    @classmethod
    def risk(
        cls,
        miner_uid:    int,
        miner_hotkey: str,
        shrinkage:    float = 0.5,
        n_factors:    int   = 3,
        model_type:   str   = "ledoit_wolf",
    ) -> "AgentSubmission":
        """Create a RISK submission."""
        return cls(miner_uid=miner_uid, miner_hotkey=miner_hotkey,
                   role=AgentRole.RISK,
                   payload={"shrinkage": shrinkage,
                            "n_factors": n_factors,
                            "model_type": model_type})

    @classmethod
    def portfolio(
        cls,
        miner_uid:     int,
        miner_hotkey:  str,
        max_weight:    float = 0.40,
        min_weight:    float = 0.0,
        max_turnover:  float = 1.0,
        method:        str   = "ic_weighted",
    ) -> "AgentSubmission":
        """Create a PORTFOLIO submission."""
        return cls(miner_uid=miner_uid, miner_hotkey=miner_hotkey,
                   role=AgentRole.PORTFOLIO,
                   payload={"max_weight":   max_weight,
                            "min_weight":   min_weight,
                            "max_turnover": max_turnover,
                            "method":       method})

    @classmethod
    def meta(
        cls,
        miner_uid:       int,
        miner_hotkey:    str,
        ic_predictions:  dict[str, float],
    ) -> "AgentSubmission":
        """Create a META submission."""
        return cls(miner_uid=miner_uid, miner_hotkey=miner_hotkey,
                   role=AgentRole.META,
                   payload={"ic_predictions": ic_predictions})

    # ── Validation ────────────────────────────────────────────────────────────

    def is_valid(self) -> tuple[bool, str]:
        """Basic format validation before routing to evaluator."""
        if self.role == AgentRole.SIGNAL:
            if not self.formula.strip():
                return False, "SIGNAL submission requires a non-empty formula."

        elif self.role == AgentRole.STRATEGY:
            sw = self.payload.get("signal_weights", {})
            if not sw:
                return False, "STRATEGY submission requires non-empty signal_weights."
            if any(v < 0 for v in sw.values()):
                return False, "Strategy weights must be non-negative."
            total = sum(sw.values())
            if abs(total - 1.0) > 0.02:
                return False, f"Strategy weights sum to {total:.4f}, expected 1.0 ±0.02."

        elif self.role == AgentRole.RISK:
            s = self.payload.get("shrinkage", 0.5)
            if not 0.0 <= s <= 1.0:
                return False, f"Shrinkage must be in [0, 1], got {s}."

        elif self.role == AgentRole.PORTFOLIO:
            mw = self.payload.get("max_weight", 0.4)
            if not 0.0 < mw <= 1.0:
                return False, f"max_weight must be in (0, 1], got {mw}."

        elif self.role == AgentRole.META:
            preds = self.payload.get("ic_predictions", {})
            if not preds:
                return False, "META submission requires non-empty ic_predictions."

        return True, ""

    def to_dict(self) -> dict:
        return {
            "miner_uid":    self.miner_uid,
            "miner_hotkey": self.miner_hotkey,
            "role":         self.role.value,
            "formula":      self.formula,
            "payload":      self.payload,
            "category":     self.category,
        }


# ── Registry ──────────────────────────────────────────────────────────────────

class AgentRegistry:
    """
    Maps AgentRole values to their evaluator names and reward budgets.

    Used by MultiAgentLoop to route submissions and by role_rewards.py
    to compute role-stratified rewards.
    """

    ROLES: list[AgentRole] = list(AgentRole)

    @staticmethod
    def budget(role: AgentRole) -> float:
        return role.reward_budget()

    @staticmethod
    def all_budgets() -> dict[str, float]:
        return {r.value: r.reward_budget() for r in AgentRole}

    @staticmethod
    def total_budget() -> float:
        return sum(r.reward_budget() for r in AgentRole)

    @staticmethod
    def roles_by_budget() -> list[AgentRole]:
        return sorted(AgentRole, key=lambda r: r.reward_budget(), reverse=True)
