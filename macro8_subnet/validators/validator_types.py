"""
validators/validator_types.py
------------------------------
Defines the four validator roles and the reward proposal schema for
the distributed validator consensus system.

In a centralised design, one validator runs all evaluations and sets
all rewards. The distributed design splits evaluation into four
specialised validators that each vote on rewards for their domain.

The four validator roles
-------------------------
    SIGNAL     evaluates signal IC quality → votes on SignalMiner rewards
    RISK       evaluates covariance models → votes on RiskMiner rewards
    PORTFOLIO  evaluates constraint sets   → votes on PortfolioMiner rewards
    META       evaluates IC predictions    → votes on MetaMiner rewards

Reward proposal flow
---------------------
    1. Each validator runs its evaluation independently
    2. Each validator produces a RewardProposal: {miner_uid → reward_weight}
    3. ConsensusEngine (consensus.py) aggregates proposals:
           final_reward = stake_weighted_average(proposals)
    4. Validators whose proposals diverge from consensus are penalised
       (Brier score of their calls)

This creates:
    - Redundancy: no single validator can corrupt rewards
    - Specialisation: each validator focuses on what it knows
    - Accountability: divergent validators lose credibility

ValidatorSubmission schema
---------------------------
    validator_uid     : int
    validator_hotkey  : str
    role              : ValidatorRole
    epoch             : int
    reward_proposals  : {miner_uid: reward_weight}
    domain_scores     : {miner_uid: raw_score}  (before normalisation)
    stake             : float  — validator's TAO stake (weights their vote)
    evaluation_hash   : str    — hash of evaluation inputs for verification
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Validator roles ───────────────────────────────────────────────────────────

class ValidatorRole(str, Enum):
    SIGNAL    = "signal"      # evaluates SignalMiner submissions (IC scoring)
    RISK      = "risk"        # evaluates RiskMiner submissions (vol prediction)
    PORTFOLIO = "portfolio"   # evaluates PortfolioMiner submissions (Sharpe)
    META      = "meta"        # evaluates MetaMiner submissions (IC prediction)

    def evaluates_agent_role(self) -> str:
        """Which AgentRole this ValidatorRole evaluates."""
        mapping = {
            ValidatorRole.SIGNAL:    "signal",
            ValidatorRole.RISK:      "risk",
            ValidatorRole.PORTFOLIO: "portfolio",
            ValidatorRole.META:      "meta",
        }
        return mapping[self]

    def description(self) -> str:
        descs = {
            ValidatorRole.SIGNAL:    "Evaluates alpha signal IC quality",
            ValidatorRole.RISK:      "Evaluates covariance model accuracy",
            ValidatorRole.PORTFOLIO: "Evaluates portfolio constraint performance",
            ValidatorRole.META:      "Evaluates IC prediction accuracy",
        }
        return descs[self]


# ── Proposal and submission types ─────────────────────────────────────────────

@dataclass
class RewardProposal:
    """
    One validator's reward vote for a set of miners in its domain.

    The proposal contains normalised reward weights — they must sum to
    1.0 (or close to it). The ConsensusEngine combines proposals from
    multiple validators using stake-weighted averaging.
    """
    validator_uid:    int
    validator_hotkey: str
    role:             ValidatorRole
    epoch:            int
    reward_weights:   dict[int, float]    # {miner_uid: weight}, sum ≈ 1.0
    domain_scores:    dict[int, float]    # {miner_uid: raw score (pre-normalise)}
    stake:            float               # validator's TAO stake

    def __post_init__(self):
        # Ensure weights are non-negative
        self.reward_weights = {
            uid: max(float(w), 0.0)
            for uid, w in self.reward_weights.items()
        }
        # Normalise if needed
        total = sum(self.reward_weights.values())
        if total > 1e-8 and abs(total - 1.0) > 0.01:
            self.reward_weights = {
                uid: w / total for uid, w in self.reward_weights.items()
            }

    def evaluation_hash(self) -> str:
        """Deterministic hash of this proposal for verification."""
        payload = json.dumps({
            "v":      self.validator_uid,
            "role":   self.role.value,
            "epoch":  self.epoch,
            "scores": {str(k): round(v, 6) for k, v in
                       sorted(self.domain_scores.items())},
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def is_valid(self) -> tuple[bool, str]:
        """Basic validation checks."""
        if not self.reward_weights:
            return False, "Empty reward_weights"
        if any(w < 0 for w in self.reward_weights.values()):
            return False, "Negative reward weight"
        total = sum(self.reward_weights.values())
        if abs(total - 1.0) > 0.05:
            return False, f"Weights sum to {total:.4f}, expected 1.0 ±0.05"
        if self.stake < 0:
            return False, f"Negative stake: {self.stake}"
        return True, ""

    def to_dict(self) -> dict:
        return {
            "validator_uid":  self.validator_uid,
            "role":           self.role.value,
            "epoch":          self.epoch,
            "stake":          round(self.stake, 4),
            "reward_weights": {str(k): round(v, 6)
                               for k, v in self.reward_weights.items()},
            "hash":           self.evaluation_hash(),
        }


@dataclass
class ValidatorSubmission:
    """
    A validator's complete submission for one epoch.

    One submission can contain proposals for multiple domains if the
    validator is a generalist (though specialist validators are preferred).
    """
    validator_uid:    int
    validator_hotkey: str
    epoch:            int
    proposals:        list[RewardProposal] = field(default_factory=list)
    stake:            float                = 1.0

    def add_proposal(self, proposal: RewardProposal) -> None:
        """Add a domain proposal. Each role can appear at most once."""
        roles = {p.role for p in self.proposals}
        if proposal.role in roles:
            # Replace existing proposal for this role
            self.proposals = [p for p in self.proposals
                              if p.role != proposal.role]
        self.proposals.append(proposal)

    def proposal_for(self, role: ValidatorRole) -> Optional[RewardProposal]:
        """Return the proposal for a specific role, or None."""
        for p in self.proposals:
            if p.role == role:
                return p
        return None

    def covered_roles(self) -> list[ValidatorRole]:
        return [p.role for p in self.proposals]

    def to_dict(self) -> dict:
        return {
            "validator_uid":  self.validator_uid,
            "validator_hotkey": self.validator_hotkey,
            "epoch":          self.epoch,
            "stake":          round(self.stake, 4),
            "covered_roles":  [r.value for r in self.covered_roles()],
            "proposals":      [p.to_dict() for p in self.proposals],
        }


# ── Registry ──────────────────────────────────────────────────────────────────

class ValidatorRegistry:
    """Maps ValidatorRole to the evaluation module it should run."""

    ROLE_TO_EVALUATOR = {
        ValidatorRole.SIGNAL:    "RiskMinerEvaluator → ICScorer",
        ValidatorRole.RISK:      "RiskMinerEvaluator",
        ValidatorRole.PORTFOLIO: "PortfolioMinerEvaluator",
        ValidatorRole.META:      "MetaMinerEvaluator",
    }

    @staticmethod
    def all_roles() -> list[ValidatorRole]:
        return list(ValidatorRole)

    @staticmethod
    def evaluator_name(role: ValidatorRole) -> str:
        return ValidatorRegistry.ROLE_TO_EVALUATOR[role]
