"""
validators/consensus.py
------------------------
Distributed validator consensus engine.

Aggregates reward proposals from multiple specialised validators
into a single authoritative reward vector. Uses stake-weighted
averaging so high-stake validators carry more weight.

Validators whose proposals consistently diverge from consensus are
penalised — this creates accountability and rewards honest evaluation.

Consensus mechanism
--------------------
For each miner in a domain:
    final_weight_i = Σ(stake_v * proposal_weight_i_v) / Σ(stake_v)

where the sum is over all validators who submitted a proposal for
this domain and this miner.

Disagreement penalty
---------------------
After consensus is computed, each validator's proposal is compared
to the consensus using a Brier-score-like divergence metric:

    divergence_v = mean(|proposal_weight_i_v - consensus_weight_i|²)

Validators with high divergence lose credibility (tracked as a
running score). Their proposals receive lower weight in future epochs.

This punishes:
    - Corrupt validators trying to inflate specific miners' rewards
    - Lazy validators who submit random or default proposals
    - Colluding validators who copy each other

It rewards:
    - Validators who independently arrive at the same conclusion
    - Validators whose domain expertise produces accurate evaluations

Final reward assembly
----------------------
After domain-specific consensus:
    1. Each domain produces a consensus weight vector (normalised)
    2. Domain weights are rescaled by role budgets (from AgentRole)
    3. Final reward = concatenation of all domain weight vectors

The final reward vector has one entry per (miner, role) pair and
sums to 1.0 globally.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

_SUBNET = Path(__file__).resolve().parent.parent
_ROOT   = _SUBNET.parent
for p in [str(_SUBNET), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from macro8_subnet.validators.validator_types import (
    ValidatorRole, RewardProposal, ValidatorSubmission
)


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class DomainConsensus:
    """Consensus reward weights for one validator domain."""
    role:              ValidatorRole
    consensus_weights: dict[int, float]    # {miner_uid: weight}, sum ≈ 1.0
    n_validators:      int
    total_stake:       float
    coverage:          float               # fraction of miners that all validators agreed on

    def to_dict(self) -> dict:
        return {
            "role":         self.role.value,
            "n_validators": self.n_validators,
            "total_stake":  round(self.total_stake, 4),
            "coverage":     round(self.coverage,    4),
            "weights":      {str(k): round(v, 6)
                             for k, v in self.consensus_weights.items()},
        }


@dataclass
class ValidatorDivergence:
    """How much a validator's proposal diverged from consensus."""
    validator_uid:  int
    role:           ValidatorRole
    divergence:     float     # mean squared divergence ∈ [0, 1]
    penalty:        float     # reward reduction factor ∈ [0, 1]; 0 = full penalty
    credibility:    float     # running credibility score ∈ [0, 1]


@dataclass
class ConsensusReport:
    """Complete consensus output for one epoch."""
    epoch:              int
    domain_consensus:   list[DomainConsensus]
    divergences:        list[ValidatorDivergence]
    final_rewards:      dict[int, float]   # global {miner_uid: weight}, sum=1.0
    n_validators:       int
    n_domains_covered:  int

    def as_weight_list(self) -> tuple[list[int], list[float]]:
        """Return (uids, weights) for subtensor.set_weights()."""
        uids    = sorted(self.final_rewards.keys())
        weights = [self.final_rewards[uid] for uid in uids]
        return uids, weights

    def to_dict(self) -> dict:
        return {
            "epoch":             self.epoch,
            "n_validators":      self.n_validators,
            "n_domains_covered": self.n_domains_covered,
            "reward_sum":        round(sum(self.final_rewards.values()), 6),
            "domain_consensus":  [d.to_dict() for d in self.domain_consensus],
            "divergences":       [
                {"uid": d.validator_uid, "role": d.role.value,
                 "divergence": round(d.divergence, 4),
                 "credibility": round(d.credibility, 4)}
                for d in self.divergences
            ],
            "final_rewards": {
                str(k): round(v, 6) for k, v in self.final_rewards.items()
            },
        }


# ── Consensus Engine ──────────────────────────────────────────────────────────

class ConsensusEngine:
    """
    Aggregates validator reward proposals into a consensus reward vector.

    Maintains running credibility scores for each validator so that
    consistently accurate validators carry more weight over time.
    """

    def __init__(
        self,
        min_validators_per_domain: int   = 1,
        disagreement_alpha:        float = 0.3,   # EMA decay for credibility
        penalty_threshold:         float = 0.10,  # divergence above this → penalty
    ):
        """
        Args:
            min_validators_per_domain: Minimum validators needed for consensus.
            disagreement_alpha:        EMA weight for credibility updates.
            penalty_threshold:         Divergence above which validator is penalised.
        """
        self.min_validators  = min_validators_per_domain
        self.alpha           = disagreement_alpha
        self.penalty_thr     = penalty_threshold

        # Running credibility per validator (uid → credibility [0,1])
        self._credibility: dict[int, float] = defaultdict(lambda: 1.0)

    # ── Public API ────────────────────────────────────────────────────────────

    def compute_consensus(
        self,
        epoch:       int,
        submissions: list[ValidatorSubmission],
        role_budgets: Optional[dict[str, float]] = None,
    ) -> ConsensusReport:
        """
        Compute consensus rewards from all validator submissions.

        Args:
            epoch:        Current epoch number.
            submissions:  List of ValidatorSubmission, one per validator.
            role_budgets: {role_value: budget_fraction} for global weighting.
                          Default: equal weights across active domains.

        Returns:
            ConsensusReport with final reward weights summing to 1.0.
        """
        if not submissions:
            return ConsensusReport(
                epoch=epoch, domain_consensus=[], divergences=[],
                final_rewards={}, n_validators=0, n_domains_covered=0,
            )

        # ── Step 1: Compute per-domain consensus ──────────────────────────────
        domain_proposals  = self._group_by_domain(submissions)
        domain_consensuses = []
        all_divergences   = []

        for role, proposals in domain_proposals.items():
            if len(proposals) < self.min_validators:
                continue

            dc = self._domain_consensus(role, proposals)
            domain_consensuses.append(dc)

            divs = self._compute_divergences(role, proposals, dc.consensus_weights)
            all_divergences.extend(divs)
            self._update_credibility(divs)

        # ── Step 2: Assemble global reward vector ─────────────────────────────
        final_rewards = self._assemble_final_rewards(
            domain_consensuses, role_budgets
        )

        return ConsensusReport(
            epoch=epoch,
            domain_consensus=domain_consensuses,
            divergences=all_divergences,
            final_rewards=final_rewards,
            n_validators=len(submissions),
            n_domains_covered=len(domain_consensuses),
        )

    def credibility(self, validator_uid: int) -> float:
        """Return current credibility score for a validator."""
        return float(self._credibility[validator_uid])

    # ── Domain consensus ──────────────────────────────────────────────────────

    def _domain_consensus(
        self,
        role:      ValidatorRole,
        proposals: list[RewardProposal],
    ) -> DomainConsensus:
        """
        Stake-weighted average of proposals for one domain.
        """
        # Collect all miner UIDs mentioned by any validator
        all_uids: set[int] = set()
        for p in proposals:
            all_uids.update(p.reward_weights.keys())

        if not all_uids:
            return DomainConsensus(role=role, consensus_weights={},
                                   n_validators=len(proposals),
                                   total_stake=0.0, coverage=0.0)

        total_stake  = sum(p.stake * self._credibility[p.validator_uid]
                           for p in proposals)
        consensus    = {}

        for uid in all_uids:
            weighted_sum = 0.0
            for p in proposals:
                w   = p.reward_weights.get(uid, 0.0)
                s   = p.stake * self._credibility[p.validator_uid]
                weighted_sum += w * s

            consensus[uid] = weighted_sum / total_stake if total_stake > 1e-8 else 0.0

        # Normalise
        total = sum(consensus.values())
        if total > 1e-8:
            consensus = {uid: w / total for uid, w in consensus.items()}

        # Coverage: fraction of miners all validators agreed on
        n_agreed = sum(
            1 for uid in all_uids
            if all(uid in p.reward_weights for p in proposals)
        )
        coverage = n_agreed / len(all_uids) if all_uids else 0.0

        return DomainConsensus(
            role=role,
            consensus_weights=consensus,
            n_validators=len(proposals),
            total_stake=total_stake,
            coverage=coverage,
        )

    # ── Divergence tracking ───────────────────────────────────────────────────

    def _compute_divergences(
        self,
        role:      ValidatorRole,
        proposals: list[RewardProposal],
        consensus: dict[int, float],
    ) -> list[ValidatorDivergence]:
        """Measure how much each validator diverged from consensus."""
        divs = []
        all_uids = sorted(consensus.keys())

        if not all_uids:
            return divs

        consensus_vec = np.array([consensus[uid] for uid in all_uids])

        for p in proposals:
            prop_vec  = np.array([p.reward_weights.get(uid, 0.0) for uid in all_uids])
            # Mean squared divergence
            divergence = float(np.mean((prop_vec - consensus_vec) ** 2))

            # Penalty: linear above threshold, 0 below
            penalty = float(np.clip(
                (divergence - self.penalty_thr) / (1.0 - self.penalty_thr),
                0.0, 1.0,
            ))

            divs.append(ValidatorDivergence(
                validator_uid=p.validator_uid,
                role=role,
                divergence=divergence,
                penalty=penalty,
                credibility=self._credibility[p.validator_uid],
            ))

        return divs

    def _update_credibility(self, divergences: list[ValidatorDivergence]) -> None:
        """Update running credibility using exponential moving average."""
        for d in divergences:
            uid       = d.validator_uid
            prev_cred = self._credibility[uid]
            # High divergence → credibility falls; low divergence → stays high
            target    = 1.0 - d.penalty
            new_cred  = (1 - self.alpha) * prev_cred + self.alpha * target
            self._credibility[uid] = float(np.clip(new_cred, 0.1, 1.0))

    # ── Final reward assembly ─────────────────────────────────────────────────

    def _assemble_final_rewards(
        self,
        domain_consensuses: list[DomainConsensus],
        role_budgets:       Optional[dict[str, float]],
    ) -> dict[int, float]:
        """
        Combine domain reward vectors into a single global reward vector.

        Each domain's consensus weights are scaled by the domain's budget
        fraction so the total sums to 1.0.
        """
        if not domain_consensuses:
            return {}

        # Default: equal budgets across active domains
        if role_budgets is None:
            n      = len(domain_consensuses)
            budgets = {dc.role.value: 1.0 / n for dc in domain_consensuses}
        else:
            # Normalise provided budgets to active domains only
            active  = {dc.role.value for dc in domain_consensuses}
            total   = sum(v for k, v in role_budgets.items() if k in active)
            budgets = {
                k: v / total for k, v in role_budgets.items()
                if k in active and total > 1e-8
            }

        final: dict[int, float] = defaultdict(float)
        for dc in domain_consensuses:
            budget = budgets.get(dc.role.value, 0.0)
            for uid, w in dc.consensus_weights.items():
                final[uid] += w * budget

        # Final normalisation
        total_final = sum(final.values())
        if total_final > 1e-8:
            final = {uid: w / total_final for uid, w in final.items()}

        return dict(final)

    # ── Grouping ──────────────────────────────────────────────────────────────

    @staticmethod
    def _group_by_domain(
        submissions: list[ValidatorSubmission],
    ) -> dict[ValidatorRole, list[RewardProposal]]:
        """Group all proposals by validator role."""
        by_role: dict[ValidatorRole, list[RewardProposal]] = defaultdict(list)
        for sub in submissions:
            for proposal in sub.proposals:
                valid, _ = proposal.is_valid()
                if valid:
                    by_role[proposal.role].append(proposal)
        return dict(by_role)
