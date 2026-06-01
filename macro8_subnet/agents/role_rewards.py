"""
agents/role_rewards.py
-----------------------
Role-stratified reward emission for the multi-agent network.

Each role has a budget fraction of total epoch TAO emission:

    SIGNAL    30% — core signal discovery
    STRATEGY  25% — signal composition
    RISK      20% — covariance modelling
    PORTFOLIO 15% — constraint optimisation
    META      10% — IC prediction

Within each role, rewards are distributed proportional to performance
score using softmax normalisation (consistent with the existing
RewardModel in reward/reward_model.py).

The final output is a flat list of (miner_uid, reward_weight) pairs
where all weights sum to 1.0 — exactly what subtensor.set_weights()
expects.

Design choices
--------------
  Role budgets are configurable but default to the values above.
  If a role has zero valid submissions, its budget redistributes
  proportionally to other active roles.
  Miners can participate in multiple roles in the same epoch — their
  rewards from each role are summed.
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

from macro8_subnet.agents.agent_roles import AgentRole, AgentRegistry


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class RoleRewardEntry:
    """Reward breakdown for one miner across all roles."""
    miner_uid:       int
    miner_hotkey:    str
    role_scores:     dict[str, float]    # role_value → raw score this epoch
    role_rewards:    dict[str, float]    # role_value → reward from this role
    total_reward:    float               # sum across all roles
    rank:            int                 # 1 = highest reward

    def to_dict(self) -> dict:
        return {
            "miner_uid":    self.miner_uid,
            "role_scores":  {k: round(v, 6) for k, v in self.role_scores.items()},
            "role_rewards": {k: round(v, 6) for k, v in self.role_rewards.items()},
            "total_reward": round(self.total_reward, 6),
            "rank":         self.rank,
        }


@dataclass
class RoleRewardReport:
    """Complete reward report for one epoch across all roles."""
    epoch:           int
    entries:         list[RoleRewardEntry]
    role_budgets:    dict[str, float]    # actual budgets used (after redistribution)
    n_active_roles:  int
    reward_sum:      float               # should be 1.0

    def as_weight_list(self) -> tuple[list[int], list[float]]:
        """Return (uids, weights) lists for subtensor.set_weights()."""
        uids    = [e.miner_uid    for e in self.entries]
        weights = [e.total_reward for e in self.entries]
        return uids, weights

    def top_n(self, n: int = 5) -> list[RoleRewardEntry]:
        return sorted(self.entries, key=lambda e: e.total_reward, reverse=True)[:n]

    def to_dict(self) -> dict:
        return {
            "epoch":          self.epoch,
            "role_budgets":   {k: round(v, 4) for k, v in self.role_budgets.items()},
            "n_active_roles": self.n_active_roles,
            "reward_sum":     round(self.reward_sum, 6),
            "entries":        [e.to_dict() for e in self.entries],
        }


# ── Role Reward Model ─────────────────────────────────────────────────────────

class RoleRewardModel:
    """
    Computes role-stratified reward weights for a multi-agent epoch.

    Usage
    -----
        model = RoleRewardModel()

        # After evaluating all role submissions:
        role_scores = {
            AgentRole.SIGNAL:    [{"uid": 0, "score": 0.05}, ...],
            AgentRole.RISK:      [{"uid": 1, "score": 0.82}, ...],
            AgentRole.PORTFOLIO: [{"uid": 2, "score": 1.12}, ...],
            ...
        }
        report = model.compute(epoch=5, role_scores=role_scores)
        uids, weights = report.as_weight_list()
    """

    def __init__(
        self,
        budgets:     Optional[dict[AgentRole, float]] = None,
        temperature: float = 1.0,
    ):
        """
        Args:
            budgets:     Custom role budget fractions (must sum to 1.0).
                         None = use AgentRole defaults.
            temperature: Softmax temperature. Higher → winner-takes-more.
        """
        if budgets:
            total = sum(budgets.values())
            self.budgets = {r: v / total for r, v in budgets.items()}
        else:
            self.budgets = {r: r.reward_budget() for r in AgentRole}
        self.temperature = temperature

    def compute(
        self,
        epoch:       int,
        role_scores: dict[AgentRole, list[dict]],
    ) -> RoleRewardReport:
        """
        Compute role-stratified rewards.

        Args:
            epoch:       Current epoch number.
            role_scores: {AgentRole: [{"uid": int, "hotkey": str, "score": float}]}
                         One list per role, one entry per miner in that role.

        Returns:
            RoleRewardReport with per-miner reward weights summing to 1.0.
        """
        # ── Step 1: Determine active roles and redistribute budgets ───────────
        active_roles = {r for r, entries in role_scores.items()
                        if entries}

        if not active_roles:
            return RoleRewardReport(
                epoch=epoch, entries=[], role_budgets={},
                n_active_roles=0, reward_sum=0.0
            )

        # Redistribute inactive role budgets proportionally
        total_active = sum(self.budgets[r] for r in active_roles)
        effective_budgets = {
            r: self.budgets[r] / total_active
            for r in active_roles
        }

        # ── Step 2: Softmax normalise within each role ────────────────────────
        # Collect all miner UIDs
        all_uids: dict[int, str] = {}
        for entries in role_scores.values():
            for e in entries:
                all_uids[e["uid"]] = e.get("hotkey", f"uid_{e['uid']}")

        # Per-role reward allocation
        role_allocs: dict[int, dict[str, float]] = {uid: {} for uid in all_uids}
        role_raw:    dict[int, dict[str, float]] = {uid: {} for uid in all_uids}

        for role in active_roles:
            entries = role_scores.get(role, [])
            if not entries:
                continue

            budget    = effective_budgets[role]
            uids      = [e["uid"]   for e in entries]
            scores    = np.array([max(float(e["score"]), 0.0) for e in entries])

            # Softmax normalisation within role
            if scores.sum() < 1e-8:
                norm_weights = np.ones(len(scores)) / len(scores)
            else:
                scaled   = scores * self.temperature
                shifted  = scaled - scaled.max()
                exp_s    = np.exp(np.clip(shifted, -500, 500))
                norm_weights = exp_s / exp_s.sum()

            for uid, raw_s, norm_w in zip(uids, scores.tolist(), norm_weights.tolist()):
                role_allocs[uid][role.value] = float(norm_w * budget)
                role_raw[uid][role.value]    = float(raw_s)

        # ── Step 3: Sum across roles ──────────────────────────────────────────
        totals = {uid: sum(allocs.values()) for uid, allocs in role_allocs.items()}

        # Final normalisation (should already sum to 1.0 but guard against rounding)
        grand_total = sum(totals.values())
        if grand_total > 1e-8:
            totals = {uid: t / grand_total for uid, t in totals.items()}

        # ── Step 4: Build entries sorted by total reward ──────────────────────
        entries_raw = sorted(totals.items(), key=lambda x: x[1], reverse=True)
        entries = []
        for rank, (uid, total_r) in enumerate(entries_raw, start=1):
            entries.append(RoleRewardEntry(
                miner_uid=uid,
                miner_hotkey=all_uids[uid],
                role_scores=role_raw.get(uid, {}),
                role_rewards=role_allocs.get(uid, {}),
                total_reward=total_r,
                rank=rank,
            ))

        return RoleRewardReport(
            epoch=epoch,
            entries=entries,
            role_budgets={r.value: effective_budgets.get(r, 0.0) for r in AgentRole},
            n_active_roles=len(active_roles),
            reward_sum=float(sum(totals.values())),
        )

    def summarise(self, report: RoleRewardReport) -> str:
        """One-line summary of a reward report."""
        top = report.top_n(3)
        top_str = " | ".join(f"uid={e.miner_uid} {e.total_reward:.4f}" for e in top)
        return (f"Epoch {report.epoch}: {len(report.entries)} miners | "
                f"{report.n_active_roles} roles | "
                f"top3: [{top_str}]")
