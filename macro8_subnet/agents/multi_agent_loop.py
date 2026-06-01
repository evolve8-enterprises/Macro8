"""
agents/multi_agent_loop.py
---------------------------
Multi-agent research loop orchestrator.

The MultiAgentLoop extends the existing ResearchLoop by accepting
mixed AgentSubmission lists containing all five agent roles and routing
each submission to the correct evaluator.

Relationship to ResearchLoop
------------------------------
    ResearchLoop:      handles SIGNAL submissions only
    MultiAgentLoop:    handles all 5 roles, uses ResearchLoop
                       internally for SIGNAL processing

The loop is fully backward-compatible: an epoch with only SIGNAL
submissions produces the same result as ResearchLoop.run_epoch().

Epoch flow
----------
    1. Split submissions by role
    2. Route SIGNAL  → ResearchLoop (IC + library management)
    3. Route STRATEGY → StrategyMinerEvaluator
    4. Route RISK    → RiskMinerEvaluator
    5. Route PORTFOLIO → PortfolioMinerEvaluator
    6. Route META    → MetaMinerEvaluator (store for next-epoch scoring)
       Score pending META predictions from previous epoch
    7. Integrate best outputs: best covariance + best constraints → portfolio
    8. Compute role-stratified rewards
    9. Return MultiEpochReport
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_SUBNET = Path(__file__).resolve().parent.parent
_ROOT   = _SUBNET.parent
for p in [str(_SUBNET), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from macro8_subnet.agents.agent_roles          import AgentRole, AgentSubmission, AgentRegistry
from macro8_subnet.agents.risk_miner_agent     import RiskMinerEvaluator, RiskEvalResult
from macro8_subnet.agents.portfolio_miner_agent import PortfolioMinerEvaluator, PortfolioEvalResult
from macro8_subnet.agents.strategy_miner_agent import StrategyMinerEvaluator, StrategyEvalResult
from macro8_subnet.agents.meta_miner_agent     import MetaMinerEvaluator, MetaEvalResult
from macro8_subnet.agents.role_rewards         import RoleRewardModel, RoleRewardReport
from macro8_subnet.alpha.research_loop         import (
    ResearchLoop, FormulaSubmission, EpochReport
)
from macro8_subnet.alpha.alpha_library         import AlphaLibrary
from macro8_subnet.alpha.meta_alpha_model      import MetaAlphaModel


# ── Multi-agent epoch report ──────────────────────────────────────────────────

@dataclass
class MultiEpochReport:
    """Complete results from one multi-agent research loop epoch."""
    epoch:               int
    elapsed_seconds:     float = 0.0

    # Signal role results (from ResearchLoop)
    signal_report:       Optional[EpochReport]           = None

    # Other role results
    strategy_results:    list[StrategyEvalResult]        = field(default_factory=list)
    risk_results:        list[RiskEvalResult]            = field(default_factory=list)
    portfolio_results:   list[PortfolioEvalResult]       = field(default_factory=list)
    meta_results:        list[MetaEvalResult]            = field(default_factory=list)

    # Integrated outputs
    best_covariance:     Optional[dict]                  = None   # from RiskMiners
    best_constraints:    Optional[dict]                  = None   # from PortfolioMiners
    best_meta_preds:     Optional[dict[str, float]]      = None   # from MetaMiners

    # Reward report
    reward_report:       Optional[RoleRewardReport]      = None

    # Aggregate stats
    n_submissions_by_role: dict[str, int]                = field(default_factory=dict)
    errors:                list[str]                     = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"═══ Multi-Agent Epoch {self.epoch} ═══"]
        for role, n in self.n_submissions_by_role.items():
            lines.append(f"  {role:<12}: {n} submissions")
        if self.signal_report:
            lines.append(f"  Library      : {self.signal_report.library_size} active signals")
        if self.reward_report:
            lines.append(f"  Reward roles : {self.reward_report.n_active_roles}/5 active")
            top1 = self.reward_report.top_n(1)
            if top1:
                lines.append(f"  Top miner    : uid={top1[0].miner_uid} "
                             f"reward={top1[0].total_reward:.4f}")
        lines.append(f"  Elapsed      : {self.elapsed_seconds:.2f}s")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "epoch":           self.epoch,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "n_submissions":   self.n_submissions_by_role,
            "signal": self.signal_report.to_dict() if self.signal_report else None,
            "strategy_results": [r.to_dict() for r in self.strategy_results],
            "risk_results":     [r.to_dict() for r in self.risk_results],
            "portfolio_results": [r.to_dict() for r in self.portfolio_results],
            "meta_results":     [r.to_dict() for r in self.meta_results],
            "best_constraints": self.best_constraints,
            "reward_report":    self.reward_report.to_dict() if self.reward_report else None,
        }


# ── Multi-Agent Loop ──────────────────────────────────────────────────────────

class MultiAgentLoop:
    """
    Multi-agent research loop: routes submissions by role and integrates outputs.

    Wraps ResearchLoop for SIGNAL processing. Adds four new evaluators
    for STRATEGY, RISK, PORTFOLIO, and META roles.
    """

    def __init__(
        self,
        prices:    pd.DataFrame,
        library:   AlphaLibrary,
        meta_model: MetaAlphaModel,
        risk_free:  float = 0.0,
        verbose:    bool  = True,
    ):
        self.prices    = prices
        self.returns   = prices.pct_change().dropna()
        self.library   = library
        self.risk_free = risk_free
        self.verbose   = verbose

        # Core signal loop (handles SIGNAL role)
        self._signal_loop  = ResearchLoop(
            prices, library, meta_model,
            run_stress=False, verbose=False,
        )

        # Role evaluators (lazily initialised as data becomes available)
        self._risk_eval:      Optional[RiskMinerEvaluator]      = None
        self._portfolio_eval: Optional[PortfolioMinerEvaluator] = None
        self._strategy_eval:  Optional[StrategyMinerEvaluator]  = None
        self._meta_eval:      MetaMinerEvaluator                = MetaMinerEvaluator()
        self._reward_model:   RoleRewardModel                   = RoleRewardModel()

    # ── Public API ────────────────────────────────────────────────────────────

    def run_epoch(
        self,
        epoch:       int,
        submissions: list[AgentSubmission],
    ) -> MultiEpochReport:
        """
        Execute one multi-agent epoch.

        Args:
            epoch:       Monotonically increasing epoch number.
            submissions: Mixed list of AgentSubmission from all roles.

        Returns:
            MultiEpochReport with all role results and reward allocation.
        """
        t_start = time.perf_counter()
        report  = MultiEpochReport(epoch=epoch)

        # ── Route submissions by role ─────────────────────────────────────────
        by_role = self._split_by_role(submissions)
        report.n_submissions_by_role = {r.value: len(v) for r, v in by_role.items()}

        self._log(f"\n{'═'*60}")
        self._log(f"  🤖  MULTI-AGENT EPOCH {epoch}")
        self._log(f"{'═'*60}")
        for role, subs in by_role.items():
            if subs:
                self._log(f"  {role.value:<12}: {len(subs)} submission(s)")

        # ── 1. SIGNAL role: ResearchLoop ──────────────────────────────────────
        if by_role[AgentRole.SIGNAL]:
            self._log("\n  [SIGNAL] Running research loop...")
            formula_subs = [
                FormulaSubmission(
                    miner_uid=s.miner_uid,
                    miner_hotkey=s.miner_hotkey,
                    formula=s.formula,
                    category=s.category,
                )
                for s in by_role[AgentRole.SIGNAL]
            ]
            report.signal_report = self._signal_loop.run_epoch(epoch, formula_subs)
            self._log(f"    Library={report.signal_report.library_size} | "
                      f"Admitted={report.signal_report.n_signals_admitted}")

        # Refresh evaluators now that library may have changed
        self._refresh_evaluators()

        # ── 2. RISK role ──────────────────────────────────────────────────────
        if by_role[AgentRole.RISK] and self._risk_eval:
            self._log("\n  [RISK] Evaluating covariance models...")
            report.risk_results = self._risk_eval.evaluate_batch(by_role[AgentRole.RISK])
            n_ok = sum(1 for r in report.risk_results if r.success)
            self._log(f"    {n_ok}/{len(report.risk_results)} models valid")
            if report.risk_results:
                best = self._risk_eval.best_covariance(report.risk_results)
                report.best_covariance = best.to_dict() if best else None

        # ── 3. STRATEGY role ──────────────────────────────────────────────────
        if by_role[AgentRole.STRATEGY] and self._strategy_eval:
            self._log("\n  [STRATEGY] Evaluating signal combinations...")
            report.strategy_results = self._strategy_eval.evaluate_batch(
                by_role[AgentRole.STRATEGY]
            )
            n_ok = sum(1 for r in report.strategy_results if r.success)
            self._log(f"    {n_ok}/{len(report.strategy_results)} strategies valid")

        # ── 4. PORTFOLIO role ─────────────────────────────────────────────────
        if by_role[AgentRole.PORTFOLIO] and self._portfolio_eval:
            self._log("\n  [PORTFOLIO] Evaluating constraint sets...")
            report.portfolio_results = self._portfolio_eval.evaluate_batch(
                by_role[AgentRole.PORTFOLIO]
            )
            n_ok = sum(1 for r in report.portfolio_results if r.success)
            self._log(f"    {n_ok}/{len(report.portfolio_results)} constraint sets valid")
            if report.portfolio_results:
                report.best_constraints = self._portfolio_eval.best_constraints(
                    report.portfolio_results
                )

        # ── 5. META role: score pending + store new ───────────────────────────
        if by_role[AgentRole.META]:
            self._log("\n  [META] Processing IC predictions...")

            # Score predictions from previous epoch
            if self.library.n_active > 0:
                actual_ics = {
                    f.name: rec.current_ic
                    for f in self.library.all_active_factors()
                    if (rec := self.library.get_record(f.name)) is not None
                }
                pending_results = self._meta_eval.score_pending(actual_ics)
                if pending_results:
                    report.meta_results.extend(pending_results)
                    self._log(f"    Scored {len(pending_results)} pending predictions")

            # Store new predictions for next epoch scoring
            for sub in by_role[AgentRole.META]:
                valid, reason = sub.is_valid()
                if valid:
                    self._meta_eval.store_predictions(sub)

            # If we have actual ICs, also do immediate scoring for rewards
            if self.library.n_active > 0:
                actual_ics = {
                    f.name: rec.current_ic
                    for f in self.library.all_active_factors()
                    if (rec := self.library.get_record(f.name)) is not None
                }
                immediate = self._meta_eval.evaluate_batch(
                    by_role[AgentRole.META], actual_ics
                )
                report.meta_results.extend(immediate)
                n_ok = sum(1 for r in report.meta_results if r.success)
                self._log(f"    {n_ok} META results")

                # Extract best predictions to guide evolution
                report.best_meta_preds = self._meta_eval.best_predictions(
                    immediate, by_role[AgentRole.META]
                )

        # ── 6. Role-stratified rewards ────────────────────────────────────────
        self._log("\n  Computing role-stratified rewards...")
        role_scores = self._build_role_scores(report)
        report.reward_report = self._reward_model.compute(epoch, role_scores)
        self._log(f"    {report.reward_report.n_active_roles} active roles | "
                  f"{len(report.reward_report.entries)} miners rewarded")

        report.elapsed_seconds = time.perf_counter() - t_start
        self._log(f"\n{report.summary()}")
        return report

    # ── Internal ──────────────────────────────────────────────────────────────

    def _split_by_role(
        self, submissions: list[AgentSubmission]
    ) -> dict[AgentRole, list[AgentSubmission]]:
        """Group submissions by role, validating each."""
        by_role: dict[AgentRole, list[AgentSubmission]] = {r: [] for r in AgentRole}
        for sub in submissions:
            ok, reason = sub.is_valid()
            if ok:
                by_role[sub.role].append(sub)
            else:
                self._log(f"  ⚠️  uid={sub.miner_uid} {sub.role.value}: {reason}")
        return by_role

    def _refresh_evaluators(self) -> None:
        """Rebuild role evaluators with current library state."""
        # RiskMinerEvaluator — needs only returns data
        if self._risk_eval is None:
            self._risk_eval = RiskMinerEvaluator(self.returns)

        # StrategyMinerEvaluator — needs library signals
        lib_sigs = self.library.all_signals()
        if lib_sigs:
            self._strategy_eval = StrategyMinerEvaluator(
                lib_sigs, self.returns, self.risk_free
            )

        # PortfolioMinerEvaluator — needs IC scores
        ic_scores = {
            f.name: rec.mean_ic
            for f in self.library.all_active_factors()
            if (rec := self.library.get_record(f.name)) is not None
        }
        self._portfolio_eval = PortfolioMinerEvaluator(
            self.returns, ic_scores, self.risk_free
        )

    def _build_role_scores(
        self, report: MultiEpochReport
    ) -> dict[AgentRole, list[dict]]:
        """Compile per-role score lists for the reward model."""
        scores: dict[AgentRole, list[dict]] = {r: [] for r in AgentRole}

        # SIGNAL: use ic_score from miner rewards in signal_report
        if report.signal_report:
            for rw in report.signal_report.miner_rewards:
                scores[AgentRole.SIGNAL].append({
                    "uid":    rw.miner_uid,
                    "hotkey": rw.miner_hotkey,
                    "score":  rw.ic_score,
                })

        # STRATEGY
        for r in report.strategy_results:
            scores[AgentRole.STRATEGY].append({
                "uid": r.miner_uid, "hotkey": f"uid_{r.miner_uid}",
                "score": r.reward_score,
            })

        # RISK
        for r in report.risk_results:
            scores[AgentRole.RISK].append({
                "uid": r.miner_uid, "hotkey": f"uid_{r.miner_uid}",
                "score": r.reward_score,
            })

        # PORTFOLIO
        for r in report.portfolio_results:
            scores[AgentRole.PORTFOLIO].append({
                "uid": r.miner_uid, "hotkey": f"uid_{r.miner_uid}",
                "score": r.reward_score,
            })

        # META
        for r in report.meta_results:
            scores[AgentRole.META].append({
                "uid": r.miner_uid, "hotkey": f"uid_{r.miner_uid}",
                "score": r.reward_score,
            })

        return scores

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)
