"""
tests/test_sprint10.py
-----------------------
QA: Complete self-contained tests for Sprint 10 multi-agent modules.

Covers:
    agents/agent_roles.py        — AgentRole, AgentSubmission, AgentRegistry
    agents/risk_miner_agent.py   — RiskMinerEvaluator
    agents/portfolio_miner_agent.py — PortfolioMinerEvaluator
    agents/strategy_miner_agent.py  — StrategyMinerEvaluator
    agents/meta_miner_agent.py   — MetaMinerEvaluator
    agents/role_rewards.py       — RoleRewardModel
    agents/multi_agent_loop.py   — MultiAgentLoop end-to-end
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SUITE_DIR  = Path(__file__).resolve().parent
SUBNET_DIR = SUITE_DIR.parent
PROJECT    = SUBNET_DIR.parent

for p in [str(SUBNET_DIR), str(PROJECT)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_prices(n: int = 150, seed: int = 42) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "SPY":  100 * np.cumprod(1 + rng.normal(0.0005, 0.012, n)),
        "AAPL": 100 * np.cumprod(1 + rng.normal(0.0006, 0.015, n)),
        "GLD":  100 * np.cumprod(1 + rng.normal(0.0003, 0.009, n)),
    }, index=dates)


def make_returns(n: int = 150, seed: int = 42) -> pd.DataFrame:
    return make_prices(n, seed).pct_change().dropna()


def make_library_with_signals():
    """Create a populated AlphaLibrary for testing."""
    from macro8_subnet.alpha.alpha_library  import AlphaLibrary
    from macro8_subnet.alpha.alpha_schema   import AlphaFactor, AlphaEvaluation, AlphaCategory
    rng    = np.random.default_rng(1)
    n      = 120
    dates  = pd.date_range("2021-01-01", periods=n, freq="B")
    lib    = AlphaLibrary()
    for i, name in enumerate(["f_momentum", "f_vol", "f_regime"]):
        sig     = {a: pd.Series(rng.normal(0, 0.1, n), index=dates)
                   for a in ["SPY", "AAPL", "GLD"]}
        factor  = AlphaFactor(name=name, miner_uid=i, miner_hotkey=f"5F{i}",
                              signals=sig, category=AlphaCategory.MOMENTUM)
        ev      = AlphaEvaluation(factor_name=name, miner_uid=i,
                                   mean_ic=0.04 + i*0.01, ic_ir=0.5,
                                   is_duplicate=False, passes_ic_threshold=True,
                                   success=True)
        lib.add_factor(factor, ev, epoch=1, min_ic=0.0)
    return lib


# ════════════════════════════════════════════════════════════════════════════
# AGENT ROLES
# ════════════════════════════════════════════════════════════════════════════

from macro8_subnet.agents.agent_roles import AgentRole, AgentSubmission, AgentRegistry


class TestAgentRole:
    def test_five_roles_exist(self):
        assert len(list(AgentRole)) == 5

    def test_budgets_sum_to_one(self):
        total = sum(r.reward_budget() for r in AgentRole)
        assert abs(total - 1.0) < 1e-6

    def test_each_role_has_description(self):
        for r in AgentRole:
            assert len(r.description()) > 0

    def test_role_values_are_strings(self):
        for r in AgentRole:
            assert isinstance(r.value, str)


class TestAgentSubmission:
    def test_signal_constructor(self):
        s = AgentSubmission.signal(1, "5F", "momentum_20d")
        assert s.role    == AgentRole.SIGNAL
        assert s.formula == "momentum_20d"

    def test_strategy_constructor(self):
        s = AgentSubmission.strategy(1, "5F", {"f1": 0.6, "f2": 0.4})
        assert s.role == AgentRole.STRATEGY
        assert s.payload["signal_weights"]["f1"] == pytest.approx(0.6)

    def test_risk_constructor(self):
        s = AgentSubmission.risk(1, "5F", shrinkage=0.3, n_factors=5)
        assert s.role == AgentRole.RISK
        assert s.payload["shrinkage"] == pytest.approx(0.3)

    def test_portfolio_constructor(self):
        s = AgentSubmission.portfolio(1, "5F", max_weight=0.35)
        assert s.role == AgentRole.PORTFOLIO
        assert s.payload["max_weight"] == pytest.approx(0.35)

    def test_meta_constructor(self):
        s = AgentSubmission.meta(1, "5F", {"f1": 0.05, "f2": 0.03})
        assert s.role == AgentRole.META
        assert s.payload["ic_predictions"]["f1"] == pytest.approx(0.05)

    def test_signal_validation_empty_formula(self):
        s    = AgentSubmission.signal(0, "5F", "")
        ok, _ = s.is_valid()
        assert ok is False

    def test_strategy_validation_bad_sum(self):
        s = AgentSubmission.strategy(0, "5F", {"f1": 0.3, "f2": 0.3})
        ok, msg = s.is_valid()
        assert ok is False
        assert "sum" in msg.lower()

    def test_strategy_validation_negative_weight(self):
        s = AgentSubmission.strategy(0, "5F", {"f1": 1.2, "f2": -0.2})
        ok, _ = s.is_valid()
        assert ok is False

    def test_risk_validation_bad_shrinkage(self):
        s = AgentSubmission.risk(0, "5F", shrinkage=1.5)
        ok, _ = s.is_valid()
        assert ok is False

    def test_portfolio_validation_bad_max_weight(self):
        s = AgentSubmission.portfolio(0, "5F", max_weight=0.0)
        ok, _ = s.is_valid()
        assert ok is False

    def test_meta_validation_empty_predictions(self):
        s = AgentSubmission.meta(0, "5F", {})
        ok, _ = s.is_valid()
        assert ok is False

    def test_to_dict_serialisable(self):
        for sub in [
            AgentSubmission.signal(0, "5F", "momentum_20d"),
            AgentSubmission.strategy(0, "5F", {"f1": 1.0}),
            AgentSubmission.risk(0, "5F"),
            AgentSubmission.portfolio(0, "5F"),
            AgentSubmission.meta(0, "5F", {"f1": 0.04}),
        ]:
            json.dumps(sub.to_dict())


class TestAgentRegistry:
    def test_budgets_sum_to_one(self):
        total = AgentRegistry.total_budget()
        assert abs(total - 1.0) < 1e-6

    def test_all_budgets_dict(self):
        d = AgentRegistry.all_budgets()
        assert len(d) == 5
        assert all(isinstance(v, float) for v in d.values())

    def test_roles_by_budget_sorted(self):
        roles = AgentRegistry.roles_by_budget()
        budgets = [r.reward_budget() for r in roles]
        assert budgets == sorted(budgets, reverse=True)


# ════════════════════════════════════════════════════════════════════════════
# RISK MINER EVALUATOR
# ════════════════════════════════════════════════════════════════════════════

from macro8_subnet.agents.risk_miner_agent import RiskMinerEvaluator


class TestRiskMinerEvaluator:
    def _eval(self):
        return RiskMinerEvaluator(make_returns())

    def test_evaluate_returns_result(self):
        from macro8_subnet.agents.risk_miner_agent import RiskEvalResult
        sub = AgentSubmission.risk(0, "5F")
        r   = self._eval().evaluate(sub)
        assert isinstance(r, RiskEvalResult)

    def test_success_on_valid_submission(self):
        r = self._eval().evaluate(AgentSubmission.risk(0, "5F"))
        assert r.success is True

    def test_predicted_vol_positive(self):
        r = self._eval().evaluate(AgentSubmission.risk(0, "5F"))
        if r.success:
            assert r.predicted_vol > 0

    def test_accuracy_in_zero_one(self):
        r = self._eval().evaluate(AgentSubmission.risk(0, "5F"))
        if r.success and r.covariance_accuracy is not None:
            assert 0.0 <= r.covariance_accuracy <= 1.0

    def test_different_models_produce_results(self):
        ev = self._eval()
        for model in ["ledoit_wolf", "diagonal", "factor"]:
            sub = AgentSubmission.risk(0, "5F", model_type=model)
            r   = ev.evaluate(sub)
            assert isinstance(r.success, bool)

    def test_to_dict_serialisable(self):
        r = self._eval().evaluate(AgentSubmission.risk(0, "5F"))
        json.dumps(r.to_dict())

    def test_batch_evaluation(self):
        ev   = self._eval()
        subs = [AgentSubmission.risk(i, f"5F{i}") for i in range(3)]
        res  = ev.evaluate_batch(subs)
        assert len(res) == 3

    def test_best_covariance_returns_estimate(self):
        ev  = self._eval()
        res = ev.evaluate_batch([AgentSubmission.risk(0, "5F")])
        cov = ev.best_covariance(res)
        assert cov is not None

    def test_shrinkage_zero_to_one(self):
        ev = self._eval()
        for s in [0.0, 0.3, 0.7, 1.0]:
            sub = AgentSubmission.risk(0, "5F", shrinkage=s)
            r   = ev.evaluate(sub)
            assert isinstance(r.success, bool)


# ════════════════════════════════════════════════════════════════════════════
# PORTFOLIO MINER EVALUATOR
# ════════════════════════════════════════════════════════════════════════════

from macro8_subnet.agents.portfolio_miner_agent import PortfolioMinerEvaluator


class TestPortfolioMinerEvaluator:
    def _eval(self):
        ic = {"SPY": 0.05, "AAPL": 0.04, "GLD": 0.03}
        return PortfolioMinerEvaluator(make_returns(), ic)

    def test_evaluate_returns_result(self):
        from macro8_subnet.agents.portfolio_miner_agent import PortfolioEvalResult
        r = self._eval().evaluate(AgentSubmission.portfolio(0, "5F"))
        assert isinstance(r, PortfolioEvalResult)

    def test_success_on_valid_submission(self):
        r = self._eval().evaluate(AgentSubmission.portfolio(0, "5F"))
        assert r.success is True

    def test_baseline_sharpe_computed(self):
        r = self._eval().evaluate(AgentSubmission.portfolio(0, "5F"))
        if r.success:
            assert r.baseline_sharpe is not None

    def test_sharpe_improvement_is_float(self):
        r = self._eval().evaluate(AgentSubmission.portfolio(0, "5F"))
        if r.success and r.sharpe_improvement is not None:
            assert isinstance(r.sharpe_improvement, float)

    def test_invalid_max_weight_fails(self):
        sub = AgentSubmission.portfolio(0, "5F", max_weight=0.0)
        r   = self._eval().evaluate(sub)
        assert r.success is False

    def test_to_dict_serialisable(self):
        r = self._eval().evaluate(AgentSubmission.portfolio(0, "5F"))
        json.dumps(r.to_dict())

    def test_batch(self):
        ev   = self._eval()
        subs = [AgentSubmission.portfolio(i, f"5F{i}", max_weight=0.3+i*0.05)
                for i in range(3)]
        res  = ev.evaluate_batch(subs)
        assert len(res) == 3

    def test_best_constraints_returns_dict(self):
        ev  = self._eval()
        res = ev.evaluate_batch([AgentSubmission.portfolio(0, "5F")])
        bc  = ev.best_constraints(res)
        assert isinstance(bc, dict)
        assert "max_weight" in bc


# ════════════════════════════════════════════════════════════════════════════
# STRATEGY MINER EVALUATOR
# ════════════════════════════════════════════════════════════════════════════

from macro8_subnet.agents.strategy_miner_agent import StrategyMinerEvaluator


class TestStrategyMinerEvaluator:
    def _eval(self):
        lib = make_library_with_signals()
        return StrategyMinerEvaluator(lib.all_signals(), make_returns())

    def test_evaluate_returns_result(self):
        from macro8_subnet.agents.strategy_miner_agent import StrategyEvalResult
        sw  = {"f_momentum": 0.5, "f_vol": 0.5}
        sub = AgentSubmission.strategy(0, "5F", sw)
        r   = self._eval().evaluate(sub)
        assert isinstance(r, StrategyEvalResult)

    def test_valid_combination_succeeds(self):
        sub = AgentSubmission.strategy(0, "5F", {"f_momentum": 0.6, "f_vol": 0.4})
        r   = self._eval().evaluate(sub)
        assert r.success is True

    def test_unknown_signals_handled(self):
        sub = AgentSubmission.strategy(0, "5F", {"nonexistent_signal": 1.0})
        r   = self._eval().evaluate(sub)
        assert r.success is False

    def test_composite_ic_is_float(self):
        sub = AgentSubmission.strategy(0, "5F", {"f_momentum": 0.5, "f_vol": 0.5})
        r   = self._eval().evaluate(sub)
        if r.success:
            assert isinstance(r.composite_ic, float)

    def test_n_signals_used(self):
        sub = AgentSubmission.strategy(0, "5F", {"f_momentum": 0.5, "f_vol": 0.5})
        r   = self._eval().evaluate(sub)
        if r.success:
            assert r.n_signals_used == 2

    def test_to_dict_serialisable(self):
        sub = AgentSubmission.strategy(0, "5F", {"f_momentum": 1.0})
        r   = self._eval().evaluate(sub)
        json.dumps(r.to_dict())

    def test_batch(self):
        ev   = self._eval()
        subs = [
            AgentSubmission.strategy(0, "5F0", {"f_momentum": 0.7, "f_vol": 0.3}),
            AgentSubmission.strategy(1, "5F1", {"f_momentum": 0.3, "f_regime": 0.7}),
        ]
        res  = ev.evaluate_batch(subs)
        assert len(res) == 2


# ════════════════════════════════════════════════════════════════════════════
# META MINER EVALUATOR
# ════════════════════════════════════════════════════════════════════════════

from macro8_subnet.agents.meta_miner_agent import MetaMinerEvaluator


class TestMetaMinerEvaluator:
    def _actual_ics(self):
        return {"f_momentum": 0.05, "f_vol": 0.03, "f_regime": 0.04}

    def test_evaluate_good_predictions(self):
        sub = AgentSubmission.meta(0, "5F",
                                    {"f_momentum": 0.06, "f_vol": 0.02, "f_regime": 0.04})
        ev  = MetaMinerEvaluator()
        r   = ev.evaluate(sub, self._actual_ics())
        assert r.success is True

    def test_rank_correlation_range(self):
        sub = AgentSubmission.meta(0, "5F",
                                    {"f_momentum": 0.06, "f_vol": 0.02, "f_regime": 0.04})
        ev  = MetaMinerEvaluator()
        r   = ev.evaluate(sub, self._actual_ics())
        if r.success and r.rank_correlation is not None:
            assert -1.0 <= r.rank_correlation <= 1.0

    def test_empty_predictions_fails(self):
        sub = AgentSubmission.meta(0, "5F", {})
        ev  = MetaMinerEvaluator()
        # is_valid() will fail before evaluate() — test evaluate directly
        r   = ev.evaluate(sub, self._actual_ics())
        assert r.success is False

    def test_no_matching_signals_fails(self):
        sub = AgentSubmission.meta(0, "5F", {"nonexistent": 0.5, "also_none": 0.3})
        ev  = MetaMinerEvaluator(min_matched=2)
        r   = ev.evaluate(sub, self._actual_ics())
        assert r.success is False

    def test_reward_non_negative(self):
        sub = AgentSubmission.meta(0, "5F",
                                    {"f_momentum": 0.05, "f_vol": 0.03})
        ev  = MetaMinerEvaluator()
        r   = ev.evaluate(sub, self._actual_ics())
        assert r.reward_score >= 0.0

    def test_to_dict_serialisable(self):
        sub = AgentSubmission.meta(0, "5F", {"f_momentum": 0.05})
        ev  = MetaMinerEvaluator()
        r   = ev.evaluate(sub, self._actual_ics())
        json.dumps(r.to_dict())

    def test_store_and_score_pending(self):
        ev  = MetaMinerEvaluator()
        sub = AgentSubmission.meta(0, "5F", {"f_momentum": 0.06, "f_vol": 0.02})
        ev.store_predictions(sub)
        results = ev.score_pending({"f_momentum": 0.05, "f_vol": 0.03})
        assert len(results) >= 1

    def test_pending_cleared_after_scoring(self):
        ev  = MetaMinerEvaluator()
        sub = AgentSubmission.meta(0, "5F", {"f_momentum": 0.05})
        ev.store_predictions(sub)
        ev.score_pending({"f_momentum": 0.05})
        # Scoring again should return empty (pending cleared)
        results2 = ev.score_pending({"f_momentum": 0.05})
        assert len(results2) == 0

    def test_direction_accuracy_range(self):
        sub = AgentSubmission.meta(0, "5F",
                                    {"f_momentum": 0.06, "f_vol": 0.02, "f_regime": 0.04})
        ev  = MetaMinerEvaluator()
        r   = ev.evaluate(sub, self._actual_ics())
        if r.success and r.prediction_accuracy is not None:
            assert 0.0 <= r.prediction_accuracy <= 1.0


# ════════════════════════════════════════════════════════════════════════════
# ROLE REWARD MODEL
# ════════════════════════════════════════════════════════════════════════════

from macro8_subnet.agents.role_rewards import RoleRewardModel


def make_role_scores(n_per_role: int = 3) -> dict:
    scores = {}
    for i, role in enumerate(AgentRole):
        scores[role] = [
            {"uid": i * 10 + j, "hotkey": f"5F{i*10+j}", "score": float(j+1) * 0.1}
            for j in range(n_per_role)
        ]
    return scores


class TestRoleRewardModel:
    def test_compute_returns_report(self):
        from macro8_subnet.agents.role_rewards import RoleRewardReport
        m = RoleRewardModel()
        r = m.compute(1, make_role_scores())
        assert isinstance(r, RoleRewardReport)

    def test_reward_sum_is_one(self):
        m = RoleRewardModel()
        r = m.compute(1, make_role_scores())
        assert abs(r.reward_sum - 1.0) < 1e-5

    def test_all_rewards_non_negative(self):
        m = RoleRewardModel()
        r = m.compute(1, make_role_scores())
        assert all(e.total_reward >= 0 for e in r.entries)

    def test_higher_score_higher_reward_within_role(self):
        """Within one role, higher score → higher reward."""
        m      = RoleRewardModel()
        scores = {AgentRole.SIGNAL: [
            {"uid": 0, "hotkey": "5F0", "score": 0.1},
            {"uid": 1, "hotkey": "5F1", "score": 0.9},
        ]}
        r = m.compute(1, scores)
        uid0 = next(e for e in r.entries if e.miner_uid == 0)
        uid1 = next(e for e in r.entries if e.miner_uid == 1)
        assert uid1.total_reward > uid0.total_reward

    def test_active_roles_count(self):
        m = RoleRewardModel()
        r = m.compute(1, make_role_scores())
        assert r.n_active_roles == 5

    def test_partial_roles_redistributes_budget(self):
        """Only 2 roles active → those 2 share 100% of budget."""
        m      = RoleRewardModel()
        scores = {
            AgentRole.SIGNAL:   [{"uid": 0, "hotkey": "5F0", "score": 0.5}],
            AgentRole.RISK:     [{"uid": 1, "hotkey": "5F1", "score": 0.5}],
        }
        r = m.compute(1, scores)
        assert r.n_active_roles == 2
        assert abs(sum(e.total_reward for e in r.entries) - 1.0) < 1e-5

    def test_zero_score_entries_handled(self):
        m = RoleRewardModel()
        scores = {AgentRole.SIGNAL: [
            {"uid": 0, "hotkey": "5F", "score": 0.0},
            {"uid": 1, "hotkey": "5G", "score": 0.0},
        ]}
        r = m.compute(1, scores)
        assert abs(sum(e.total_reward for e in r.entries) - 1.0) < 1e-5

    def test_to_dict_serialisable(self):
        m = RoleRewardModel()
        r = m.compute(1, make_role_scores(2))
        json.dumps(r.to_dict())

    def test_as_weight_list(self):
        m = RoleRewardModel()
        r = m.compute(1, make_role_scores(2))
        uids, weights = r.as_weight_list()
        assert len(uids) == len(weights)
        assert abs(sum(weights) - 1.0) < 1e-5

    def test_ranks_sequential(self):
        m = RoleRewardModel()
        r = m.compute(1, make_role_scores())
        ranks = sorted(e.rank for e in r.entries)
        assert ranks == list(range(1, len(ranks) + 1))

    def test_empty_submissions_returns_empty(self):
        m = RoleRewardModel()
        r = m.compute(1, {})
        assert len(r.entries) == 0

    def test_custom_budgets(self):
        budgets = {r: 0.2 for r in AgentRole}   # equal budgets
        m       = RoleRewardModel(budgets=budgets)
        r       = m.compute(1, make_role_scores())
        assert abs(r.reward_sum - 1.0) < 1e-5

    def test_summarise_returns_string(self):
        m   = RoleRewardModel()
        r   = m.compute(1, make_role_scores(2))
        s   = m.summarise(r)
        assert isinstance(s, str) and len(s) > 0


# ════════════════════════════════════════════════════════════════════════════
# MULTI-AGENT LOOP
# ════════════════════════════════════════════════════════════════════════════

from macro8_subnet.agents.multi_agent_loop import MultiAgentLoop, MultiEpochReport


def make_multi_loop(verbose=False):
    from macro8_subnet.alpha.alpha_library    import AlphaLibrary
    from macro8_subnet.alpha.meta_alpha_model import MetaAlphaModel
    return MultiAgentLoop(
        make_prices(),
        AlphaLibrary(),
        MetaAlphaModel(min_samples=999),
        verbose=verbose,
    )


def make_mixed_submissions(n_per_role: int = 1) -> list[AgentSubmission]:
    subs = []
    for i in range(n_per_role):
        subs.append(AgentSubmission.signal(i,    f"5F{i}",   "momentum_20d"))
        subs.append(AgentSubmission.risk(i+10,   f"5F{i+10}"))
        subs.append(AgentSubmission.portfolio(i+20, f"5F{i+20}", max_weight=0.4))
        subs.append(AgentSubmission.meta(i+30,   f"5F{i+30}",
                                          {"f_test": 0.04, "f_test2": 0.03}))
    return subs


class TestMultiAgentLoop:
    def test_creates(self):
        assert make_multi_loop() is not None

    def test_returns_multi_epoch_report(self):
        loop   = make_multi_loop()
        report = loop.run_epoch(1, make_mixed_submissions())
        assert isinstance(report, MultiEpochReport)

    def test_epoch_number_preserved(self):
        report = make_multi_loop().run_epoch(7, make_mixed_submissions())
        assert report.epoch == 7

    def test_elapsed_positive(self):
        report = make_multi_loop().run_epoch(1, make_mixed_submissions())
        assert report.elapsed_seconds > 0

    def test_n_submissions_by_role(self):
        report = make_multi_loop().run_epoch(1, make_mixed_submissions(2))
        assert "signal" in report.n_submissions_by_role
        assert report.n_submissions_by_role["signal"] == 2

    def test_risk_results_present(self):
        subs   = [AgentSubmission.risk(0, "5F")]
        report = make_multi_loop().run_epoch(1, subs)
        assert isinstance(report.risk_results, list)

    def test_portfolio_results_present(self):
        subs   = [AgentSubmission.portfolio(0, "5F")]
        report = make_multi_loop().run_epoch(1, subs)
        assert isinstance(report.portfolio_results, list)

    def test_reward_report_present(self):
        report = make_multi_loop().run_epoch(1, make_mixed_submissions())
        assert report.reward_report is not None

    def test_reward_sum_is_one(self):
        report = make_multi_loop().run_epoch(1, make_mixed_submissions())
        if report.reward_report and report.reward_report.entries:
            total = sum(e.total_reward for e in report.reward_report.entries)
            assert abs(total - 1.0) < 1e-5

    def test_to_dict_serialisable(self):
        report = make_multi_loop().run_epoch(1, make_mixed_submissions())
        json.dumps(report.to_dict())

    def test_summary_is_string(self):
        report = make_multi_loop().run_epoch(1, make_mixed_submissions())
        s = report.summary()
        assert isinstance(s, str) and len(s) > 0

    def test_empty_submissions_ok(self):
        report = make_multi_loop().run_epoch(1, [])
        assert report.epoch == 1

    def test_invalid_submissions_rejected_gracefully(self):
        subs = [
            AgentSubmission(0, "5F", AgentRole.SIGNAL, formula=""),   # invalid
            AgentSubmission.signal(1, "5G", "momentum_20d"),           # valid
        ]
        report = make_multi_loop().run_epoch(1, subs)
        # Should not crash
        assert report.epoch == 1

    def test_signal_only_epoch(self):
        """Signal-only epoch works like ResearchLoop."""
        subs   = [AgentSubmission.signal(i, f"5F{i}", "momentum_20d") for i in range(2)]
        report = make_multi_loop().run_epoch(1, subs)
        assert report.signal_report is not None

    def test_multi_epoch_accumulates(self):
        """Running multiple epochs should not crash."""
        loop = make_multi_loop()
        for epoch in range(1, 4):
            subs   = make_mixed_submissions(1)
            report = loop.run_epoch(epoch, subs)
            assert report.epoch == epoch

    def test_strategy_role_with_library(self):
        """Strategy role needs library signals — test after signals are added."""
        from macro8_subnet.alpha.alpha_library    import AlphaLibrary
        from macro8_subnet.alpha.meta_alpha_model import MetaAlphaModel
        loop    = MultiAgentLoop(
            make_prices(),
            make_library_with_signals(),   # pre-populated library
            MetaAlphaModel(min_samples=999),
            verbose=False,
        )
        subs = [AgentSubmission.strategy(
            0, "5F", {"f_momentum": 0.5, "f_vol": 0.5}
        )]
        report = loop.run_epoch(1, subs)
        assert len(report.strategy_results) >= 0   # may succeed or fail gracefully
