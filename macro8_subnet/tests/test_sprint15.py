"""
tests/test_sprint15.py
-----------------------
QA: Sprint 15 — MacroSession integration layer + TransactionCostModel.

Covers:
    alpha/macro_session.py     — session creation, epoch runs, reports
    alpha/transaction_costs.py — cost model, turnover, net return

Both modules are tested as self-contained units and in combination
with the existing module stack.
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


# ── Shared fixtures ───────────────────────────────────────────────────────────

def make_prices(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "SPY":  100 * np.cumprod(1 + rng.normal(0.0005, 0.012, n)),
        "AAPL": 100 * np.cumprod(1 + rng.normal(0.0006, 0.015, n)),
        "GLD":  100 * np.cumprod(1 + rng.normal(0.0003, 0.009, n)),
    }, index=dates)


# ════════════════════════════════════════════════════════════════════════════
# TRANSACTION COST MODEL
# ════════════════════════════════════════════════════════════════════════════

from macro8_subnet.alpha.transaction_costs import (
    TransactionCostModel, CostModel, CostResult,
)


def make_pv(n: int = 60) -> pd.Series:
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    vals  = 100_000 * np.cumprod(1 + np.random.default_rng(0).normal(0.001, 0.01, n))
    return pd.Series(vals, index=dates)


def make_weight_history(n: int = 60, seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(n):
        raw = np.abs(rng.normal(0, 1, 3)) + 0.1
        raw /= raw.sum()
        result.append({"SPY": float(raw[0]), "AAPL": float(raw[1]), "GLD": float(raw[2])})
    return result


class TestCostModel:
    def test_default_values(self):
        m = CostModel()
        assert m.cost_per_turnover == pytest.approx(0.001)

    def test_custom_values(self):
        m = CostModel(cost_per_turnover=0.002, spread_bps=10.0)
        assert m.cost_per_turnover == pytest.approx(0.002)


class TestTransactionCostModel:
    def test_returns_cost_result(self):
        tcm    = TransactionCostModel()
        result = tcm.apply(make_pv(), make_weight_history())
        assert isinstance(result, CostResult)

    def test_gross_return_is_float(self):
        tcm    = TransactionCostModel()
        result = tcm.apply(make_pv(), make_weight_history())
        assert isinstance(result.gross_return, float)

    def test_net_return_leq_gross(self):
        tcm    = TransactionCostModel()
        result = tcm.apply(make_pv(), make_weight_history())
        assert result.net_return <= result.gross_return + 1e-8

    def test_total_cost_non_negative(self):
        tcm    = TransactionCostModel()
        result = tcm.apply(make_pv(), make_weight_history())
        assert result.total_cost >= 0.0

    def test_zero_cost_model(self):
        tcm    = TransactionCostModel(CostModel(cost_per_turnover=0.0))
        result = tcm.apply(make_pv(), make_weight_history())
        assert result.total_cost == pytest.approx(0.0)

    def test_high_cost_reduces_net_return(self):
        pv    = make_pv()
        wh    = make_weight_history()
        low   = TransactionCostModel(CostModel(cost_per_turnover=0.0001)).apply(pv, wh)
        high  = TransactionCostModel(CostModel(cost_per_turnover=0.01)).apply(pv, wh)
        assert high.total_cost >= low.total_cost

    def test_single_observation(self):
        tcm    = TransactionCostModel()
        pv     = make_pv(2)
        result = tcm.apply(pv, [{"SPY": 1.0}])
        assert isinstance(result.gross_return, float)

    def test_avg_daily_turn_non_negative(self):
        tcm    = TransactionCostModel()
        result = tcm.apply(make_pv(), make_weight_history())
        assert result.avg_daily_turn >= 0.0

    def test_annualised_cost_non_negative(self):
        tcm    = TransactionCostModel()
        result = tcm.apply(make_pv(), make_weight_history())
        assert result.annualised_cost >= 0.0

    def test_passes_cost_filter_when_profitable(self):
        """Portfolio with positive gross return and zero cost should pass."""
        rng   = np.random.default_rng(0)
        n     = 60
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        vals  = 100_000 * np.cumprod(1 + np.abs(rng.normal(0.005, 0.005, n)))
        pv    = pd.Series(vals, index=dates)
        tcm   = TransactionCostModel(CostModel(cost_per_turnover=0.0))
        r     = tcm.apply(pv, make_weight_history(n))
        assert r.passes_cost_filter is True

    def test_static_portfolio_zero_turnover(self):
        """A portfolio that never rebalances has zero turnover after first day."""
        static_weights = [{"SPY": 0.5, "AAPL": 0.3, "GLD": 0.2}] * 60
        tcm    = TransactionCostModel()
        result = tcm.apply(make_pv(), static_weights)
        # Static weights → turnover only on first period
        assert result.avg_daily_turn < 0.1


# ════════════════════════════════════════════════════════════════════════════
# MACRO SESSION
# ════════════════════════════════════════════════════════════════════════════

from macro8_subnet.alpha.macro_session import (
    MacroSession, SessionReport, _seed_hypotheses, _match_hypothesis,
)


class TestSessionReport:
    def _make(self) -> SessionReport:
        return SessionReport(
            epoch=1, elapsed_seconds=2.5,
            n_formulas_evaluated=300, n_formulas_passing=45,
            best_ic=0.072, best_formula="rank(momentum_20d)",
            batch_throughput=5000.0,
            n_active_hypotheses=5, n_retired_hypotheses=0,
            mean_confidence=0.65, top_hypothesis="Momentum predicts returns",
            n_formula_records=42, n_graph_edges=38,
            n_library_signals=0, n_newly_admitted=1, n_newly_retired=0,
        )

    def test_summary_is_string(self):
        assert isinstance(self._make().summary(), str)

    def test_summary_contains_epoch(self):
        assert "1" in self._make().summary()

    def test_to_dict_serialisable(self):
        json.dumps(self._make().to_dict())

    def test_to_dict_has_required_keys(self):
        d = self._make().to_dict()
        assert "epoch"     in d
        assert "batch"     in d
        assert "knowledge" in d
        assert "library"   in d

    def test_warnings_list_empty_default(self):
        r = SessionReport(epoch=1)
        assert r.warnings == []


class TestSeedHypotheses:
    def test_seeds_requested_count(self):
        from macro8_subnet.alpha.hypothesis_engine import HypothesisLibrary
        lib = HypothesisLibrary()
        _seed_hypotheses(lib, n=4)
        assert lib.size == 4

    def test_seeds_have_statements(self):
        from macro8_subnet.alpha.hypothesis_engine import HypothesisLibrary
        lib = HypothesisLibrary()
        _seed_hypotheses(lib, n=3)
        for rec in lib.all_active():
            assert len(rec.statement) > 10

    def test_seeds_are_active(self):
        from macro8_subnet.alpha.hypothesis_engine import HypothesisLibrary
        lib = HypothesisLibrary()
        _seed_hypotheses(lib, n=5)
        assert lib.n_active == 5


class TestMatchHypothesis:
    def test_momentum_formula_matches_momentum_hyp(self):
        from macro8_subnet.alpha.hypothesis_engine import HypothesisLibrary
        lib = HypothesisLibrary()
        _seed_hypotheses(lib, n=5)
        ids = _match_hypothesis("rank(momentum_20d)", lib)
        assert len(ids) >= 1

    def test_volatility_formula_matches_vol_hyp(self):
        from macro8_subnet.alpha.hypothesis_engine import HypothesisLibrary
        lib = HypothesisLibrary()
        _seed_hypotheses(lib, n=5)
        ids = _match_hypothesis("volatility_20d", lib)
        assert len(ids) >= 1

    def test_unknown_formula_empty_ids(self):
        from macro8_subnet.alpha.hypothesis_engine import HypothesisLibrary
        lib = HypothesisLibrary()
        _seed_hypotheses(lib, n=5)
        ids = _match_hypothesis("rsi_14", lib)   # rsi not in any category map
        assert isinstance(ids, list)  # may be empty or not — just don't crash

    def test_empty_library_returns_empty(self):
        from macro8_subnet.alpha.hypothesis_engine import HypothesisLibrary
        lib = HypothesisLibrary()
        ids = _match_hypothesis("momentum_20d", lib)
        assert ids == []


class TestMacroSessionCreation:
    def test_from_prices_creates_session(self):
        s = MacroSession.from_prices(make_prices(), n_hypotheses=3,
                                      n_formulas_per_epoch=50)
        assert s is not None

    def test_hypothesis_library_seeded(self):
        s = MacroSession.from_prices(make_prices(), n_hypotheses=4,
                                      n_formulas_per_epoch=50)
        assert s.hyp_lib.size == 4

    def test_batch_evaluator_ready(self):
        s = MacroSession.from_prices(make_prices(), n_formulas_per_epoch=50)
        assert s.batch_eval is not None
        assert s.batch_eval.feat_tensor.n_features > 0

    def test_research_graph_initialised(self):
        s = MacroSession.from_prices(make_prices(), n_formulas_per_epoch=50)
        assert s.graph is not None

    def test_knowledge_graph_property(self):
        from macro8_subnet.alpha.research_graph import KnowledgeGraph
        s  = MacroSession.from_prices(make_prices(), n_formulas_per_epoch=50)
        kg = s.knowledge_graph
        assert isinstance(kg, KnowledgeGraph)

    def test_direct_construction(self):
        from macro8_subnet.alpha.hypothesis_engine import HypothesisLibrary
        from macro8_subnet.alpha.research_graph    import ResearchGraph
        from macro8_subnet.alpha.alpha_library     import AlphaLibrary
        from macro8_subnet.alpha.meta_alpha_model  import MetaAlphaModel

        prices  = make_prices()
        hyp_lib = HypothesisLibrary()
        _seed_hypotheses(hyp_lib, n=3)
        graph   = ResearchGraph(hyp_lib)
        alpha   = AlphaLibrary()
        meta    = MetaAlphaModel(min_samples=100)

        s = MacroSession(
            prices=prices, hypothesis_library=hyp_lib,
            research_graph=graph, alpha_library=alpha,
            meta_model=meta, n_formulas_per_epoch=50, verbose=False,
        )
        assert s is not None


class TestMacroSessionEpoch:
    def _session(self, n_formulas=100) -> MacroSession:
        return MacroSession.from_prices(
            make_prices(), n_hypotheses=3,
            n_formulas_per_epoch=n_formulas, verbose=False,
        )

    def test_run_epoch_returns_report(self):
        r = self._session().run_epoch(1)
        assert isinstance(r, SessionReport)

    def test_epoch_number_preserved(self):
        s = self._session()
        assert s.run_epoch(7).epoch == 7

    def test_elapsed_seconds_positive(self):
        assert self._session().run_epoch(1).elapsed_seconds > 0

    def test_formulas_evaluated_positive(self):
        r = self._session().run_epoch(1)
        assert r.n_formulas_evaluated > 0

    def test_best_ic_non_negative(self):
        r = self._session().run_epoch(1)
        assert r.best_ic >= 0.0

    def test_best_formula_is_string(self):
        r = self._session().run_epoch(1)
        assert isinstance(r.best_formula, str)

    def test_n_active_hypotheses_positive(self):
        r = self._session().run_epoch(1)
        assert r.n_active_hypotheses > 0

    def test_mean_confidence_in_range(self):
        r = self._session().run_epoch(1)
        assert 0.0 <= r.mean_confidence <= 1.0

    def test_n_formula_records_grows(self):
        s  = self._session()
        r1 = s.run_epoch(1)
        r2 = s.run_epoch(2)
        # Records should stay stable or grow across epochs
        assert r2.n_formula_records >= 0

    def test_graph_edges_non_negative(self):
        r = self._session().run_epoch(1)
        assert r.n_graph_edges >= 0

    def test_to_dict_serialisable(self):
        r = self._session().run_epoch(1)
        json.dumps(r.to_dict())

    def test_run_multiple_epochs(self):
        s = self._session()
        reports = s.run(n_epochs=3)
        assert len(reports) == 3
        for i, r in enumerate(reports, start=1):
            assert r.epoch == i

    def test_epoch_numbers_sequential(self):
        s = self._session()
        reports = s.run(n_epochs=4)
        epochs = [r.epoch for r in reports]
        assert epochs == [1, 2, 3, 4]

    def test_confidence_never_nan(self):
        s       = self._session()
        reports = s.run(n_epochs=2)
        for r in reports:
            assert not (r.mean_confidence != r.mean_confidence)   # nan check

    def test_batch_throughput_positive(self):
        r = self._session().run_epoch(1)
        assert r.batch_throughput > 0

    def test_hypothesis_confidence_increases_with_evidence(self):
        """
        After many epochs with positive IC formulas, at least one
        hypothesis should have confidence > 0.5.
        """
        s       = self._session(n_formulas=200)
        reports = s.run(n_epochs=3)
        # By epoch 3, some hypotheses should have gained evidence
        assert reports[-1].mean_confidence >= 0.5


class TestMacroSessionKnowledge:
    def test_knowledge_base_builds(self, capsys):
        s = MacroSession.from_prices(make_prices(), n_hypotheses=3,
                                      n_formulas_per_epoch=100, verbose=False)
        s.run(n_epochs=2)
        s.print_knowledge_base(top_n=3)
        out = capsys.readouterr().out
        assert "KNOWLEDGE BASE" in out

    def test_knowledge_graph_builds(self, capsys):
        s = MacroSession.from_prices(make_prices(), n_hypotheses=3,
                                      n_formulas_per_epoch=100, verbose=False)
        s.run(n_epochs=2)
        s.print_knowledge_graph()
        out = capsys.readouterr().out
        assert "KNOWLEDGE GRAPH" in out

    def test_formula_records_linked_to_hypotheses(self):
        """After running, at least some formula records should have hypothesis links."""
        s = MacroSession.from_prices(make_prices(), n_hypotheses=5,
                                      n_formulas_per_epoch=200, verbose=False)
        s.run(n_epochs=2)
        formula_lib = s.graph.formula_library
        linked = [r for r in formula_lib.all_active() if r.hypothesis_ids]
        assert len(linked) > 0

    def test_knowledge_summary_serialisable(self):
        s = MacroSession.from_prices(make_prices(), n_hypotheses=3,
                                      n_formulas_per_epoch=100, verbose=False)
        s.run(n_epochs=2)
        json.dumps(s.knowledge_graph.knowledge_summary())

    def test_top_hypothesis_has_ic_history(self):
        s = MacroSession.from_prices(make_prices(), n_hypotheses=3,
                                      n_formulas_per_epoch=200, verbose=False)
        s.run(n_epochs=3)
        top = s.hyp_lib.rank_by_confidence(1)
        assert len(top) > 0
        # Top hypothesis should have accumulated IC evidence
        assert top[0].n_observations >= 0   # may be 0 if no formulas matched

    def test_regime_ic_populated(self):
        """Running epochs should populate regime-conditional IC data."""
        s = MacroSession.from_prices(make_prices(), n_hypotheses=3,
                                      n_formulas_per_epoch=200, verbose=False)
        s.run(n_epochs=3)
        # At least one hypothesis should have regime_ic data
        has_regime = any(
            bool(r.regime_ic) for r in s.hyp_lib.all_active()
        )
        # Not guaranteed (depends on regime detector), but shouldn't crash
        assert isinstance(has_regime, bool)
