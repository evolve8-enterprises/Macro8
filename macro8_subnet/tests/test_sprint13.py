"""
tests/test_sprint13.py
-----------------------
QA: Complete self-contained tests for Sprint 13 — research graph.

Covers:
    FormulaRecord     — properties, serialisation, lifecycle
    FormulaLibrary    — CRUD, ranking, retirement
    ResearchGraph     — graph construction, evidence propagation, queries
    KnowledgeGraph    — rich queries, regime matrix, summary
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SUITE_DIR  = Path(__file__).resolve().parent
SUBNET_DIR = SUITE_DIR.parent
PROJECT    = SUBNET_DIR.parent
for p in [str(SUBNET_DIR), str(PROJECT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from macro8_subnet.alpha.hypothesis_engine import (
    HypothesisLibrary, HypothesisCategory, BayesianUpdater, HypothesisStatus,
)
from macro8_subnet.alpha.research_graph import (
    FormulaRecord, FormulaLibrary, ResearchGraph, KnowledgeGraph,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_hyp_lib() -> HypothesisLibrary:
    lib  = HypothesisLibrary()
    lib.add("Momentum predicts returns",      HypothesisCategory.MOMENTUM,   0, 1)
    lib.add("Volatility predicts returns",    HypothesisCategory.VOLATILITY, 1, 1)
    lib.add("Regime switching improves alpha",HypothesisCategory.REGIME,     2, 1)
    return lib


def make_formula_record(
    formula: str = "rank(momentum_20d)",
    uid:     int = 0,
) -> FormulaRecord:
    lib = FormulaLibrary()
    return lib.register(formula, uid)


def make_populated_graph() -> tuple[ResearchGraph, list[str], list[str]]:
    """Returns (graph, [hypothesis_ids], [formula_ids])."""
    hyp_lib = make_hyp_lib()
    graph   = ResearchGraph(hyp_lib)
    h_ids   = [r.hypothesis_id for r in hyp_lib.all_active()]

    f_ids = [
        graph.register_formula("rank(momentum_20d)",                     0, 1, [h_ids[0]]),
        graph.register_formula("rank(momentum_20d) - rank(volatility_20d)", 0, 1, [h_ids[0], h_ids[1]]),
        graph.register_formula("zscore(cross_momentum)",                 1, 1, [h_ids[0]]),
        graph.register_formula("regime_signal * momentum_20d",           2, 1, [h_ids[2]]),
        graph.register_formula("volatility_20d",                         1, 1, [h_ids[1]]),
    ]
    return graph, h_ids, f_ids


# ════════════════════════════════════════════════════════════════════════════
# FormulaRecord
# ════════════════════════════════════════════════════════════════════════════

class TestFormulaRecord:
    def test_mean_ic_empty(self):
        r = make_formula_record()
        assert r.mean_ic == 0.0

    def test_mean_ic_computed(self):
        r = make_formula_record()
        r.ic_history = [0.04, 0.05, 0.03]
        assert r.mean_ic == pytest.approx(0.04)

    def test_mean_msc_empty(self):
        r = make_formula_record()
        assert r.mean_msc == 0.0

    def test_ic_stability_all_positive(self):
        r = make_formula_record()
        r.ic_history = [0.04, 0.03, 0.05]
        assert r.ic_stability == pytest.approx(1.0)

    def test_ic_stability_mixed(self):
        r = make_formula_record()
        r.ic_history = [0.04, -0.02, 0.03, -0.01]
        assert r.ic_stability == pytest.approx(0.5)

    def test_n_evaluations(self):
        r = make_formula_record()
        r.ic_history = [0.04, 0.05]
        assert r.n_evaluations == 2

    def test_is_evolved_false_for_gen0(self):
        r = make_formula_record()
        assert r.is_evolved is False

    def test_is_evolved_true_for_gen1(self):
        lib = FormulaLibrary()
        r   = lib.register("rank(momentum_20d)", 0, generation=1)
        assert r.is_evolved is True

    def test_n_supported_hypotheses(self):
        r = make_formula_record()
        r.hypothesis_ids = ["h1", "h2"]
        assert r.n_supported_hypotheses == 2

    def test_best_regime(self):
        r = make_formula_record()
        r.regime_ic = {"low_vol": [0.06, 0.07], "risk_off": [-0.01]}
        assert r.best_regime() == "low_vol"

    def test_best_regime_no_data(self):
        r = make_formula_record()
        assert r.best_regime() is None

    def test_ic_in_regime(self):
        r = make_formula_record()
        r.regime_ic = {"low_vol": [0.05, 0.07]}
        assert r.ic_in_regime("low_vol") == pytest.approx(0.06)

    def test_ic_in_regime_missing(self):
        r = make_formula_record()
        assert r.ic_in_regime("nonexistent") is None

    def test_should_retire_insufficient_obs(self):
        r = make_formula_record()
        r.ic_history = [0.001, -0.001]   # only 2 obs
        assert r.should_retire() is False

    def test_should_retire_poor_ic(self):
        r = make_formula_record()
        r.ic_history = [-0.01, -0.02, -0.01, -0.02, -0.01, 0.001]
        assert r.should_retire() is True

    def test_should_not_retire_good_ic(self):
        r = make_formula_record()
        r.ic_history = [0.04, 0.05, 0.03, 0.04, 0.05]
        assert r.should_retire() is False

    def test_to_dict_has_required_keys(self):
        r = make_formula_record()
        d = r.to_dict()
        for key in ("formula_id", "formula_string", "mean_ic",
                    "n_evaluations", "is_evolved", "generation"):
            assert key in d

    def test_to_dict_serialisable(self):
        r = make_formula_record()
        r.ic_history = [0.04, 0.05]
        json.dumps(r.to_dict())

    def test_formula_id_deterministic(self):
        lib = FormulaLibrary()
        r1  = lib.register("rank(momentum_20d)", 0)
        r2  = lib.register("rank(momentum_20d)", 1)   # different miner
        assert r1.formula_id == r2.formula_id   # same formula string → same id

    def test_summary_line_is_string(self):
        r = make_formula_record()
        assert isinstance(r.summary_line(), str)


# ════════════════════════════════════════════════════════════════════════════
# FormulaLibrary
# ════════════════════════════════════════════════════════════════════════════

class TestFormulaLibrary:
    def test_register_creates_record(self):
        lib = FormulaLibrary()
        r   = lib.register("momentum_20d", 0)
        assert r is not None
        assert lib.size == 1

    def test_register_same_formula_returns_existing(self):
        lib = FormulaLibrary()
        r1  = lib.register("momentum_20d", 0)
        r2  = lib.register("momentum_20d", 1)   # different miner
        assert r1 is r2
        assert lib.size == 1

    def test_different_formulas_different_records(self):
        lib = FormulaLibrary()
        lib.register("momentum_20d", 0)
        lib.register("volatility_20d", 0)
        assert lib.size == 2

    def test_get_returns_record(self):
        lib = FormulaLibrary()
        r1  = lib.register("momentum_20d", 0)
        r2  = lib.get(r1.formula_id)
        assert r2 is r1

    def test_get_by_string(self):
        lib = FormulaLibrary()
        lib.register("rank(momentum_20d)", 0)
        r   = lib.get_by_string("rank(momentum_20d)")
        assert r is not None

    def test_get_unknown_returns_none(self):
        lib = FormulaLibrary()
        assert lib.get("nonexistent") is None

    def test_all_active_excludes_retired(self):
        lib = FormulaLibrary()
        r   = lib.register("momentum_20d", 0)
        lib.retire(r.formula_id)
        assert len(lib.all_active()) == 0

    def test_all_evolved_filters_by_generation(self):
        lib = FormulaLibrary()
        lib.register("f1", 0, generation=0)
        lib.register("f2", 0, generation=1)
        lib.register("f3", 0, generation=2)
        assert len(lib.all_evolved()) == 2

    def test_rank_by_ic_sorted(self):
        lib = FormulaLibrary()
        for i, f in enumerate(["f1", "f2", "f3"]):
            r = lib.register(f, 0)
            r.ic_history = [float(i+1) * 0.01]
        top = lib.rank_by_ic(3)
        ics = [r.mean_ic for r in top]
        assert ics == sorted(ics, reverse=True)

    def test_rank_by_msc_sorted(self):
        lib = FormulaLibrary()
        for i, f in enumerate(["f1", "f2", "f3"]):
            r = lib.register(f, 0)
            r.msc_history = [float(i+1) * 0.005]
        top = lib.rank_by_msc(3)
        mscs = [r.mean_msc for r in top]
        assert mscs == sorted(mscs, reverse=True)

    def test_by_hypothesis(self):
        lib = FormulaLibrary()
        r1  = lib.register("f1", 0)
        r2  = lib.register("f2", 0)
        r1.hypothesis_ids = ["h1", "h2"]
        r2.hypothesis_ids = ["h2"]
        h2_formulas = lib.by_hypothesis("h2")
        assert len(h2_formulas) == 2

    def test_update_ic_appends(self):
        lib = FormulaLibrary()
        r   = lib.register("f1", 0)
        lib.update_ic(r.formula_id, 0.04, msc=0.01, regime="low_vol")
        assert len(r.ic_history) == 1
        assert len(r.msc_history) == 1
        assert "low_vol" in r.regime_ic

    def test_update_ic_auto_retires_poor_formula(self):
        lib = FormulaLibrary()
        r   = lib.register("bad_formula", 0)
        for _ in range(6):
            lib.update_ic(r.formula_id, -0.05)
        assert r.is_retired is True

    def test_n_active_property(self):
        lib = FormulaLibrary()
        r   = lib.register("f1", 0)
        lib.register("f2", 0)
        lib.retire(r.formula_id)
        assert lib.n_active == 1

    def test_n_evolved_property(self):
        lib = FormulaLibrary()
        lib.register("f1", 0, generation=0)
        lib.register("f2", 0, generation=1)
        assert lib.n_evolved == 1


# ════════════════════════════════════════════════════════════════════════════
# ResearchGraph
# ════════════════════════════════════════════════════════════════════════════

class TestResearchGraph:
    def test_register_formula_returns_id(self):
        graph = ResearchGraph(make_hyp_lib())
        fid   = graph.register_formula("momentum_20d", 0)
        assert isinstance(fid, str) and len(fid) > 0

    def test_register_creates_formula_record(self):
        graph = ResearchGraph(make_hyp_lib())
        fid   = graph.register_formula("momentum_20d", 0)
        assert graph.formula_library.get(fid) is not None

    def test_n_formulas_increments(self):
        graph = ResearchGraph(make_hyp_lib())
        graph.register_formula("f1", 0)
        graph.register_formula("f2", 0)
        assert graph.n_formulas == 2

    def test_link_creates_edges(self):
        hyp_lib = make_hyp_lib()
        graph   = ResearchGraph(hyp_lib)
        hid     = hyp_lib.all_active()[0].hypothesis_id
        fid     = graph.register_formula("f1", 0)
        graph.link(fid, hid)
        assert graph.n_edges == 1

    def test_link_is_bidirectional(self):
        hyp_lib = make_hyp_lib()
        graph   = ResearchGraph(hyp_lib)
        hid     = hyp_lib.all_active()[0].hypothesis_id
        fid     = graph.register_formula("f1", 0)
        graph.link(fid, hid)
        assert len(graph.formulas_for(hid)) == 1
        assert len(graph.hypotheses_for(fid)) == 1

    def test_link_idempotent(self):
        hyp_lib = make_hyp_lib()
        graph   = ResearchGraph(hyp_lib)
        hid     = hyp_lib.all_active()[0].hypothesis_id
        fid     = graph.register_formula("f1", 0)
        graph.link(fid, hid)
        graph.link(fid, hid)   # duplicate
        assert graph.n_edges == 1

    def test_unlink_removes_edge(self):
        hyp_lib = make_hyp_lib()
        graph   = ResearchGraph(hyp_lib)
        hid     = hyp_lib.all_active()[0].hypothesis_id
        fid     = graph.register_formula("f1", 0)
        graph.link(fid, hid)
        graph.unlink(fid, hid)
        assert graph.n_edges == 0

    def test_register_with_hypothesis_ids_creates_edges(self):
        hyp_lib = make_hyp_lib()
        graph   = ResearchGraph(hyp_lib)
        h_ids   = [r.hypothesis_id for r in hyp_lib.all_active()][:2]
        fid     = graph.register_formula("f1", 0, hypothesis_ids=h_ids)
        assert graph.n_supported_hypotheses(fid) == 2

    def test_propagate_evidence_updates_formula(self):
        graph, h_ids, f_ids = make_populated_graph()
        graph.propagate_evidence(f_ids[0], ic=0.04, epoch=2)
        rec = graph.formula_library.get(f_ids[0])
        assert len(rec.ic_history) == 1
        assert rec.ic_history[0] == pytest.approx(0.04)

    def test_propagate_evidence_updates_linked_hypotheses(self):
        graph, h_ids, f_ids = make_populated_graph()
        # f_ids[0] is linked to h_ids[0]
        hrec = graph.hypothesis_library.get(h_ids[0])
        before = hrec.confidence_score
        graph.propagate_evidence(f_ids[0], ic=0.05, epoch=2)
        assert hrec.confidence_score > before

    def test_propagate_evidence_returns_hypothesis_confidences(self):
        graph, h_ids, f_ids = make_populated_graph()
        result = graph.propagate_evidence(f_ids[0], ic=0.04, epoch=2)
        assert isinstance(result, dict)
        assert h_ids[0] in result

    def test_propagate_evidence_multi_hypothesis_formula(self):
        """f_ids[1] is linked to h_ids[0] AND h_ids[1]."""
        graph, h_ids, f_ids = make_populated_graph()
        result = graph.propagate_evidence(f_ids[1], ic=0.04, epoch=2)
        assert h_ids[0] in result
        assert h_ids[1] in result

    def test_propagate_evidence_regime_tracked(self):
        graph, h_ids, f_ids = make_populated_graph()
        graph.propagate_evidence(f_ids[0], ic=0.04, regime="low_vol", epoch=2)
        rec = graph.formula_library.get(f_ids[0])
        assert "low_vol" in rec.regime_ic

    def test_propagate_evidence_msc_tracked(self):
        graph, h_ids, f_ids = make_populated_graph()
        graph.propagate_evidence(f_ids[0], ic=0.04, msc=0.015, epoch=2)
        rec = graph.formula_library.get(f_ids[0])
        assert len(rec.msc_history) == 1

    def test_propagate_batch(self):
        graph, h_ids, f_ids = make_populated_graph()
        ic_map = {f_ids[0]: 0.04, f_ids[1]: 0.03, f_ids[2]: 0.05}
        result = graph.propagate_batch(ic_map, epoch=2)
        assert len(result) == 3

    def test_multi_formula_faster_confidence_update(self):
        """Hypothesis with 2 supporting formulas updates confidence faster."""
        hyp_lib = make_hyp_lib()
        hrec    = hyp_lib.all_active()[0]

        graph1 = ResearchGraph(hyp_lib)
        fid1   = graph1.register_formula("f1", 0, hypothesis_ids=[hrec.hypothesis_id])
        fid2   = graph1.register_formula("f2", 0, hypothesis_ids=[hrec.hypothesis_id])

        # Propagate 2 successes in one epoch (2 linked formulas)
        graph1.propagate_evidence(fid1, ic=0.05, epoch=1)
        graph1.propagate_evidence(fid2, ic=0.05, epoch=1)
        conf_two_formulas = hrec.confidence_score

        # Rebuild with one formula only
        hyp_lib2 = make_hyp_lib()
        hrec2    = hyp_lib2.all_active()[0]
        graph2   = ResearchGraph(hyp_lib2)
        fid3     = graph2.register_formula("f1", 0, hypothesis_ids=[hrec2.hypothesis_id])
        graph2.propagate_evidence(fid3, ic=0.05, epoch=1)
        conf_one_formula = hrec2.confidence_score

        assert conf_two_formulas > conf_one_formula

    def test_formulas_for_hypothesis(self):
        graph, h_ids, f_ids = make_populated_graph()
        formulas = graph.formulas_for(h_ids[0])
        assert len(formulas) >= 2   # h_ids[0] linked to f_ids[0], [1], [2]

    def test_hypotheses_for_formula(self):
        graph, h_ids, f_ids = make_populated_graph()
        # f_ids[1] links to h_ids[0] and h_ids[1]
        hyps = graph.hypotheses_for(f_ids[1])
        assert len(hyps) == 2

    def test_n_supporting_formulas(self):
        graph, h_ids, f_ids = make_populated_graph()
        assert graph.n_supporting_formulas(h_ids[0]) == 3

    def test_shared_formulas(self):
        graph, h_ids, f_ids = make_populated_graph()
        shared = graph.shared_formulas(h_ids[0], h_ids[1])
        assert len(shared) == 1
        assert shared[0].formula_id == f_ids[1]

    def test_shared_formulas_none(self):
        graph, h_ids, f_ids = make_populated_graph()
        # h_ids[0] and h_ids[2] have no shared formula
        shared = graph.shared_formulas(h_ids[0], h_ids[2])
        assert len(shared) == 0

    def test_n_isolated_formulas(self):
        graph = ResearchGraph(make_hyp_lib())
        graph.register_formula("isolated_formula", 0)   # no hypothesis linked
        assert graph.n_isolated_formulas == 1

    def test_n_isolated_hypotheses(self):
        hyp_lib = make_hyp_lib()
        graph   = ResearchGraph(hyp_lib)
        # No formulas registered → all hypotheses are isolated
        assert graph.n_isolated_hypotheses == len(hyp_lib.all_active())

    def test_formula_evolution_tree(self):
        graph    = ResearchGraph(make_hyp_lib())
        parent   = graph.register_formula("momentum_20d", 0)
        child    = graph.register_formula("rank(momentum_20d)", 0,
                                           generation=1, parent_ids=[parent])
        tree = graph.formula_evolution_tree(parent)
        assert tree["formula_id"] == parent
        assert isinstance(tree["children"], list)

    def test_evolved_formula_parent_link(self):
        graph    = ResearchGraph(make_hyp_lib())
        parent   = graph.register_formula("f_parent", 0)
        child_id = graph.register_formula("f_child", 0,
                                           generation=1, parent_ids=[parent])
        child_rec = graph.formula_library.get(child_id)
        assert parent in child_rec.parent_ids


# ════════════════════════════════════════════════════════════════════════════
# KnowledgeGraph
# ════════════════════════════════════════════════════════════════════════════

class TestKnowledgeGraph:
    def _kg(self) -> tuple[KnowledgeGraph, ResearchGraph, list[str], list[str]]:
        hyp_lib = make_hyp_lib()
        graph   = ResearchGraph(hyp_lib)
        h_ids   = [r.hypothesis_id for r in hyp_lib.all_active()]
        f_ids   = [
            graph.register_formula("rank(momentum_20d)",           0, 1, [h_ids[0]]),
            graph.register_formula("zscore(cross_momentum)",       0, 1, [h_ids[0], h_ids[1]]),
            graph.register_formula("volatility_20d",               1, 1, [h_ids[1]]),
            graph.register_formula("regime_signal * momentum_20d", 2, 1, [h_ids[2]]),
        ]
        # Add some IC history
        for fid, ic in zip(f_ids, [0.04, 0.03, 0.02, 0.05]):
            graph.propagate_evidence(fid, ic, regime="low_vol", epoch=2)

        kg = KnowledgeGraph(hyp_lib, graph.formula_library, graph)
        return kg, graph, h_ids, f_ids

    def test_most_supported_hypotheses_returns_list(self):
        kg, *_ = self._kg()
        result = kg.most_supported_hypotheses(3)
        assert isinstance(result, list)

    def test_most_supported_sorted_by_n_formulas(self):
        kg, *_ = self._kg()
        result = kg.most_supported_hypotheses()
        n_vals = [r["n_formulas"] for r in result]
        assert n_vals == sorted(n_vals, reverse=True)

    def test_most_versatile_formulas_returns_list(self):
        kg, *_ = self._kg()
        result = kg.most_versatile_formulas(3)
        assert isinstance(result, list)

    def test_most_versatile_sorted_by_n_hypotheses(self):
        kg, *_ = self._kg()
        result = kg.most_versatile_formulas()
        n_vals = [r["n_hypotheses"] for r in result]
        assert n_vals == sorted(n_vals, reverse=True)

    def test_best_formulas_for_hypothesis(self):
        kg, _, h_ids, _ = self._kg()
        result = kg.best_formulas_for_hypothesis(h_ids[0])
        assert isinstance(result, list)
        if len(result) >= 2:
            ics = [r["mean_ic"] for r in result]
            assert ics == sorted(ics, reverse=True)

    def test_regime_performance_matrix(self):
        kg, *_ = self._kg()
        matrix = kg.regime_performance_matrix()
        assert isinstance(matrix, dict)
        # low_vol evidence was added for all formulas
        # At least one category should have low_vol data
        has_regime = any("low_vol" in v for v in matrix.values())
        assert has_regime

    def test_knowledge_summary_keys(self):
        kg, *_ = self._kg()
        s = kg.knowledge_summary()
        for key in ("n_hypotheses", "n_formulas", "n_edges",
                    "most_supported", "most_versatile"):
            assert key in s

    def test_knowledge_summary_serialisable(self):
        kg, *_ = self._kg()
        json.dumps(kg.knowledge_summary())

    def test_n_edges_correct(self):
        kg, graph, *_ = self._kg()
        s = kg.knowledge_summary()
        assert s["n_edges"] == graph.n_edges

    def test_print_summary_runs(self, capsys):
        kg, *_ = self._kg()
        kg.print_summary()
        out = capsys.readouterr().out
        assert "KNOWLEDGE GRAPH" in out


# ════════════════════════════════════════════════════════════════════════════
# END-TO-END: Research Graph + Hypothesis Engine
# ════════════════════════════════════════════════════════════════════════════

class TestEndToEnd:
    def test_full_epoch_evidence_flow(self):
        """Simulate a complete epoch: register formulas, propagate evidence,
        check hypothesis confidence updates correctly."""
        hyp_lib = HypothesisLibrary()
        hrec    = hyp_lib.add("Momentum predicts returns",
                              HypothesisCategory.MOMENTUM, 0, 1)
        graph   = ResearchGraph(hyp_lib)

        # Miner submits two formulas for one hypothesis
        f1 = graph.register_formula("rank(momentum_20d)",           0, 1, [hrec.hypothesis_id])
        f2 = graph.register_formula("rank(momentum_20d) - rank(volatility_20d)", 0, 1,
                                     [hrec.hypothesis_id])

        # Epoch 1: both formulas evaluated, both show positive IC
        graph.propagate_evidence(f1, ic=0.04, regime="Low-Vol Trending", epoch=1)
        graph.propagate_evidence(f2, ic=0.03, regime="Low-Vol Trending", epoch=1)

        # 2 successes → alpha = 3, beta = 1, confidence = 3/4 = 0.75
        assert hrec.confidence_score == pytest.approx(0.75)
        assert hrec.status.value in ("active", "pending")

        # Epoch 2: one formula fails
        graph.propagate_evidence(f1, ic=-0.01, epoch=2)
        # 1 failure → alpha=3, beta=2, confidence = 3/5 = 0.60
        assert hrec.confidence_score == pytest.approx(0.60)

    def test_hypothesis_accumulates_evidence_from_many_formulas(self):
        """10 formulas supporting one hypothesis → confidence converges."""
        hyp_lib = HypothesisLibrary()
        hrec    = hyp_lib.add("Strong momentum claim",
                              HypothesisCategory.MOMENTUM, 0, 1)
        graph   = ResearchGraph(hyp_lib)

        for i in range(10):
            fid = graph.register_formula(f"formula_{i}", 0, 1, [hrec.hypothesis_id])
            graph.propagate_evidence(fid, ic=0.05, epoch=1)   # all positive

        # 10 successes → alpha=11, beta=1 → confidence ≈ 0.917
        assert hrec.confidence_score > 0.85

    def test_weak_hypothesis_retires(self):
        """After many failures, hypothesis should retire."""
        hyp_lib = HypothesisLibrary()
        hrec    = hyp_lib.add("Bad hypothesis", HypothesisCategory.UNKNOWN, 0, 1)
        graph   = ResearchGraph(hyp_lib)

        fid = graph.register_formula("bad_formula", 0, 1, [hrec.hypothesis_id])
        for i in range(12):
            graph.propagate_evidence(fid, ic=-0.05, epoch=i+1)

        assert hrec.status == HypothesisStatus.RETIRED

    def test_formula_retirement_does_not_break_graph(self):
        """Retiring a formula should not affect linked hypotheses."""
        hyp_lib = HypothesisLibrary()
        hrec    = hyp_lib.add("Test", HypothesisCategory.MOMENTUM, 0, 1)
        graph   = ResearchGraph(hyp_lib)

        fid = graph.register_formula("retiring_formula", 0, 1, [hrec.hypothesis_id])
        # Add poor IC to trigger auto-retirement
        for _ in range(6):
            graph.propagate_evidence(fid, ic=-0.05)

        frec = graph.formula_library.get(fid)
        assert frec.is_retired is True
        # Hypothesis still accessible
        assert graph.hypothesis_library.get(hrec.hypothesis_id) is not None

    def test_knowledge_graph_after_full_epoch(self):
        """KnowledgeGraph summary is consistent after evidence propagation."""
        hyp_lib = make_hyp_lib()
        graph   = ResearchGraph(hyp_lib)
        h_ids   = [r.hypothesis_id for r in hyp_lib.all_active()]

        for i, hid in enumerate(h_ids):
            for j in range(2):
                fid = graph.register_formula(f"formula_{i}_{j}", i, 1, [hid])
                graph.propagate_evidence(fid, ic=0.03 + j*0.01, regime="low_vol", epoch=1)

        kg = KnowledgeGraph(hyp_lib, graph.formula_library, graph)
        s  = kg.knowledge_summary()

        assert s["n_formulas"]  == 6
        assert s["n_edges"]     == 6
        assert s["n_isolated_hypotheses"] == 0
        assert s["n_isolated_formulas"]   == 0
        import json
        json.dumps(s)   # must be serialisable
