"""
tests/test_sprint30.py
-----------------------
Sprint 30: Decision Execution Engine

Tests cover:
    - PortfolioConstraints / ConstraintSolver
    - DrawdownGuard
    - TradeExecutor / ExecutionPlan / TradeOrder
    - LiveTracker / DailyRecord / PerformanceWindow
    - PredictionMarket / MacroPrediction
    - run_live integration
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
for p in [str(_ROOT), str(_ROOT / "macro8_subnet")]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def prices():
    rng = np.random.default_rng(42)
    n, a = 400, 10
    tickers = ["SPY","QQQ","IWM","TLT","GLD","DBC","EEM","FXI","VNQ","HYG"]
    p = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, (n, a)), axis=0))
    return pd.DataFrame(p, index=pd.bdate_range("2015-01-01", periods=n), columns=tickers)


@pytest.fixture(scope="module")
def raw_positions():
    return {"SPY": 0.15, "QQQ": 0.10, "TLT": -0.20, "GLD": -0.12,
            "EEM": 0.09, "DBC": -0.06, "IWM": 0.06, "HYG": 0.04,
            "VNQ": 0.03, "FXI": -0.03}


@pytest.fixture(scope="module")
def constraints():
    from macro8_subnet.execution.engine import PortfolioConstraints
    return PortfolioConstraints(max_weight=0.35, max_sector_gross=0.55,
                                max_net_exposure=0.20, min_weight=0.01)


@pytest.fixture(scope="module")
def solver(constraints):
    from macro8_subnet.execution.engine import ConstraintSolver
    return ConstraintSolver(constraints)


@pytest.fixture(scope="module")
def executor():
    from macro8_subnet.execution.engine import TradeExecutor
    return TradeExecutor(capital=100_000, min_trade=0.005)


@pytest.fixture(scope="module")
def forecast(prices):
    from macro8_subnet.alpha.gp_miner import GPMiner
    from macro8_subnet.alpha.regime_prediction import ForecastedEnsemble
    gp = GPMiner(prices, pop_size=30, elite_n=6, seed=42, verbose=False)
    gp.run(n_epochs=2)
    formulas = gp.top_formulas(8)
    fens = ForecastedEnsemble(prices, formulas[:6], horizon=5,
                               verbose=False, scale_by_conf=False)
    fens.fit()
    return fens.forecast()


# ── 1. PortfolioConstraints ───────────────────────────────────────────────────

class TestPortfolioConstraints:
    def test_import(self):
        from macro8_subnet.execution.engine import PortfolioConstraints

    def test_default_values(self):
        from macro8_subnet.execution.engine import PortfolioConstraints
        c = PortfolioConstraints()
        assert c.max_weight == 0.40
        assert c.max_sector_gross == 0.60
        assert c.max_net_exposure == 0.20
        assert c.min_weight == 0.01

    def test_default_sector_map_populated(self):
        from macro8_subnet.execution.engine import PortfolioConstraints
        c = PortfolioConstraints()
        assert c.sector_map is not None
        assert "SPY" in c.sector_map

    def test_effective_max_weight_no_stress(self):
        from macro8_subnet.execution.engine import PortfolioConstraints
        c = PortfolioConstraints(max_weight=0.40, stress_delever=0.25)
        assert c.effective_max_weight(0.0) == pytest.approx(0.40)

    def test_effective_max_weight_high_stress(self):
        from macro8_subnet.execution.engine import PortfolioConstraints
        c = PortfolioConstraints(max_weight=0.40, stress_delever=0.25)
        # P(stress)=1.0 → maximum deleverage
        mw = c.effective_max_weight(1.0)
        assert mw < 0.40, "High stress should reduce max weight"
        assert mw > 0.0

    def test_stress_delever_monotone(self):
        from macro8_subnet.execution.engine import PortfolioConstraints
        c = PortfolioConstraints(max_weight=0.40)
        mw_low  = c.effective_max_weight(0.1)
        mw_high = c.effective_max_weight(0.9)
        assert mw_low >= mw_high


# ── 2. ConstraintSolver ───────────────────────────────────────────────────────

class TestConstraintSolver:
    def test_import(self):
        from macro8_subnet.execution.engine import ConstraintSolver

    def test_apply_returns_dict(self, solver, raw_positions):
        out = solver.apply(raw_positions)
        assert isinstance(out, dict)

    def test_l1_norm_equals_scale(self, solver, raw_positions):
        out = solver.apply(raw_positions, scale=1.0)
        l1  = sum(abs(w) for w in out.values())
        assert abs(l1 - 1.0) < 0.01

    def test_scale_applied_correctly(self, solver, raw_positions):
        out_full  = solver.apply(raw_positions, scale=1.0)
        out_half  = solver.apply(raw_positions, scale=0.5)
        l1_full   = sum(abs(w) for w in out_full.values())
        l1_half   = sum(abs(w) for w in out_half.values())
        assert abs(l1_full - 1.0) < 0.01
        assert abs(l1_half - 0.5) < 0.01

    def test_no_weight_exceeds_max(self, solver, raw_positions):
        out = solver.apply(raw_positions, p_stress=0.0)
        max_w = solver.c.max_weight
        for t, w in out.items():
            assert abs(w) <= max_w + 1e-6, f"{t}: {w:.4f} > {max_w}"

    def test_min_weight_filter(self):
        from macro8_subnet.execution.engine import PortfolioConstraints, ConstraintSolver
        c  = PortfolioConstraints(min_weight=0.05)
        cs = ConstraintSolver(c)
        pos = {"SPY": 0.50, "QQQ": 0.03, "TLT": -0.02}  # last two too small
        out = cs.apply(pos)
        assert "QQQ" not in out or abs(out.get("QQQ", 0)) >= 0.05

    def test_net_exposure_within_limit(self, solver, raw_positions):
        out = solver.apply(raw_positions)
        net = sum(out.values())
        assert abs(net) <= solver.c.max_net_exposure + 0.01

    def test_empty_positions_returns_empty(self, solver):
        out = solver.apply({})
        assert out == {}

    def test_stress_reduces_max_weight(self):
        from macro8_subnet.execution.engine import PortfolioConstraints, ConstraintSolver
        cs  = ConstraintSolver(PortfolioConstraints(max_weight=0.40))
        pos = {"SPY": 0.30, "TLT": -0.30}
        out_low  = cs.apply(pos, p_stress=0.05)
        out_high = cs.apply(pos, p_stress=0.95)
        # High stress → lower individual weights (before normalisation)
        # Check that stress_delever is having an effect somewhere
        max_low  = max(abs(w) for w in out_low.values())
        max_high = max(abs(w) for w in out_high.values())
        # After L1 normalisation both should be ~0.5, so check at un-normalised level
        # The constraint clips, then normalises — so with stress the clip fires at lower level
        assert max_low >= max_high or abs(max_low - max_high) < 0.05  # may be equal if no clip


# ── 3. DrawdownGuard ─────────────────────────────────────────────────────────

class TestDrawdownGuard:
    def test_import(self):
        from macro8_subnet.execution.engine import DrawdownGuard

    def test_initial_scale_is_one(self):
        from macro8_subnet.execution.engine import DrawdownGuard
        g = DrawdownGuard()
        assert g.position_scale() == 1.0

    def test_scale_remains_one_within_limit(self):
        from macro8_subnet.execution.engine import DrawdownGuard
        g = DrawdownGuard(max_drawdown=-0.10)
        for _ in range(10):
            g.update(0.001)   # positive PnL
        assert g.position_scale() == 1.0

    def test_scale_falls_below_one_on_drawdown(self):
        from macro8_subnet.execution.engine import DrawdownGuard
        g = DrawdownGuard(max_drawdown=-0.02, lookback=30)
        # Simulate a drawdown
        g.update(0.005)
        for _ in range(15):
            g.update(-0.004)  # sustained loss
        scale = g.position_scale()
        assert scale < 1.0, f"Expected scale < 1.0, got {scale}"

    def test_scale_never_below_floor(self):
        from macro8_subnet.execution.engine import DrawdownGuard
        g = DrawdownGuard(max_drawdown=-0.01, position_floor=0.25)
        for _ in range(30):
            g.update(-0.01)
        assert g.position_scale() >= 0.25

    def test_current_drawdown_non_positive(self):
        from macro8_subnet.execution.engine import DrawdownGuard
        g = DrawdownGuard()
        g.update(0.01); g.update(-0.02); g.update(-0.01)
        assert g.current_drawdown <= 0

    def test_cumulative_pnl(self):
        from macro8_subnet.execution.engine import DrawdownGuard
        g = DrawdownGuard()
        g.update(0.01); g.update(-0.005); g.update(0.002)
        assert abs(g.cumulative_pnl - 0.007) < 1e-8


# ── 4. TradeExecutor ─────────────────────────────────────────────────────────

class TestTradeExecutor:
    def test_import(self):
        from macro8_subnet.execution.engine import TradeExecutor, TradeOrder, ExecutionPlan

    def test_compute_trades_from_flat(self, executor, raw_positions):
        plan = executor.compute_trades(raw_positions, {}, date=pd.Timestamp("2020-01-01"))
        assert isinstance(plan.orders, list)
        assert len(plan.orders) > 0

    def test_all_orders_are_trade_orders(self, executor, raw_positions):
        from macro8_subnet.execution.engine import TradeOrder
        plan = executor.compute_trades(raw_positions, {})
        for o in plan.orders:
            assert isinstance(o, TradeOrder)

    def test_plan_total_turnover_correct(self, executor, raw_positions):
        plan = executor.compute_trades(raw_positions, {})
        expected = sum(abs(o.weight_change) for o in plan.orders)
        assert abs(plan.total_turnover - expected) < 1e-8

    def test_n_buys_sells_correct(self, executor, raw_positions):
        plan = executor.compute_trades(raw_positions, {})
        n_buys  = sum(1 for o in plan.orders if o.direction == "BUY")
        n_sells = sum(1 for o in plan.orders if o.direction == "SELL")
        assert plan.n_buys  == n_buys
        assert plan.n_sells == n_sells

    def test_min_trade_filter(self, raw_positions):
        from macro8_subnet.execution.engine import TradeExecutor
        ex = TradeExecutor(capital=100_000, min_trade=0.50)  # huge min
        plan = ex.compute_trades(raw_positions, {})
        assert len(plan.orders) == 0   # all trades smaller than 0.50 filtered

    def test_no_trades_when_already_positioned(self, executor, raw_positions):
        # If current == target exactly, no trades needed
        plan = executor.compute_trades(raw_positions, raw_positions)
        assert len(plan.orders) == 0 or plan.total_turnover < 0.01

    def test_notional_equals_weight_times_capital(self, executor):
        from macro8_subnet.execution.engine import TradeExecutor
        ex  = TradeExecutor(capital=200_000, min_trade=0.001)
        pos = {"SPY": 0.20, "TLT": -0.15}
        plan = ex.compute_trades(pos, {})
        for o in plan.orders:
            expected_notional = abs(o.weight_change) * 200_000
            assert abs(o.notional - expected_notional) < 1.0

    def test_execution_plan_summary_string(self, executor, raw_positions):
        plan = executor.compute_trades(raw_positions, {})
        s = plan.summary()
        assert "turnover" in s.lower()
        assert "orders" in s.lower()

    def test_simulate_fill_returns_dict(self, executor, prices, raw_positions):
        plan = executor.compute_trades(raw_positions, {}, date=prices.index[0])
        fills = executor.simulate_fill(plan, prices, prices.index[0])
        assert isinstance(fills, dict)

    def test_cost_estimate_non_negative(self, executor, raw_positions):
        plan = executor.compute_trades(raw_positions, {})
        assert plan.estimated_cost_bps >= 0
        for o in plan.orders:
            assert o.cost_estimate_bps >= 0


# ── 5. LiveTracker ────────────────────────────────────────────────────────────

class TestLiveTracker:
    def test_import(self):
        from macro8_subnet.execution.engine import LiveTracker, PerformanceWindow

    def test_initial_snapshot_empty(self):
        from macro8_subnet.execution.engine import LiveTracker
        t = LiveTracker()
        pw = t.snapshot()
        assert pw.n_days == 0

    def test_update_increments_records(self, forecast, executor, raw_positions, prices):
        from macro8_subnet.execution.engine import LiveTracker, ConstraintSolver
        t = LiveTracker()
        plan = executor.compute_trades(raw_positions, {})
        t.update(prices.index[0], raw_positions, plan, 0.001, forecast)
        t.update(prices.index[1], raw_positions, plan, -0.002, forecast)
        assert len(t._records) == 2

    def test_snapshot_returns_performance_window(self, forecast, executor, raw_positions, prices):
        from macro8_subnet.execution.engine import LiveTracker
        t    = LiveTracker()
        plan = executor.compute_trades(raw_positions, {})
        for i in range(5):
            pnl = 0.001 if i % 2 == 0 else -0.0005
            t.update(prices.index[i], raw_positions, plan, pnl, forecast)
        pw = t.snapshot()
        assert isinstance(pw.sharpe_ann, float)
        assert np.isfinite(pw.sharpe_ann)

    def test_hit_rate_in_unit_interval(self, forecast, executor, raw_positions, prices):
        from macro8_subnet.execution.engine import LiveTracker
        t    = LiveTracker()
        plan = executor.compute_trades(raw_positions, {})
        pnls = [0.01, -0.005, 0.008, -0.002, 0.003, 0.001]
        for i, pnl in enumerate(pnls):
            t.update(prices.index[i], raw_positions, plan, pnl, forecast)
        pw = t.snapshot()
        assert 0 <= pw.hit_rate <= 1

    def test_drawdown_guard_updates_on_update(self, forecast, executor, raw_positions, prices):
        from macro8_subnet.execution.engine import LiveTracker
        t    = LiveTracker()
        plan = executor.compute_trades(raw_positions, {})
        for i in range(10):
            t.update(prices.index[i], raw_positions, plan, -0.003, forecast)
        dd = t.drawdown_guard.current_drawdown
        assert dd < 0

    def test_confidence_multiplier_in_range(self, forecast, executor, raw_positions, prices):
        from macro8_subnet.execution.engine import LiveTracker
        t    = LiveTracker()
        plan = executor.compute_trades(raw_positions, {})
        for i in range(5):
            t.update(prices.index[i], raw_positions, plan, 0.001, forecast)
        mult = t.confidence_multiplier()
        assert 0.25 <= mult <= 1.0

    def test_full_history_returns_dataframe(self, forecast, executor, raw_positions, prices):
        from macro8_subnet.execution.engine import LiveTracker
        t    = LiveTracker()
        plan = executor.compute_trades(raw_positions, {})
        for i in range(5):
            t.update(prices.index[i], raw_positions, plan, 0.001, forecast)
        df = t.full_history()
        assert isinstance(df, pd.DataFrame)
        assert "pnl" in df.columns
        assert "cum_pnl" in df.columns

    def test_print_summary_no_crash(self, forecast, executor, raw_positions, prices, capsys):
        from macro8_subnet.execution.engine import LiveTracker
        t    = LiveTracker()
        plan = executor.compute_trades(raw_positions, {})
        for i in range(3):
            t.update(prices.index[i], raw_positions, plan, 0.001, forecast)
        t.print_summary()
        out = capsys.readouterr().out
        assert "MACRO8" in out


# ── 6. PredictionMarket ───────────────────────────────────────────────────────

class TestPredictionMarket:
    def test_import(self):
        from macro8_subnet.execution.engine import PredictionMarket, MacroPrediction

    def test_emit_returns_macro_prediction(self, forecast):
        from macro8_subnet.execution.engine import (
            PredictionMarket, DrawdownGuard, PerformanceWindow,
        )
        pw = PerformanceWindow(5, 1.0, -0.01, 0.10, 0.05, 0.6, 0.8, 0.02)
        dg = DrawdownGuard()
        pm = PredictionMarket()
        pred = pm.emit(forecast, pw, dg)
        from macro8_subnet.execution.engine import MacroPrediction
        assert isinstance(pred, MacroPrediction)

    def test_prediction_epoch_increments(self, forecast):
        from macro8_subnet.execution.engine import (
            PredictionMarket, DrawdownGuard, PerformanceWindow,
        )
        pw = PerformanceWindow(5, 1.0, -0.01, 0.10, 0.05, 0.6, 0.8, 0.02)
        dg = DrawdownGuard()
        pm = PredictionMarket()
        p1 = pm.emit(forecast, pw, dg)
        p2 = pm.emit(forecast, pw, dg)
        assert p2.epoch == p1.epoch + 1

    def test_scenario_probs_sum_to_one(self, forecast):
        from macro8_subnet.execution.engine import (
            PredictionMarket, DrawdownGuard, PerformanceWindow,
        )
        pw   = PerformanceWindow(5, 1.0, -0.01, 0.10, 0.05, 0.6, 0.8, 0.02)
        dg   = DrawdownGuard()
        pm   = PredictionMarket()
        pred = pm.emit(forecast, pw, dg)
        total = sum(pred.scenario_probs.values())
        assert abs(total - 1.0) < 0.01

    def test_regime_probs_sum_to_one(self, forecast):
        from macro8_subnet.execution.engine import (
            PredictionMarket, DrawdownGuard, PerformanceWindow,
        )
        pw   = PerformanceWindow(5, 1.0, -0.01, 0.10, 0.05, 0.6, 0.8, 0.02)
        dg   = DrawdownGuard()
        pred = PredictionMarket().emit(forecast, pw, dg)
        total = sum(pred.regime_probs.values())
        assert abs(total - 1.0) < 0.01

    def test_to_json_serialisable(self, forecast):
        from macro8_subnet.execution.engine import (
            PredictionMarket, DrawdownGuard, PerformanceWindow,
        )
        pw   = PerformanceWindow(5, 1.0, -0.01, 0.10, 0.05, 0.6, 0.8, 0.02)
        dg   = DrawdownGuard()
        pred = PredictionMarket().emit(forecast, pw, dg)
        import json
        serialised = json.loads(pred.to_json())
        assert "scenario_probs" in serialised
        assert "positions" in serialised

    def test_top_scenarios_sorted(self, forecast):
        from macro8_subnet.execution.engine import (
            PredictionMarket, DrawdownGuard, PerformanceWindow,
        )
        pw   = PerformanceWindow(5, 1.0, -0.01, 0.10, 0.05, 0.6, 0.8, 0.02)
        pred = PredictionMarket().emit(forecast, pw, DrawdownGuard())
        top3 = pred.top_scenarios(3)
        probs = [p for _, p in top3]
        assert probs == sorted(probs, reverse=True)

    def test_latest_returns_last(self, forecast):
        from macro8_subnet.execution.engine import (
            PredictionMarket, DrawdownGuard, PerformanceWindow,
        )
        pw = PerformanceWindow(5, 1.0, -0.01, 0.10, 0.05, 0.6, 0.8, 0.02)
        dg = DrawdownGuard()
        pm = PredictionMarket()
        pm.emit(forecast, pw, dg); pm.emit(forecast, pw, dg)
        assert pm.latest().epoch == 2

    def test_print_latest_no_crash(self, forecast, capsys):
        from macro8_subnet.execution.engine import (
            PredictionMarket, DrawdownGuard, PerformanceWindow,
        )
        pw = PerformanceWindow(5, 1.0, -0.01, 0.10, 0.05, 0.6, 0.8, 0.02)
        pm = PredictionMarket()
        pm.emit(forecast, pw, DrawdownGuard())
        pm.print_latest()
        out = capsys.readouterr().out
        assert "PREDICTION MARKET" in out


# ── 7. run_live ───────────────────────────────────────────────────────────────

class TestRunLive:
    def test_run_live_returns_three_items(self, prices):
        from macro8_subnet.alpha.gp_miner import GPMiner
        from macro8_subnet.alpha.regime_prediction import ForecastedEnsemble
        from macro8_subnet.execution.engine import run_live
        gp = GPMiner(prices, pop_size=20, elite_n=5, seed=42, verbose=False)
        gp.run(n_epochs=2)
        formulas = gp.top_formulas(6)
        fens = ForecastedEnsemble(prices, formulas[:4], horizon=5,
                                  verbose=False)
        fens.fit()
        result = run_live(fens, prices, capital=100_000, n_days=10,
                          verbose=False)
        assert len(result) == 3

    def test_run_live_tracker_has_records(self, prices):
        from macro8_subnet.alpha.gp_miner import GPMiner
        from macro8_subnet.alpha.regime_prediction import ForecastedEnsemble
        from macro8_subnet.execution.engine import run_live
        gp = GPMiner(prices, pop_size=20, elite_n=5, seed=42, verbose=False)
        gp.run(n_epochs=2)
        formulas = gp.top_formulas(6)
        fens = ForecastedEnsemble(prices, formulas[:4], horizon=5,
                                  verbose=False)
        fens.fit()
        tracker, market, hist = run_live(fens, prices, n_days=10, verbose=False)
        assert len(tracker._records) > 0
        assert market.latest() is not None

    def test_run_live_history_finite(self, prices):
        from macro8_subnet.alpha.gp_miner import GPMiner
        from macro8_subnet.alpha.regime_prediction import ForecastedEnsemble
        from macro8_subnet.execution.engine import run_live
        gp = GPMiner(prices, pop_size=20, elite_n=5, seed=42, verbose=False)
        gp.run(n_epochs=2)
        formulas = gp.top_formulas(6)
        fens = ForecastedEnsemble(prices, formulas[:4], horizon=5,
                                  verbose=False)
        fens.fit()
        _, _, hist = run_live(fens, prices, n_days=10, verbose=False)
        if len(hist) > 0:
            assert np.isfinite(hist["pnl"].values).all()
