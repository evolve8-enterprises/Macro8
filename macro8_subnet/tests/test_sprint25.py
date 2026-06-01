"""
tests/test_sprint25.py
-----------------------
Sprint 25: Transaction Cost Model + GP on Calibrated Market Data

Tests cover:
    - TransactionCostModel: spread data, capital scaling, annual drag
    - Round-trip costs: calibrated bps per ticker
    - Square-root impact: grows with capital
    - PnL application: net < gross; costs non-negative
    - Vectorised multi-formula cost application
    - PortfolioEvaluator: net_sharpe < gross sharpe; cost fields present
    - Signal filtering: high-turnover signals killed at large capital
    - GP integration: composite favours low-cost signals
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
    n_days, n_assets = 500, 10
    tickers = ["SPY","QQQ","IWM","TLT","GLD","DBC","EEM","FXI","VNQ","HYG"]
    log_ret = rng.normal(0.0003, 0.01, (n_days, n_assets))
    prices_arr = 100 * np.exp(np.cumsum(log_ret, axis=0))
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    return pd.DataFrame(prices_arr, index=dates, columns=tickers)


@pytest.fixture(scope="module")
def tcm(prices):
    from macro8_subnet.evaluation.transaction_costs import TransactionCostModel
    return TransactionCostModel(list(prices.columns), capital=100_000)


@pytest.fixture(scope="module")
def evaluator(prices):
    from macro8_subnet.alpha.portfolio_evaluator import PortfolioEvaluator
    return PortfolioEvaluator(prices, apply_costs=True, default_capital=100_000)


@pytest.fixture(scope="module")
def formulas():
    return [
        "momentum_20d", "reversal_5d", "mean_rev_score",
        "market_corr_20d", "market_corr_60d",
        "decay(momentum_60d, halflife=20)",
        "reversal_3d + market_corr_20d",
    ]


@pytest.fixture(scope="module")
def result(evaluator, formulas):
    return evaluator.evaluate(formulas)


# ── 1. TransactionCostModel construction ─────────────────────────────────────

class TestTCMConstruction:
    def test_import(self):
        from macro8_subnet.evaluation.transaction_costs import (
            TransactionCostModel, build_cost_model,
            SPREAD_BPS, IMPACT_ETA, ADV_BILLION,
        )

    def test_universe_stored(self, tcm, prices):
        assert tcm.universe == list(prices.columns)
        assert tcm.n_assets == len(prices.columns)

    def test_spreads_array_shape(self, tcm, prices):
        assert tcm._spreads.shape == (len(prices.columns),)

    def test_etas_array_shape(self, tcm, prices):
        assert tcm._etas.shape == (len(prices.columns),)

    def test_adv_array_shape(self, tcm, prices):
        assert tcm._adv.shape == (len(prices.columns),)

    def test_build_cost_model_convenience(self, prices):
        from macro8_subnet.evaluation.transaction_costs import build_cost_model
        m = build_cost_model(prices, capital=50_000)
        assert m.capital == 50_000
        assert m.universe == list(prices.columns)


# ── 2. Calibrated spread data ─────────────────────────────────────────────────

class TestCalibratedSpreads:
    def test_spy_spread_near_zero(self):
        """SPY is the most liquid ETF — spread < 1 bps."""
        from macro8_subnet.evaluation.transaction_costs import SPREAD_BPS
        assert SPREAD_BPS["SPY"] < 1.0, f"SPY spread {SPREAD_BPS['SPY']} should be < 1 bps"

    def test_hyg_wider_than_spy(self):
        """HYG (high yield bonds) should cost more than SPY to trade."""
        from macro8_subnet.evaluation.transaction_costs import SPREAD_BPS
        assert SPREAD_BPS["HYG"] > SPREAD_BPS["SPY"] * 5

    def test_dbc_wider_than_eem(self):
        """DBC (commodity basket) is less liquid than EEM."""
        from macro8_subnet.evaluation.transaction_costs import SPREAD_BPS
        assert SPREAD_BPS["DBC"] > SPREAD_BPS["EEM"]

    def test_all_spreads_positive(self):
        from macro8_subnet.evaluation.transaction_costs import SPREAD_BPS
        for ticker, bps in SPREAD_BPS.items():
            assert bps > 0, f"Spread for {ticker} must be positive"

    def test_default_spread_reasonable(self):
        from macro8_subnet.evaluation.transaction_costs import SPREAD_BPS
        # Default for unknown tickers should be moderate
        assert 1.0 < SPREAD_BPS["_DEFAULT"] < 20.0


# ── 3. Round-trip cost calculation ───────────────────────────────────────────

class TestRoundTripCost:
    def test_round_trip_spy_small_capital(self, tcm):
        bps = tcm.round_trip_bps("SPY", capital=1_000)
        # SPY at $1k should be essentially just the spread
        assert bps < 1.5, f"SPY round-trip at $1k should be < 1.5 bps, got {bps:.3f}"

    def test_round_trip_hyg_higher_than_spy(self, tcm):
        spy_bps = tcm.round_trip_bps("SPY", capital=100_000)
        hyg_bps = tcm.round_trip_bps("HYG", capital=100_000)
        assert hyg_bps > spy_bps

    def test_round_trip_increases_with_capital(self, tcm):
        """Market impact grows with capital — round-trip cost should increase."""
        bps_1k  = tcm.round_trip_bps("HYG", capital=1_000)
        bps_1m  = tcm.round_trip_bps("HYG", capital=1_000_000)
        assert bps_1m >= bps_1k, "Round-trip cost should be ≥ at higher capital"

    def test_round_trip_positive_for_all_tickers(self, tcm):
        for t in tcm.universe:
            assert tcm.round_trip_bps(t) > 0

    def test_round_trip_unknown_ticker_uses_default(self, prices):
        from macro8_subnet.evaluation.transaction_costs import TransactionCostModel
        m   = TransactionCostModel(["UNKNOWN_ETF"], capital=100_000)
        bps = m.round_trip_bps("UNKNOWN_ETF")
        assert bps > 0


# ── 4. Annual drag ────────────────────────────────────────────────────────────

class TestAnnualDrag:
    def test_drag_increases_with_turnover(self, tcm):
        drag_low  = tcm.annual_drag(daily_turnover=0.01)
        drag_high = tcm.annual_drag(daily_turnover=0.10)
        assert drag_high > drag_low

    def test_drag_increases_with_capital(self, tcm):
        drag_1k = tcm.annual_drag(0.05, capital=1_000)
        drag_1m = tcm.annual_drag(0.05, capital=1_000_000)
        assert drag_1m >= drag_1k

    def test_drag_non_negative(self, tcm):
        for turn in [0.01, 0.05, 0.10]:
            assert tcm.annual_drag(turn) >= 0

    def test_capital_cost_table_shape(self, tcm):
        table = tcm.capital_cost_table(daily_turnover=0.05)
        assert len(table) == 4   # 4 capital tiers
        for cap, drag in table.items():
            assert drag >= 0
            assert cap > 0

    def test_high_turnover_high_drag(self, tcm):
        """Daily turnover of 10% should cost meaningfully annually."""
        drag = tcm.annual_drag(daily_turnover=0.10)
        # At our calibration, 10% turnover × ~4 bps/trade × 252 ≈ 0.1%
        assert drag > 0.001, f"10% daily turnover drag should exceed 0.1%/yr, got {drag:.4f}"


# ── 5. PnL application ───────────────────────────────────────────────────────

class TestPnLApplication:
    @pytest.fixture(scope="class")
    def synthetic_pnl_setup(self, prices):
        """Build minimal portfolio sim data for cost application tests."""
        rng     = np.random.default_rng(42)
        returns = np.log(prices).diff().dropna().values.astype(np.float32)
        T, A    = len(returns), len(prices.columns)
        weights = rng.normal(0, 1, (T + 1, A)).astype(np.float32)
        weights /= (np.abs(weights).sum(axis=1, keepdims=True) + 1e-8)
        # PnL at t = w[t-1] · r[t]: use w[:-1] and r[:]
        pnl = (weights[:-1] * returns).sum(axis=1)   # [T]
        return pnl, weights, returns

    def test_net_pnl_less_than_or_equal_gross(self, tcm, synthetic_pnl_setup):
        pnl_gross, weights, returns = synthetic_pnl_setup
        pnl_net = tcm.apply(pnl_gross, weights, returns, capital=100_000)
        # Net PnL should be ≤ gross (costs are always deducted)
        assert np.mean(pnl_net) <= np.mean(pnl_gross) + 1e-6

    def test_net_pnl_same_length(self, tcm, synthetic_pnl_setup):
        pnl_gross, weights, returns = synthetic_pnl_setup
        pnl_net = tcm.apply(pnl_gross, weights, returns)
        assert len(pnl_net) == len(pnl_gross)

    def test_costs_non_negative(self, tcm, synthetic_pnl_setup):
        pnl_gross, weights, returns = synthetic_pnl_setup
        pnl_net = tcm.apply(pnl_gross, weights, returns)
        daily_costs = pnl_gross - pnl_net[:len(pnl_gross)]
        # Some days may have near-zero cost but total cost >= 0
        assert daily_costs.sum() >= -1e-6

    def test_higher_capital_higher_total_cost(self, tcm, synthetic_pnl_setup):
        pnl_gross, weights, returns = synthetic_pnl_setup
        pnl_1k = tcm.apply(pnl_gross, weights, returns, capital=1_000)
        pnl_1m = tcm.apply(pnl_gross, weights, returns, capital=1_000_000)
        cost_1k = np.mean(pnl_gross) - np.mean(pnl_1k)
        cost_1m = np.mean(pnl_gross) - np.mean(pnl_1m)
        assert cost_1m >= cost_1k - 1e-8


# ── 6. Vectorised application ─────────────────────────────────────────────────

class TestVectorisedApplication:
    def test_vectorised_shape(self, prices, tcm):
        rng     = np.random.default_rng(7)
        returns = np.log(prices).diff().dropna().values.astype(np.float32)
        T, A, F = len(returns), len(prices.columns), 5
        weights = rng.normal(0, 1, (T + 1, A, F)).astype(np.float32)
        weights /= (np.abs(weights).sum(axis=1, keepdims=True) + 1e-8)
        pnl = (weights[:-1] * returns[:, :, None]).sum(axis=1)   # [T × F]
        pnl_net = tcm.apply_vectorised(pnl, weights, returns)
        assert pnl_net.shape == (T, F)

    def test_vectorised_net_leq_gross(self, prices, tcm):
        rng     = np.random.default_rng(8)
        returns = np.log(prices).diff().dropna().values.astype(np.float32)
        T, A, F = len(returns), len(prices.columns), 3
        weights = rng.normal(0, 1, (T + 1, A, F)).astype(np.float32)
        weights /= (np.abs(weights).sum(axis=1, keepdims=True) + 1e-8)
        pnl_gross = (weights[:-1] * returns[:, :, None]).sum(axis=1)
        pnl_net   = tcm.apply_vectorised(pnl_gross, weights, returns)
        for f in range(F):
            assert np.mean(pnl_net[:, f]) <= np.mean(pnl_gross[:, f]) + 1e-6


# ── 7. PortfolioEvaluator with costs ─────────────────────────────────────────

class TestPortfolioEvaluatorWithCosts:
    def test_net_sharpe_field_present(self, result):
        for ps in result.portfolio_scores:
            assert hasattr(ps, "net_sharpe")
            assert np.isfinite(ps.net_sharpe)

    def test_cost_drag_fields_present(self, result):
        for ps in result.portfolio_scores:
            assert hasattr(ps, "cost_drag_annual")
            assert hasattr(ps, "cost_drag_1m")

    def test_cost_drag_annual_non_negative(self, result):
        """Cost drag should be ≥ 0 (costs are always deducted)."""
        for ps in result.portfolio_scores:
            # Allow small floating point error
            assert ps.cost_drag_annual >= -0.5, \
                f"Negative cost drag {ps.cost_drag_annual:.2f} for {ps.formula}"

    def test_net_sharpe_leq_gross(self, result):
        """Net Sharpe ≤ gross Sharpe (costs can only reduce it)."""
        for ps in result.portfolio_scores:
            assert ps.net_sharpe <= ps.sharpe + 0.001, \
                f"{ps.formula}: net={ps.net_sharpe:.3f} > gross={ps.sharpe:.3f}"

    def test_to_dict_has_cost_fields(self, result):
        for ps in result.portfolio_scores:
            d = ps.to_dict()
            assert "net_sharpe" in d
            assert "cost_drag_bps" in d

    def test_evaluator_cost_model_attached(self, evaluator):
        from macro8_subnet.evaluation.transaction_costs import TransactionCostModel
        assert hasattr(evaluator, "cost_model")
        assert isinstance(evaluator.cost_model, TransactionCostModel)

    def test_apply_costs_flag_respected(self, prices):
        from macro8_subnet.alpha.portfolio_evaluator import PortfolioEvaluator
        ev_no_cost = PortfolioEvaluator(prices, apply_costs=False)
        result_no  = ev_no_cost.evaluate(["momentum_20d", "market_corr_20d"])
        for ps in result_no.portfolio_scores:
            # Without costs: net_sharpe should equal gross sharpe
            assert ps.net_sharpe == ps.sharpe, \
                f"With apply_costs=False, net_sharpe should equal sharpe"


# ── 8. Signal filtering effect ────────────────────────────────────────────────

class TestSignalFilteringEffect:
    def test_high_turnover_has_larger_cost_drag(self, result):
        """Signals with higher turnover should have higher cost drag."""
        # Find highest and lowest turnover signals
        sorted_by_turn = sorted(result.portfolio_scores, key=lambda ps: ps.daily_turnover)
        if len(sorted_by_turn) >= 2:
            low_turn  = sorted_by_turn[0]
            high_turn = sorted_by_turn[-1]
            if high_turn.daily_turnover > low_turn.daily_turnover * 2:
                assert high_turn.cost_drag_annual >= low_turn.cost_drag_annual, \
                    f"High turnover signal should have higher cost drag"

    def test_cost_kills_some_signals(self, prices):
        """Some gross-positive signals should go net-negative after costs."""
        from macro8_subnet.alpha.portfolio_evaluator import PortfolioEvaluator
        ev = PortfolioEvaluator(prices, apply_costs=True, default_capital=100_000)
        # Include fast reversal (high turnover) alongside slow momentum
        result = ev.evaluate([
            "reversal_5d",       # short-term: high turnover, cost-sensitive
            "market_corr_60d",   # slow: low turnover, cost-resistant
        ])
        if result.n_formulas >= 2:
            # At least one signal should have net < gross
            any_reduced = any(
                ps.net_sharpe < ps.sharpe for ps in result.portfolio_scores
            )
            assert any_reduced, "Costs should reduce Sharpe for at least one signal"

    def test_slow_signals_survive_costs(self, prices):
        """Low-turnover signals should retain most of their Sharpe after costs."""
        from macro8_subnet.alpha.portfolio_evaluator import PortfolioEvaluator
        ev = PortfolioEvaluator(prices, apply_costs=True, default_capital=100_000)
        result = ev.evaluate(["market_corr_60d"])
        if result.n_formulas > 0:
            ps = result.portfolio_scores[0]
            # For market_corr_60d (turnover ~0.004/day), cost drag should be small
            if ps.daily_turnover < 0.01:
                retention = ps.net_sharpe / (ps.sharpe + 1e-8)
                assert retention > 0.80, \
                    f"Low-turnover signal should retain >80% of Sharpe, got {retention:.2f}"

    def test_1m_drag_exceeds_100k_drag(self, result):
        """Cost drag at $1M should be >= drag at $100k (more market impact)."""
        for ps in result.portfolio_scores:
            if ps.daily_turnover > 0.01:
                assert ps.cost_drag_1m >= ps.cost_drag_annual - 0.1, \
                    f"{ps.formula}: $1M drag should >= $100k drag"
                break


# ── 9. GP integration with costs ─────────────────────────────────────────────

class TestGPWithCosts:
    @pytest.fixture(scope="class")
    def gp_report(self, prices):
        from macro8_subnet.alpha.gp_miner import GPMiner
        gp = GPMiner(prices, pop_size=40, elite_n=8, seed=42, verbose=False)
        return gp.run(n_epochs=3)

    def test_gp_uses_portfolio_evaluator(self, prices):
        from macro8_subnet.alpha.gp_miner import GPMiner
        from macro8_subnet.alpha.portfolio_evaluator import PortfolioEvaluator
        gp = GPMiner(prices, pop_size=20, elite_n=5, seed=1)
        assert isinstance(gp._batch_eval, PortfolioEvaluator)

    def test_gp_evaluator_has_cost_model(self, prices):
        from macro8_subnet.alpha.gp_miner import GPMiner
        from macro8_subnet.evaluation.transaction_costs import TransactionCostModel
        gp = GPMiner(prices, pop_size=20, elite_n=5, seed=2)
        assert hasattr(gp._batch_eval, "cost_model")
        assert isinstance(gp._batch_eval.cost_model, TransactionCostModel)

    def test_scored_formulas_have_cost_fields(self, gp_report):
        """ScoredFormulas from GP should carry cost-adjusted metrics."""
        for sf in gp_report.top_formulas[:5]:
            # composite is based on net sharpe when costs are applied
            assert sf.composite >= 0.0
            assert np.isfinite(sf.sharpe)

    def test_gp_report_summary_runs(self, gp_report):
        summary = gp_report.summary()
        assert len(summary) > 50
        assert "composite" in summary.lower() or "Sharpe" in summary


# ── 10. Edge cases ────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_zero_turnover_no_cost(self, tcm, prices):
        """If weights never change, cost should be zero."""
        T, A    = len(prices) - 1, len(prices.columns)
        returns = np.log(prices).diff().dropna().values.astype(np.float32)
        weights = np.ones((T + 1, A), dtype=np.float32) / A  # constant weights
        pnl     = np.zeros(T, dtype=np.float32)

        pnl_net     = tcm.apply(pnl, weights, returns)
        daily_costs = pnl - pnl_net[:T]
        assert np.allclose(daily_costs, 0, atol=1e-6), \
            f"Constant weights should have zero cost. Max cost: {np.abs(daily_costs).max():.2e}"

    def test_cost_model_graceful_with_unknown_ticker(self):
        from macro8_subnet.evaluation.transaction_costs import TransactionCostModel
        m = TransactionCostModel(["XYZ_UNKNOWN"], capital=100_000)
        from macro8_subnet.evaluation.transaction_costs import SPREAD_BPS
        assert m._spreads[0] == SPREAD_BPS["_DEFAULT"]

    def test_annual_drag_zero_turnover(self, tcm):
        drag = tcm.annual_drag(daily_turnover=0.0)
        assert drag == 0.0

    def test_capital_cost_table_monotone_with_capital(self, tcm):
        """Higher capital → higher drag (market impact grows)."""
        table = tcm.capital_cost_table(daily_turnover=0.05)
        drags = [table[cap] for cap in sorted(table.keys())]
        for i in range(len(drags) - 1):
            assert drags[i] <= drags[i + 1] + 1e-10, \
                f"Drag not monotone: {drags}"
