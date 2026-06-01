"""
execution/engine.py
--------------------
Decision Execution Engine — Sprint 30.

Completes the loop from research to live deployment:

    Research  (GP miner)
    Forecast  (ForecastedEnsemble)
    Constrain (PortfolioConstraints → ConstraintSolver)
    Execute   (TradeExecutor)
    Track     (LiveTracker)
    Publish   (PredictionMarket)
    Adapt     (DrawdownGuard, retrain signal)

Four components
---------------

1. PortfolioConstraints / ConstraintSolver
   Applies hard limits after position sizing:
       max_weight:         no single position exceeds this (default 0.40)
       max_sector_gross:   sector gross exposure cap (default 0.60)
       max_net_exposure:   |sum(positions)| ≤ limit (default 0.20, i.e. near market-neutral)
       min_weight:         drop positions smaller than this (avoids tiny round-trips)
       stress_delever:     automatically reduce max_weight when P(stress) is high

   DrawdownGuard (part of constraint layer):
       Monitors rolling equity curve and scales down position sizes if
       realised drawdown exceeds max_drawdown threshold.
       Scale = max(0.25, 1 − 0.5 × severity)
       Recovery: scale restores as drawdown recovers.

2. TradeExecutor
   Converts target positions to actual trade orders:
       TradeOrder:     ticker, direction (BUY/SELL), notional_weight, cost_estimate_bps
       ExecutionPlan:  full list of trades, total_turnover, estimated_cost
   
   Slippage model: spread/2 + market impact for each trade.
   Min trade size: filter trades smaller than min_trade (avoids churning).
   Rebalance timing: daily (default), weekly, or threshold-based.

3. LiveTracker
   Tracks every decision the system makes and its outcome:
       DailyRecord:   date, positions_held, trades, pnl, regime_predicted,
                      regime_actual, confidence, drawdown_scale
       PerformanceWindow: rolling stats (sharpe, max_dd, vol, turnover, hit_rate)
       retrain_signal(): returns True when performance degrades below threshold

   Feedback loop: compares regime_predicted to regime_actual 20 days later.
   Accuracy falls → confidence multiplier falls → positions scale down.

4. PredictionMarket
   Emits structured probabilistic forecasts as JSON-serialisable records.
   This is the external-facing prediction layer:
       MacroPrediction:  full snapshot of the system's current view
           timestamp, horizon_days, regime_probs, scenario_probs,
           policy_state, confidence, positions, performance_context

   Can emit to stdout, JSON file, or dict for downstream consumers.
   Designed for integration with prediction market protocols (Metaculus,
   Manifold, or bespoke on-chain markets via Bittensor).

Usage
-----
    from macro8_subnet.execution.engine import (
        PortfolioConstraints, ConstraintSolver, DrawdownGuard,
        TradeExecutor, LiveTracker, PredictionMarket, run_live,
    )

    # Build the execution engine
    constraints = PortfolioConstraints(max_weight=0.35, max_net_exposure=0.15)
    executor    = TradeExecutor(capital=100_000)
    tracker     = LiveTracker()
    market      = PredictionMarket()

    # Each day:
    forecast    = fens.forecast()                      # from ForecastedEnsemble
    constrained = ConstraintSolver(constraints).apply(
        forecast.positions,
        p_stress=forecast.regime_forecast.stress,
        scale=tracker.drawdown_guard.position_scale(),
    )
    plan = executor.compute_trades(constrained, current_holdings)
    tracker.update(date, constrained, plan, realized_pnl, forecast)
    market.emit(forecast, tracker.snapshot())

    # run_live(): the full loop in one call
    run_live(fens, prices, capital=100_000, n_days=252)
"""

from __future__ import annotations

import json
import sys
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_SUBNET = Path(__file__).resolve().parent.parent
_ROOT   = _SUBNET.parent
for p in [str(_SUBNET), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Sector map for the default 10-ticker universe ─────────────────────────────
DEFAULT_SECTOR_MAP: dict[str, str] = {
    "SPY": "equity_us",   "QQQ": "equity_us",   "IWM": "equity_us",
    "TLT": "rates",       "HYG": "credit",
    "GLD": "commodities", "DBC": "commodities",
    "EEM": "equity_em",   "FXI": "equity_em",
    "VNQ": "real_estate",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. PORTFOLIO CONSTRAINTS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PortfolioConstraints:
    """
    Hard limits applied to position sizes after signal generation.

    Parameters
    ----------
    max_weight:         float — maximum absolute weight per ticker (default 0.40).
    max_sector_gross:   float — max total gross exposure per sector (default 0.60).
    max_net_exposure:   float — |sum of all weights| ≤ this (default 0.20).
    min_weight:         float — positions smaller than this are dropped (default 0.01).
    sector_map:         dict  — {ticker: sector}. None = use DEFAULT_SECTOR_MAP.
    stress_delever:     float — fraction by which max_weight shrinks per unit
                               P(stress) above 0.30 (default 0.25).
                               E.g. stress_delever=0.25, P(stress)=0.8 →
                               max_weight multiplied by (1 − 0.25 × (0.8−0.3)/0.7) = 0.82×.
    """
    max_weight:       float               = 0.40
    max_sector_gross: float               = 0.60
    max_net_exposure: float               = 0.20
    min_weight:       float               = 0.01
    sector_map:       Optional[dict]      = None
    stress_delever:   float               = 0.25   # gradual deleverage
    kill_switch:      float               = 0.65   # hard gross-exposure cap when
                                                    # p_stress exceeds this threshold
    kill_switch_exposure: float           = 0.30   # gross exposure cap in kill mode

    def __post_init__(self):
        if self.sector_map is None:
            self.sector_map = DEFAULT_SECTOR_MAP

    def effective_max_weight(self, p_stress: float = 0.0) -> float:
        """Stress-adjusted max weight."""
        excess_stress = max(0.0, p_stress - 0.30) / 0.70
        return self.max_weight * (1 - self.stress_delever * excess_stress)


class ConstraintSolver:
    """
    Applies PortfolioConstraints to a raw position dict.

    Steps (applied in order):
        1. Stress-adjusted max_weight clip
        2. Min-weight filter
        3. Sector gross exposure cap
        4. Net exposure control
        5. Re-normalise L1 to 1 (or to scale if scale < 1.0)
    """

    def __init__(self, constraints: PortfolioConstraints = None):
        self.c = constraints or PortfolioConstraints()

    def apply(
        self,
        positions:  dict[str, float],
        p_stress:   float = 0.0,
        scale:      float = 1.0,      # DrawdownGuard scale
    ) -> dict[str, float]:
        """
        Apply all constraints to raw positions.

        Args:
            positions: {ticker: weight} from signal generator.
            p_stress:  Current P(stress) from RegimeForecast.
            scale:     DrawdownGuard position scale (1.0 = no scaling).

        Returns:
            Constrained {ticker: weight} with L1 norm = scale.
        """
        if not positions:
            return {}

        c = self.c
        max_w = c.effective_max_weight(p_stress)

        # Step 1: clip per-ticker weight
        out = {t: float(np.clip(w, -max_w, max_w)) for t, w in positions.items()}

        # Step 2: drop below min_weight
        out = {t: w for t, w in out.items() if abs(w) >= c.min_weight}
        if not out:
            return {}

        # Step 3: sector gross cap
        if c.sector_map:
            out = self._apply_sector_cap(out, c.sector_map, c.max_sector_gross)

        # Step 4: net exposure control
        net = sum(out.values())
        if abs(net) > c.max_net_exposure:
            out = self._reduce_net(out, net, c.max_net_exposure)

        # Step 5: stress kill switch — override the scale target.
        # When p_stress > kill_switch threshold, the regime layer acts as a
        # RISK CONTROLLER: it reduces target gross exposure from `scale` to
        # kill_switch_exposure, regardless of what the signal engine wants.
        # Design: kill switch overrides scale (not clip-then-renormalise,
        # which would undo the cap in step 6).
        effective_scale = scale
        if p_stress > c.kill_switch:
            # Linear ramp: at kill_switch → kill_switch_exposure * scale
            #              at p_stress=1 → kill_switch_exposure * 0.5 * scale
            severity        = (p_stress - c.kill_switch) / (1.0 - c.kill_switch + 1e-8)
            exposure_target = c.kill_switch_exposure * (1 - 0.5 * severity)
            effective_scale = min(scale, exposure_target)

        # Step 6: re-normalise to effective_scale (respects kill switch)
        l1 = sum(abs(w) for w in out.values())
        if l1 > 1e-6:
            out = {t: w / l1 * effective_scale for t, w in out.items()}

        return out

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _apply_sector_cap(
        self,
        positions: dict,
        sector_map: dict,
        cap: float,
    ) -> dict:
        out = dict(positions)
        sectors = set(sector_map.values())
        for sector in sectors:
            tickers     = [t for t in out if sector_map.get(t) == sector]
            sector_gross = sum(abs(out[t]) for t in tickers)
            if sector_gross > cap:
                scale = cap / sector_gross
                for t in tickers:
                    out[t] *= scale
        return out

    def _reduce_net(
        self,
        positions: dict,
        current_net: float,
        target_net: float,
    ) -> dict:
        out     = dict(positions)
        excess  = abs(current_net) - target_net
        sign    = float(np.sign(current_net))
        # Find all positions contributing to the excess direction
        side    = [t for t, w in out.items() if float(np.sign(w)) == sign]
        side_total = sum(abs(out[t]) for t in side)
        if side_total > 1e-6:
            reduction = excess / side_total
            for t in side:
                out[t] *= (1.0 - reduction)
        return out


class DrawdownGuard:
    """
    Monitors realised drawdown and scales down positions when a threshold
    is breached.

    Behaviour:
        - Tracks a rolling window of daily PnL
        - Computes peak-to-trough drawdown within the window
        - If drawdown < max_drawdown: scale = max(floor, 1 − severity × 0.5)
        - Recovery: scale recovers proportionally as drawdown recovers
        - Hard floor: never below position_floor (default 0.25)

    Parameters
    ----------
    max_drawdown:   float — trigger threshold (default −0.05 = −5%).
    lookback:       int   — rolling window in days (default 20).
    position_floor: float — minimum scale regardless of drawdown (default 0.25).
    """

    def __init__(
        self,
        max_drawdown:   float = -0.05,
        lookback:       int   = 20,
        position_floor: float = 0.25,
    ):
        self.max_drawdown   = max_drawdown
        self.lookback       = lookback
        self.position_floor = position_floor
        self._pnl:          deque[float] = deque(maxlen=lookback)

    def update(self, daily_pnl: float) -> None:
        """Record today's portfolio PnL."""
        self._pnl.append(daily_pnl)

    def position_scale(self) -> float:
        """
        Compute current position scale factor ∈ [floor, 1.0].

        Returns 1.0 (no scaling) when drawdown is within limits.
        Returns < 1.0 when drawdown exceeds threshold.
        """
        if len(self._pnl) < 2:
            return 1.0
        pnl    = np.array(self._pnl)
        cum    = np.cumsum(pnl)
        peak   = np.maximum.accumulate(cum)
        dd     = (cum - peak).min()

        if dd >= self.max_drawdown:
            return 1.0

        severity = abs(dd) / abs(self.max_drawdown)
        scale    = 1.0 - 0.5 * min(severity, 2.0)
        return max(self.position_floor, scale)

    @property
    def current_drawdown(self) -> float:
        """Current peak-to-trough drawdown in the lookback window."""
        if len(self._pnl) < 2:
            return 0.0
        pnl  = np.array(self._pnl)
        cum  = np.cumsum(pnl)
        peak = np.maximum.accumulate(cum)
        return float((cum - peak).min())

    @property
    def cumulative_pnl(self) -> float:
        return float(sum(self._pnl))


# ══════════════════════════════════════════════════════════════════════════════
# 2. TRADE EXECUTOR
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TradeOrder:
    """A single trade instruction."""
    ticker:           str
    direction:        str       # "BUY" | "SELL"
    weight_change:    float     # signed weight delta (positive = buy)
    notional:         float     # dollar notional (weight × capital)
    cost_estimate_bps: float    # estimated round-trip cost in basis points

    def __repr__(self) -> str:
        return (
            f"TradeOrder({self.direction} {self.ticker} "
            f"Δw={self.weight_change:+.4f} "
            f"${self.notional:,.0f} "
            f"cost≈{self.cost_estimate_bps:.1f}bps)"
        )


@dataclass
class ExecutionPlan:
    """Complete list of trades for one rebalance."""
    date:               pd.Timestamp
    orders:             list[TradeOrder]
    total_turnover:     float     # sum |weight_change|
    estimated_cost_bps: float     # total cost estimate
    n_buys:             int
    n_sells:            int

    def summary(self) -> str:
        return (
            f"ExecutionPlan [{self.date.date()}]: "
            f"{len(self.orders)} orders "
            f"(buys={self.n_buys} sells={self.n_sells}) | "
            f"turnover={self.total_turnover:.4f} | "
            f"cost≈{self.estimated_cost_bps:.1f}bps"
        )

    def print(self) -> None:
        print(f"\n  {self.summary()}")
        for o in sorted(self.orders, key=lambda x: abs(x.weight_change), reverse=True):
            print(f"    {o}")


class TradeExecutor:
    """
    Converts target positions → trade orders with cost estimation.

    Cost model: spread/2 + square-root impact (from TransactionCostModel).

    Parameters
    ----------
    capital:    float — portfolio dollar size (for notional calculation).
    min_trade:  float — minimum weight change to generate an order (default 0.005).
    """

    # Default spread assumptions (bps) per ticker if TransactionCostModel unavailable
    _DEFAULT_SPREADS = {
        "SPY": 0.32, "QQQ": 0.50, "IWM": 0.60, "TLT": 4.00, "HYG": 5.50,
        "GLD": 5.00, "DBC": 8.00, "EEM": 2.50, "FXI": 3.50, "VNQ": 3.00,
    }

    def __init__(self, capital: float = 100_000, min_trade: float = 0.005):
        self.capital   = capital
        self.min_trade = min_trade

    def compute_trades(
        self,
        target_positions:  dict[str, float],
        current_positions: dict[str, float],
        date:              Optional[pd.Timestamp] = None,
        p_stress:          float = 0.0,
    ) -> ExecutionPlan:
        """
        Compute the trades needed to move from current to target positions.

        Args:
            target_positions:  {ticker: weight} desired portfolio.
            current_positions: {ticker: weight} current holdings (default = flat).
            date:              Date for the execution plan.
            p_stress:          Probability of stress regime from RegimeForecast.
                               Used to widen spread estimates: in stress regimes
                               bid-ask spreads empirically widen 2-4x for ETFs.
                               Model: cost_bps *= (1 + 2 * p_stress).

        Returns:
            ExecutionPlan with individual TradeOrders.
        """
        date = date or pd.Timestamp.now()
        # Stress multiplier: 1× at p_stress=0, 3× at p_stress=1
        # Empirically, ETF spreads widen 2-4× during VIX spikes (Amihud 2002)
        stress_mult = 1.0 + 2.0 * float(np.clip(p_stress, 0.0, 1.0))

        all_tickers = set(list(target_positions.keys()) + list(current_positions.keys()))
        orders = []

        for ticker in all_tickers:
            target  = target_positions.get(ticker, 0.0)
            current = current_positions.get(ticker, 0.0)
            delta   = target - current

            if abs(delta) < self.min_trade:
                continue

            base_spread_bps = self._DEFAULT_SPREADS.get(ticker, 5.0)
            cost_bps        = (base_spread_bps / 2) * stress_mult  # one-way, stress-adjusted
            notional        = abs(delta) * self.capital
            direction       = "BUY" if delta > 0 else "SELL"

            orders.append(TradeOrder(
                ticker=ticker,
                direction=direction,
                weight_change=delta,
                notional=notional,
                cost_estimate_bps=cost_bps,
            ))

        total_turnover = sum(abs(o.weight_change) for o in orders)
        avg_cost_bps   = (
            sum(o.cost_estimate_bps * abs(o.weight_change) for o in orders)
            / (total_turnover + 1e-8)
        )

        return ExecutionPlan(
            date=date,
            orders=orders,
            total_turnover=total_turnover,
            estimated_cost_bps=avg_cost_bps,
            n_buys=sum(1 for o in orders if o.direction == "BUY"),
            n_sells=sum(1 for o in orders if o.direction == "SELL"),
        )

    def simulate_fill(
        self,
        plan:   ExecutionPlan,
        prices: pd.DataFrame,
        date:   pd.Timestamp,
    ) -> dict[str, float]:
        """
        Simulate trade fills with slippage.

        Returns actual filled positions after slippage (slightly worse than target).
        Slippage model: each trade gets a price 0.5 × spread worse than mid.
        """
        fills = {}
        for order in plan.orders:
            ticker = order.ticker
            if ticker not in prices.columns:
                fills[ticker] = order.weight_change
                continue
            # Slippage: spread/2 penalty on the weight (expressed as return drag)
            slippage_return = self._DEFAULT_SPREADS.get(ticker, 5.0) / 2 / 10_000
            if order.direction == "BUY":
                fills[ticker] = order.weight_change * (1 - slippage_return)
            else:
                fills[ticker] = order.weight_change * (1 + slippage_return)
        return fills


# ══════════════════════════════════════════════════════════════════════════════
# 3. LIVE TRACKER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DailyRecord:
    """A single day's decision and outcome record."""
    date:              pd.Timestamp
    positions:         dict[str, float]
    trades:            ExecutionPlan
    pnl:               float           # realised daily log-return PnL
    pnl_net:           float           # PnL after estimated costs
    regime_predicted:  str
    regime_actual:     str             # known after N days (filled in later)
    confidence:        float
    drawdown_scale:    float
    p_stress:          float


@dataclass
class PerformanceWindow:
    """Rolling performance statistics over a window of DailyRecords."""
    n_days:          int
    sharpe_ann:      float
    max_drawdown:    float
    ann_vol:         float
    daily_turnover:  float
    hit_rate:        float    # fraction of profitable days
    regime_accuracy: float    # fraction where regime_predicted == regime_actual
    cum_pnl:         float

    def summary(self) -> str:
        return (
            f"Sharpe={self.sharpe_ann:+.2f}  "
            f"MaxDD={self.max_drawdown:.3f}  "
            f"Vol={self.ann_vol:.1%}  "
            f"Turn={self.daily_turnover:.3f}  "
            f"Hit={self.hit_rate:.0%}  "
            f"RegAcc={self.regime_accuracy:.0%}"
        )


class LiveTracker:
    """
    Tracks every decision and outcome; drives the feedback loop.

    The feedback loop:
        1. Every day, record (positions, trades, PnL, regime_predicted).
        2. After N=5 days, fill in regime_actual and compute accuracy.
        3. If rolling regime_accuracy < accuracy_threshold:
               confidence_multiplier *= 0.9  (decay trust in predictions)
        4. If running_sharpe < sharpe_floor for 20 days:
               retrain_signal() returns True.
        5. After retraining, confidence_multiplier resets to 1.0.

    Parameters
    ----------
    accuracy_threshold: float — minimum regime prediction accuracy (default 0.55).
    sharpe_floor:       float — minimum rolling Sharpe before retrain (default −0.5).
    window:             int   — rolling performance window in days (default 20).
    regime_lag:         int   — days before actual regime is known (default 5).
    """

    def __init__(
        self,
        accuracy_threshold: float = 0.55,
        sharpe_floor:       float = -0.50,
        window:             int   = 20,
        regime_lag:         int   = 5,
    ):
        self.accuracy_threshold  = accuracy_threshold
        self.sharpe_floor        = sharpe_floor
        self.window              = window
        self.regime_lag          = regime_lag
        self.drawdown_guard      = DrawdownGuard()
        self._records:           list[DailyRecord]  = []
        self._pending_actuals:   dict[int, str]     = {}  # idx → actual regime
        self._confidence_mult:   float              = 1.0
        self._retrain_triggered: bool               = False

    def update(
        self,
        date:             pd.Timestamp,
        positions:        dict[str, float],
        plan:             ExecutionPlan,
        pnl:              float,
        forecast,         # ForecastResult from ForecastedEnsemble
    ) -> None:
        """
        Record one day's decision and outcome.

        Args:
            date:       Today's date.
            positions:  Held positions (after constraints).
            plan:       Execution plan (trades).
            pnl:        Today's realised PnL (log-return basis).
            forecast:   ForecastResult from ForecastedEnsemble.
        """
        pnl_net = pnl - plan.total_turnover * plan.estimated_cost_bps / 10_000
        self.drawdown_guard.update(pnl)

        record = DailyRecord(
            date=date,
            positions=dict(positions),
            trades=plan,
            pnl=pnl,
            pnl_net=pnl_net,
            regime_predicted=forecast.regime_forecast.most_likely,
            regime_actual="unknown",   # filled in after lag
            confidence=forecast.confidence,
            drawdown_scale=self.drawdown_guard.position_scale(),
            p_stress=forecast.regime_forecast.stress,
        )
        self._records.append(record)

        # Fill in actuals from lag-ago predictions
        lag_idx = len(self._records) - 1 - self.regime_lag
        if lag_idx >= 0 and self._records[lag_idx].regime_actual == "unknown":
            # Use the current regime as the "actual" for the lagged record
            self._records[lag_idx].regime_actual = forecast.regime_current

        # Update confidence multiplier based on rolling regime accuracy
        self._update_confidence()

    def fill_actual_regime(self, idx: int, actual: str) -> None:
        """Retroactively fill in the actual regime for a past record."""
        if 0 <= idx < len(self._records):
            self._records[idx].regime_actual = actual

    def snapshot(self) -> PerformanceWindow:
        """Compute rolling performance statistics over the last `window` days."""
        recent = [r for r in self._records[-self.window:]
                  if r.pnl is not None]
        if not recent:
            return PerformanceWindow(0, 0, 0, 0, 0, 0, 0, 0)

        pnl_arr   = np.array([r.pnl for r in recent])
        cum_pnl   = float(np.sum(pnl_arr))
        sharpe    = float(pnl_arr.mean() / (pnl_arr.std() + 1e-8) * np.sqrt(252))
        cum       = np.cumsum(pnl_arr)
        max_dd    = float((cum - np.maximum.accumulate(cum)).min())
        ann_vol   = float(pnl_arr.std() * np.sqrt(252))
        hit_rate  = float(np.mean(pnl_arr > 0))
        turnover  = float(np.mean([r.trades.total_turnover for r in recent]))

        # Regime accuracy on resolved records
        resolved = [r for r in self._records if r.regime_actual not in ("unknown", "")]
        reg_acc  = (float(np.mean([
            r.regime_predicted == r.regime_actual for r in resolved
        ])) if resolved else 0.5)

        return PerformanceWindow(
            n_days=len(recent),
            sharpe_ann=sharpe,
            max_drawdown=max_dd,
            ann_vol=ann_vol,
            daily_turnover=turnover,
            hit_rate=hit_rate,
            regime_accuracy=reg_acc,
            cum_pnl=cum_pnl,
        )

    def confidence_multiplier(self) -> float:
        """
        Returns the current confidence multiplier ∈ [0.25, 1.0].

        Falls when regime predictions are systematically wrong.
        Recovers slowly when accuracy improves.
        """
        return float(np.clip(self._confidence_mult, 0.25, 1.0))

    def retrain_signal(self) -> bool:
        """
        Returns True when conditions warrant retraining the GP/ensemble.

        Trigger: rolling Sharpe below floor for at least `window` days.
        After returning True once, resets (won't trigger again until performance
        degrades again).
        """
        if len(self._records) < self.window:
            return False
        pw = self.snapshot()
        if pw.sharpe_ann < self.sharpe_floor:
            if not self._retrain_triggered:
                self._retrain_triggered = True
                return True
        else:
            self._retrain_triggered = False
            self._confidence_mult    = min(1.0, self._confidence_mult * 1.05)
        return False

    def full_history(self) -> pd.DataFrame:
        """Return full record history as a DataFrame."""
        if not self._records:
            return pd.DataFrame()
        rows = []
        for r in self._records:
            rows.append({
                "date":             r.date,
                "pnl":              r.pnl,
                "pnl_net":          r.pnl_net,
                "cum_pnl":          None,
                "regime_predicted": r.regime_predicted,
                "regime_actual":    r.regime_actual,
                "confidence":       r.confidence,
                "drawdown_scale":   r.drawdown_scale,
                "p_stress":         r.p_stress,
                "turnover":         r.trades.total_turnover,
                "n_trades":         len(r.trades.orders),
            })
        df = pd.DataFrame(rows).set_index("date")
        df["cum_pnl"] = df["pnl"].cumsum()
        return df

    def print_summary(self) -> None:
        """Print current performance summary."""
        pw  = self.snapshot()
        n   = len(self._records)
        dd  = self.drawdown_guard.current_drawdown
        mul = self.confidence_multiplier()
        print(f"\n  ╔{'═'*65}╗")
        print(f"  ║  MACRO8 LIVE TRACKER — {n} days recorded{' '*max(0,24-len(str(n)))}║")
        print(f"  ╠{'═'*65}╣")
        print(f"  ║  Performance ({pw.n_days}d window): {pw.summary()}")
        print(f"  ║  Drawdown guard:   {dd:+.4f}  (scale={self.drawdown_guard.position_scale():.2f})")
        print(f"  ║  Confidence mult:  {mul:.2f}  (regime accuracy={pw.regime_accuracy:.0%})")
        print(f"  ╚{'═'*65}╝")

    def _update_confidence(self) -> None:
        """Update confidence multiplier based on rolling regime accuracy."""
        resolved = [r for r in self._records if r.regime_actual not in ("unknown", "")]
        if len(resolved) < 5:
            return
        recent_resolved = resolved[-10:]  # last 10 resolved records
        accuracy = np.mean([
            r.regime_predicted == r.regime_actual for r in recent_resolved
        ])
        if accuracy < self.accuracy_threshold:
            # Decay confidence
            self._confidence_mult *= 0.95
        else:
            # Slow recovery
            self._confidence_mult = min(1.0, self._confidence_mult * 1.02)


# ══════════════════════════════════════════════════════════════════════════════
# 4. PREDICTION MARKET
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MacroPrediction:
    """
    A complete prediction market record.

    This is the external-facing output of the system: a structured,
    serialisable snapshot of what the system believes about the future.

    Fields match the interface for prediction market protocols:
        - Metaculus-style: P(event) for specific macro events
        - Manifold-style: continuous probability over scenarios
        - Bittensor on-chain: JSON payload for validator scoring
    """
    timestamp:        str           # ISO 8601
    epoch:            int
    horizon_days:     int

    # Core forecast
    regime_current:   str
    regime_probs:     dict[str, float]   # {calm, normal, stress}
    regime_forecast:  str               # most likely next regime

    # Scenario probabilities (the prediction market)
    scenario_probs:   dict[str, float]  # 8 scenarios, sum to 1

    # Policy state
    policy_state:     dict[str, str]    # {indicator: rising/flat/falling}

    # Confidence
    confidence:       float
    confidence_level: str               # HIGH / MEDIUM / LOW

    # Positions (the actionable output)
    positions:        dict[str, float]
    active_formulas:  list[str]
    n_clusters:       int

    # Performance context (from LiveTracker)
    running_sharpe:   float
    current_drawdown: float
    drawdown_scale:   float
    regime_accuracy:  float

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def top_scenarios(self, n: int = 3) -> list[tuple[str, float]]:
        return sorted(self.scenario_probs.items(), key=lambda x: x[1], reverse=True)[:n]


class PredictionMarket:
    """
    Emits and tracks MacroPrediction records.

    Maintains a history of predictions for:
        - Scoring predictions against outcomes
        - Calibration analysis (are P=0.7 events right 70% of the time?)
        - Bittensor validator submission

    Parameters
    ----------
    max_history: int — predictions to retain in memory (default 252 = 1 year).
    """

    def __init__(self, max_history: int = 252):
        self.max_history = max_history
        self._history:   deque[MacroPrediction] = deque(maxlen=max_history)
        self._epoch      = 0

    def emit(
        self,
        forecast,           # ForecastResult from ForecastedEnsemble
        perf:               PerformanceWindow,
        drawdown_guard:     DrawdownGuard,
        confidence_mult:    float = 1.0,
    ) -> MacroPrediction:
        """
        Create and store a prediction from the current system state.

        Args:
            forecast:        ForecastResult from ForecastedEnsemble.forecast().
            perf:            PerformanceWindow from LiveTracker.snapshot().
            drawdown_guard:  DrawdownGuard from LiveTracker.
            confidence_mult: Feedback confidence multiplier.

        Returns:
            MacroPrediction — the structured prediction record.
        """
        self._epoch += 1
        rf = forecast.regime_forecast

        # Convert policy state to direction labels
        ps      = forecast.policy_state
        pol_dict = {
            "rates":     "rising" if ps.rate_rising()        else "falling" if ps.rate_falling()    else "flat",
            "inflation": "rising" if ps.inflation_rising()   else "flat",
            "liquidity": "tightening" if ps.credit_tightening() else "easing",
            "dollar":    "strong" if ps.dollar_strong()      else "weak",
            "breadth":   "broadening" if ps.breadth_broadening() else "narrowing",
        }

        confidence = float(forecast.confidence * confidence_mult)

        pred = MacroPrediction(
            timestamp=datetime.now().isoformat(),
            epoch=self._epoch,
            horizon_days=rf.horizon_days,
            regime_current=forecast.regime_current,
            regime_probs={"calm": rf.calm, "normal": rf.normal, "stress": rf.stress},
            regime_forecast=rf.most_likely,
            scenario_probs=forecast.scenario_probs,
            policy_state=pol_dict,
            confidence=round(confidence, 4),
            confidence_level="HIGH" if confidence > 0.70 else "MEDIUM" if confidence > 0.45 else "LOW",
            positions=forecast.positions,
            active_formulas=forecast.active_formulas,
            n_clusters=forecast.n_clusters,
            running_sharpe=round(perf.sharpe_ann, 3),
            current_drawdown=round(drawdown_guard.current_drawdown, 5),
            drawdown_scale=round(drawdown_guard.position_scale(), 3),
            regime_accuracy=round(perf.regime_accuracy, 3),
        )
        self._history.append(pred)
        return pred

    def latest(self) -> Optional[MacroPrediction]:
        """Most recent prediction."""
        return self._history[-1] if self._history else None

    def calibration_score(self) -> dict[str, float]:
        """
        Compute calibration score per scenario.

        For each scenario, measures whether the predicted probabilities
        match the historical resolution rate. Returns Brier scores
        (lower = better calibrated, 0 = perfect).

        Note: requires resolved predictions (actual outcomes known).
        Returns empty dict if insufficient history.
        """
        resolved = [p for p in self._history
                    if hasattr(p, "_resolved") and p._resolved]
        if len(resolved) < 10:
            return {}

        scores = {}
        for scenario in ALL_SCENARIOS:
            probs    = np.array([p.scenario_probs.get(scenario, 0) for p in resolved])
            outcomes = np.array([getattr(p, "_outcome", {}).get(scenario, 0)
                                 for p in resolved])
            scores[scenario] = float(np.mean((probs - outcomes) ** 2))
        return scores

    def print_latest(self) -> None:
        """Print the most recent prediction in human-readable format."""
        pred = self.latest()
        if pred is None:
            print("[PredictionMarket] No predictions yet")
            return
        print(f"\n  ╔{'═'*68}╗")
        print(f"  ║  MACRO8 PREDICTION MARKET  [{pred.timestamp[:19]}]{' '*10}║")
        print(f"  ╠{'═'*68}╣")
        print(f"  ║  Epoch: {pred.epoch}  |  Horizon: {pred.horizon_days}d  |  Confidence: {pred.confidence:.2f} [{pred.confidence_level}]")
        print(f"  ║  Current: {pred.regime_current:<10}  Forecast: {pred.regime_forecast:<10}  Scale: {pred.drawdown_scale:.2f}")
        print(f"  ║  P(calm)={pred.regime_probs['calm']:.2f}  P(normal)={pred.regime_probs['normal']:.2f}  P(stress)={pred.regime_probs['stress']:.2f}")
        print(f"  ║  Policy: {' '.join(f'{k}={v}' for k,v in pred.policy_state.items())}")
        print(f"  ╠{'═'*68}╣")
        print(f"  ║  Top scenarios:")
        for name, prob in pred.top_scenarios(4):
            bar = "█" * int(prob * 48)
            print(f"  ║    {name:<28} {prob:>5.1%}  {bar}")
        print(f"  ╠{'═'*68}╣")
        print(f"  ║  Performance: Sharpe={pred.running_sharpe:+.2f}  DD={pred.current_drawdown:.4f}  RegAcc={pred.regime_accuracy:.0%}")
        print(f"  ║  Positions:  {len(pred.positions)} assets | Formulas: {len(pred.active_formulas)}")
        print(f"  ╚{'═'*68}╝")


# ── Import constant from regime_prediction ────────────────────────────────────
try:
    from macro8_subnet.alpha.regime_prediction import ALL_SCENARIOS
except ImportError:
    ALL_SCENARIOS = [
        "rates_up_200bps", "rates_down_100bps", "equity_crash_30pct",
        "oil_spike_50pct", "china_crisis", "soft_landing",
        "stagflation", "ai_boom",
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 5. RUN_LIVE — the complete loop
# ══════════════════════════════════════════════════════════════════════════════

def run_live(
    fens,                           # ForecastedEnsemble, already fitted
    prices:      pd.DataFrame,      # full price history
    capital:     float = 100_000,
    n_days:      int   = 252,
    constraints: PortfolioConstraints = None,
    verbose:     bool  = True,
    print_every: int   = 20,        # print summary every N days
) -> tuple[LiveTracker, PredictionMarket, pd.DataFrame]:
    """
    Run the complete decision loop for `n_days`.

    Pipeline per day:
        1. Forecast (ForecastedEnsemble.forecast)
        2. Constrain (ConstraintSolver)
        3. Execute (TradeExecutor.compute_trades)
        4. Track (LiveTracker.update)
        5. Emit (PredictionMarket.emit)
        6. Adapt (DrawdownGuard, retrain_signal check)

    Args:
        fens:        Fitted ForecastedEnsemble.
        prices:      Full price DataFrame.
        capital:     Portfolio dollar size.
        n_days:      Days to simulate.
        constraints: PortfolioConstraints. None = defaults.
        verbose:     Print progress.
        print_every: Summary print frequency.

    Returns:
        (tracker, market, history_df) — full run results.
    """
    constraints = constraints or PortfolioConstraints()
    solver      = ConstraintSolver(constraints)
    executor    = TradeExecutor(capital=capital)
    tracker     = LiveTracker()
    market      = PredictionMarket()

    log_ret  = np.log(prices).diff().dropna()
    holdings = {}  # start flat

    start_idx = len(prices) - n_days - 1
    if start_idx < 0:
        start_idx = 0

    if verbose:
        print(f"[run_live] Starting {n_days}-day live simulation "
              f"from {prices.index[start_idx].date()}")

    for i in range(n_days):
        idx  = start_idx + i
        if idx + 1 >= len(prices):
            break
        date = prices.index[idx]

        # 1. Get forecast using data up to today
        recent_prices = prices.iloc[max(0, idx - 500): idx + 1]
        try:
            forecast = fens.forecast(prices=recent_prices, date=date)
        except Exception:
            # Fall back to simple position from base ensemble
            forecast = fens.forecast()

        # 2. Apply constraints (stress-aware + drawdown scale)
        dd_scale    = tracker.drawdown_guard.position_scale()
        conf_mult   = tracker.confidence_multiplier()
        raw_pos     = forecast.positions
        constrained = solver.apply(
            raw_pos,
            p_stress=forecast.regime_forecast.stress,
            scale=dd_scale * conf_mult,
        )

        # 3. Compute trades (pass p_stress so executor uses stress-adjusted spreads)
        plan = executor.compute_trades(
            constrained, holdings, date=date,
            p_stress=forecast.regime_forecast.stress,
        )

        # 4. Realise PnL (next day's return on today's held positions)
        if idx + 1 < len(log_ret) + 1:
            ret_row = log_ret.iloc[idx] if idx < len(log_ret) else pd.Series(0, index=prices.columns)
            pnl     = sum(
                holdings.get(t, 0) * float(ret_row.get(t, 0))
                for t in holdings
            )
        else:
            pnl = 0.0

        # 5. Track
        tracker.update(date, constrained, plan, pnl, forecast)

        # 6. Emit to prediction market
        perf = tracker.snapshot()
        market.emit(forecast, perf, tracker.drawdown_guard, tracker.confidence_multiplier())

        # 7. Update holdings
        holdings = dict(constrained)

        # 8. Check retrain signal
        if tracker.retrain_signal() and verbose:
            print(f"  [run_live] Day {i+1}: RETRAIN SIGNAL triggered "
                  f"(Sharpe={perf.sharpe_ann:.2f})")

        # 9. Periodic print
        if verbose and (i + 1) % print_every == 0:
            perf = tracker.snapshot()
            dd   = tracker.drawdown_guard.current_drawdown
            print(
                f"  Day {i+1:4d} [{date.date()}] | "
                f"{perf.summary()} | "
                f"DD={dd:.4f} | scale={dd_scale:.2f}"
            )

    if verbose:
        tracker.print_summary()
        market.print_latest()

    return tracker, market, tracker.full_history()
