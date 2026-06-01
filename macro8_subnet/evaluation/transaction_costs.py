"""
evaluation/transaction_costs.py
---------------------------------
Transaction Cost Model — calibrated to real ETF market data.

Two cost components (industry standard):

    total_cost = spread_cost + market_impact

1. Spread cost (fixed per trade)
   ─────────────────────────────
   Each round-trip pays half the bid-ask spread on entry and half on exit.
   Calibrated from ETF.com and Natixis data (2024):

       SPY:  0.32 bps   (most liquid ETF on earth)
       QQQ:  0.50 bps
       IWM:  0.60 bps
       TLT:  4.00 bps   (bond ETF, slower underlying)
       GLD:  5.00 bps   (commodity ETF, futures-based)
       DBC:  8.00 bps   (commodity basket, less liquid)
       EEM:  2.50 bps   (EM equity, wider than US)
       FXI:  3.50 bps   (China ETF, regime risk)
       VNQ:  3.00 bps   (REIT ETF)
       HYG:  5.50 bps   (high-yield bond, illiquid underlying)

   Sources:
       - ETF.com: "SPY average spread 0.0032%"
       - Natixis (2024): US equity ETF spreads 1-2 bps; EM 4-10 bps;
         fixed income 3-5 bps; commodity 4-7 bps
       - alphaexcapital.com: "TLT 3-5 bps; GLD 4-7 bps"

2. Market impact (square-root model)
   ────────────────────────────────────
   Price impact grows as √(order_size / daily_volume).
   Academic standard: Almgren et al. (2005), Bouchaud et al. (2004),
   empirically validated across 500,000 trades.

       impact_bps = η × σ × √(Q / V)

   where:
       η = market impact coefficient (~0.1–0.5, asset-specific)
       σ = daily return volatility (annualised/√252)
       Q = order size as fraction of daily volume
       V = average daily volume fraction (≈ 1.0 by convention)

   Approximated for capital-aware backtesting as:
       impact_bps = η × σ_daily × √(capital_fraction)

   where capital_fraction = portfolio_value / daily_dollar_volume.

Per-trade total cost
────────────────────
    cost_per_trade = (spread_bps/2 + impact_bps) / 10_000

Applied to PnL
──────────────
    PnL_net[t] = PnL_gross[t] − cost[t]
    cost[t]    = sum_assets(|Δweight[t,a]| × cost_per_trade[a])

   The |Δweight| is the daily portfolio turnover — the fraction of the
   portfolio being rebalanced. High-turnover strategies (reversal) pay
   this cost every day; low-turnover strategies (long-term momentum) pay
   it infrequently.

Capital scaling
───────────────
   Market impact increases with capital:
       impact[capital] = η × σ × √(capital / ADV)

   For a $1k portfolio, impact is negligible.
   For a $10M portfolio trading illiquid ETFs, impact can exceed spread.

Usage
─────
    from macro8_subnet.evaluation.transaction_costs import TransactionCostModel

    tcm = TransactionCostModel(universe=['SPY','QQQ','TLT',...])

    # Apply to a PnL series
    pnl_net = tcm.apply(pnl_gross, weights, capital=100_000)

    # Query cost for one asset
    cost_bps = tcm.round_trip_bps('HYG', capital=1_000_000)
"""

from __future__ import annotations

import sys
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


# ── Calibrated spread data (real ETF market data, 2024) ──────────────────────
# Source: ETF.com, Natixis Investment Managers, alphaexcapital.com
# Units: basis points (1 bps = 0.01%)

SPREAD_BPS: dict[str, float] = {
    # Ultra-liquid US equity ETFs: 0.3–1 bps
    "SPY": 0.32,   # S&P 500 — most liquid ETF, $388B AUM
    "QQQ": 0.50,   # Nasdaq 100 — second most liquid
    "IWM": 0.60,   # Russell 2000 — slightly wider, smaller cap

    # Fixed income ETFs: 3–5 bps (slower underlying bond markets)
    "TLT": 4.00,   # 20+ Year Treasury — slower bond mkts
    "HYG": 5.50,   # High Yield — illiquid underlying credits
    "LQD": 3.50,   # Investment Grade — moderate liquidity

    # Commodity ETFs: 4–8 bps (futures-based, roll costs)
    "GLD": 5.00,   # Gold — futures-based, some roll cost
    "DBC": 8.00,   # Commodity basket — wider, less liquid
    "USO": 9.00,   # Oil ETF — very wide spreads

    # Emerging market ETFs: 2–8 bps (foreign market costs)
    "EEM": 2.50,   # Broad EM — large, liquid
    "FXI": 3.50,   # China-specific — political risk in spread
    "EWZ": 6.00,   # Brazil — wider EM

    # Real estate
    "VNQ": 3.00,   # REIT ETF — less liquid than equity

    # Default for unknown tickers
    "_DEFAULT": 5.00,
}

# Market impact coefficient η per asset class
# Higher η = more market-sensitive = higher impact per unit of capital
# Calibrated from Almgren et al. (2005) and Bouchaud et al. (2004)
IMPACT_ETA: dict[str, float] = {
    "SPY": 0.05,   # Ultra-liquid — impact nearly zero for small caps
    "QQQ": 0.07,
    "IWM": 0.10,   # Less liquid than SPY/QQQ
    "TLT": 0.08,   # Bond mkts — larger but slower
    "GLD": 0.12,   # Commodity ETF — futures carry adds cost
    "DBC": 0.18,
    "EEM": 0.15,   # EM — wider underlying spreads
    "FXI": 0.20,
    "VNQ": 0.12,
    "HYG": 0.20,   # High yield — very illiquid in stress
    "LQD": 0.10,
    "USO": 0.25,
    "_DEFAULT": 0.15,
}

# Approximate average daily dollar volume ($B) for each ETF
# Used to calibrate capital-relative order size
ADV_BILLION: dict[str, float] = {
    "SPY": 25.0,   # ~$25B/day
    "QQQ": 12.0,
    "IWM":  3.5,
    "TLT":  1.5,
    "GLD":  1.2,
    "DBC":  0.15,
    "EEM":  0.80,
    "FXI":  0.30,
    "VNQ":  0.40,
    "HYG":  0.60,
    "LQD":  0.50,
    "USO":  0.10,
    "_DEFAULT": 0.20,
}


@dataclass
class CostBreakdown:
    """Per-asset cost decomposition for one period."""
    asset:         str
    spread_bps:    float
    impact_bps:    float
    total_bps:     float
    turnover_frac: float   # |Δweight| — fraction of portfolio rebalanced
    cost_return:   float   # cost as return drag (negative)

    def __repr__(self) -> str:
        return (
            f"CostBreakdown({self.asset}: "
            f"spread={self.spread_bps:.2f}bps, "
            f"impact={self.impact_bps:.2f}bps, "
            f"total={self.total_bps:.2f}bps, "
            f"turn={self.turnover_frac:.4f}, "
            f"drag={self.cost_return*100:.4f}%)"
        )


@dataclass
class CostSummary:
    """Aggregate cost statistics for a strategy."""
    # Per-period costs
    annual_cost_bps:   float   # annualised total cost in bps
    daily_cost_mean:   float   # mean daily cost as return
    daily_cost_std:    float   # std of daily cost

    # Cost decomposition
    spread_fraction:   float   # fraction of total cost from spread
    impact_fraction:   float   # fraction from market impact

    # Capital sensitivity
    cost_at_1k:        float   # annualised cost drag at $1k
    cost_at_100k:      float   # at $100k
    cost_at_1m:        float   # at $1M

    # Impact on Sharpe
    gross_sharpe:      float
    net_sharpe:        float
    sharpe_drag:       float   # gross - net

    def print(self) -> None:
        print(f"\n  Transaction Cost Summary")
        print(f"  {'─'*45}")
        print(f"  Annual cost:     {self.annual_cost_bps:.1f} bps/year")
        print(f"  Spread fraction: {self.spread_fraction:.0%}")
        print(f"  Impact fraction: {self.impact_fraction:.0%}")
        print(f"  Capital sensitivity:")
        print(f"    $1k:   {self.cost_at_1k*100:.3f}%/yr drag")
        print(f"    $100k: {self.cost_at_100k*100:.3f}%/yr drag")
        print(f"    $1M:   {self.cost_at_1m*100:.3f}%/yr drag")
        print(f"  Sharpe: {self.gross_sharpe:.3f} (gross) → "
              f"{self.net_sharpe:.3f} (net)  [{self.sharpe_drag:+.3f}]")


class TransactionCostModel:
    """
    Realistic transaction cost model for ETF cross-sectional strategies.

    Implements:
        1. Bid-ask spread costs (calibrated to 2024 ETF market data)
        2. Square-root market impact (Almgren/Bouchaud model)
        3. Capital-aware scaling

    Parameters
    ----------
    universe:    list[str]  — ETF tickers in the portfolio.
    capital:     float      — Default portfolio size in USD.
    annualise:   int        — Trading days per year (252).
    """

    def __init__(
        self,
        universe:  list[str],
        capital:   float = 100_000,
        annualise: int   = 252,
    ):
        self.universe  = universe
        self.capital   = capital
        self.annualise = annualise
        self.n_assets  = len(universe)

        # Pre-build per-asset parameter arrays (aligned with universe)
        self._spreads = np.array([
            SPREAD_BPS.get(t, SPREAD_BPS["_DEFAULT"])
            for t in universe
        ], dtype=np.float32)

        self._etas = np.array([
            IMPACT_ETA.get(t, IMPACT_ETA["_DEFAULT"])
            for t in universe
        ], dtype=np.float32)

        self._adv = np.array([
            ADV_BILLION.get(t, ADV_BILLION["_DEFAULT"]) * 1e9
            for t in universe
        ], dtype=np.float64)  # in USD

    # ── Public API ────────────────────────────────────────────────────────────

    def apply(
        self,
        pnl_gross:     np.ndarray,   # [T] daily gross PnL
        weights:       np.ndarray,   # [T+1 × A] portfolio weights
        returns:       np.ndarray,   # [T × A] daily returns (for vol estimate)
        capital:       Optional[float] = None,
        verbose:       bool = False,
    ) -> np.ndarray:
        """
        Apply transaction costs to a gross PnL series.

        Args:
            pnl_gross:  Gross daily PnL [T].
            weights:    Portfolio weights [T+1 × A].
            returns:    Daily returns [T × A] for volatility estimation.
            capital:    Portfolio size (overrides default).
            verbose:    Print cost summary.

        Returns:
            Net PnL series [T] after transaction costs.
        """
        cap      = capital or self.capital
        T        = len(pnl_gross)
        A        = self.n_assets

        # Daily turnover = |Δweights| [T × A]
        turnover = np.abs(np.diff(weights, axis=0))   # [T × A]
        # Align: turnover[t] applies to pnl_gross[t]
        T_use    = min(T, len(turnover))
        turnover = turnover[:T_use]

        # Daily volatility estimate per asset [A]
        vol_daily = np.std(returns, axis=0) if len(returns) > 5 else np.ones(A) * 0.01

        # Cost per unit turnover per asset [A] — in return units (not bps)
        spread_cost_per_unit = self._spreads / 2 / 10_000         # half-spread on entry
        impact_cost_per_unit = self._impact_per_unit(vol_daily, cap)

        total_cost_per_unit = spread_cost_per_unit + impact_cost_per_unit

        # Daily cost [T]
        daily_cost = (turnover * total_cost_per_unit[np.newaxis, :]).sum(axis=1)

        # Net PnL
        pnl_net             = np.zeros(T, dtype=np.float64)
        pnl_net[:T_use]     = pnl_gross[:T_use] - daily_cost
        pnl_net[T_use:]     = pnl_gross[T_use:]   # no turnover data beyond here

        if verbose:
            self._print_summary(pnl_gross[:T_use], pnl_net[:T_use], daily_cost,
                                spread_cost_per_unit, impact_cost_per_unit,
                                turnover, cap)

        return pnl_net

    def apply_vectorised(
        self,
        pnl_gross:  np.ndarray,   # [T × F] gross PnL for F formulas
        weights:    np.ndarray,   # [T+1 × A × F] portfolio weights
        returns:    np.ndarray,   # [T × A] daily returns
        capital:    float = None,
    ) -> np.ndarray:
        """
        Vectorised transaction cost application for a batch of F formulas.

        Args:
            pnl_gross:  [T × F] gross PnL.
            weights:    [T+1 × A × F] portfolio weights.
            returns:    [T × A] daily returns.
            capital:    Portfolio size.

        Returns:
            Net PnL [T × F].
        """
        cap     = capital or self.capital
        T, F    = pnl_gross.shape
        A       = self.n_assets

        # Turnover [T × A × F]
        turnover = np.abs(np.diff(weights, axis=0))     # [T × A × F]
        T_use    = min(T, len(turnover))

        vol_daily = np.std(returns, axis=0) if len(returns) > 5 else np.ones(A) * 0.01

        spread_cost = self._spreads / 2 / 10_000           # [A]
        impact_cost = self._impact_per_unit(vol_daily, cap) # [A]
        cost_per_unit = (spread_cost + impact_cost)         # [A]

        # Daily cost [T × F]
        # sum over assets: turnover[t,a,f] × cost[a]
        daily_cost = np.einsum('taf,a->tf', turnover[:T_use], cost_per_unit)

        pnl_net              = pnl_gross.copy()
        pnl_net[:T_use, :]  -= daily_cost

        return pnl_net

    def round_trip_bps(self, ticker: str, capital: float = None) -> float:
        """
        Total round-trip cost in basis points for one trade in a given asset.

        Includes half-spread on entry, half-spread on exit, and market impact.
        Market impact uses a typical daily turnover of 5% of capital.

        Args:
            ticker:  ETF ticker symbol.
            capital: Portfolio size (default: self.capital).

        Returns:
            Round-trip cost in basis points.
        """
        cap = capital or self.capital

        spread = SPREAD_BPS.get(ticker, SPREAD_BPS["_DEFAULT"])
        eta    = IMPACT_ETA.get(ticker, IMPACT_ETA["_DEFAULT"])
        adv    = ADV_BILLION.get(ticker, ADV_BILLION["_DEFAULT"]) * 1e9

        # Assume 5% daily turnover per asset (typical cross-sectional strategy)
        order_size    = cap * 0.05
        capital_frac  = order_size / adv
        impact        = eta * 0.01 * np.sqrt(capital_frac) * 10_000  # in bps

        return spread + 2 * impact   # round trip = 2 × one-way impact

    def annual_drag(
        self,
        daily_turnover: float,
        capital:        float = None,
        universe:       Optional[list[str]] = None,
    ) -> float:
        """
        Estimate annual return drag from transaction costs.

        Args:
            daily_turnover:  Mean fraction of portfolio rebalanced per day.
            capital:         Portfolio size.
            universe:        Tickers (uses self.universe if None).

        Returns:
            Annual drag as a decimal (e.g. 0.015 = 1.5%/year).
        """
        cap = capital or self.capital
        tickers = universe or self.universe

        # Mean cost per unit turnover across universe
        spread_mean = np.mean([
            SPREAD_BPS.get(t, SPREAD_BPS["_DEFAULT"]) for t in tickers
        ]) / 2 / 10_000

        vol_mean   = 0.01   # ~1% daily vol, typical ETF
        eta_mean   = np.mean([
            IMPACT_ETA.get(t, IMPACT_ETA["_DEFAULT"]) for t in tickers
        ])
        adv_mean   = np.mean([
            ADV_BILLION.get(t, ADV_BILLION["_DEFAULT"]) for t in tickers
        ]) * 1e9
        order_size = cap * daily_turnover / len(tickers)
        impact     = eta_mean * vol_mean * np.sqrt(order_size / adv_mean)

        daily_cost = daily_turnover * (spread_mean + impact)
        return daily_cost * self.annualise

    def capital_cost_table(
        self,
        daily_turnover: float,
        capitals:       list[float] = None,
    ) -> dict[float, float]:
        """
        Return annual drag for each capital tier.

        Args:
            daily_turnover:  Mean daily |Δweight| per asset.
            capitals:        List of capital sizes to evaluate.

        Returns:
            {capital: annual_drag_fraction}
        """
        caps = capitals or [1_000, 10_000, 100_000, 1_000_000]
        return {
            cap: self.annual_drag(daily_turnover, capital=cap)
            for cap in caps
        }

    # ── Private ───────────────────────────────────────────────────────────────

    def _impact_per_unit(
        self,
        vol_daily:  np.ndarray,   # [A] daily return std
        capital:    float,
    ) -> np.ndarray:
        """
        Square-root market impact per unit of turnover [A].

        impact[a] = η[a] × σ[a] × √(capital / ADV[a])

        The √(capital/ADV) term captures how our order size relative
        to daily volume grows with capital — the core capital-scaling effect.

        At $1k: capital/ADV ≈ 1e-7 → impact ≈ 0
        At $1M: capital/ADV ≈ 1e-4 → impact ≈ 0.1 × σ × 0.01 ≈ tiny
        At $100M: impact starts to matter for illiquid ETFs
        """
        capital_frac = capital / self._adv        # [A]
        impact = self._etas * vol_daily * np.sqrt(np.clip(capital_frac, 0, 1))
        return impact.astype(np.float32)

    def _print_summary(
        self,
        pnl_gross:        np.ndarray,
        pnl_net:          np.ndarray,
        daily_cost:       np.ndarray,
        spread_cost_unit: np.ndarray,   # [A]
        impact_cost_unit: np.ndarray,   # [A]
        turnover:         np.ndarray,   # [T × A]
        capital:          float,
    ) -> None:
        """Print a formatted cost summary."""
        ann = self.annualise
        gross_sharpe = pnl_gross.mean() / (pnl_gross.std() + 1e-10) * np.sqrt(ann)
        net_sharpe   = pnl_net.mean()   / (pnl_net.std()   + 1e-10) * np.sqrt(ann)
        total_spread = (turnover * spread_cost_unit).sum()
        total_impact = (turnover * impact_cost_unit).sum()
        total_cost   = total_spread + total_impact
        spread_frac  = total_spread / (total_cost + 1e-10)
        impact_frac  = 1 - spread_frac
        annual_bps   = daily_cost.mean() * ann * 10_000

        print(f"\n  Transaction Cost Summary (capital=${capital:,.0f})")
        print(f"  {'─'*50}")
        print(f"  Annual cost drag:  {annual_bps:.1f} bps/yr  "
              f"({daily_cost.mean()*100:.4f}%/day mean)")
        print(f"  Spread fraction:   {spread_frac:.0%}")
        print(f"  Impact fraction:   {impact_frac:.0%}")
        print(f"  Sharpe (gross→net): {gross_sharpe:.3f} → {net_sharpe:.3f}  "
              f"[{net_sharpe-gross_sharpe:+.3f}]")
        print()
        print(f"  Per-asset cost (bps, one-way):")
        for i, t in enumerate(self.universe):
            sp = spread_cost_unit[i] * 10_000
            im = impact_cost_unit[i] * 10_000
            print(f"    {t:<6}  spread={sp:.2f}  impact={im:.2f}  "
                  f"total={sp+im:.2f}")


def build_cost_model(prices: pd.DataFrame, capital: float = 100_000) -> TransactionCostModel:
    """
    Convenience constructor: build a TransactionCostModel from a price DataFrame.

    Args:
        prices:  Price DataFrame with tickers as columns.
        capital: Default portfolio size.

    Returns:
        TransactionCostModel configured for the given universe.
    """
    return TransactionCostModel(
        universe=list(prices.columns),
        capital=capital,
    )
