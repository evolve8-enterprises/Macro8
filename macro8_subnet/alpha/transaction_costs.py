"""
alpha/transaction_costs.py
---------------------------
Transaction cost models for realistic alpha evaluation.

Net return after costs:
    net_return = gross_return - turnover * cost_per_turn
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class CostModel:
    cost_per_turnover: float = 0.001
    spread_bps:        float = 5.0
    market_impact:     float = 0.0005
    max_turnover:      float = 1.0


@dataclass
class CostResult:
    gross_return:         float
    net_return:           float
    total_cost:           float
    avg_daily_turn:       float
    annualised_cost:      float
    cost_adjusted_sharpe: Optional[float]
    passes_cost_filter:   bool


class TransactionCostModel:
    """Applies realistic transaction costs to portfolio return series."""

    def __init__(self, model: CostModel = CostModel()):
        self.model = model

    def apply(
        self,
        portfolio_value: pd.Series,
        weight_history:  list[dict[str, float]],
        risk_free:       float = 0.0,
    ) -> CostResult:
        if len(weight_history) < 2:
            gross_ret = float((portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1)
            return CostResult(
                gross_return=gross_ret, net_return=gross_ret,
                total_cost=0.0, avg_daily_turn=0.0, annualised_cost=0.0,
                cost_adjusted_sharpe=None, passes_cost_filter=True,
            )

        turnovers    = self._compute_turnover(weight_history)
        avg_turn     = float(np.mean(turnovers))
        daily_cost   = turnovers * self.model.cost_per_turnover
        total_cost   = float(daily_cost.sum())

        gross_ret    = float((portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1)
        ann_cost     = avg_turn * self.model.cost_per_turnover * 252
        net_ret      = gross_ret - total_cost

        daily_rets   = portfolio_value.pct_change().dropna()
        net_daily    = daily_rets - daily_cost[:len(daily_rets)]
        std          = float(net_daily.std())
        net_sharpe   = (net_daily.mean() * 252 - risk_free) / (std * np.sqrt(252)) \
                       if std > 1e-8 else None

        return CostResult(
            gross_return=gross_ret, net_return=net_ret,
            total_cost=total_cost, avg_daily_turn=avg_turn,
            annualised_cost=ann_cost, cost_adjusted_sharpe=net_sharpe,
            passes_cost_filter=net_ret > 0,
        )

    @staticmethod
    def _compute_turnover(weight_history: list[dict[str, float]]) -> pd.Series:
        all_assets = sorted({a for w in weight_history for a in w})
        turnovers  = []
        prev       = {a: 0.0 for a in all_assets}
        for weights in weight_history:
            total = sum(abs(weights.get(a, 0.0) - prev.get(a, 0.0)) for a in all_assets)
            turnovers.append(0.5 * total)
            prev = {a: weights.get(a, 0.0) for a in all_assets}
        return pd.Series(turnovers)
