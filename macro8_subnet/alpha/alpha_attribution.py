"""
alpha/alpha_attribution.py
--------------------------
Alpha attribution engine — measures each signal's individual
contribution to portfolio performance.

Elite quant firms don't just ask "did the portfolio perform well?"
They ask "which signals drove performance and by how much?"

This module provides four attribution metrics:

1. Marginal Sharpe Contribution (MSC)
   How much Sharpe ratio does each signal add?
   MSC_i = w_i * (Σ^-1 μ)_i / total_sharpe

2. Variance Decomposition
   What fraction of portfolio variance comes from each signal?
   VC_i = w_i * (Σw)_i / w'Σw

3. Return Attribution
   What fraction of portfolio return comes from each signal?
   RA_i = w_i * μ_i / Σ(w_j * μ_j)

4. Risk-Adjusted Attribution
   Information Ratio contribution: IC contribution weighted by capacity
   RAA_i = (IC_i * capacity_i) / Σ(IC_j * capacity_j)

Industry context
----------------
MSC is the standard metric at Renaissance Technologies-style firms.
It answers the fundamental question: "If I remove signal i and replace
it with an equal-weight allocation to the rest, does the portfolio
Sharpe go up or down?"

Positive MSC = signal helps the portfolio
Negative MSC = signal hurts the portfolio (should be removed)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class SignalAttribution:
    """Attribution metrics for one signal."""
    signal_name:     str
    weight:          float     # current portfolio weight for this signal
    msc:             float     # Marginal Sharpe Contribution
    variance_contrib: float    # fraction of portfolio variance explained
    return_contrib:  float     # fraction of portfolio return explained
    ic_contribution: float     # IC-weighted contribution (regime-adjusted)
    is_drag:         bool      # True if MSC < 0 (signal is hurting portfolio)

    def to_dict(self) -> dict:
        return {
            "signal_name":      self.signal_name,
            "weight":           round(self.weight,           6),
            "msc":              round(self.msc,              6),
            "variance_contrib": round(self.variance_contrib, 6),
            "return_contrib":   round(self.return_contrib,   6),
            "ic_contribution":  round(self.ic_contribution,  6),
            "is_drag":          self.is_drag,
        }


@dataclass
class AttributionReport:
    """Full attribution decomposition for a portfolio of signals."""
    portfolio_sharpe:   float
    portfolio_return:   float
    portfolio_vol:      float
    attributions:       list[SignalAttribution]
    top_contributors:   list[str]    # signal names sorted by MSC desc
    drags:              list[str]    # signal names with negative MSC
    diversification_ratio: float    # measures portfolio diversification

    def to_dict(self) -> dict:
        return {
            "portfolio_sharpe":       round(self.portfolio_sharpe,       6),
            "portfolio_return":       round(self.portfolio_return,        6),
            "portfolio_vol":          round(self.portfolio_vol,           6),
            "diversification_ratio":  round(self.diversification_ratio,  4),
            "top_contributors":       self.top_contributors[:5],
            "drags":                  self.drags,
            "attributions":           [a.to_dict() for a in self.attributions],
        }

    def summary(self) -> str:
        n_drag = len(self.drags)
        n_good = len(self.attributions) - n_drag
        return (
            f"Portfolio Sharpe={self.portfolio_sharpe:.3f} | "
            f"{n_good} contributing signals | "
            f"{n_drag} drag signals | "
            f"Diversification={self.diversification_ratio:.3f}"
        )


# ── Attribution Engine ────────────────────────────────────────────────────────

class AlphaAttributionEngine:
    """
    Decomposes portfolio performance into individual signal contributions.

    Designed to work with the output of SignalAggregator and
    PortfolioOptimizer — takes signal-level data and returns
    attribution metrics for each signal.
    """

    def __init__(self, risk_free_rate: float = 0.0):
        self.risk_free_rate = risk_free_rate

    # ── Public API ────────────────────────────────────────────────────────────

    def attribute(
        self,
        signal_returns:  pd.DataFrame,
        signal_weights:  dict[str, float],
        ic_scores:       Optional[dict[str, float]] = None,
        capacities:      Optional[dict[str, float]] = None,
    ) -> AttributionReport:
        """
        Compute full attribution decomposition.

        Args:
            signal_returns: DataFrame where columns = signal names,
                            rows = dates, values = signal daily returns.
                            Each column is treated as one "alpha strategy".
            signal_weights: Current portfolio weight per signal {name: weight}.
            ic_scores:      Optional {name: mean_ic} for IC-weighted attribution.
            capacities:     Optional {name: capacity} for risk-adjusted attribution.

        Returns:
            AttributionReport with MSC and other metrics per signal.
        """
        signals = [s for s in signal_weights if s in signal_returns.columns]
        if not signals:
            return self._empty_report()

        ret_data = signal_returns[signals].dropna()
        if len(ret_data) < 5:
            return self._empty_report()

        w_vec  = np.array([signal_weights[s] for s in signals])
        mu_vec = ret_data.mean().values * 252        # annualised expected return
        cov    = ret_data.cov().values * 252         # annualised covariance

        # ── Portfolio-level stats ─────────────────────────────────────────────
        port_ret  = float(w_vec @ mu_vec)
        port_var  = float(w_vec @ cov @ w_vec)
        port_vol  = float(np.sqrt(max(port_var, 1e-12)))
        port_sharpe = (port_ret - self.risk_free_rate) / port_vol

        # ── Marginal Sharpe Contribution ──────────────────────────────────────
        # MSC_i = w_i * (Σ^-1 μ)_i normalised to sum to portfolio Sharpe
        try:
            cov_inv  = np.linalg.pinv(cov)
            msc_raw  = w_vec * (cov_inv @ mu_vec)
            # Normalise so MSCs sum to portfolio Sharpe
            msc_sum  = msc_raw.sum()
            if abs(msc_sum) > 1e-8:
                msc_vec = msc_raw * (port_sharpe / msc_sum)
            else:
                msc_vec = msc_raw
        except Exception:
            msc_vec = np.zeros(len(signals))

        # ── Variance decomposition ────────────────────────────────────────────
        # VC_i = w_i * (Σw)_i / w'Σw
        cov_w    = cov @ w_vec
        if port_var > 1e-12:
            vc_vec = (w_vec * cov_w) / port_var
        else:
            vc_vec = np.ones(len(signals)) / len(signals)

        # ── Return attribution ────────────────────────────────────────────────
        # RA_i = w_i * μ_i / Σ(w_j * μ_j)
        contrib = w_vec * mu_vec
        total_c = contrib.sum()
        ra_vec  = contrib / total_c if abs(total_c) > 1e-12 else np.zeros(len(signals))

        # ── IC-weighted attribution ───────────────────────────────────────────
        if ic_scores and capacities:
            ic_cap = np.array([
                ic_scores.get(s, 0.0) * capacities.get(s, 1.0)
                for s in signals
            ])
            ic_total = ic_cap.sum()
            ic_vec   = ic_cap / ic_total if ic_total > 1e-8 else np.zeros(len(signals))
        elif ic_scores:
            ic_vals  = np.array([ic_scores.get(s, 0.0) for s in signals])
            ic_total = ic_vals.sum()
            ic_vec   = ic_vals / ic_total if ic_total > 1e-8 else np.zeros(len(signals))
        else:
            ic_vec   = np.ones(len(signals)) / len(signals)

        # ── Diversification ratio ─────────────────────────────────────────────
        # weighted avg of individual vols / portfolio vol
        indiv_vols   = np.sqrt(np.diag(cov))
        weighted_vol = float(w_vec @ indiv_vols)
        div_ratio    = weighted_vol / port_vol if port_vol > 1e-8 else 1.0

        # ── Build per-signal attributions ─────────────────────────────────────
        attributions = []
        for i, name in enumerate(signals):
            attributions.append(SignalAttribution(
                signal_name=name,
                weight=float(w_vec[i]),
                msc=float(msc_vec[i]),
                variance_contrib=float(vc_vec[i]),
                return_contrib=float(ra_vec[i]),
                ic_contribution=float(ic_vec[i]),
                is_drag=float(msc_vec[i]) < 0,
            ))

        # Sort by MSC descending
        attributions.sort(key=lambda a: a.msc, reverse=True)
        top_contributors = [a.signal_name for a in attributions if a.msc >= 0]
        drags            = [a.signal_name for a in attributions if a.is_drag]

        return AttributionReport(
            portfolio_sharpe=port_sharpe,
            portfolio_return=port_ret,
            portfolio_vol=port_vol,
            attributions=attributions,
            top_contributors=top_contributors,
            drags=drags,
            diversification_ratio=div_ratio,
        )

    def leave_one_out_sharpe(
        self,
        signal_returns: pd.DataFrame,
        signal_weights: dict[str, float],
    ) -> dict[str, float]:
        """
        Compute Sharpe ratio when each signal is removed one at a time.

        Returns {signal_name: sharpe_without_it} — signals where
        sharpe_without < portfolio_sharpe are valuable contributors.

        Args:
            signal_returns: Daily signal return DataFrame.
            signal_weights: Current portfolio weights.

        Returns:
            Dict of {signal_name: sharpe_without_this_signal}.
        """
        signals  = [s for s in signal_weights if s in signal_returns.columns]
        ret_data = signal_returns[signals].dropna()

        results = {}
        for excl in signals:
            remaining = [s for s in signals if s != excl]
            if not remaining:
                results[excl] = 0.0
                continue

            w_r  = np.array([signal_weights[s] for s in remaining])
            total = w_r.sum()
            if total < 1e-8:
                results[excl] = 0.0
                continue
            w_r  /= total

            mu   = ret_data[remaining].mean().values * 252
            cov  = ret_data[remaining].cov().values * 252
            ret  = float(w_r @ mu)
            vol  = float(np.sqrt(max(w_r @ cov @ w_r, 1e-12)))
            results[excl] = (ret - self.risk_free_rate) / vol if vol > 1e-8 else 0.0

        return results

    @staticmethod
    def _empty_report() -> AttributionReport:
        return AttributionReport(
            portfolio_sharpe=0.0,
            portfolio_return=0.0,
            portfolio_vol=0.0,
            attributions=[],
            top_contributors=[],
            drags=[],
            diversification_ratio=1.0,
        )
