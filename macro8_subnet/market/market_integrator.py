"""
market/market_integrator.py
----------------------------
Integrates prediction market prices into portfolio construction.

This module is the bridge between the signal prediction market and
the portfolio optimizer. It answers the question:

    "Given IC history AND market beliefs, how should we weight signals?"

The integration formula
-----------------------
    combined_weight_i = α * ic_score_i + β * market_confidence_i

    Default: α=0.70, β=0.30

Rationale:
  - IC history is a backward-looking measure (past predictive power)
  - Market confidence is a forward-looking measure (predicted future IC)
  - A 70/30 blend puts more weight on observed evidence than beliefs,
    while still incorporating the market's forward-looking signal

This mirrors how professional allocators blend quantitative scores
with analyst conviction ratings.

Special cases
-------------
  - Signal with no market positions: uses IC score only (β treated as 0)
  - Signal with IC=0 but high market confidence: receives minimum weight
  - Negative market price (bearish consensus): reduces signal weight

Market confidence directional adjustment
-----------------------------------------
The market price ∈ [-1, 1]:
    +1.0 = everyone strongly bullish → full confidence boost
    -1.0 = everyone strongly bearish → zero or negative boost

    market_confidence = max(market_price, 0)
    (negative consensus removes upside boost but doesn't flip weight)

Usage
-----
    integrator = MarketIntegrator()
    blended    = integrator.market_weighted_ics(ic_scores, market)
    # Use blended weights in PortfolioOptimizer instead of raw IC scores
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_SUBNET = Path(__file__).resolve().parent.parent
_ROOT   = _SUBNET.parent
for p in [str(_SUBNET), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from macro8_subnet.market.signal_market import SignalMarket


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class BlendedSignalScore:
    """Combined IC + market score for one signal."""
    signal_name:        str
    ic_score:           float          # raw IC from history
    market_price:       float          # market price ∈ [-1, 1]
    market_confidence:  float          # max(market_price, 0) ∈ [0, 1]
    blended_score:      float          # final combined weight
    has_market_data:    bool           # False if no market positions

    def to_dict(self) -> dict:
        return {
            "signal_name":       self.signal_name,
            "ic_score":          round(self.ic_score,          6),
            "market_price":      round(self.market_price,      4),
            "market_confidence": round(self.market_confidence, 4),
            "blended_score":     round(self.blended_score,     6),
            "has_market_data":   self.has_market_data,
        }


# ── Integrator ────────────────────────────────────────────────────────────────

class MarketIntegrator:
    """
    Blends IC history with market confidence for portfolio signal weighting.

    alpha: weight on IC history (backward-looking)
    beta:  weight on market confidence (forward-looking)
    """

    def __init__(
        self,
        alpha: float = 0.70,   # IC history weight
        beta:  float = 0.30,   # market confidence weight
    ):
        if abs(alpha + beta - 1.0) > 0.01:
            raise ValueError(f"alpha + beta must equal 1.0, got {alpha + beta:.4f}")
        self.alpha = alpha
        self.beta  = beta

    # ── Public API ────────────────────────────────────────────────────────────

    def market_weighted_ics(
        self,
        ic_scores:  dict[str, float],
        market:     SignalMarket,
    ) -> dict[str, float]:
        """
        Blend IC scores with market confidence into portfolio weights.

        Signals with both high IC and strong market conviction receive
        the highest weight. Signals the market is bearish on are
        discounted even if their IC history is good.

        Args:
            ic_scores: {signal_name: mean_ic} from the alpha library.
            market:    Current SignalMarket with open positions.

        Returns:
            {signal_name: blended_score} — use as signal strengths in
            PortfolioOptimizer.optimize(signals=...).
        """
        blended_scores  = self._compute_blended(ic_scores, market)
        return {s.signal_name: max(s.blended_score, 0.0) for s in blended_scores}

    def signal_confidence_vector(
        self,
        signal_names: list[str],
        market:       SignalMarket,
    ) -> dict[str, float]:
        """
        Return normalised market confidence scores for a list of signals.

        Signals not in the market receive equal baseline confidence (0.5).
        Used as a portfolio weight modifier independent of IC history.

        Args:
            signal_names: All signal names to include.
            market:       Current SignalMarket.

        Returns:
            {signal_name: confidence ∈ [0, 1]}
        """
        confidences = {}
        for name in signal_names:
            price = market.price_signal(name)
            # Shift from [-1,1] to [0,1]; no-data signals get 0.5
            if name in market.tracked_signals():
                confidences[name] = float(np.clip((price + 1.0) / 2.0, 0.0, 1.0))
            else:
                confidences[name] = 0.5   # neutral baseline
        return confidences

    def blended_scores_detail(
        self,
        ic_scores: dict[str, float],
        market:    SignalMarket,
    ) -> list[BlendedSignalScore]:
        """
        Return detailed blended scores including all components.
        Useful for logging and explainability.
        """
        return self._compute_blended(ic_scores, market)

    # ── Core computation ──────────────────────────────────────────────────────

    def _compute_blended(
        self,
        ic_scores: dict[str, float],
        market:    SignalMarket,
    ) -> list[BlendedSignalScore]:
        """Compute blended scores for all signals in ic_scores."""
        results        = []
        market_prices  = market.all_prices()

        # Normalise IC scores to [0, 1] range for blending
        ic_values  = np.array(list(ic_scores.values()))
        ic_max     = float(ic_values.max()) if len(ic_values) > 0 else 1.0
        ic_min     = float(ic_values.min()) if len(ic_values) > 0 else 0.0
        ic_range   = ic_max - ic_min

        for name, ic in ic_scores.items():
            # Normalise IC to [0, 1]
            ic_norm   = ((ic - ic_min) / ic_range) if ic_range > 1e-8 else 0.5

            # Market data (use 0 if no positions — neutral, not bearish)
            m_price   = market_prices.get(name, 0.0)
            m_conf    = float(max(m_price, 0.0))   # bearish = 0 contribution
            has_data  = name in market_prices

            # Effective weights: if no market data, use IC only
            if has_data:
                a, b  = self.alpha, self.beta
            else:
                a, b  = 1.0, 0.0

            blended   = a * ic_norm + b * m_conf

            results.append(BlendedSignalScore(
                signal_name=name,
                ic_score=float(ic),
                market_price=m_price,
                market_confidence=m_conf,
                blended_score=float(blended),
                has_market_data=has_data,
            ))

        return results
