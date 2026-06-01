"""
market/signal_market.py
------------------------
Signal Prediction Market for the Macro8 subnet.

Each alpha signal in the library becomes a tradeable market asset.
Miners stake on signals they believe will have positive IC next epoch.
The aggregate of these beliefs produces a market confidence price that
informs portfolio construction.

Market design
-------------
Positions:
    LONG  — bet that signal will have IC > 0 next epoch
    SHORT — bet that signal will have IC ≤ 0

Confidence:
    Each position carries a confidence ∈ [0, 1]:
        1.0 = certain the outcome will occur
        0.5 = coin flip
    Miners who report truthfully are optimal under the quadratic
    scoring rule (proper scoring rule — see market_rewards.py).

Market price per signal:
    price = Σ(stake_i * confidence_i * direction_i) / Σ(stake_i)
    where direction = +1 for LONG, -1 for SHORT
    Result ∈ [-1, 1]; positive = bullish consensus.

Settlement:
    After each epoch, actual ICs are observed.
    Positions are settled: IC > 0 = LONG wins, IC ≤ 0 = SHORT wins.
    P&L is computed per position using the quadratic scoring rule.

Why this works
--------------
Prediction markets are the best known mechanism for aggregating
distributed private information. Each miner who stakes a position
reveals their private belief about the signal's quality. The aggregate
of these beliefs is often more accurate than any individual prediction
(Wisdom of Crowds theorem).

For Macro8, this means:
  - New signals get priced by the network immediately
  - Market prices update faster than IC history (1 epoch vs many epochs)
  - Miners have financial incentive to research signal quality

Usage
-----
    market = SignalMarket()
    market.open_position(SignalPosition(uid=0, signal_name="f_momentum",
                                         direction=PositionDirection.LONG,
                                         stake=0.1, confidence=0.8))
    price = market.price_signal("f_momentum")
    results = market.settle_epoch({"f_momentum": 0.05})  # IC > 0 → LONG wins
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np


# ── Position types ────────────────────────────────────────────────────────────

class PositionDirection(str, Enum):
    LONG  = "long"    # bet IC > 0
    SHORT = "short"   # bet IC ≤ 0

    def sign(self) -> float:
        return 1.0 if self == PositionDirection.LONG else -1.0


@dataclass
class SignalPosition:
    """
    One miner's market position on a signal.

    Attributes
    ----------
    miner_uid      : int — the miner placing this bet
    miner_hotkey   : str
    signal_name    : str — name of signal in the alpha library
    direction      : PositionDirection — LONG (bullish) or SHORT (bearish)
    stake          : float — relative stake weight [0, 1]
    confidence     : float — belief strength [0.5, 1.0]
                     0.5 = uncertain, 1.0 = certain
    epoch          : int — epoch when this position was opened
    """
    miner_uid:    int
    miner_hotkey: str
    signal_name:  str
    direction:    PositionDirection
    stake:        float              # relative weight, normalised later
    confidence:   float              # [0.5, 1.0]
    epoch:        int                = 0

    def __post_init__(self):
        self.stake      = float(np.clip(self.stake,      0.0,  1.0))
        self.confidence = float(np.clip(self.confidence, 0.5,  1.0))

    @property
    def signed_confidence(self) -> float:
        """Directional confidence: +confidence for LONG, -confidence for SHORT."""
        return self.direction.sign() * self.confidence

    def to_dict(self) -> dict:
        return {
            "miner_uid":    self.miner_uid,
            "signal_name":  self.signal_name,
            "direction":    self.direction.value,
            "stake":        round(self.stake,      4),
            "confidence":   round(self.confidence, 4),
            "epoch":        self.epoch,
        }


@dataclass
class SettlementResult:
    """Settlement outcome for one position after epoch resolution."""
    position:      SignalPosition
    actual_ic:     float
    ic_positive:   bool       # True if IC > 0
    long_won:      bool       # True if LONG positions were correct
    prediction_correct: bool  # True if this position's direction was correct
    pnl_score:     float      # quadratic scoring rule P&L ∈ [-1, 1]

    def to_dict(self) -> dict:
        return {
            "miner_uid":          self.position.miner_uid,
            "signal_name":        self.position.signal_name,
            "direction":          self.position.direction.value,
            "actual_ic":          round(self.actual_ic,  6),
            "ic_positive":        self.ic_positive,
            "prediction_correct": self.prediction_correct,
            "pnl_score":          round(self.pnl_score,  6),
        }


# ── Market Book ───────────────────────────────────────────────────────────────

@dataclass
class MarketBook:
    """
    Order book for one signal — all open positions and their aggregate price.
    """
    signal_name:  str
    positions:    list[SignalPosition] = field(default_factory=list)

    # ── Price computation ─────────────────────────────────────────────────────

    def market_price(self) -> float:
        """
        Stake-weighted market price ∈ [-1, 1].
        Positive = bullish consensus (more/stronger LONG stakers).
        Negative = bearish consensus.
        """
        if not self.positions:
            return 0.0

        total_stake = sum(p.stake for p in self.positions)
        if total_stake < 1e-8:
            return 0.0

        weighted = sum(p.stake * p.signed_confidence for p in self.positions)
        return float(np.clip(weighted / total_stake, -1.0, 1.0))

    def confidence_score(self) -> float:
        """
        Market confidence ∈ [0, 1] — absolute value of market price.
        Used to weight signals in portfolio construction.
        """
        return abs(self.market_price())

    def bullish_fraction(self) -> float:
        """Fraction of stake on the LONG (bullish) side."""
        total = sum(p.stake for p in self.positions)
        if total < 1e-8:
            return 0.5
        long_stake = sum(p.stake for p in self.positions
                         if p.direction == PositionDirection.LONG)
        return long_stake / total

    def n_longs(self) -> int:
        return sum(1 for p in self.positions if p.direction == PositionDirection.LONG)

    def n_shorts(self) -> int:
        return sum(1 for p in self.positions if p.direction == PositionDirection.SHORT)

    def to_dict(self) -> dict:
        return {
            "signal_name":     self.signal_name,
            "n_positions":     len(self.positions),
            "market_price":    round(self.market_price(),      4),
            "confidence":      round(self.confidence_score(),  4),
            "bullish_fraction": round(self.bullish_fraction(), 4),
            "n_longs":         self.n_longs(),
            "n_shorts":        self.n_shorts(),
        }


# ── Signal Market ─────────────────────────────────────────────────────────────

class SignalMarket:
    """
    Prediction market for alpha signals.

    Maintains one MarketBook per signal. Miners open positions by
    calling open_position(). At epoch end, settle_epoch() compares
    predictions to actual ICs and computes settlement scores.
    """

    def __init__(self):
        self._books:  dict[str, MarketBook] = {}
        self._history: list[dict]            = []   # settled epoch records

    # ── Position management ───────────────────────────────────────────────────

    def open_position(self, position: SignalPosition) -> bool:
        """
        Register a miner's prediction for a signal.

        Args:
            position: SignalPosition to add to the market.

        Returns:
            True if position was accepted, False if invalid.
        """
        if position.stake < 1e-6:
            return False
        if position.signal_name not in self._books:
            self._books[position.signal_name] = MarketBook(position.signal_name)
        self._books[position.signal_name].positions.append(position)
        return True

    def open_positions_batch(self, positions: list[SignalPosition]) -> int:
        """Open multiple positions. Returns count of accepted positions."""
        return sum(1 for p in positions if self.open_position(p))

    # ── Market prices ─────────────────────────────────────────────────────────

    def price_signal(self, signal_name: str) -> float:
        """
        Current market price for one signal ∈ [-1, 1].
        Positive = bullish consensus, negative = bearish.
        Returns 0.0 if no positions exist.
        """
        book = self._books.get(signal_name)
        return book.market_price() if book else 0.0

    def confidence_score(self, signal_name: str) -> float:
        """
        Market confidence ∈ [0, 1] — how strongly the market has a view.
        Used to weight signals in portfolio construction.
        """
        book = self._books.get(signal_name)
        return book.confidence_score() if book else 0.0

    def all_prices(self) -> dict[str, float]:
        """Return {signal_name: market_price} for all tracked signals."""
        return {name: book.market_price() for name, book in self._books.items()}

    def all_confidences(self) -> dict[str, float]:
        """Return {signal_name: confidence_score} for all tracked signals."""
        return {name: book.confidence_score() for name, book in self._books.items()}

    # ── Settlement ────────────────────────────────────────────────────────────

    def settle_epoch(
        self,
        actual_ics: dict[str, float],
        epoch:      int = 0,
    ) -> list[SettlementResult]:
        """
        Settle all open positions against observed ICs.

        For each signal with an open book:
          - IC > 0 → LONG positions win
          - IC ≤ 0 → SHORT positions win

        Quadratic scoring rule P&L:
          correct direction:   pnl =  2 * confidence - 1  ∈ [0, 1]
          wrong direction:     pnl = -(2 * confidence - 1) ∈ [-1, 0]

        Args:
            actual_ics: {signal_name: mean_ic} from this epoch's evaluation.
            epoch:      Current epoch number (for record-keeping).

        Returns:
            List of SettlementResult, one per position.
        """
        results  = []
        epoch_summary = {"epoch": epoch, "n_settled": 0, "signals": {}}

        for signal_name, book in self._books.items():
            ic = actual_ics.get(signal_name)
            if ic is None or not book.positions:
                continue

            ic_positive = float(ic) > 0.0
            long_won    = ic_positive

            for pos in book.positions:
                correct  = (pos.direction == PositionDirection.LONG) == long_won
                # Quadratic scoring: incentive-compatible scoring rule
                c        = pos.confidence
                pnl      = (2 * c - 1) if correct else -(2 * c - 1)

                results.append(SettlementResult(
                    position=pos,
                    actual_ic=float(ic),
                    ic_positive=ic_positive,
                    long_won=long_won,
                    prediction_correct=correct,
                    pnl_score=pnl,
                ))

            epoch_summary["signals"][signal_name] = {
                "actual_ic": round(float(ic), 6),
                "long_won":  long_won,
                "n_settled": len(book.positions),
            }
            epoch_summary["n_settled"] += len(book.positions)

        self._history.append(epoch_summary)

        # Clear settled books for next epoch
        settled_names = list(epoch_summary["signals"].keys())
        for name in settled_names:
            self._books.pop(name, None)

        return results

    # ── Queries ───────────────────────────────────────────────────────────────

    def book(self, signal_name: str) -> Optional[MarketBook]:
        return self._books.get(signal_name)

    def tracked_signals(self) -> list[str]:
        return list(self._books.keys())

    def n_open_positions(self) -> int:
        return sum(len(b.positions) for b in self._books.values())

    def market_summary(self) -> list[dict]:
        return [b.to_dict() for b in self._books.values()]

    @property
    def epoch_history(self) -> list[dict]:
        return list(self._history)
