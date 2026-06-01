"""
evaluation/leaderboard.py
--------------------------
Strategy Discovery Leaderboard for Macro8.

Tracks and ranks submitted signals across four dimensions, making
the subnet transparent and encouraging miners to diversify.

Four leaderboard categories
----------------------------
    🏆  Top IC Signals        — highest mean IC (raw predictive power)
    ⏳  Longest Surviving     — formulas still active after N epochs
    🔀  Most Orthogonal       — lowest correlation with rest of library
    💰  Highest Capacity      — most scalable (best capacity score)

Why a leaderboard matters
--------------------------
    1. Transparency: miners can see what kinds of signals earn rewards.
       This guides research in productive directions.

    2. Diversity incentive: showing the "most orthogonal" category
       explicitly rewards miners who explore different signal families.

    3. Regime signals: a separate "best across regimes" track rewards
       signals that survive multiple market conditions.

    4. Metagraph health: leaderboard entries reveal subnet concentration
       (is one miner dominating?) and signal library quality.

Usage
-----
    board = Leaderboard()

    # After each epoch, register results
    for formula_id, result in scored_results.items():
        board.register(result, epoch=epoch)

    # Print current state
    board.print_full()

    # Query programmatically
    top_ic = board.top_by_ic(5)
    survivors = board.top_by_longevity(10)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

_SUBNET = Path(__file__).resolve().parent.parent
_ROOT   = _SUBNET.parent
for p in [str(_SUBNET), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from macro8_subnet.evaluation.signal_scorer import SignalScoreResult
from macro8_subnet.alpha.capacity_model     import LifecycleState


# ── Leaderboard Entry ─────────────────────────────────────────────────────────

@dataclass
class LeaderboardEntry:
    """
    One formula's complete leaderboard record.

    Updated each epoch a formula is submitted and evaluated.
    """
    formula_id:      str
    formula_string:  str
    miner_uid:       int
    miner_hotkey:    str              = ""

    # Performance metrics (best ever)
    best_ic:         float            = 0.0
    best_stability:  float            = 0.0    # IC-IR
    best_capacity:   float            = 0.0
    novelty_score:   float            = 1.0    # 1 - max_correlation
    regime_score:    float            = 0.5    # min regime IC (normalised)

    # Current state
    current_reward:  float            = 0.0
    ema_weight:      float            = 0.0
    lifecycle:       LifecycleState   = LifecycleState.EXPERIMENTAL
    n_epochs_active: int              = 0      # epochs since first submission
    n_epochs_production: int          = 0      # epochs in PRODUCTION state

    # History
    epoch_born:      int              = 0
    epoch_last_seen: int              = 0
    ic_history:      list[float]      = field(default_factory=list)
    reward_history:  list[float]      = field(default_factory=list)

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def mean_ic(self) -> float:
        return float(np.mean(self.ic_history)) if self.ic_history else 0.0

    @property
    def ic_ir(self) -> float:
        if len(self.ic_history) < 2:
            return 0.0
        std = float(np.std(self.ic_history))
        return self.mean_ic / (std + 1e-6) if std > 0 else 0.0

    @property
    def is_active(self) -> bool:
        return self.lifecycle not in (LifecycleState.RETIRED,)

    @property
    def longevity_score(self) -> float:
        """Normalised longevity: n_epochs_active / 50 (full credit at 50 epochs)."""
        return float(min(self.n_epochs_active / 50.0, 1.0))

    def to_dict(self) -> dict:
        return {
            "formula_id":          self.formula_id,
            "formula_string":      self.formula_string[:60],
            "miner_uid":           self.miner_uid,
            "mean_ic":             round(self.mean_ic,       6),
            "ic_ir":               round(self.ic_ir,         4),
            "best_ic":             round(self.best_ic,       6),
            "novelty_score":       round(self.novelty_score, 4),
            "capacity_score":      round(self.best_capacity, 4),
            "regime_score":        round(self.regime_score,  4),
            "n_epochs_active":     self.n_epochs_active,
            "lifecycle":           self.lifecycle.value,
            "current_reward":      round(self.current_reward, 6),
            "ema_weight":          round(self.ema_weight,     6),
        }

    def one_liner(self, rank: int, category: str) -> str:
        bar_len = int(self.current_reward * 30)
        bar     = "█" * bar_len
        return (
            f"  #{rank:<3} │ uid={self.miner_uid:<4} │ "
            f"IC={self.mean_ic:6.4f} │ "
            f"IR={self.ic_ir:5.2f} │ "
            f"age={self.n_epochs_active:3d}ep │ "
            f"{self.lifecycle.value:<12} │ "
            f"{bar:<30} "
            f"{'  ' + self.formula_string[:30]}"
        )


# ── Leaderboard ───────────────────────────────────────────────────────────────

class Leaderboard:
    """
    Multi-dimensional signal discovery leaderboard.

    Maintains one LeaderboardEntry per formula, updated each epoch.
    Provides four sorted views:
        top_by_ic()         — highest raw IC
        top_by_longevity()  — longest surviving signals
        top_by_novelty()    — most orthogonal to library
        top_by_capacity()   — highest capacity scores
    """

    def __init__(self, max_entries: int = 500):
        self.max_entries = max_entries
        self._entries:   dict[str, LeaderboardEntry] = {}
        self._epoch:     int = 0

    # ── Update ────────────────────────────────────────────────────────────────

    def register(
        self,
        result:    SignalScoreResult,
        miner_uid: int = 0,
        hotkey:    str = "",
        epoch:     int = 0,
    ) -> LeaderboardEntry:
        """
        Register or update a formula's leaderboard entry.

        Args:
            result:    SignalScoreResult from the scoring engine.
            miner_uid: Submitting miner's UID.
            hotkey:    Miner's SS58 address.
            epoch:     Current epoch number.

        Returns:
            Updated LeaderboardEntry.
        """
        self._epoch = max(self._epoch, epoch)
        fid         = result.formula_id

        if fid not in self._entries:
            entry = LeaderboardEntry(
                formula_id=fid,
                formula_string=result.formula_string,
                miner_uid=miner_uid,
                miner_hotkey=hotkey,
                epoch_born=epoch,
            )
            self._entries[fid] = entry
        else:
            entry = self._entries[fid]

        # Extract component scores
        ic_comp       = next((c for c in result.components if c.name == "ic"),       None)
        stab_comp     = next((c for c in result.components if c.name == "stability"), None)
        cap_comp      = next((c for c in result.components if c.name == "capacity"),  None)
        nov_comp      = next((c for c in result.components if c.name == "novelty"),   None)
        regime_comp   = next((c for c in result.components if c.name == "regime"),    None)

        raw_ic = ic_comp.raw if ic_comp else 0.0

        # Update bests
        entry.best_ic       = max(entry.best_ic,       raw_ic)
        entry.best_stability = max(entry.best_stability, stab_comp.raw if stab_comp else 0.0)
        entry.best_capacity  = max(entry.best_capacity,  cap_comp.score if cap_comp else 0.0)
        if nov_comp:
            entry.novelty_score = nov_comp.score
        if regime_comp:
            entry.regime_score  = regime_comp.score

        # Update current state
        entry.current_reward  = result.final_reward
        entry.ema_weight      = result.ema_weight
        entry.lifecycle       = result.lifecycle_state
        entry.epoch_last_seen = epoch

        # Update histories
        if raw_ic != 0:
            entry.ic_history.append(raw_ic)
            if len(entry.ic_history) > 50:
                entry.ic_history.pop(0)

        entry.reward_history.append(result.final_reward)
        if len(entry.reward_history) > 50:
            entry.reward_history.pop(0)

        # Increment counters
        entry.n_epochs_active += 1
        if result.lifecycle_state == LifecycleState.PRODUCTION:
            entry.n_epochs_production += 1

        # Evict oldest entries if over limit
        if len(self._entries) > self.max_entries:
            self._evict_oldest()

        return entry

    def retire(self, formula_id: str) -> None:
        """Mark a formula as retired."""
        entry = self._entries.get(formula_id)
        if entry:
            entry.lifecycle = LifecycleState.RETIRED

    # ── Ranked views ──────────────────────────────────────────────────────────

    def top_by_ic(self, n: int = 10) -> list[LeaderboardEntry]:
        """Top N by mean IC."""
        active = [e for e in self._entries.values() if e.is_active]
        return sorted(active, key=lambda e: e.mean_ic, reverse=True)[:n]

    def top_by_longevity(self, n: int = 10) -> list[LeaderboardEntry]:
        """Top N by epochs active (longest-surviving signals)."""
        active = [e for e in self._entries.values() if e.is_active]
        return sorted(active, key=lambda e: e.n_epochs_active, reverse=True)[:n]

    def top_by_novelty(self, n: int = 10) -> list[LeaderboardEntry]:
        """Top N by novelty (most orthogonal to the rest of library)."""
        active = [e for e in self._entries.values() if e.is_active]
        return sorted(active, key=lambda e: e.novelty_score, reverse=True)[:n]

    def top_by_capacity(self, n: int = 10) -> list[LeaderboardEntry]:
        """Top N by capacity score (most scalable signals)."""
        active = [e for e in self._entries.values() if e.is_active]
        return sorted(active, key=lambda e: e.best_capacity, reverse=True)[:n]

    def top_by_reward(self, n: int = 10) -> list[LeaderboardEntry]:
        """Top N by current EMA reward weight."""
        active = [e for e in self._entries.values() if e.is_active]
        return sorted(active, key=lambda e: e.ema_weight, reverse=True)[:n]

    # ── Statistics ────────────────────────────────────────────────────────────

    @property
    def n_total(self) -> int:
        return len(self._entries)

    @property
    def n_active(self) -> int:
        return sum(1 for e in self._entries.values() if e.is_active)

    @property
    def n_production(self) -> int:
        return sum(1 for e in self._entries.values()
                   if e.lifecycle == LifecycleState.PRODUCTION)

    @property
    def mean_ic(self) -> float:
        active = [e for e in self._entries.values() if e.is_active and e.mean_ic != 0]
        return float(np.mean([e.mean_ic for e in active])) if active else 0.0

    @property
    def n_unique_miners(self) -> int:
        return len({e.miner_uid for e in self._entries.values() if e.is_active})

    def concentration(self) -> float:
        """
        Herfindahl-Hirschman index of reward concentration.
        HHI = sum(share_i^2). Range [1/n, 1]. Lower = more diverse.
        """
        active     = [e for e in self._entries.values() if e.is_active]
        if not active:
            return 1.0
        total      = sum(e.ema_weight for e in active)
        if total < 1e-8:
            return 1.0 / len(active)
        shares     = [e.ema_weight / total for e in active]
        return float(sum(s**2 for s in shares))

    def stats(self) -> dict:
        return {
            "epoch":          self._epoch,
            "n_total":        self.n_total,
            "n_active":       self.n_active,
            "n_production":   self.n_production,
            "n_unique_miners": self.n_unique_miners,
            "mean_ic":        round(self.mean_ic, 6),
            "concentration":  round(self.concentration(), 4),
        }

    # ── Display ───────────────────────────────────────────────────────────────

    def print_full(self, top_n: int = 10) -> None:
        """Print the complete leaderboard with all four categories."""
        W = 80
        print(f"\n{'═'*W}")
        print(f"  🏆  MACRO8 STRATEGY DISCOVERY LEADERBOARD  "
              f"│ Epoch {self._epoch} │ {self.n_active} active signals")
        print(f"{'═'*W}")
        s = self.stats()
        print(f"  Total: {s['n_total']}  │  Active: {s['n_active']}  │  "
              f"Production: {s['n_production']}  │  "
              f"Miners: {s['n_unique_miners']}  │  "
              f"Mean IC: {s['mean_ic']:.4f}  │  "
              f"HHI: {s['concentration']:.3f}")
        print(f"{'─'*W}")

        categories = [
            ("🏆 Top IC Signals",       self.top_by_ic(top_n),       "ic"),
            ("⏳ Longest Surviving",    self.top_by_longevity(top_n), "longevity"),
            ("🔀 Most Orthogonal",      self.top_by_novelty(top_n),   "novelty"),
            ("💰 Highest Capacity",     self.top_by_capacity(top_n),  "capacity"),
        ]

        for title, entries, cat in categories:
            print(f"\n  {title}")
            print(f"  {'─'*75}")
            if not entries:
                print("  (no entries yet)")
            else:
                hdr = (f"  {'#':<4} │ {'uid':<6} │ {'IC':>7} │ "
                       f"{'IR':>6} │ {'age':>5} │ {'lifecycle':<12} │ "
                       f"{'reward':>8} │ formula")
                print(hdr)
                for rank, entry in enumerate(entries[:top_n], start=1):
                    print(entry.one_liner(rank, cat))

        print(f"\n{'═'*W}\n")

    def print_summary(self) -> None:
        """One-line summary suitable for validator epoch logging."""
        s = self.stats()
        top = self.top_by_ic(1)
        best = f"best={top[0].formula_string[:25]} IC={top[0].mean_ic:.4f}" if top else "empty"
        print(f"[Leaderboard] epoch={s['epoch']} active={s['n_active']} "
              f"production={s['n_production']} "
              f"mean_IC={s['mean_ic']:.4f} "
              f"HHI={s['concentration']:.3f} | {best}")

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "stats":          self.stats(),
            "top_ic":         [e.to_dict() for e in self.top_by_ic(20)],
            "top_longevity":  [e.to_dict() for e in self.top_by_longevity(20)],
            "top_novelty":    [e.to_dict() for e in self.top_by_novelty(20)],
            "top_capacity":   [e.to_dict() for e in self.top_by_capacity(20)],
        }

    # ── Eviction ──────────────────────────────────────────────────────────────

    def _evict_oldest(self) -> None:
        """Remove the oldest retired entries when over max_entries."""
        retired = [
            (e.epoch_last_seen, fid)
            for fid, e in self._entries.items()
            if e.lifecycle == LifecycleState.RETIRED
        ]
        if not retired:
            return
        retired.sort()
        n_to_remove = len(self._entries) - self.max_entries
        for _, fid in retired[:n_to_remove]:
            del self._entries[fid]
