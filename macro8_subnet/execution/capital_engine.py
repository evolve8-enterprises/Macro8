"""
execution/capital_engine.py
-----------------------------
Multi-horizon capital allocation engine.

Each signal has a natural rebalancing frequency. momentum_60d run daily
earns Sharpe +0.12; run at 90-day rebalancing it earns +0.89 — the same
signal, 7× better. The capital engine allocates capital across four
rebalancing schedules and learns, via softmax, which frequency is working.

Architecture
------------
Four parallel "buckets" share the same signal but trade at different speeds:

    1d  bucket  — rebalances every day   (fast, high turnover)
    7d  bucket  — rebalances every 5 days (weekly)
    30d bucket  — rebalances every 21 days (monthly)
    90d bucket  — rebalances every 63 days (quarterly, lowest cost)

At each reallocation event (default: weekly):
    1. Score each bucket by rolling Sharpe over the past N days
    2. Apply softmax to scores → new allocation weights
    3. Apply EMA smoothing to prevent over-reaction
    4. Enforce a minimum floor so no bucket is fully starved

Compounding
-----------
Each bucket's capital grows with its daily PnL:
    bucket_capital *= (1 + daily_pnl)

Total capital = sum of all bucket capitals.

Usage
-----
    engine = CapitalEngine(initial_capital=100_000, signal_name="momentum_60d")
    engine.record_pnl("2024-01-15", {"1d": 0.002, "7d": 0.003, "30d": 0.001, "90d": 0.004})
    engine.maybe_reallocate("2024-01-15")
    allocations = engine.get_allocations()   # {"1d": 25000, "7d": 30000, ...}
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ── Constants ──────────────────────────────────────────────────────────────────

HORIZONS: list[str] = ["1d", "7d", "30d", "90d"]

# Rebalancing period in trading days per horizon
REBAL_DAYS: dict[str, int] = {
    "1d":  1,
    "7d":  5,
    "30d": 21,
    "90d": 63,
}

# How often (in trading days) to run the softmax reallocation
DEFAULT_REALLOC_FREQ = 5   # weekly

# EMA weight for reallocation smoothing (α): lower = more stable weights
# 0.20 means new allocation is 20% new signal, 80% previous allocation
REALLOC_EMA_ALPHA = 0.20

# Minimum fraction of capital any bucket may hold (prevents starvation)
MIN_BUCKET_WEIGHT = 0.05   # 5%

# Rolling window for bucket Sharpe estimation (in trading days)
SHARPE_WINDOW = 60


# ── CapitalEngine ──────────────────────────────────────────────────────────────

class CapitalEngine:
    """
    Allocates capital across four rebalancing-frequency buckets for one signal.

    Parameters
    ----------
    initial_capital : float
        Total starting capital in dollars.
    signal_name : str
        Name of the signal this engine manages (used for logging).
    realloc_freq : int
        How often (in trading days) to run the softmax reallocation.
    state_path : Path | None
        If provided, persist state to disk (survives restarts).
    """

    def __init__(
        self,
        initial_capital: float = 100_000,
        signal_name:     str   = "signal",
        realloc_freq:    int   = DEFAULT_REALLOC_FREQ,
        state_path:      Optional[Path] = None,
    ):
        self.signal_name   = signal_name
        self.realloc_freq  = realloc_freq
        self.state_path    = state_path

        # Capital split equally at start
        initial_each = initial_capital / len(HORIZONS)
        self.buckets: dict[str, float] = {h: initial_each for h in HORIZONS}

        # EMA weights (initialised to equal)
        self._weights: np.ndarray = np.ones(len(HORIZONS)) / len(HORIZONS)

        # PnL history: {horizon: [daily_pnl, ...]}
        self._pnl_history: dict[str, list[float]] = {h: [] for h in HORIZONS}

        # Day counter (for reallocation scheduling)
        self._day_count: int = 0

        # Rebalance counters per horizon bucket
        self._days_since_rebal: dict[str, int] = {h: 0 for h in HORIZONS}

        # Load persisted state if available
        if self.state_path and Path(self.state_path).exists():
            self._load_state()

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def total_capital(self) -> float:
        """Total capital across all buckets."""
        return sum(self.buckets.values())

    def record_pnl(
        self,
        date:          str,
        pnl_by_horizon: dict[str, float],
    ) -> None:
        """
        Record each bucket's daily PnL and compound its capital.

        Args:
            date:            Trading date string (YYYY-MM-DD).
            pnl_by_horizon:  {horizon: daily_return} where daily_return is
                             the portfolio log-return for that bucket today.
        """
        for h in HORIZONS:
            ret = float(pnl_by_horizon.get(h, 0.0))
            # Compound: bucket_t = bucket_{t-1} × (1 + ret)
            self.buckets[h] = self.buckets[h] * (1.0 + ret)
            # Store history for Sharpe estimation (max SHARPE_WINDOW days)
            hist = self._pnl_history[h]
            hist.append(ret)
            if len(hist) > SHARPE_WINDOW:
                hist.pop(0)

        self._day_count += 1

        # Increment rebalance counters
        for h in HORIZONS:
            self._days_since_rebal[h] += 1

        if self.state_path:
            self._save_state()

    def should_rebalance(self, horizon: str) -> bool:
        """
        Returns True if bucket `horizon` is due for a position update.
        Called daily before computing new positions.
        """
        due = self._days_since_rebal.get(horizon, 0) >= REBAL_DAYS[horizon]
        if due:
            self._days_since_rebal[horizon] = 0
        return due

    def maybe_reallocate(self, date: str) -> bool:
        """
        Run softmax reallocation if the reallocation period has elapsed.

        Returns True if reallocation was performed.
        """
        if self._day_count % self.realloc_freq != 0:
            return False

        # Estimate rolling Sharpe for each bucket
        scores = self._bucket_sharpes()
        self._reallocate(scores)

        if self.state_path:
            self._save_state()

        return True

    def get_allocations(self) -> dict[str, float]:
        """
        Return current capital per bucket.

        Returns
        -------
        {horizon: capital_dollars}
        """
        return dict(self.buckets)

    def get_weights(self) -> dict[str, float]:
        """
        Return current allocation weights (sum to 1).
        """
        total = self.total_capital
        if total < 1e-8:
            return {h: 1.0/len(HORIZONS) for h in HORIZONS}
        return {h: self.buckets[h] / total for h in HORIZONS}

    def summary(self) -> str:
        """One-line summary for logging."""
        total = self.total_capital
        w     = self.get_weights()
        parts = "  ".join(f"{h}={w[h]:.0%}" for h in HORIZONS)
        return (f"[CapitalEngine:{self.signal_name}] "
                f"total=${total:,.0f}  {parts}")

    # ── Internal ───────────────────────────────────────────────────────────────

    def _bucket_sharpes(self) -> dict[str, float]:
        """
        Compute rolling Sharpe for each bucket over the past SHARPE_WINDOW days.
        Returns a floored value of 0 for buckets with insufficient history.
        """
        sharpes = {}
        for h in HORIZONS:
            hist = self._pnl_history[h]
            if len(hist) < 5:
                sharpes[h] = 0.0   # no history: neutral score
                continue
            arr   = np.array(hist)
            mu    = arr.mean()
            sigma = arr.std() + 1e-9
            ann   = float(mu / sigma * math.sqrt(252))
            sharpes[h] = ann
        return sharpes

    def _reallocate(self, scores: dict[str, float]) -> None:
        """
        Apply softmax over bucket Sharpe scores to get new allocation weights.

        Steps:
        1. Softmax over raw scores
        2. EMA blend with previous weights (stability)
        3. Clip to minimum floor
        4. Renormalise to sum = 1
        5. Redistribute total capital proportionally
        """
        score_arr = np.array([scores[h] for h in HORIZONS], dtype=float)

        # Softmax (temperature=1 — no sharpening needed, EMA handles stability)
        # Subtract max for numerical stability
        score_arr -= score_arr.max()
        raw_weights = np.exp(score_arr)
        raw_weights /= raw_weights.sum()

        # EMA smoothing: blend with current weights
        new_weights = (
            REALLOC_EMA_ALPHA * raw_weights
            + (1.0 - REALLOC_EMA_ALPHA) * self._weights
        )

        # Minimum floor: no bucket below MIN_BUCKET_WEIGHT
        new_weights = np.clip(new_weights, MIN_BUCKET_WEIGHT, None)
        new_weights /= new_weights.sum()

        self._weights = new_weights

        # Redistribute total capital according to new weights
        total = self.total_capital
        for i, h in enumerate(HORIZONS):
            self.buckets[h] = total * float(new_weights[i])

    def _save_state(self) -> None:
        """Persist state to JSON for restart recovery."""
        if not self.state_path:
            return
        state = {
            "buckets":          self.buckets,
            "weights":          self._weights.tolist(),
            "pnl_history":      self._pnl_history,
            "day_count":        self._day_count,
            "days_since_rebal": self._days_since_rebal,
        }
        Path(self.state_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.state_path).write_text(json.dumps(state, indent=2))

    def _load_state(self) -> None:
        """Restore persisted state."""
        try:
            state = json.loads(Path(self.state_path).read_text())
            self.buckets           = state["buckets"]
            self._weights          = np.array(state["weights"])
            self._pnl_history      = state["pnl_history"]
            self._day_count        = state["day_count"]
            self._days_since_rebal = state["days_since_rebal"]
        except Exception:
            pass   # corrupt state: start fresh


# ── MultiSignalCapitalEngine ───────────────────────────────────────────────────

class MultiSignalCapitalEngine:
    """
    Manages one CapitalEngine per active signal and combines their positions.

    This is the top-level class used by PaperTrader and live_runner.
    Each signal gets its own rebalancing schedule; capital flows toward
    signals+frequencies that are currently compounding best.

    Parameters
    ----------
    signal_names : list[str]
        Active signal formulas or short names.
    initial_capital : float
        Total capital split equally across signals at start.
    state_dir : Path | None
        Directory for per-signal state files.
    """

    def __init__(
        self,
        signal_names:    list[str],
        initial_capital: float = 100_000,
        state_dir:       Optional[Path] = None,
    ):
        self.signal_names = signal_names
        n = len(signal_names)
        per_signal = initial_capital / max(n, 1)

        self.engines: dict[str, CapitalEngine] = {}
        for name in signal_names:
            # Sanitise name for filename use
            safe = name[:20].replace(" ", "_").replace("/", "_")
            sp   = (Path(state_dir) / f"{safe}_capital.json"
                    if state_dir else None)
            self.engines[name] = CapitalEngine(
                initial_capital=per_signal,
                signal_name=name,
                state_path=sp,
            )

    @property
    def total_capital(self) -> float:
        return sum(e.total_capital for e in self.engines.values())

    def record_pnl(
        self,
        date:            str,
        pnl_by_signal:   dict[str, dict[str, float]],
    ) -> None:
        """
        Record PnL for each signal × horizon pair.

        Args:
            date:           Trading date.
            pnl_by_signal:  {signal_name: {horizon: daily_return}}
        """
        for name, engine in self.engines.items():
            pnl = pnl_by_signal.get(name, {h: 0.0 for h in HORIZONS})
            engine.record_pnl(date, pnl)
            engine.maybe_reallocate(date)

    def signal_capital(self, signal_name: str) -> float:
        """Total capital allocated to one signal across all horizons."""
        e = self.engines.get(signal_name)
        return e.total_capital if e else 0.0

    def horizon_capital(self, signal_name: str, horizon: str) -> float:
        """Capital in one bucket for one signal."""
        e = self.engines.get(signal_name)
        return e.buckets.get(horizon, 0.0) if e else 0.0

    def should_rebalance(self, signal_name: str, horizon: str) -> bool:
        """True if this signal/horizon bucket is due for a position update."""
        e = self.engines.get(signal_name)
        return e.should_rebalance(horizon) if e else True

    def print_summary(self) -> None:
        for name, engine in self.engines.items():
            print(engine.summary())
