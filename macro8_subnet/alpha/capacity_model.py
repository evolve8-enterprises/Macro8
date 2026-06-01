"""
alpha/capacity_model.py
------------------------
Alpha Lifecycle + Capacity Model for the Macro8 platform.

Prevents the alpha decay trap: without lifecycle control, the library
fills with stale signals that degrade portfolio quality.

Three cooperating components
-----------------------------
    LifecycleEngine     Manages signal state transitions
    CapacityEstimator   Estimates signal weight and crowding
    DecayEstimator      Fits exponential decay to IC history

Lifecycle states
----------------
    EXPERIMENTAL  < MIN_EPOCHS observations — unproven
    VALIDATED     IC > threshold, MIN_EPOCHS consecutive epochs
    PRODUCTION    Validated + positive MSC + high IC stability
    DECAYING      IC declining persistently
    RETIRED       Decaying too long, or forced retirement

Portfolio weight adjustment
---------------------------
    adjusted_weight_i = IC_i × lifecycle_multiplier(state_i)
                               × capacity_score_i
                               × (1 - crowding_score_i)

    Lifecycle multipliers:
        EXPERIMENTAL: 0.30  (trial weight — don't commit capital)
        VALIDATED:    0.70  (reasonable allocation)
        PRODUCTION:   1.00  (full allocation)
        DECAYING:     0.30  (wind down)
        RETIRED:      0.00  (excluded from portfolio)

Decay estimation
----------------
Fits IC(t) ≈ IC₀ · exp(−λ·t) to the recent IC history.
    ic_half_life = ln(2) / λ   (in epochs)
    Short half-life → fast decay → DECAYING state
    Long half-life → persistent → stays PRODUCTION

Usage
-----
    engine = LifecycleEngine()

    # After each epoch, assess all formula records
    transitions = engine.assess_all(formula_library.all_active())
    print(f"Transitions: {transitions}")

    # Get adjusted weight for portfolio construction
    weight = engine.adjusted_weight(formula_record, raw_ic=0.042)
    print(f"Portfolio weight modifier: {weight:.3f}")
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

_SUBNET = Path(__file__).resolve().parent.parent
_ROOT   = _SUBNET.parent
for p in [str(_SUBNET), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ── Lifecycle state ───────────────────────────────────────────────────────────

class LifecycleState(str, Enum):
    EXPERIMENTAL = "experimental"
    VALIDATED    = "validated"
    PRODUCTION   = "production"
    DECAYING     = "decaying"
    RETIRED      = "retired"

    @property
    def weight_multiplier(self) -> float:
        """Portfolio weight multiplier for this lifecycle state."""
        return {
            LifecycleState.EXPERIMENTAL: 0.30,
            LifecycleState.VALIDATED:    0.70,
            LifecycleState.PRODUCTION:   1.00,
            LifecycleState.DECAYING:     0.30,
            LifecycleState.RETIRED:      0.00,
        }[self]

    @property
    def is_active(self) -> bool:
        return self not in (LifecycleState.RETIRED,)


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class LifecycleTransition:
    """Records one state change for a formula."""
    formula_id:   str
    from_state:   LifecycleState
    to_state:     LifecycleState
    reason:       str
    epoch:        int

    def __str__(self) -> str:
        return (f"{self.formula_id[:8]} "
                f"{self.from_state.value} → {self.to_state.value} "
                f"({self.reason})")


@dataclass
class DecayEstimate:
    """Exponential decay fit for one formula's IC history."""
    formula_id:    str
    ic_half_life:  Optional[float]   # epochs; None if not enough data
    decay_rate:    float              # λ in IC(t) = IC₀·exp(−λ·t)
    ic0:           float              # estimated initial IC
    r_squared:     float              # goodness of fit [0, 1]
    is_decaying:   bool               # True if half_life < DECAY_THRESHOLD

    def to_dict(self) -> dict:
        return {
            "formula_id":   self.formula_id,
            "ic_half_life": round(self.ic_half_life, 2) if self.ic_half_life else None,
            "decay_rate":   round(self.decay_rate,   6),
            "is_decaying":  self.is_decaying,
            "r_squared":    round(self.r_squared,    4),
        }


@dataclass
class CapacityReport:
    """Capacity + crowding assessment for one formula."""
    formula_id:     str
    lifecycle:      LifecycleState
    capacity_score: float   # [0, 1] — how much weight this signal can bear
    crowding_score: float   # [0, 1] — how crowded (higher = worse)
    weight_modifier: float  # combined: lifecycle × capacity × (1-crowding)
    ic_half_life:   Optional[float]

    def to_dict(self) -> dict:
        return {
            "formula_id":      self.formula_id,
            "lifecycle":       self.lifecycle.value,
            "capacity_score":  round(self.capacity_score,  4),
            "crowding_score":  round(self.crowding_score,  4),
            "weight_modifier": round(self.weight_modifier, 4),
            "ic_half_life":    round(self.ic_half_life, 2) if self.ic_half_life else None,
        }


# ── Decay Estimator ───────────────────────────────────────────────────────────

class DecayEstimator:
    """
    Fits exponential decay to IC history.

    Model: IC(t) = IC₀ · exp(−λ · t)
    Transformed: log|IC(t)| = log|IC₀| − λ·t

    A linear regression on log-transformed IC gives λ and IC₀.
    """

    def __init__(
        self,
        min_obs:         int   = 4,
        decay_threshold: float = 10.0,   # half-life < this → decaying
    ):
        self.min_obs         = min_obs
        self.decay_threshold = decay_threshold

    def estimate(self, formula_id: str, ic_history: list[float]) -> DecayEstimate:
        """
        Estimate decay parameters from IC history.

        Args:
            formula_id:  Identifier for logging.
            ic_history:  List of IC values ordered by epoch.

        Returns:
            DecayEstimate with half-life, decay rate, and fit quality.
        """
        if len(ic_history) < self.min_obs:
            return DecayEstimate(
                formula_id=formula_id,
                ic_half_life=None, decay_rate=0.0,
                ic0=float(np.mean(ic_history)) if ic_history else 0.0,
                r_squared=0.0, is_decaying=False,
            )

        ics  = np.array(ic_history, dtype=float)
        n    = len(ics)
        t    = np.arange(n, dtype=float)

        # Use only non-zero, finite values
        mask = np.isfinite(ics) & (np.abs(ics) > 1e-8)
        if mask.sum() < self.min_obs:
            return DecayEstimate(
                formula_id=formula_id, ic_half_life=None,
                decay_rate=0.0, ic0=float(np.mean(ics[np.isfinite(ics)])),
                r_squared=0.0, is_decaying=False,
            )

        log_ic  = np.log(np.abs(ics[mask]))
        t_valid = t[mask]

        # Linear regression: log|IC| = a - λt
        try:
            coeffs  = np.polyfit(t_valid, log_ic, 1)
            lam     = float(-coeffs[0])   # positive λ = decay
            log_ic0 = float(coeffs[1])
            ic0     = float(np.exp(log_ic0))

            # R² of the log-linear fit
            y_hat   = np.polyval(coeffs, t_valid)
            ss_res  = float(np.sum((log_ic - y_hat) ** 2))
            ss_tot  = float(np.sum((log_ic - log_ic.mean()) ** 2))
            r2      = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0

            half_life = float(np.log(2) / lam) if lam > 1e-6 else None
            decaying  = (half_life is not None and
                         half_life < self.decay_threshold and
                         lam > 0)

        except Exception:
            return DecayEstimate(
                formula_id=formula_id, ic_half_life=None,
                decay_rate=0.0, ic0=float(np.mean(ics)),
                r_squared=0.0, is_decaying=False,
            )

        return DecayEstimate(
            formula_id=formula_id,
            ic_half_life=half_life,
            decay_rate=max(lam, 0.0),
            ic0=ic0,
            r_squared=float(np.clip(r2, 0.0, 1.0)),
            is_decaying=decaying,
        )


# ── Capacity Estimator ────────────────────────────────────────────────────────

class CapacityEstimator:
    """
    Estimates how much portfolio weight a signal can support.

    Capacity is derived from:
        - IC stability (fraction of positive IC epochs)
        - Observation count (more data = more confident)
        - Lifecycle state (production signals earn full capacity)
        - MSC history (signals that demonstrably help the portfolio)
    """

    # Minimum IC stability for capacity > baseline
    MIN_IC_STABILITY = 0.40

    def estimate(
        self,
        ic_history:    list[float],
        msc_history:   list[float],
        lifecycle:     LifecycleState,
        crowding:      float = 0.0,
    ) -> float:
        """
        Compute capacity score ∈ [0, 1].

        Args:
            ic_history:  IC values over epochs.
            msc_history: MSC values over epochs.
            lifecycle:   Current lifecycle state.
            crowding:    Crowding score from orthogonality filter.

        Returns:
            capacity_score ∈ [0, 1].
        """
        if lifecycle == LifecycleState.RETIRED:
            return 0.0

        n = len(ic_history)
        if n == 0:
            return 0.20 * lifecycle.weight_multiplier

        # IC stability component
        positive_ic = sum(1 for ic in ic_history if ic > 0)
        stability   = positive_ic / n

        # Observation confidence (more data = more confidence, up to 20 obs)
        obs_conf    = min(n / 20.0, 1.0)

        # MSC component (positive MSC = signal actually helps portfolio)
        if msc_history:
            msc_mean  = float(np.mean(msc_history))
            msc_score = float(np.clip((msc_mean + 0.1) / 0.2, 0.0, 1.0))
        else:
            msc_score = 0.5   # neutral default

        # Crowding penalty
        crowding_pen = float(1.0 - min(crowding, 1.0))

        # Combine: stability is most important
        raw_capacity = (
            0.50 * stability +
            0.25 * obs_conf +
            0.15 * msc_score +
            0.10 * crowding_pen
        )

        # Scale by lifecycle multiplier
        capacity = raw_capacity * lifecycle.weight_multiplier

        return float(np.clip(capacity, 0.0, 1.0))


# ── Lifecycle Engine ──────────────────────────────────────────────────────────

class LifecycleEngine:
    """
    Manages state transitions for formula records.

    Evaluates each formula's IC history, MSC history, and decay estimate
    to determine lifecycle state transitions. Applied once per epoch.

    State transition rules
    ----------------------
    EXPERIMENTAL → VALIDATED:
        mean_ic > min_ic AND n_obs >= min_epochs

    VALIDATED → PRODUCTION:
        ic_stability > 0.60 AND mean_msc >= 0

    PRODUCTION → DECAYING:
        decay_estimate.is_decaying OR ic_stability < 0.35

    DECAYING → RETIRED:
        has been DECAYING for retire_after_epochs

    Any → RETIRED (forced):
        mean_ic < min_ic_retire AND n_obs >= min_epochs
    """

    def __init__(
        self,
        min_ic:              float = 0.01,   # minimum IC for VALIDATED
        min_ic_retire:       float = 0.005,  # below this → force retire
        min_epochs:          int   = 3,      # observations before graduation
        min_ic_stability:    float = 0.60,   # for PRODUCTION
        retire_after_epochs: int   = 5,      # epochs in DECAYING before retire
    ):
        self.min_ic              = min_ic
        self.min_ic_retire       = min_ic_retire
        self.min_epochs          = min_epochs
        self.min_ic_stability    = min_ic_stability
        self.retire_after_epochs = retire_after_epochs

        self._decay_estimator    = DecayEstimator()
        self._capacity_estimator = CapacityEstimator()

        # Persistent state
        self._states:    dict[str, LifecycleState] = {}
        self._decay_epochs: dict[str, int]         = {}   # epochs in DECAYING

    # ── Public API ────────────────────────────────────────────────────────────

    def assess(
        self,
        formula_id:  str,
        ic_history:  list[float],
        msc_history: list[float],
        crowding:    float = 0.0,
        epoch:       int   = 0,
    ) -> tuple[LifecycleState, Optional[LifecycleTransition], CapacityReport]:
        """
        Assess one formula and apply lifecycle transitions.

        Args:
            formula_id:  Unique identifier.
            ic_history:  List of IC values (oldest first).
            msc_history: List of MSC values.
            crowding:    Crowding score from orthogonality.
            epoch:       Current epoch number.

        Returns:
            (new_state, transition_or_None, capacity_report)
        """
        prev_state  = self._states.get(formula_id, LifecycleState.EXPERIMENTAL)
        decay_est   = self._decay_estimator.estimate(formula_id, ic_history)
        new_state   = self._transition(formula_id, prev_state, ic_history,
                                        msc_history, decay_est, epoch)

        # Update persistent state
        self._states[formula_id] = new_state

        # Track DECAYING duration
        if new_state == LifecycleState.DECAYING:
            self._decay_epochs[formula_id] = \
                self._decay_epochs.get(formula_id, 0) + 1
        else:
            self._decay_epochs.pop(formula_id, None)

        # Build transition record
        transition = None
        if new_state != prev_state:
            transition = LifecycleTransition(
                formula_id=formula_id,
                from_state=prev_state,
                to_state=new_state,
                reason=self._transition_reason(prev_state, new_state),
                epoch=epoch,
            )

        # Capacity report
        capacity = self._capacity_estimator.estimate(
            ic_history, msc_history, new_state, crowding
        )
        report = CapacityReport(
            formula_id=formula_id,
            lifecycle=new_state,
            capacity_score=capacity,
            crowding_score=crowding,
            weight_modifier=float(capacity * (1.0 - min(crowding, 1.0))),
            ic_half_life=decay_est.ic_half_life,
        )

        return new_state, transition, report

    def assess_all(
        self,
        formula_records: list,   # list of FormulaRecord
        epoch:           int = 0,
    ) -> list[LifecycleTransition]:
        """
        Assess all formula records and return all transitions.

        Args:
            formula_records: List of FormulaRecord from FormulaLibrary.
            epoch:           Current epoch number.

        Returns:
            List of LifecycleTransition for any state changes.
        """
        transitions = []
        for rec in formula_records:
            # Compute crowding from correlation with library signals
            # (simplified: use ic_stability as crowding proxy)
            ic_hist  = getattr(rec, "ic_history",  [])
            msc_hist = getattr(rec, "msc_history", [])
            crowding = 0.0   # no orthogonality context here; caller can provide

            _, transition, capacity_report = self.assess(
                rec.formula_id, ic_hist, msc_hist, crowding, epoch
            )
            if transition:
                transitions.append(transition)

        return transitions

    def adjusted_weight(
        self,
        formula_id:  str,
        raw_ic:      float,
        crowding:    float = 0.0,
    ) -> float:
        """
        Compute lifecycle-adjusted portfolio weight for a formula.

        adjusted_weight = raw_ic × lifecycle_multiplier × (1 - crowding)

        Args:
            formula_id: Formula identifier.
            raw_ic:     Raw IC score from evaluation.
            crowding:   Crowding score [0, 1].

        Returns:
            Adjusted weight ∈ [0, raw_ic].
        """
        state      = self._states.get(formula_id, LifecycleState.EXPERIMENTAL)
        multiplier = state.weight_multiplier
        return float(max(raw_ic, 0.0) * multiplier * (1.0 - min(crowding, 1.0)))

    def state_of(self, formula_id: str) -> LifecycleState:
        return self._states.get(formula_id, LifecycleState.EXPERIMENTAL)

    def summary(self) -> dict[str, int]:
        """Count of formulas in each lifecycle state."""
        counts: dict[str, int] = {s.value: 0 for s in LifecycleState}
        for state in self._states.values():
            counts[state.value] += 1
        return counts

    # ── State machine ─────────────────────────────────────────────────────────

    def _transition(
        self,
        formula_id:  str,
        prev_state:  LifecycleState,
        ic_history:  list[float],
        msc_history: list[float],
        decay_est:   DecayEstimate,
        epoch:       int,
    ) -> LifecycleState:
        """Apply state machine rules and return new state."""
        n = len(ic_history)
        if n == 0:
            return LifecycleState.EXPERIMENTAL

        mean_ic      = float(np.mean(ic_history))
        stability    = float(sum(1 for ic in ic_history if ic > 0) / n)
        mean_msc     = float(np.mean(msc_history)) if msc_history else 0.0

        # ── Force retirement (overrides all other transitions) ────────────────
        if n >= self.min_epochs and mean_ic < self.min_ic_retire:
            return LifecycleState.RETIRED

        # ── Check decay duration ──────────────────────────────────────────────
        if (prev_state == LifecycleState.DECAYING and
                self._decay_epochs.get(formula_id, 0) >= self.retire_after_epochs):
            return LifecycleState.RETIRED

        # ── State-specific transitions ────────────────────────────────────────
        if prev_state == LifecycleState.EXPERIMENTAL:
            if n >= self.min_epochs and mean_ic >= self.min_ic:
                return LifecycleState.VALIDATED
            return LifecycleState.EXPERIMENTAL

        if prev_state == LifecycleState.VALIDATED:
            if decay_est.is_decaying or stability < 0.35:
                return LifecycleState.DECAYING
            if stability >= self.min_ic_stability and mean_msc >= 0:
                return LifecycleState.PRODUCTION
            return LifecycleState.VALIDATED

        if prev_state == LifecycleState.PRODUCTION:
            if decay_est.is_decaying or stability < 0.35:
                return LifecycleState.DECAYING
            return LifecycleState.PRODUCTION

        if prev_state == LifecycleState.DECAYING:
            # Can recover if IC improves
            if stability >= 0.50 and not decay_est.is_decaying:
                return LifecycleState.VALIDATED
            return LifecycleState.DECAYING

        return prev_state

    @staticmethod
    def _transition_reason(
        from_state: LifecycleState,
        to_state:   LifecycleState,
    ) -> str:
        reasons = {
            (LifecycleState.EXPERIMENTAL, LifecycleState.VALIDATED):  "IC > threshold for min epochs",
            (LifecycleState.VALIDATED,    LifecycleState.PRODUCTION):  "high stability + positive MSC",
            (LifecycleState.PRODUCTION,   LifecycleState.DECAYING):    "IC declining / low stability",
            (LifecycleState.VALIDATED,    LifecycleState.DECAYING):    "IC declining",
            (LifecycleState.DECAYING,     LifecycleState.RETIRED):     "prolonged decay",
            (LifecycleState.DECAYING,     LifecycleState.VALIDATED):   "IC recovered",
        }
        return reasons.get((from_state, to_state), f"{from_state.value}→{to_state.value}")
