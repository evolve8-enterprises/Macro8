"""
evaluation/signal_scorer.py
----------------------------
Multi-Component Signal Scoring Engine for Macro8.

Scores each submitted alpha formula across five dimensions:

    Component         Weight   What it measures
    ──────────────────────────────────────────────────────────────
    IC score          40%      Predictive power (Spearman rank IC)
    Stability score   20%      Consistency (IC IR — mean/std)
    Decay score       15%      Longevity (IC half-life)
    Novelty score     15%      Orthogonality (low correlation with library)
    Capacity score    10%      Scalability (turnover-adjusted capacity)
    ──────────────────────────────────────────────────────────────
    Total             100%

Then multiplied by lifecycle weight:
    final_reward = composite_score × lifecycle_multiplier

Why multi-component scoring is critical
----------------------------------------
Single-metric systems (IC only) are immediately gamed:
    - Miners overfit to historical data → high IC, zero out-of-sample
    - Miners submit correlated formulas → rewards without discovery
    - Miners find short-lived patterns → IC spikes then collapses

The five-component filter is much harder to game simultaneously:
    - High IC + low stability → noise (penalised)
    - High IC + fast decay    → overfit (penalised)
    - High IC + high corr     → crowded (penalised)
    - High IC + low capacity  → tiny market (penalised)

Real alpha looks like:
    IC = moderate (0.02–0.06)
    stability = high (IC_IR > 0.5)
    decay = slow (half-life > 20 epochs)
    novelty = high (correlation < 0.4 with library)
    capacity = large (low turnover)

EMA weight smoothing
--------------------
Validators should not swing weights every epoch.
    weight_t = α × reward_t + (1−α) × weight_{t−1}
Default α = 0.20 (slow-moving, stable allocations).

Usage
-----
    scorer = SignalScorer(prices, library_signals)

    result = scorer.score(
        formula="rank(momentum_20d) - rank(volatility_60d)",
        formula_id="abc123",
        ic_history=[0.04, 0.035, 0.042, 0.038],  # from library
        msc_history=[0.01, 0.008, 0.012],
    )

    print(f"reward={result.final_reward:.4f}")
    print(result.breakdown())
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

from macro8_subnet.alpha.capacity_model import (
    DecayEstimator, CapacityEstimator, LifecycleEngine, LifecycleState,
)
from macro8_subnet.alpha.batch_evaluator  import BatchEvaluator
from macro8_subnet.alpha.portfolio_evaluator import PortfolioEvaluator, PortfolioResult
from macro8_subnet.alpha.orthogonality    import OrthogonalityFilter


# ── Score weights — adjustable without code changes ──────────────────────────

DEFAULT_WEIGHTS = {
    "ic":        0.40,
    "stability": 0.20,
    "decay":     0.15,
    "novelty":   0.15,
    "capacity":  0.10,
}

# EMA smoothing coefficient (α): lower = more smoothing
DEFAULT_EMA_ALPHA = 0.20

# Normalisation anchors — IC considered "good" at this value (score=1.0)
IC_REFERENCE      = 0.05   # 5% IC → full IC score
STABILITY_REFERENCE = 1.0  # IC_IR of 1.0 → full stability score
HALF_LIFE_FULL    = 30.0   # half-life ≥ 30 epochs → full decay score
MAX_CAPACITY_LOG  = 10.0   # log(1 + max_capacity) normaliser


# ── Per-component result types ────────────────────────────────────────────────

@dataclass
class ScoreComponent:
    """One dimension of the multi-component score."""
    name:     str
    raw:      float    # raw metric value (IC, IC_IR, half-life, etc.)
    score:    float    # normalised to [0, 1]
    weight:   float    # contribution weight
    weighted: float    # score × weight

    def to_dict(self) -> dict:
        return {
            "name":     self.name,
            "raw":      round(self.raw,      6),
            "score":    round(self.score,    4),
            "weight":   self.weight,
            "weighted": round(self.weighted, 6),
        }


@dataclass
class SignalScoreResult:
    """
    Complete scoring result for one alpha signal.

    Attributes
    ----------
    formula_id       : str — unique formula identifier
    formula_string   : str — the formula text
    composite_score  : float ∈ [0, 1] — weighted sum of components
    lifecycle_state  : LifecycleState — current lifecycle stage
    lifecycle_mult   : float — multiplier from lifecycle state
    final_reward     : float ∈ [0, 1] — composite × lifecycle_mult
    components       : list[ScoreComponent] — per-dimension breakdown
    ema_weight       : float — EMA-smoothed weight (after update)
    success          : bool — False if evaluation failed
    error            : str — error message if success=False
    """
    formula_id:       str
    formula_string:   str
    composite_score:  float
    lifecycle_state:  LifecycleState
    lifecycle_mult:   float
    final_reward:     float
    components:       list[ScoreComponent]   = field(default_factory=list)
    ema_weight:       float                  = 0.0
    success:          bool                   = True
    error:            str                    = ""

    def breakdown(self) -> str:
        """One-line score breakdown for logging."""
        parts = " | ".join(
            f"{c.name}={c.score:.3f}(×{c.weight})"
            for c in self.components
        )
        return (
            f"{self.formula_string[:35]:<35} "
            f"composite={self.composite_score:.4f} "
            f"× {self.lifecycle_state.value}={self.lifecycle_mult:.2f} "
            f"→ reward={self.final_reward:.4f}  [{parts}]"
        )

    def to_dict(self) -> dict:
        return {
            "formula_id":      self.formula_id,
            "formula_string":  self.formula_string,
            "composite_score": round(self.composite_score, 6),
            "lifecycle":       self.lifecycle_state.value,
            "lifecycle_mult":  round(self.lifecycle_mult, 3),
            "final_reward":    round(self.final_reward,   6),
            "ema_weight":      round(self.ema_weight,     6),
            "components":      [c.to_dict() for c in self.components],
            "success":         self.success,
        }


# ── Signal Scorer ─────────────────────────────────────────────────────────────

class SignalScorer:
    """
    Multi-component signal scoring engine.

    Evaluates each submitted formula across five dimensions and
    combines them into a single reward weight using configurable
    weights. Applies EMA smoothing across epochs.

    Parameters
    ----------
    batch_eval          : BatchEvaluator or PortfolioEvaluator — for IC/portfolio scoring
    orth_filter         : OrthogonalityFilter — for novelty
    lifecycle_engine    : LifecycleEngine — for lifecycle weighting
    decay_estimator     : DecayEstimator — for half-life
    weights             : Component weight dict (default: DEFAULT_WEIGHTS)
    ema_alpha           : EMA smoothing coefficient (default: 0.20)
    """

    def __init__(
        self,
        batch_eval:       BatchEvaluator,    # accepts PortfolioEvaluator (subclass)
        orth_filter:      Optional[OrthogonalityFilter] = None,
        lifecycle_engine: Optional[LifecycleEngine]     = None,
        decay_estimator:  Optional[DecayEstimator]      = None,
        weights:          Optional[dict]                = None,
        ema_alpha:        float                         = DEFAULT_EMA_ALPHA,
    ):
        self.batch_eval   = batch_eval
        self.orth_filter  = orth_filter  or OrthogonalityFilter(threshold=0.90)
        self.lifecycle    = lifecycle_engine or LifecycleEngine()
        self.decay_est    = decay_estimator  or DecayEstimator()
        self.cap_est      = CapacityEstimator()
        self.weights      = weights or dict(DEFAULT_WEIGHTS)
        self.ema_alpha    = ema_alpha

        # Validate weights sum to 1.0
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            self.weights = {k: v / total for k, v in self.weights.items()}

        # EMA state: formula_id → smoothed weight
        self._ema_weights: dict[str, float] = {}

        # Library signal cache for novelty scoring
        self._library_signals: dict[str, dict] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def score_batch(
        self,
        formulas:     list[str],
        formula_ids:  list[str],
        ic_histories: dict[str, list[float]],    # formula_id → list[float]
        msc_histories: dict[str, list[float]],   # formula_id → list[float]
        library_signals: Optional[dict] = None,  # {name: {asset: pd.Series}}
        epoch:        int = 0,
    ) -> list[SignalScoreResult]:
        """
        Score a batch of formulas.

        Args:
            formulas:        Formula strings in order.
            formula_ids:     Corresponding formula IDs.
            ic_histories:    Per-formula IC history dicts.
            msc_histories:   Per-formula MSC history dicts.
            library_signals: Current library for novelty scoring.
            epoch:           Current epoch number.

        Returns:
            List of SignalScoreResult, one per formula.
        """
        if library_signals:
            self._library_signals = library_signals

        # ── Step 1: Batch evaluation (IC + portfolio if PortfolioEvaluator) ──
        batch_result = self.batch_eval.evaluate(formulas)

        # Detect whether we have a full PortfolioResult
        is_portfolio = isinstance(batch_result, PortfolioResult)

        # Build per-formula IC map (always present — backward compat)
        batch_ic_map: dict[str, tuple] = {}
        for i, f in enumerate(batch_result.formulas):
            batch_ic_map[f] = (
                float(batch_result.mean_ics[i]),
                float(batch_result.ic_irs[i]),
            )

        # Build per-formula portfolio map (only if PortfolioEvaluator)
        portfolio_map: dict[str, object] = {}
        if is_portfolio:
            for ps in batch_result.portfolio_scores:
                portfolio_map[ps.formula] = ps

        # ── Step 2: Score each formula ────────────────────────────────────────
        results = []
        for formula, fid in zip(formulas, formula_ids):
            try:
                result = self._score_one(
                    formula, fid,
                    ic_histories.get(fid, []),
                    msc_histories.get(fid, []),
                    batch_ic_map.get(formula, (0.0, 0.0)),
                    epoch,
                    portfolio_score=portfolio_map.get(formula),
                )
            except Exception as e:
                result = SignalScoreResult(
                    formula_id=fid, formula_string=formula,
                    composite_score=0.0,
                    lifecycle_state=LifecycleState.EXPERIMENTAL,
                    lifecycle_mult=0.3, final_reward=0.0,
                    success=False, error=str(e)[:200],
                )
            results.append(result)

        return results

    def score_simple(
        self,
        formula:     str,
        formula_id:  str,
        ic_history:  list[float],
        msc_history: list[float] = (),
        epoch:       int         = 0,
    ) -> SignalScoreResult:
        """
        Score a single formula (convenience wrapper).
        Uses only the IC history provided — no batch evaluation.
        """
        if not ic_history:
            return SignalScoreResult(
                formula_id=formula_id, formula_string=formula,
                composite_score=0.0,
                lifecycle_state=LifecycleState.EXPERIMENTAL,
                lifecycle_mult=LifecycleState.EXPERIMENTAL.weight_multiplier,
                final_reward=0.0, success=False,
                error="No IC history",
            )

        mean_ic = float(np.mean(ic_history))
        std_ic  = float(np.std(ic_history))
        ic_ir   = mean_ic / (std_ic + 1e-6) if std_ic > 0 else 0.0

        return self._score_one(
            formula, formula_id,
            ic_history, list(msc_history),
            (mean_ic, ic_ir),
            epoch,
        )

    def update_ema(self, formula_id: str, new_reward: float) -> float:
        """
        Update the EMA weight for a formula and return the smoothed value.

        weight_t = α × reward_t + (1−α) × weight_{t−1}
        """
        prev = self._ema_weights.get(formula_id, new_reward)
        smoothed = self.ema_alpha * new_reward + (1 - self.ema_alpha) * prev
        self._ema_weights[formula_id] = smoothed
        return smoothed

    def get_ema_weight(self, formula_id: str) -> float:
        return self._ema_weights.get(formula_id, 0.0)

    # ── Component scorers ─────────────────────────────────────────────────────

    def _ic_score(self, mean_ic: float) -> ScoreComponent:
        """
        Component 1: IC Score (40%)

        Normalised IC relative to a reference "good" IC level.
        References: IC=0.05 → score=1.0; IC=0 → score=0; IC<0 → score=0.

        Uses a soft saturation function so higher IC still improves score
        but with diminishing returns above the reference.
        """
        ic_clamped = max(mean_ic, 0.0)
        # Soft saturation: score = 1 - exp(-IC / IC_reference)
        # At IC=IC_REFERENCE: score ≈ 0.632
        # At IC=2×IC_REFERENCE: score ≈ 0.865
        raw_score = 1.0 - float(np.exp(-ic_clamped / IC_REFERENCE))
        score     = float(np.clip(raw_score, 0.0, 1.0))
        w         = self.weights["ic"]
        return ScoreComponent("ic", mean_ic, score, w, score * w)

    def _stability_score(self, ic_ir: float) -> ScoreComponent:
        """
        Component 2: Stability Score (20%)

        IC Information Ratio = mean(IC) / std(IC).
        Measures consistency. Noisy signals have low IC_IR even with high mean.

        Reference: IC_IR=1.0 → score=1.0 (standard risk-adjusted benchmark).
        """
        ir_clamped = max(ic_ir, 0.0)
        # Soft cap: score = 1 - exp(-IR / reference)
        score = float(np.clip(
            1.0 - np.exp(-ir_clamped / STABILITY_REFERENCE),
            0.0, 1.0,
        ))
        w = self.weights["stability"]
        return ScoreComponent("stability", ic_ir, score, w, score * w)

    def _decay_score(self, ic_history: list[float], formula_id: str) -> ScoreComponent:
        """
        Component 3: Decay Score (15%)

        IC half-life from exponential decay fit.
        Signals with longer half-life → more durable alpha → higher score.

        Reference: half-life ≥ 30 epochs → score=1.0 (full credit).
        """
        if len(ic_history) < 4:
            # Insufficient history — neutral score, not penalised
            score = 0.5
            return ScoreComponent("decay", 0.0, score, self.weights["decay"],
                                   score * self.weights["decay"])

        decay_est = self.decay_est.estimate(formula_id, ic_history)
        half_life = decay_est.ic_half_life or 0.0

        if half_life <= 0 or decay_est.decay_rate <= 0:
            # Non-decaying (stable) → full score
            score = 1.0
            raw   = float("inf")
        else:
            # Linear ramp up to HALF_LIFE_FULL
            score = float(np.clip(half_life / HALF_LIFE_FULL, 0.0, 1.0))
            raw   = half_life

        w = self.weights["decay"]
        return ScoreComponent("decay", raw, score, w, score * w)

    def _novelty_score(self, formula: str, signals: Optional[dict]) -> ScoreComponent:
        """
        Component 4: Novelty Score (15%)

        Orthogonality relative to the current alpha library.
        novelty_score = 1 - max_correlation_with_library

        Identical signals → max_corr=1.0 → novelty=0 (no reward for copies).
        Uncorrelated signal → max_corr=0 → novelty=1 (full reward).
        """
        if not signals or not self._library_signals:
            # No library yet — new signal is trivially novel
            score = 1.0
            raw   = 0.0
        else:
            try:
                max_corr, _ = self.orth_filter.correlate_with_library(
                    formula, signals, self._library_signals
                )
                max_corr = float(np.clip(abs(max_corr or 0.0), 0.0, 1.0))
                score    = float(np.clip(1.0 - max_corr, 0.0, 1.0))
                raw      = max_corr
            except Exception:
                score = 0.5   # neutral on error
                raw   = 0.0

        w = self.weights["novelty"]
        return ScoreComponent("novelty", raw, score, w, score * w)

    def _capacity_score(self, ic_history: list[float], msc_history: list[float],
                         lifecycle: LifecycleState) -> ScoreComponent:
        """
        Component 5: Capacity Score (10%)

        Approximates signal scalability via:
            capacity = CapacityEstimator.estimate(...)
        which uses IC stability, MSC, and observation count.

        Higher capacity → signal can support more capital → higher weight.
        """
        cap_val = self.cap_est.estimate(ic_history, msc_history, lifecycle)
        score   = float(np.clip(cap_val, 0.0, 1.0))
        w       = self.weights["capacity"]
        return ScoreComponent("capacity", cap_val, score, w, score * w)

    # ── Core scoring logic ────────────────────────────────────────────────────

    def _regime_score(
        self,
        regime_histories: dict[str, list[float]],
    ) -> ScoreComponent:
        """
        Component 6 (optional): Regime Robustness Score.

        Penalises signals that only work in one market regime.
        Real alpha is robust: it earns positive IC across bull, bear,
        and volatile periods.

        Scoring rule: min(positive_regime_ICs) across all observed regimes.
        A signal that has IC=0.05 in bull markets but IC=-0.03 in bear markets
        earns a low regime score because of the negative bear-market IC.

        If fewer than 2 regimes are populated, returns a neutral 0.5 score
        (insufficient data to penalise).

        regime_histories: {regime_name: [ic_values_in_that_regime]}
        """
        w = self.weights.get("regime", 0.0)
        if not regime_histories or w == 0:
            return ScoreComponent("regime", 0.0, 0.5, w, 0.5 * w)

        # Only consider regimes with enough observations
        regime_means = {
            r: float(np.mean(ics))
            for r, ics in regime_histories.items()
            if len(ics) >= 3
        }

        if len(regime_means) < 2:
            return ScoreComponent("regime", 0.0, 0.5, w, 0.5 * w)

        # Worst-regime IC (we want signals that are positive everywhere)
        worst_ic = min(regime_means.values())
        # Score: clamp worst_ic to [0, IC_REFERENCE], normalise
        score = float(np.clip(
            1.0 - np.exp(-max(worst_ic, 0.0) / IC_REFERENCE),
            0.0, 1.0,
        ))
        return ScoreComponent("regime", worst_ic, score, w, score * w)

    def _score_one(
        self,
        formula:          str,
        formula_id:       str,
        ic_history:       list[float],
        msc_history:      list[float],
        batch_ic:         tuple[float, float],   # (mean_ic, ic_ir) from evaluator
        epoch:            int,
        regime_histories: Optional[dict[str, list[float]]] = None,
        portfolio_score:  Optional[object] = None,  # PortfolioScore | None
    ) -> SignalScoreResult:
        """
        Score one formula across all components.

        When `portfolio_score` is provided (PortfolioEvaluator in use),
        the IC score component is replaced with the multi-horizon portfolio
        composite, and a 6th scalability component is added.
        """
        mean_ic, ic_ir = batch_ic

        # If we have IC history, blend with batch eval result
        if ic_history:
            hist_mean = float(np.mean(ic_history))
            hist_std  = float(np.std(ic_history))
            hist_ir   = hist_mean / (hist_std + 1e-6) if hist_std > 0 else 0.0
            mean_ic = 0.60 * mean_ic + 0.40 * hist_mean
            ic_ir   = 0.60 * ic_ir   + 0.40 * hist_ir
            full_history = ic_history + [batch_ic[0]] if batch_ic[0] != 0 else ic_history
        else:
            full_history = [mean_ic] if mean_ic != 0 else []

        # ── Lifecycle state ───────────────────────────────────────────────────
        lifecycle_state, _, _ = self.lifecycle.assess(
            formula_id, full_history, list(msc_history), epoch=epoch
        )
        lifecycle_mult = lifecycle_state.weight_multiplier

        # ── Core scoring: IC or multi-horizon portfolio ───────────────────────
        if portfolio_score is not None:
            # Multi-horizon IC replaces single-day IC score
            # Use portfolio's weighted IC (1d×0.4 + 7d×0.3 + 30d×0.2 + 90d×0.1)
            c1 = self._ic_score(portfolio_score.ic_weighted)

            # Stability: use Sharpe-based score instead of pure IC-IR
            # Soft-saturate Sharpe at 0.5 reference
            sharpe_clamped = max(portfolio_score.sharpe, 0.0)
            sharpe_score   = 1.0 - np.exp(-sharpe_clamped / 0.5)
            w2 = self.weights.get("stability", 0.20)
            c2 = ScoreComponent(
                "stability",
                portfolio_score.sharpe,
                float(sharpe_score),
                w2,
                float(sharpe_score * w2),
            )
        else:
            c1 = self._ic_score(mean_ic)
            c2 = self._stability_score(ic_ir)

        c3 = self._decay_score(full_history, formula_id)
        c4 = self._novelty_score(formula, None)
        c5 = self._capacity_score(full_history, list(msc_history), lifecycle_state)

        components = [c1, c2, c3, c4, c5]

        # ── Optional regime robustness component ──────────────────────────────
        if regime_histories and self.weights.get("regime", 0) > 0:
            c6 = self._regime_score(regime_histories)
            components.append(c6)

        # ── Optional scalability component (portfolio only) ───────────────────
        if portfolio_score is not None:
            cap_1m    = portfolio_score.capital_scores.get(1_000_000, 0.0)
            turnover  = portfolio_score.daily_turnover

            # Scalability: prefer positive cap_1m + low turnover
            # Soft-saturate — cap_1m in range [-∞, +∞], normalise to [0,1]
            cap_score  = float(np.clip(cap_1m / (abs(cap_1m) + 0.5) * 0.5 + 0.5, 0.0, 1.0))
            # Turnover penalty: high turnover → low scalability
            turn_score = float(np.clip(1.0 - turnover / 0.15, 0.0, 1.0))
            scale_val  = 0.6 * cap_score + 0.4 * turn_score

            w_scale = self.weights.get("scalability", 0.10)
            c_scale = ScoreComponent(
                "scalability",
                cap_1m,
                scale_val,
                w_scale,
                float(scale_val * w_scale),
            )
            components.append(c_scale)

        # ── Composite score ───────────────────────────────────────────────────
        composite = float(np.clip(sum(c.weighted for c in components), 0.0, 1.0))
        final     = float(np.clip(composite * lifecycle_mult, 0.0, 1.0))

        # ── EMA smoothing ─────────────────────────────────────────────────────
        ema = self.update_ema(formula_id, final)

        return SignalScoreResult(
            formula_id=formula_id,
            formula_string=formula,
            composite_score=composite,
            lifecycle_state=lifecycle_state,
            lifecycle_mult=lifecycle_mult,
            final_reward=final,
            components=components,
            ema_weight=ema,
            success=True,
        )


# ── Reward normalisation ──────────────────────────────────────────────────────

def normalise_rewards(
    scores:      dict[str, float],   # formula_id → final_reward
    temperature: float = 2.0,        # softmax sharpness
) -> dict[str, float]:
    """
    Softmax-normalise final rewards into allocation weights summing to 1.0.

    Higher temperature → winner-takes-more allocation.

    Args:
        scores:      {formula_id: final_reward}
        temperature: Softmax sharpness.

    Returns:
        {formula_id: allocation_weight} with weights summing to 1.0.
    """
    if not scores:
        return {}

    ids    = list(scores.keys())
    vals   = np.array([max(scores[i], 0.0) for i in ids], dtype=np.float64)
    total  = vals.sum()

    if total < 1e-8:
        equal = 1.0 / len(ids)
        return {i: equal for i in ids}

    # Softmax
    scaled  = vals * temperature
    shifted = scaled - scaled.max()
    exp_v   = np.exp(np.clip(shifted, -500, 500))
    weights = exp_v / exp_v.sum()

    return dict(zip(ids, [float(w) for w in weights]))
