"""
alpha/hypothesis_engine.py
---------------------------
The Macro8 Hypothesis Engine — turns the network from a signal search
system into a market science engine.

Every alpha signal is backed by a hypothesis: a falsifiable statement
about market behaviour. As signals are evaluated, the hypothesis gains
or loses confidence. Over many epochs the network builds a knowledge
base of which economic principles actually produce alpha.

Core scientific objects
-----------------------
    HypothesisRecord    — one testable claim about markets
    BayesianUpdater     — updates confidence from IC observations
    HypothesisLibrary   — persistent store, query interface
    HypothesisEvolution — guides formula evolution toward strong hypotheses

Example flow
------------
Epoch 1:
    Miner submits: "Momentum predicts returns" + formula rank(momentum_20d)
    IC evaluated: 0.032
    → hypothesis.confidence updates: 0.50 → 0.55

Epoch 20:
    Same hypothesis, but regime-conditional IC shows:
    LOW_VOL_TREND: IC=0.071 | RISK_OFF: IC=-0.012
    → hypothesis.regime_notes updated: "works in low-vol trending"
    → confidence: 0.68

Epoch 50:
    10 supporting signals, mostly positive IC
    → confidence: 0.79 → hypothesis "confirmed"

Design: Beta-Binomial conjugate model
--------------------------------------
Every hypothesis maintains a Beta(α, β) posterior over P(IC > threshold).
    - Prior: Beta(1, 1) — flat, uninformed
    - Update: IC > threshold → α += 1 (success)
              IC ≤ threshold → β += 1 (failure)
    - Confidence = posterior mean = α / (α + β)

This is the simplest proper Bayesian model for a binary outcome.
It converges to the true success probability as evidence accumulates,
starts at 0.5 (maximum uncertainty), and never reaches 0 or 1.

Regime-conditional tracking
-----------------------------
IC is broken down by market regime (using RegimeDetector output).
This reveals "when" hypotheses work, not just "if":
    LOW_VOL_TREND:    IC=0.071  (momentum works here)
    HIGH_VOL_TREND:   IC=0.031  (weaker)
    RISK_OFF:         IC=-0.012 (fails here)
    INFLATION:        IC=0.044  (moderate)
    LIQUIDITY_CRISIS: IC=-0.028 (fails)

HypothesisEvolution
--------------------
Uses hypothesis strength to bias formula mutation:
    - Strong hypotheses (conf > 0.65) → suggest their feature family
    - Weak hypotheses (conf < 0.35) → avoid their feature family
    - Seed formulas from high-confidence hypotheses
    - Weight tournament selection toward hypothesis-consistent formulas
"""

from __future__ import annotations

import hashlib
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

_SUBNET = Path(__file__).resolve().parent.parent
_ROOT   = _SUBNET.parent
for p in [str(_SUBNET), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ── Enums ─────────────────────────────────────────────────────────────────────

class HypothesisCategory(str, Enum):
    """Economic category of a market hypothesis."""
    MOMENTUM     = "momentum"
    MEAN_REVERSION = "mean_reversion"
    MACRO        = "macro"
    RISK         = "risk"
    CARRY        = "carry"
    VOLATILITY   = "volatility"
    CROSS_ASSET  = "cross_asset"
    REGIME       = "regime"
    UNKNOWN      = "unknown"

    def suggested_features(self) -> list[str]:
        """Feature store features most relevant to this category."""
        mapping = {
            HypothesisCategory.MOMENTUM:     ["momentum_5d", "momentum_20d", "momentum_60d",
                                               "cross_momentum"],
            HypothesisCategory.MEAN_REVERSION: ["zscore_20d", "zscore_60d", "rsi_14"],
            HypothesisCategory.MACRO:         ["regime_signal", "momentum_60d"],
            HypothesisCategory.RISK:          ["volatility_20d", "volatility_60d",
                                               "relative_vol"],
            HypothesisCategory.CARRY:         ["momentum_60d", "relative_vol"],
            HypothesisCategory.VOLATILITY:    ["volatility_10d", "volatility_20d",
                                               "volatility_60d", "relative_vol"],
            HypothesisCategory.CROSS_ASSET:   ["cross_momentum", "relative_vol",
                                               "regime_signal"],
            HypothesisCategory.REGIME:        ["regime_signal", "momentum_20d",
                                               "volatility_20d"],
            HypothesisCategory.UNKNOWN:       ["momentum_20d"],
        }
        return mapping.get(self, ["momentum_20d"])


class HypothesisStatus(str, Enum):
    ACTIVE   = "active"
    RETIRED  = "retired"    # confidence fell below threshold
    ARCHIVED = "archived"   # superseded by a more refined hypothesis
    PENDING  = "pending"    # not enough evidence yet


# ── HypothesisRecord ──────────────────────────────────────────────────────────

@dataclass
class HypothesisRecord:
    """
    One testable market hypothesis with Bayesian confidence tracking.

    Attributes
    ----------
    hypothesis_id       : str — unique identifier (hash of statement)
    statement           : str — human-readable testable claim
    category            : HypothesisCategory
    miner_uid           : int — originator
    epoch_born          : int
    supporting_signals  : list[str] — alpha library signal names that support this
    ic_history          : list[float] — IC per epoch since hypothesis was born
    regime_ic           : dict[str, list[float]] — IC broken down by regime name
    alpha_param         : float — Beta distribution α (successes + 1)
    beta_param          : float — Beta distribution β (failures + 1)
    ic_threshold        : float — IC above this = "success" for Bayesian update
    epoch_last_updated  : int
    status              : HypothesisStatus
    tags                : list[str] — free-form metadata tags
    notes               : str — accumulated observations
    """
    hypothesis_id:     str
    statement:         str
    category:          HypothesisCategory
    miner_uid:         int
    epoch_born:        int
    supporting_signals: list[str]           = field(default_factory=list)
    ic_history:        list[float]          = field(default_factory=list)
    regime_ic:         dict[str, list[float]] = field(default_factory=dict)
    alpha_param:       float                = 1.0    # Beta prior: uninformed
    beta_param:        float                = 1.0
    ic_threshold:      float                = 0.02
    epoch_last_updated: int                 = 0
    status:            HypothesisStatus     = HypothesisStatus.PENDING
    tags:              list[str]            = field(default_factory=list)
    notes:             str                  = ""

    # ── Bayesian confidence ───────────────────────────────────────────────────

    @property
    def confidence_score(self) -> float:
        """
        Posterior mean of P(IC > threshold): α / (α + β).
        Range: (0, 1). New hypothesis = 0.5. Perfect track record → 1.0.
        """
        return float(self.alpha_param / (self.alpha_param + self.beta_param))

    @property
    def confidence_interval(self) -> tuple[float, float]:
        """95% credible interval for confidence using Beta distribution."""
        from scipy.stats import beta as beta_dist
        lo = float(beta_dist.ppf(0.025, self.alpha_param, self.beta_param))
        hi = float(beta_dist.ppf(0.975, self.alpha_param, self.beta_param))
        return lo, hi

    @property
    def n_observations(self) -> int:
        return len(self.ic_history)

    @property
    def n_successes(self) -> int:
        return sum(1 for ic in self.ic_history if ic > self.ic_threshold)

    @property
    def mean_ic(self) -> float:
        return float(np.mean(self.ic_history)) if self.ic_history else 0.0

    @property
    def ic_stability(self) -> float:
        """Fraction of epochs with positive IC."""
        if not self.ic_history:
            return 0.0
        return float(sum(1 for ic in self.ic_history if ic > 0) / len(self.ic_history))

    def best_regime(self) -> Optional[str]:
        """Return the regime with the highest average IC, or None."""
        if not self.regime_ic:
            return None
        avgs = {r: float(np.mean(ics)) for r, ics in self.regime_ic.items() if ics}
        if not avgs:
            return None
        return max(avgs, key=avgs.get)

    def worst_regime(self) -> Optional[str]:
        """Return the regime with the lowest average IC, or None."""
        if not self.regime_ic:
            return None
        avgs = {r: float(np.mean(ics)) for r, ics in self.regime_ic.items() if ics}
        if not avgs:
            return None
        return min(avgs, key=avgs.get)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        lo, hi = self.confidence_interval
        return {
            "hypothesis_id":     self.hypothesis_id,
            "statement":         self.statement,
            "category":          self.category.value,
            "miner_uid":         self.miner_uid,
            "epoch_born":        self.epoch_born,
            "status":            self.status.value,
            "confidence_score":  round(self.confidence_score, 4),
            "confidence_lo":     round(lo, 4),
            "confidence_hi":     round(hi, 4),
            "n_observations":    self.n_observations,
            "n_successes":       self.n_successes,
            "mean_ic":           round(self.mean_ic, 6),
            "ic_stability":      round(self.ic_stability, 4),
            "best_regime":       self.best_regime(),
            "worst_regime":      self.worst_regime(),
            "supporting_signals": self.supporting_signals,
            "tags":              self.tags,
            "notes":             self.notes,
        }

    def knowledge_entry(self) -> str:
        """One-line summary for the knowledge base display."""
        conf     = f"{self.confidence_score:.2f}"
        best_r   = self.best_regime() or "all"
        return (f"[{self.category.value:<12}] {self.statement[:55]:<55} "
                f"| conf={conf} | IC={self.mean_ic:.3f} | best={best_r}")


# ── Bayesian Updater ──────────────────────────────────────────────────────────

class BayesianUpdater:
    """
    Updates hypothesis confidence from IC observations using
    Beta-Binomial conjugate Bayesian inference.

    The Beta distribution is the conjugate prior for a Bernoulli
    likelihood, making this analytically tractable:
        prior:     Beta(α, β)
        likelihood: IC > threshold → success (1), else failure (0)
        posterior: Beta(α + success, β + (1-success))

    This is the simplest proper Bayesian model for "does this work?"
    """

    def __init__(self, ic_threshold: float = 0.02):
        self.ic_threshold = ic_threshold

    def update(
        self,
        record:       HypothesisRecord,
        ic:           float,
        regime:       Optional[str] = None,
        epoch:        int           = 0,
    ) -> HypothesisRecord:
        """
        Update a HypothesisRecord with one new IC observation.

        Modifies the record in-place and returns it.

        Args:
            record:  HypothesisRecord to update.
            ic:      Observed IC value for this epoch.
            regime:  Current market regime name (optional).
            epoch:   Current epoch number.

        Returns:
            Updated HypothesisRecord (modified in-place).
        """
        # Record raw IC
        record.ic_history.append(float(ic))
        record.epoch_last_updated = epoch

        # Bayesian update: success if IC > threshold
        if ic > self.ic_threshold:
            record.alpha_param += 1.0   # success
        else:
            record.beta_param  += 1.0   # failure

        # Regime-conditional tracking
        if regime:
            if regime not in record.regime_ic:
                record.regime_ic[regime] = []
            record.regime_ic[regime].append(float(ic))

        # Update status based on confidence and evidence
        record.status = self._determine_status(record)

        return record

    def update_batch(
        self,
        record:        HypothesisRecord,
        ic_values:     list[float],
        regimes:       Optional[list[str]] = None,
        epoch:         int                 = 0,
    ) -> HypothesisRecord:
        """Update with multiple IC observations at once."""
        for i, ic in enumerate(ic_values):
            regime = regimes[i] if regimes and i < len(regimes) else None
            self.update(record, ic, regime, epoch)
        return record

    def predict_next_ic(
        self,
        record:     HypothesisRecord,
        regime:     Optional[str] = None,
    ) -> float:
        """
        Predict next-period IC using the posterior mean (confidence × threshold).

        If regime is specified, uses regime-conditional history if available.
        """
        if regime and regime in record.regime_ic and record.regime_ic[regime]:
            # Use regime-specific history
            return float(np.mean(record.regime_ic[regime]))
        # Fall back to overall posterior mean IC
        return record.mean_ic * record.confidence_score

    @staticmethod
    def _determine_status(record: HypothesisRecord) -> HypothesisStatus:
        """Determine hypothesis lifecycle status from evidence."""
        n = record.n_observations
        c = record.confidence_score
        if n < 3:
            return HypothesisStatus.PENDING
        elif c < 0.30:
            return HypothesisStatus.RETIRED
        elif c > 0.50:
            return HypothesisStatus.ACTIVE
        else:
            return HypothesisStatus.PENDING


# ── Hypothesis Library ────────────────────────────────────────────────────────

class HypothesisLibrary:
    """
    Persistent store and query interface for market hypotheses.

    The library is the network's growing knowledge base — each epoch
    adds evidence for or against market theories.
    """

    def __init__(self):
        self._records: dict[str, HypothesisRecord] = {}

    # ── Admission ─────────────────────────────────────────────────────────────

    def add(
        self,
        statement:  str,
        category:   HypothesisCategory,
        miner_uid:  int,
        epoch:      int,
        tags:       list[str] = (),
        ic_threshold: float   = 0.02,
    ) -> HypothesisRecord:
        """
        Register a new hypothesis.
        If an identical statement already exists, returns the existing record.

        Returns:
            The new (or existing) HypothesisRecord.
        """
        hid = self._make_id(statement)
        if hid in self._records:
            return self._records[hid]

        record = HypothesisRecord(
            hypothesis_id=hid,
            statement=statement,
            category=category,
            miner_uid=miner_uid,
            epoch_born=epoch,
            ic_threshold=ic_threshold,
            tags=list(tags),
        )
        self._records[hid] = record
        return record

    def add_supporting_signal(self, hypothesis_id: str, signal_name: str) -> None:
        """Link a library signal as evidence for a hypothesis."""
        rec = self._records.get(hypothesis_id)
        if rec and signal_name not in rec.supporting_signals:
            rec.supporting_signals.append(signal_name)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get(self, hypothesis_id: str) -> Optional[HypothesisRecord]:
        return self._records.get(hypothesis_id)

    def get_by_statement(self, statement: str) -> Optional[HypothesisRecord]:
        hid = self._make_id(statement)
        return self._records.get(hid)

    def all_active(self) -> list[HypothesisRecord]:
        """Return all non-retired, non-archived hypotheses."""
        return [r for r in self._records.values()
                if r.status not in (HypothesisStatus.RETIRED,
                                    HypothesisStatus.ARCHIVED)]

    def by_category(self, category: HypothesisCategory) -> list[HypothesisRecord]:
        return [r for r in self.all_active() if r.category == category]

    def rank_by_confidence(self, top_n: int = 10) -> list[HypothesisRecord]:
        """Return top N hypotheses sorted by confidence descending."""
        active = self.all_active()
        return sorted(active, key=lambda r: r.confidence_score, reverse=True)[:top_n]

    def rank_by_ic(self, top_n: int = 10) -> list[HypothesisRecord]:
        """Return top N by mean IC descending."""
        active = self.all_active()
        return sorted(active, key=lambda r: r.mean_ic, reverse=True)[:top_n]

    def pending(self) -> list[HypothesisRecord]:
        return [r for r in self._records.values()
                if r.status == HypothesisStatus.PENDING]

    def retired(self) -> list[HypothesisRecord]:
        return [r for r in self._records.values()
                if r.status == HypothesisStatus.RETIRED]

    # ── Updates ───────────────────────────────────────────────────────────────

    def update_with_ic(
        self,
        hypothesis_id: str,
        ic:            float,
        regime:        Optional[str] = None,
        epoch:         int           = 0,
    ) -> Optional[HypothesisRecord]:
        """Update hypothesis confidence with a new IC observation."""
        rec = self._records.get(hypothesis_id)
        if rec is None:
            return None
        updater = BayesianUpdater(ic_threshold=rec.ic_threshold)
        return updater.update(rec, ic, regime, epoch)

    def retire(self, hypothesis_id: str) -> None:
        rec = self._records.get(hypothesis_id)
        if rec:
            rec.status = HypothesisStatus.RETIRED

    def archive(self, hypothesis_id: str, note: str = "") -> None:
        rec = self._records.get(hypothesis_id)
        if rec:
            rec.status = HypothesisStatus.ARCHIVED
            if note:
                rec.notes += f" [ARCHIVED: {note}]"

    # ── Knowledge base ────────────────────────────────────────────────────────

    def knowledge_base(self) -> list[dict]:
        """Return all active hypotheses as a structured knowledge base."""
        return [r.to_dict() for r in self.rank_by_confidence()]

    def print_knowledge_base(self, top_n: int = 20) -> None:
        """Print a formatted knowledge base summary."""
        records  = self.rank_by_confidence(top_n)
        print(f"\n  📚  MACRO8 KNOWLEDGE BASE  ({self.size} hypotheses)")
        print(f"  {'─'*72}")
        print(f"  {'Statement':<55} {'Conf':>5} {'IC':>7} {'n':>4}")
        print(f"  {'─'*72}")
        for r in records:
            print(f"  {r.statement[:55]:<55} "
                  f"{r.confidence_score:5.2f} "
                  f"{r.mean_ic:7.4f} "
                  f"{r.n_observations:4d}")
        print(f"  {'─'*72}\n")

    # ── Statistics ────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._records)

    @property
    def n_active(self) -> int:
        return len(self.all_active())

    @property
    def n_retired(self) -> int:
        return len(self.retired())

    @property
    def mean_confidence(self) -> float:
        active = self.all_active()
        if not active:
            return 0.0
        return float(np.mean([r.confidence_score for r in active]))

    @staticmethod
    def _make_id(statement: str) -> str:
        """Deterministic ID from statement text."""
        return hashlib.sha256(statement.strip().lower().encode()).hexdigest()[:12]


# ── Hypothesis Evolution ──────────────────────────────────────────────────────

class HypothesisEvolution:
    """
    Uses hypothesis confidence to guide formula evolution.

    High-confidence hypotheses suggest which features to explore.
    Low-confidence hypotheses suggest features to avoid.
    The evolution engine is biased toward hypothesis-consistent mutations.
    """

    # Minimum confidence to actively suggest features
    SUGGEST_THRESHOLD = 0.60
    # Maximum confidence below which we actively avoid features
    AVOID_THRESHOLD   = 0.35

    def __init__(self, library: HypothesisLibrary):
        self.library = library

    # ── Feature guidance ──────────────────────────────────────────────────────

    def suggested_features(self) -> list[str]:
        """
        Return features suggested by high-confidence hypotheses.

        Features from strong hypotheses appear multiple times → higher
        probability of selection in evolution mutations.
        """
        features = []
        for rec in self.library.all_active():
            if rec.confidence_score >= self.SUGGEST_THRESHOLD:
                # Weight by confidence: more confidence → more repetitions
                weight = int((rec.confidence_score - self.SUGGEST_THRESHOLD) * 10) + 1
                features.extend(rec.category.suggested_features() * weight)
        return features if features else ["momentum_20d", "cross_momentum"]

    def avoided_features(self) -> list[str]:
        """Features to avoid based on low-confidence hypotheses."""
        features = set()
        for rec in self.library.all_active():
            if rec.confidence_score < self.AVOID_THRESHOLD:
                features.update(rec.category.suggested_features())
        return list(features)

    def seed_formulas(self, n: int = 5) -> list[str]:
        """
        Generate seed formula strings from high-confidence hypotheses.

        These are used to initialise or refresh the evolution population
        with hypothesis-grounded starting points.
        """
        top     = self.library.rank_by_confidence(top_n=min(n * 2, 10))
        seeds   = []
        rng     = random.Random(42)

        for rec in top:
            if len(seeds) >= n:
                break
            feats  = rec.category.suggested_features()
            if not feats:
                continue
            # Generate a simple formula template per hypothesis type
            f1     = rng.choice(feats)
            f2     = rng.choice(feats)
            if rec.category in (HypothesisCategory.MOMENTUM,
                                HypothesisCategory.CARRY):
                seeds.append(f"rank({f1})")
                if f1 != f2:
                    seeds.append(f"rank({f1}) - rank({f2})")
            elif rec.category in (HypothesisCategory.MEAN_REVERSION,
                                  HypothesisCategory.VOLATILITY):
                seeds.append(f"zscore({f1})")
            elif rec.category == HypothesisCategory.REGIME:
                seeds.append(f"regime_signal * {f1}")
            else:
                seeds.append(f"cross_momentum")

        return seeds[:n]

    def regime_filtered_seeds(
        self,
        target_regime: str,
        n: int = 3,
    ) -> list[str]:
        """
        Seed formulas focused on hypotheses that work in a specific regime.

        Args:
            target_regime: Regime name to filter by.
            n:             Number of seeds to generate.
        """
        # Find hypotheses that perform well in this regime
        regime_performers = []
        for rec in self.library.all_active():
            if target_regime in rec.regime_ic and rec.regime_ic[target_regime]:
                avg_ic = float(np.mean(rec.regime_ic[target_regime]))
                regime_performers.append((avg_ic, rec))

        if not regime_performers:
            return self.seed_formulas(n)

        regime_performers.sort(reverse=True)
        seeds = []
        for _, rec in regime_performers[:n]:
            feats = rec.category.suggested_features()
            if feats:
                f = random.choice(feats)
                seeds.append(f"regime_signal * {f}")

        return seeds[:n]

    def weight_population(
        self,
        formulas: list[str],
        feature_store_names: list[str],
    ) -> list[float]:
        """
        Assign probability weights to a formula population based on
        how well they align with high-confidence hypotheses.

        Args:
            formulas:            List of formula strings in the population.
            feature_store_names: All valid feature names.

        Returns:
            List of probability weights (sum = 1.0) for tournament selection.
        """
        suggested = set(self.suggested_features())
        avoided   = set(self.avoided_features())

        weights = []
        for formula in formulas:
            score = 1.0
            # Boost for suggested features
            for feat in suggested:
                if feat in formula:
                    score += 0.5
            # Penalty for avoided features
            for feat in avoided:
                if feat in formula and feat not in suggested:
                    score *= 0.5
            weights.append(max(score, 0.1))

        total = sum(weights)
        return [w / total for w in weights] if total > 0 else [1.0 / len(formulas)] * len(formulas)


# ── Submission and Report types ───────────────────────────────────────────────

@dataclass
class HypothesisSubmission:
    """
    A miner's hypothesis submission.

    Miners can submit new hypotheses alongside (or independently of)
    formula submissions. Each hypothesis is a testable claim about
    markets with a suggested formula implementation.
    """
    miner_uid:       int
    miner_hotkey:    str
    statement:       str                    # "Momentum predicts 5-day returns"
    category:        HypothesisCategory
    supporting_formula: str                 # matching formula string
    tags:            list[str]              = field(default_factory=list)
    description:     str                    = ""
    ic_threshold:    float                  = 0.02   # miner's own success threshold

    def is_valid(self) -> tuple[bool, str]:
        if not self.statement.strip():
            return False, "Empty statement"
        if len(self.statement) < 10:
            return False, f"Statement too short: {len(self.statement)} chars"
        if len(self.statement) > 200:
            return False, f"Statement too long: {len(self.statement)} chars"
        return True, ""

    def to_dict(self) -> dict:
        return {
            "miner_uid":           self.miner_uid,
            "statement":           self.statement,
            "category":            self.category.value,
            "supporting_formula":  self.supporting_formula,
            "tags":                self.tags,
        }


@dataclass
class HypothesisReport:
    """
    Per-epoch hypothesis engine report.

    Included in EpochReport as a new field.
    """
    epoch:              int
    n_updated:          int
    n_newly_active:     int     # became ACTIVE this epoch
    n_retired:          int     # became RETIRED this epoch
    n_total:            int
    top_5:              list[dict]        # top 5 by confidence
    knowledge_entries:  list[str]         # one-line summaries
    mean_confidence:    float

    def to_dict(self) -> dict:
        return {
            "epoch":           self.epoch,
            "n_updated":       self.n_updated,
            "n_newly_active":  self.n_newly_active,
            "n_retired":       self.n_retired,
            "n_total":         self.n_total,
            "mean_confidence": round(self.mean_confidence, 4),
            "top_5":           self.top_5,
        }

    def print_summary(self) -> None:
        print(f"\n  🧬  Hypothesis Engine — Epoch {self.epoch}")
        print(f"      Updated={self.n_updated} | "
              f"Active→={self.n_newly_active} | "
              f"Retired={self.n_retired} | "
              f"Total={self.n_total} | "
              f"Mean conf={self.mean_confidence:.3f}")
        for entry in self.knowledge_entries[:5]:
            print(f"      {entry}")


# ── Epoch update helper ───────────────────────────────────────────────────────

def update_hypotheses_from_epoch(
    library:        HypothesisLibrary,
    signal_ic_map:  dict[str, float],
    signal_to_hyp:  dict[str, str],    # {signal_name → hypothesis_id}
    regime:         Optional[str],
    epoch:          int,
) -> HypothesisReport:
    """
    Update all hypotheses from this epoch's IC results.

    Call this at the end of each ResearchLoop epoch, after IC scoring.

    Args:
        library:       HypothesisLibrary (mutated in-place).
        signal_ic_map: {signal_name: mean_ic} from this epoch.
        signal_to_hyp: Maps signal names to their hypothesis IDs.
        regime:        Current market regime name (optional).
        epoch:         Current epoch number.

    Returns:
        HypothesisReport summarising what changed.
    """
    n_updated      = 0
    n_newly_active = 0
    n_retired      = 0

    for signal_name, ic in signal_ic_map.items():
        hyp_id = signal_to_hyp.get(signal_name)
        if hyp_id is None:
            continue

        rec = library.get(hyp_id)
        if rec is None:
            continue

        prev_status = rec.status
        library.update_with_ic(hyp_id, ic, regime, epoch)
        n_updated += 1

        new_status = rec.status
        if prev_status != HypothesisStatus.ACTIVE and new_status == HypothesisStatus.ACTIVE:
            n_newly_active += 1
        elif prev_status == HypothesisStatus.ACTIVE and new_status == HypothesisStatus.RETIRED:
            n_retired += 1

    top_5   = library.rank_by_confidence(5)
    entries = [r.knowledge_entry() for r in library.rank_by_confidence(10)]

    return HypothesisReport(
        epoch=epoch,
        n_updated=n_updated,
        n_newly_active=n_newly_active,
        n_retired=n_retired,
        n_total=library.size,
        top_5=[r.to_dict() for r in top_5],
        knowledge_entries=entries,
        mean_confidence=library.mean_confidence,
    )
