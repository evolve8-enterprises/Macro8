# Sprint 16 — Capacity Model + Representation Learning Engine

## System A: Alpha Lifecycle + Capacity Model

### The alpha decay trap (precise diagnosis)
Without lifecycle control:
  - Library grows indefinitely (retirement is manual)
  - Signals get re-admitted after they've already decayed
  - Portfolio optimizer over-weights stale signals

### New module: alpha/capacity_model.py

Contains two cooperating systems:

#### 1. LifecycleEngine
Manages FormulaRecord state transitions:

    EXPERIMENTAL  → created, < MIN_EPOCHS observations
    VALIDATED     → IC > threshold for MIN_EPOCHS consecutive epochs
    PRODUCTION    → VALIDATED + positive MSC + IC stability > 0.6
    DECAYING      → IC declining (negative decay_rate for N epochs)
    RETIRED       → DECAYING for RETIRE_EPOCHS, or forced retirement

Transitions evaluated each epoch via LifecycleEngine.assess_all()

#### 2. CapacityEstimator
Estimates how much weight a signal should receive:

    capacity_score ∈ [0, 1]
    crowding_score ∈ [0, 1] (higher = more crowded = worse)

    capacity_score:
        Derived from IC stability and observation count.
        New signals get 0.3. PRODUCTION signals get up to 1.0.
        DECAYING signals get max(0.1, score * decay_penalty)

    crowding_score:
        Correlation with existing library signals.
        Already handled by OrthogonalityFilter (threshold=0.90).
        CapacityEstimator reads the orthogonality report.

    adjusted_weight_i = IC_i × lifecycle_multiplier × (1 - crowding_score)

    Lifecycle multipliers:
        EXPERIMENTAL: 0.3
        VALIDATED:    0.7
        PRODUCTION:   1.0
        DECAYING:     0.3
        RETIRED:      0.0

#### 3. DecayEstimator
Fits exponential decay to IC history:
    IC(t) ≈ IC_0 × exp(-λ × t)
    IC_half_life = ln(2) / λ

    λ = -slope of log(|IC|) over recent window
    Half-life in epochs: shorter = faster decay

Integration with FormulaRecord:
    FormulaRecord.lifecycle_state added via LifecycleEngine
    FormulaRecord.capacity_score updated by CapacityEstimator
    FormulaRecord.crowding_score updated from OrthogonalityFilter
    FormulaRecord.ic_half_life added by DecayEstimator

Integration with MacroSession:
    MacroSession.run_epoch() calls LifecycleEngine.assess_all()
    After lifecycle update, portfolio weights are adjusted

---

## System B: Representation Learning Engine

### The insight
Current system searches formula space over fixed features.
Representation learning expands this to feature space — the system
discovers new predictive features that no one explicitly programmed.

### New module: alpha/representation_engine.py

#### Architecture: three unsupervised methods

1. PCA (Principal Component Analysis)
   - Fast, interpretable, no hyperparameters
   - Extracts linear combinations of existing features
   - Output: n_components latent factors

2. Autoencoder (1-layer bottleneck)
   - Non-linear compression and reconstruction
   - Discovers non-linear market structures
   - Input: feature tensor → bottleneck → reconstruction
   - Output: bottleneck activations as latent features

3. Rolling PCA (regime-aware)
   - PCA computed on rolling windows
   - Captures regime-varying factor structure
   - Output: time-varying latent features

#### Output: LatentFeatureSet
    latent_name:     "latent_0", "latent_1", ...
    method:          "pca" | "autoencoder" | "rolling_pca"
    time_series:     dict[asset → pd.Series]   (same shape as FeatureStore output)
    explained_var:   float (fraction of variance explained, PCA only)
    ic_history:      []  (populated after IC evaluation)

#### Integration with FeatureStore
    FeatureStore.add_latent_features(latent_set)
    → adds latent_0, latent_1, ... to feature dict
    → BatchEvaluator can use latent features in formulas

#### Integration with FormulaEngine + BatchEvaluator
    ALLOWED_FEATURES extended with discovered latent names
    BatchEvaluator.feat_tensor rebuilt with latent features
    FormulaEncoder updated to handle latent feature names

#### Integration with HypothesisEngine
    When a latent feature shows IC > threshold:
    → auto-generate hypothesis statement:
      "Latent market structure {name} predicts returns"
    → category: HypothesisCategory.UNKNOWN
      (human miners can refine the interpretation)

---

## Build order
1. capacity_model.py — LifecycleEngine + CapacityEstimator + DecayEstimator
2. representation_engine.py — PCA + Autoencoder + LatentFeatureSet
3. test_sprint16.py — full self-contained test suite for both

## Contracts
- No modifications to existing modules
- All new types JSON-serialisable
- DecayEstimator handles < 3 observations gracefully (returns None)
- Autoencoder uses only numpy/sklearn (no torch dependency)
- Latent features are named "latent_pca_0", "latent_ae_0", etc.
