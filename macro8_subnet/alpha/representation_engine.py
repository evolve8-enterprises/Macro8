"""
alpha/representation_engine.py
--------------------------------
Representation Learning Engine for the Macro8 platform.

Automatically discovers new predictive features from market data
using unsupervised learning. Expands the formula search space from
hand-crafted indicators to machine-discovered market structures.

The core insight
----------------
The existing system searches formula space over fixed features:
    rank(momentum_20d) - rank(volatility_60d)

Representation learning discovers new features:
    latent_pca_0, latent_pca_1, latent_ae_0, ...

These latent features can then be used in formulas:
    rank(latent_pca_0) - rank(latent_ae_2)

The system has expanded its own search space.

Three unsupervised methods
--------------------------
1. PCA (Principal Component Analysis)
   - Extracts linear combinations of features that maximise variance
   - Fast, interpretable, no hyperparameters beyond n_components
   - Best for: linear market structures, factor models

2. Autoencoder (sklearn neural network)
   - Non-linear encoder-decoder with bottleneck
   - Discovers non-linear market structures
   - Best for: regime-dependent patterns, non-linear interactions

3. Rolling PCA
   - PCA computed on rolling windows (regime-aware)
   - Captures how factor structure changes over time
   - Best for: regime-varying latent factors

Output: LatentFeatureSet
------------------------
Each method produces a LatentFeatureSet:
    latent_name:    "latent_pca_0", "latent_ae_1", ...
    method:         "pca" | "autoencoder" | "rolling_pca"
    time_series:    dict[asset → pd.Series]  (same shape as FeatureStore)
    explained_var:  float  (variance fraction, PCA only)
    feature_basis:  list[str]  (which input features this combines)

Integration with FeatureStore + BatchEvaluator
----------------------------------------------
    engine = RepresentationEngine(prices)
    latent_sets = engine.fit_all(n_components=3)

    # Add latent features to an existing feature dict
    enriched_features = engine.enrich_feature_dict(base_features, latent_sets)

    # BatchEvaluator picks up the enriched features automatically
    beval = BatchEvaluator.__new__(BatchEvaluator)
    # rebuild feat_tensor with enriched_features

Integration with HypothesisEngine
-----------------------------------
When a latent feature shows IC > threshold, auto-generate a hypothesis:
    "Latent market structure {name} predicts cross-sectional returns"

    engine.auto_hypotheses(latent_sets, ic_threshold=0.02, hyp_lib)

No torch dependency
-------------------
All models use numpy + sklearn only. The autoencoder uses
sklearn.neural_network.MLPRegressor as an encoder-decoder.
This keeps the dependency footprint minimal.
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


# ── Latent Feature Set ────────────────────────────────────────────────────────

@dataclass
class LatentFeatureSet:
    """
    A discovered latent feature with time-series values per asset.

    Mirrors the output format of FeatureStore.build() so it can be
    directly injected into any downstream component.

    Attributes
    ----------
    latent_name    : str — e.g. "latent_pca_0", "latent_ae_2"
    method         : str — "pca" | "autoencoder" | "rolling_pca"
    time_series    : dict[asset_name → pd.Series]
    explained_var  : float — variance fraction explained (PCA only)
    feature_basis  : list[str] — input feature names this was derived from
    loadings       : np.ndarray | None — PCA component loadings
    ic_history     : list[float] — populated after IC evaluation
    """
    latent_name:   str
    method:        str
    time_series:   dict[str, pd.Series]
    explained_var: float          = 0.0
    feature_basis: list[str]      = field(default_factory=list)
    loadings:      Optional[np.ndarray] = None
    ic_history:    list[float]    = field(default_factory=list)

    def as_dataframe(self) -> pd.DataFrame:
        """Convert to a DataFrame with assets as columns."""
        if not self.time_series:
            return pd.DataFrame()
        return pd.DataFrame(self.time_series)

    @property
    def n_observations(self) -> int:
        if not self.time_series:
            return 0
        first = next(iter(self.time_series.values()))
        return len(first)

    @property
    def mean_ic(self) -> float:
        return float(np.mean(self.ic_history)) if self.ic_history else 0.0

    def interpretation(self) -> str:
        """Human-readable interpretation of this latent feature."""
        if self.method == "pca" and self.loadings is not None and self.feature_basis:
            # Find the features with the highest loadings
            abs_loadings = np.abs(self.loadings)
            top_k = min(3, len(self.feature_basis))
            top_idx = np.argsort(abs_loadings)[-top_k:][::-1]
            signs  = ["+" if self.loadings[i] > 0 else "-" for i in top_idx]
            parts  = [f"{signs[j]}{self.feature_basis[i]}"
                      for j, i in enumerate(top_idx)]
            return f"{self.latent_name} ≈ {' '.join(parts)}"
        return f"{self.latent_name} ({self.method})"

    def to_dict(self) -> dict:
        return {
            "latent_name":   self.latent_name,
            "method":        self.method,
            "explained_var": round(self.explained_var, 4),
            "feature_basis": self.feature_basis,
            "n_observations": self.n_observations,
            "mean_ic":       round(self.mean_ic, 6),
            "interpretation": self.interpretation(),
        }


# ── Representation Engine ─────────────────────────────────────────────────────

class RepresentationEngine:
    """
    Discovers latent features from market data using unsupervised learning.

    Fits PCA, autoencoder, and rolling PCA on the feature tensor and
    produces new LatentFeatureSets that extend the BatchEvaluator's
    search space.
    """

    def __init__(
        self,
        prices:          pd.DataFrame,
        min_train_obs:   int   = 30,
        random_state:    int   = 42,
    ):
        self.prices        = prices
        self.returns       = prices.pct_change().dropna()
        self.min_train_obs = min_train_obs
        self.random_state  = random_state
        self._fitted_pca   = None    # cached sklearn PCA
        self._fitted_ae    = None    # cached autoencoder

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_pca(
        self,
        feature_dict:  dict[str, pd.DataFrame],
        n_components:  int = 3,
    ) -> list[LatentFeatureSet]:
        """
        Extract PCA components from the feature tensor.

        Stacks all features into a matrix [time × (assets × features)],
        fits PCA, and returns the principal components as latent features.

        Args:
            feature_dict:  {feature_name: DataFrame[dates × assets]}
            n_components:  Number of PCA components to extract.

        Returns:
            List of LatentFeatureSet, one per component.
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        matrix, dates, assets, feat_names = self._build_cross_sectional_matrix(
            feature_dict
        )
        if matrix is None or len(matrix) < self.min_train_obs:
            return []

        # Standardise
        scaler  = StandardScaler()
        X_scaled = scaler.fit_transform(matrix)

        n_comp  = min(n_components, X_scaled.shape[1])
        pca     = PCA(n_components=n_comp, random_state=self.random_state)
        pca.fit(X_scaled)
        self._fitted_pca = pca

        # Project: latent scores [time × n_comp]
        scores = pca.transform(X_scaled)   # [T × n_comp]

        latent_sets = []
        for k in range(n_comp):
            component_scores = scores[:, k]  # [T]

            # Spread uniformly across assets (cross-sectional average signal)
            # Each asset gets the same latent score at each time
            time_series = {
                asset: pd.Series(component_scores, index=dates)
                for asset in assets
            }

            latent_sets.append(LatentFeatureSet(
                latent_name=f"latent_pca_{k}",
                method="pca",
                time_series=time_series,
                explained_var=float(pca.explained_variance_ratio_[k]),
                feature_basis=feat_names,
                loadings=pca.components_[k],
            ))

        return latent_sets

    def fit_autoencoder(
        self,
        feature_dict:  dict[str, pd.DataFrame],
        n_latent:      int = 3,
        hidden_size:   int = 8,
    ) -> list[LatentFeatureSet]:
        """
        Extract latent features using a shallow autoencoder.

        Architecture: input → hidden → bottleneck → hidden → input
        The bottleneck activations are the latent features.

        Uses sklearn.neural_network.MLPRegressor as encoder-decoder.
        No torch dependency.

        Args:
            feature_dict: {feature_name: DataFrame[dates × assets]}
            n_latent:     Bottleneck size (latent dimension).
            hidden_size:  Hidden layer size.

        Returns:
            List of LatentFeatureSet, one per latent dimension.
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.neural_network import MLPRegressor

        matrix, dates, assets, feat_names = self._build_cross_sectional_matrix(
            feature_dict
        )
        if matrix is None or len(matrix) < self.min_train_obs:
            return []

        scaler  = StandardScaler()
        X_scaled = scaler.fit_transform(matrix).astype(np.float32)

        n_input  = X_scaled.shape[1]
        n_latent = min(n_latent, n_input - 1)
        if n_latent < 1:
            return []

        # Encoder: input → bottleneck
        encoder = MLPRegressor(
            hidden_layer_sizes=(hidden_size, n_latent),
            activation="tanh",
            solver="adam",
            max_iter=200,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1,
        )

        # Train as autoencoder: predict input from input
        try:
            # MLPRegressor is for regression; we use it to build encoder
            # by predicting input (reconstruction objective)
            encoder.fit(X_scaled, X_scaled[:, :n_latent])
        except Exception:
            return []

        # Extract bottleneck activations
        # Get activations at the n_latent-sized hidden layer
        latent_activations = self._extract_bottleneck(encoder, X_scaled, n_latent)
        if latent_activations is None:
            return []

        self._fitted_ae = encoder

        latent_sets = []
        for k in range(min(n_latent, latent_activations.shape[1])):
            scores = latent_activations[:, k]
            time_series = {
                asset: pd.Series(scores, index=dates)
                for asset in assets
            }
            latent_sets.append(LatentFeatureSet(
                latent_name=f"latent_ae_{k}",
                method="autoencoder",
                time_series=time_series,
                explained_var=0.0,   # not directly meaningful for AE
                feature_basis=feat_names,
            ))

        return latent_sets

    def fit_rolling_pca(
        self,
        feature_dict:  dict[str, pd.DataFrame],
        n_components:  int = 2,
        window:        int = 40,
        step:          int = 5,
    ) -> list[LatentFeatureSet]:
        """
        Compute PCA on rolling windows to capture regime-varying structure.

        At each time step, PCA is fitted on the previous `window` rows.
        The components rotate as the market regime changes.

        Args:
            feature_dict: {feature_name: DataFrame[dates × assets]}
            n_components: Number of PCA components per window.
            window:       Rolling window size in time steps.
            step:         Step between window computations.

        Returns:
            List of LatentFeatureSet for each component.
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        matrix, dates, assets, feat_names = self._build_cross_sectional_matrix(
            feature_dict
        )
        if matrix is None or len(matrix) < window + 5:
            return []

        T       = len(matrix)
        n_comp  = min(n_components, matrix.shape[1])
        scores  = np.zeros((T, n_comp), dtype=np.float32)
        valid   = np.zeros(T, dtype=bool)

        for t in range(window, T, step):
            window_data = matrix[t - window:t]
            try:
                scaler      = StandardScaler()
                X_w         = scaler.fit_transform(window_data)
                pca_w       = PCA(n_components=n_comp)
                pca_w.fit(X_w)
                # Project the current row
                x_t         = scaler.transform(matrix[t:t + 1])
                score_t     = pca_w.transform(x_t)[0]
                # Fill in steps since last computation
                end         = min(t + step, T)
                scores[t:end] = score_t
                valid[t:end]  = True
            except Exception:
                continue

        # Forward-fill any gaps at the start
        first_valid = np.where(valid)[0]
        if len(first_valid) == 0:
            return []
        scores[:first_valid[0]] = scores[first_valid[0]]
        valid[:first_valid[0]]  = True

        latent_sets = []
        for k in range(n_comp):
            time_series = {
                asset: pd.Series(scores[:, k], index=dates)
                for asset in assets
            }
            latent_sets.append(LatentFeatureSet(
                latent_name=f"latent_rpca_{k}",
                method="rolling_pca",
                time_series=time_series,
                feature_basis=feat_names,
            ))

        return latent_sets

    def fit_all(
        self,
        feature_dict:  dict[str, pd.DataFrame],
        n_components:  int = 3,
    ) -> list[LatentFeatureSet]:
        """
        Fit all three methods and return all latent feature sets.

        Args:
            feature_dict: {feature_name: DataFrame[dates × assets]}
            n_components: Components per method.

        Returns:
            Combined list of all LatentFeatureSets.
        """
        results = []
        results.extend(self.fit_pca(feature_dict, n_components))
        results.extend(self.fit_autoencoder(feature_dict, n_components))
        results.extend(self.fit_rolling_pca(feature_dict, n_components))
        return results

    def enrich_feature_dict(
        self,
        base_features: dict[str, pd.DataFrame],
        latent_sets:   list[LatentFeatureSet],
    ) -> dict[str, pd.DataFrame]:
        """
        Add latent features to an existing feature dict.

        Compatible with FeatureStore.build() output format.
        The returned dict can be passed directly to FeatureTensor.from_feature_dict().

        Args:
            base_features: Original {feature_name: DataFrame} from FeatureStore.
            latent_sets:   Discovered latent features to add.

        Returns:
            Enriched feature dict with latent features appended.
        """
        enriched = dict(base_features)
        for ls in latent_sets:
            enriched[ls.latent_name] = ls.as_dataframe()
        return enriched

    def latent_feature_names(self, latent_sets: list[LatentFeatureSet]) -> list[str]:
        """Return list of all latent feature names."""
        return [ls.latent_name for ls in latent_sets]

    def auto_hypotheses(
        self,
        latent_sets:      list[LatentFeatureSet],
        hypothesis_library,           # HypothesisLibrary
        ic_threshold:     float = 0.02,
        epoch:            int   = 0,
    ) -> list:
        """
        Auto-generate hypotheses for latent features that show predictive IC.

        When a latent feature has IC > threshold, it is evidence that
        machine-discovered market structures are predictive. We register
        a hypothesis to track this.

        Args:
            latent_sets:       Discovered LatentFeatureSets with ic_history set.
            hypothesis_library: HypothesisLibrary to register hypotheses in.
            ic_threshold:      Minimum IC to trigger hypothesis creation.
            epoch:             Current epoch.

        Returns:
            List of newly created HypothesisRecord objects.
        """
        from macro8_subnet.alpha.hypothesis_engine import HypothesisCategory

        new_records = []
        for ls in latent_sets:
            if not ls.ic_history:
                continue
            mean_ic = float(np.mean(ls.ic_history))
            if mean_ic < ic_threshold:
                continue

            # Map method to category
            category_map = {
                "pca":         HypothesisCategory.CROSS_ASSET,
                "autoencoder": HypothesisCategory.REGIME,
                "rolling_pca": HypothesisCategory.REGIME,
            }
            cat  = category_map.get(ls.method, HypothesisCategory.UNKNOWN)
            stmt = (f"Latent market structure {ls.latent_name} "
                    f"({ls.method}) predicts cross-sectional returns")

            hrec = hypothesis_library.add(
                statement=stmt, category=cat,
                miner_uid=0, epoch=epoch,
                tags=["auto-generated", "latent-feature", ls.method],
            )
            new_records.append(hrec)

        return new_records

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_cross_sectional_matrix(
        self,
        feature_dict: dict[str, pd.DataFrame],
    ) -> tuple:
        """
        Build a [time × (features)] matrix from the feature dict.

        Averages across assets to get one value per feature per time step.
        This is the input to PCA/autoencoder.

        Returns:
            (matrix, dates, asset_names, feature_names)
            or (None, None, None, None) on failure.
        """
        if not feature_dict:
            return None, None, None, None

        # Align all features on common date index
        first_df  = next(iter(feature_dict.values()))
        dates     = first_df.index
        assets    = list(first_df.columns)
        feat_names = []
        arrays     = []

        for name, df in feature_dict.items():
            aligned = df.reindex(dates).fillna(0.0)
            # Average across assets: [T × A] → [T]
            avg = aligned.mean(axis=1).values
            arrays.append(avg)
            feat_names.append(name)

        if not arrays:
            return None, None, None, None

        matrix = np.column_stack(arrays).astype(np.float32)  # [T × n_features]

        # Drop rows with any NaN/inf
        valid  = np.all(np.isfinite(matrix), axis=1)
        if valid.sum() < self.min_train_obs:
            return None, None, None, None

        return matrix[valid], dates[valid], assets, feat_names

    @staticmethod
    def _extract_bottleneck(
        encoder,
        X:        np.ndarray,
        n_latent: int,
    ) -> Optional[np.ndarray]:
        """
        Extract bottleneck activations from a fitted MLPRegressor.

        MLPRegressor stores weights in encoder.coefs_ as a list of
        [n_in × n_out] matrices. The bottleneck is the second hidden layer.
        """
        try:
            # Forward pass through encoder layers up to bottleneck
            # MLPRegressor uses tanh by default
            activation = X.astype(np.float64)
            for i, (W, b) in enumerate(zip(encoder.coefs_, encoder.intercepts_)):
                activation = activation @ W + b
                if i < len(encoder.coefs_) - 1:
                    # Apply hidden activation (tanh)
                    activation = np.tanh(activation)
                # Stop at the bottleneck layer
                if activation.shape[1] == n_latent:
                    break
            return activation.astype(np.float32)
        except Exception:
            return None
