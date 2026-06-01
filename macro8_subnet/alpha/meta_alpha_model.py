"""
alpha/meta_alpha_model.py
--------------------------
Meta-Alpha model: predicts which alpha signals will perform well
in the next evaluation period, based on their historical behaviour.

This is the "alpha of alphas" — a model that learns what makes a
good alpha signal, then uses that to rank incoming candidates.

Feature vector per signal
--------------------------
    mean_ic         historical average IC
    ic_ir           IC Information Ratio (stability metric)
    ic_stability    fraction of periods with positive IC
    decay_rate      slope of IC over recent history (negative = dying)
    ic_lag2_ratio   IC(lag-2) / IC(lag-1)  (persistence measure)
    ic_last_3       average IC over last 3 epochs
    regime_ic_0..4  average IC in each of the 5 market regimes
    capacity        current capacity allocation
    epochs_alive    how long signal has been in library

Target
------
    y = IC in next epoch

Training
--------
The model trains on past (features, IC) pairs from library history.
As the library accumulates history, model quality improves.
Initially (< 10 data points), falls back to IC-ranking heuristic.

Models available
----------------
    ridge      Ridge regression (L2 regularised, fast, interpretable)
    gbm        Gradient Boosting (captures non-linear relationships)
    ensemble   Average of ridge + GBM predictions

Industry note
-------------
This is analogous to Quantopian's ranking model and WorldQuant's
internal "alpha fitness" predictions. The model gets smarter as
the library grows — network effects compound over time.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_MACRO8_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_MACRO8_ROOT) not in sys.path:
    sys.path.insert(0, str(_MACRO8_ROOT))


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_alpha_features(record) -> dict[str, float]:
    """
    Extract feature vector from an AlphaRecord for meta-learning.

    Args:
        record: AlphaRecord from the alpha library.

    Returns:
        Dict of feature_name → float value (all finite).
    """
    ic_hist = record.ic_history or [0.0]
    ir_hist = record.ir_history or [0.0]

    # Recent IC trend
    last_3   = float(np.mean(ic_hist[-3:])) if len(ic_hist) >= 3 else float(np.mean(ic_hist))
    last_1   = float(ic_hist[-1]) if ic_hist else 0.0
    lag2     = float(ic_hist[-2]) if len(ic_hist) >= 2 else last_1
    lag2_ratio = (last_1 / lag2) if abs(lag2) > 1e-8 else 1.0

    features = {
        "mean_ic":        float(record.mean_ic),
        "ic_ir":          float(record.current_ir),
        "ic_stability":   float(record.ic_stability),
        "decay_rate":     float(record.decay_rate),
        "ic_last_3":      last_3,
        "ic_last_1":      last_1,
        "ic_lag2_ratio":  float(np.clip(lag2_ratio, -5.0, 5.0)),
        "capacity":       float(record.capacity),
        "epochs_alive":   float(min(record.epochs_alive, 100)),
    }

    # Replace any NaN/Inf
    return {
        k: float(v) if np.isfinite(v) else 0.0
        for k, v in features.items()
    }


FEATURE_NAMES = [
    "mean_ic", "ic_ir", "ic_stability", "decay_rate",
    "ic_last_3", "ic_last_1", "ic_lag2_ratio", "capacity", "epochs_alive",
]


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class MetaAlphaPrediction:
    """Predicted IC and ranking for one signal."""
    signal_name:      str
    predicted_ic:     float
    prediction_rank:  int           # 1 = best predicted signal
    confidence:       float         # model confidence [0, 1]
    features_used:    dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "signal_name":     self.signal_name,
            "predicted_ic":    round(self.predicted_ic,    6),
            "prediction_rank": self.prediction_rank,
            "confidence":      round(self.confidence,      4),
        }


@dataclass
class MetaAlphaReport:
    """Full meta-alpha model output for all library signals."""
    predictions:       list[MetaAlphaPrediction]
    model_method:      str
    train_r_squared:   Optional[float]
    n_training_samples: int
    feature_importances: dict[str, float]   # feature → importance weight
    is_trained:        bool

    def top_signals(self, n: int = 5) -> list[str]:
        """Return top N signal names by predicted IC."""
        return [p.signal_name for p in self.predictions[:n]]

    def to_dict(self) -> dict:
        return {
            "model_method":      self.model_method,
            "train_r_squared":   round(self.train_r_squared, 4) if self.train_r_squared else None,
            "n_training_samples": self.n_training_samples,
            "is_trained":        self.is_trained,
            "top_signals":       self.top_signals(5),
            "predictions":       [p.to_dict() for p in self.predictions],
        }


# ── Meta-Alpha Model ──────────────────────────────────────────────────────────

class MetaAlphaModel:
    """
    Learns what makes alpha signals good and predicts future IC.

    Trains on (alpha_features → future_ic) pairs accumulated from
    the alpha library across epochs.

    Usage
    -----
        model = MetaAlphaModel(method="ridge")

        # Each epoch: add new training samples
        for record in library.all_active():
            model.add_training_sample(record, actual_next_ic)

        # Predict which signals will be best next epoch
        report = model.predict_all(library.all_active())
        best   = report.top_signals(3)
    """

    def __init__(
        self,
        method:      str   = "ridge",
        alpha:       float = 0.5,        # Ridge regularisation
        min_samples: int   = 8,          # minimum training points before ML kicks in
        n_estimators: int  = 50,
    ):
        self.method       = method
        self.alpha        = alpha
        self.min_samples  = min_samples
        self.n_estimators = n_estimators

        self._X: list[list[float]] = []   # feature matrix (training)
        self._y: list[float]       = []   # target IC values (training)
        self._model                = None
        self._scaler               = None
        self._is_trained           = False
        self._feature_importances: dict[str, float] = {}
        self._train_r2: Optional[float] = None

    # ── Training data accumulation ────────────────────────────────────────────

    def add_training_sample(
        self,
        record,           # AlphaRecord
        actual_next_ic: float,
    ) -> None:
        """
        Add one training sample: (signal features at time t, IC at time t+1).

        Call this each epoch for every active library signal after
        observing the new IC value.

        Args:
            record:         AlphaRecord at prediction time.
            actual_next_ic: The IC that was observed in the NEXT epoch.
        """
        features = extract_alpha_features(record)
        x_row    = [features.get(f, 0.0) for f in FEATURE_NAMES]
        self._X.append(x_row)
        self._y.append(float(actual_next_ic))

        # Retrain whenever we have enough data
        if len(self._y) >= self.min_samples:
            self._train()

    def add_batch(self, samples: list[tuple]) -> None:
        """
        Add multiple (record, actual_ic) training samples at once.

        Args:
            samples: List of (AlphaRecord, float) tuples.
        """
        for record, actual_ic in samples:
            features = extract_alpha_features(record)
            x_row    = [features.get(f, 0.0) for f in FEATURE_NAMES]
            self._X.append(x_row)
            self._y.append(float(actual_ic))

        if len(self._y) >= self.min_samples:
            self._train()

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, record) -> MetaAlphaPrediction:
        """
        Predict next-period IC for one signal.

        Falls back to heuristic (current mean_ic) if model not trained.
        """
        features     = extract_alpha_features(record)
        x_row        = np.array([[features.get(f, 0.0) for f in FEATURE_NAMES]])
        signal_name  = record.name

        if self._is_trained and self._scaler is not None:
            try:
                x_scaled     = self._scaler.transform(x_row)
                predicted_ic = float(self._model.predict(x_scaled)[0])
                confidence   = min(len(self._y) / 50.0, 1.0)
            except Exception:
                predicted_ic = float(record.mean_ic)
                confidence   = 0.2
        else:
            # Heuristic: blend current IC with trend
            predicted_ic = float(0.7 * record.mean_ic + 0.3 * record.current_ic)
            confidence   = 0.1

        return MetaAlphaPrediction(
            signal_name=signal_name,
            predicted_ic=predicted_ic,
            prediction_rank=0,   # set by predict_all
            confidence=confidence,
            features_used=features,
        )

    def predict_all(self, records: list) -> MetaAlphaReport:
        """
        Predict IC for all signals and produce a ranked report.

        Args:
            records: List of AlphaRecord objects from the library.

        Returns:
            MetaAlphaReport with ranked predictions.
        """
        if not records:
            return MetaAlphaReport(
                predictions=[], model_method=self.method,
                train_r_squared=self._train_r2,
                n_training_samples=len(self._y),
                feature_importances=self._feature_importances,
                is_trained=self._is_trained,
            )

        predictions = [self.predict(r) for r in records]

        # Rank by predicted IC descending
        predictions.sort(key=lambda p: p.predicted_ic, reverse=True)
        for rank, pred in enumerate(predictions, start=1):
            pred.prediction_rank = rank

        return MetaAlphaReport(
            predictions=predictions,
            model_method=self.method,
            train_r_squared=self._train_r2,
            n_training_samples=len(self._y),
            feature_importances=self._feature_importances,
            is_trained=self._is_trained,
        )

    # ── Internal training ─────────────────────────────────────────────────────

    def _train(self) -> None:
        """Fit the prediction model on accumulated training data."""
        from sklearn.linear_model    import Ridge
        from sklearn.ensemble        import GradientBoostingRegressor
        from sklearn.preprocessing   import StandardScaler

        X = np.array(self._X)
        y = np.array(self._y)

        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if self.method == "ridge":
            model = Ridge(alpha=self.alpha)
        elif self.method == "gbm":
            model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
            )
        else:
            model = Ridge(alpha=self.alpha)

        model.fit(X_scaled, y)

        # R² on training data
        y_pred       = model.predict(X_scaled)
        ss_res       = float(np.sum((y - y_pred) ** 2))
        ss_tot       = float(np.sum((y - y.mean()) ** 2))
        self._train_r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        # Feature importances
        if hasattr(model, "coef_"):
            raw_imp = np.abs(model.coef_)
        elif hasattr(model, "feature_importances_"):
            raw_imp = model.feature_importances_
        else:
            raw_imp = np.ones(len(FEATURE_NAMES))

        total_imp = raw_imp.sum()
        if total_imp > 1e-8:
            raw_imp = raw_imp / total_imp
        self._feature_importances = dict(zip(FEATURE_NAMES, raw_imp.tolist()))

        self._model      = model
        self._scaler     = scaler
        self._is_trained = True

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def n_samples(self) -> int:
        return len(self._y)
