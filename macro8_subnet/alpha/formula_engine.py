"""
alpha/formula_engine.py
------------------------
Alpha Formula Engine — miners submit signal formulas, not precomputed values.

Instead of submitting a pd.Series of computed alpha values, miners submit
a string formula that is evaluated by the validator against the shared
feature store. This creates a combinatorial search space similar to
WorldQuant's WebSim platform and Two Sigma's internal research tools.

Formula examples
----------------
    "zscore(momentum_20d)"
    "rank(momentum_20d) - rank(volatility_60d)"
    "regime_signal * momentum_5d"
    "decay(zscore(rsi_14), halflife=10)"
    "clip(cross_momentum, -2, 2)"
    "lag(momentum_5d, n=2)"

Available operators
-------------------
    zscore(x)              Standardise x to zero mean, unit variance
    rank(x)                Cross-sectional rank (0=lowest, 1=highest)
    decay(x, halflife=N)   Exponential decay / smoothing
    neutralize(x)          Demean cross-sectionally (subtract universe mean)
    clip(x, lo, hi)        Clip values to [lo, hi]
    lag(x, n=1)            Shift signal backwards by n days
    sign(x)                Element-wise sign: -1, 0, +1
    abs(x)                 Element-wise absolute value

Available inputs (FeatureStore features)
------------------------------------------
    momentum_5d, momentum_10d, momentum_20d, momentum_60d
    volatility_10d, volatility_20d, volatility_60d
    zscore_20d, zscore_60d
    rsi_14
    cross_momentum, relative_vol, regime_signal

Combinatorial operators
-----------------------
    +, -, *, /  (element-wise arithmetic between features)

Security
--------
The formula is evaluated in a RESTRICTED environment with no access to
Python builtins, file system, or network. Only whitelisted operations
are permitted. Any formula that fails evaluation is rejected.

Usage
-----
    engine  = FormulaEngine(feature_store)
    signal  = engine.evaluate("rank(momentum_20d) - rank(volatility_60d)")
    # signal is a dict[str, pd.Series] — asset → daily values
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_MACRO8_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_MACRO8_ROOT) not in sys.path:
    sys.path.insert(0, str(_MACRO8_ROOT))


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class FormulaResult:
    """Result of evaluating one alpha formula."""
    formula:  str
    signals:  Optional[dict[str, pd.Series]]   # asset → signal series
    success:  bool
    error:    Optional[str]           = None
    n_obs:    int                     = 0

    def to_alpha_signals(self) -> dict[str, pd.Series]:
        """Return signals dict, empty if evaluation failed."""
        return self.signals or {}


# ── Formula Engine ────────────────────────────────────────────────────────────

class FormulaEngine:
    """
    Parses and evaluates alpha signal formulas against a feature store.

    Provides a safe, restricted evaluation environment — only
    whitelisted operations are permitted.
    """

    # Whitelisted feature names
    ALLOWED_FEATURES = {
        "momentum_5d", "momentum_10d", "momentum_20d", "momentum_60d",
        "volatility_10d", "volatility_20d", "volatility_60d",
        "zscore_20d", "zscore_60d",
        "rsi_14",
        "cross_momentum", "relative_vol", "regime_signal",
    }

    # Allowed operator names in formulas
    ALLOWED_OPS = {"zscore", "rank", "decay", "neutralize", "clip", "lag", "sign", "abs"}

    def __init__(self, feature_store):
        """
        Args:
            feature_store: FeatureStore instance with computed features.
        """
        self._fs = feature_store

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(self, formula: str) -> FormulaResult:
        """
        Evaluate a formula string against the feature store.

        Args:
            formula: Alpha formula string (see module docstring for syntax).

        Returns:
            FormulaResult with signal dict on success, error string on failure.
        """
        formula = formula.strip()

        # Security: validate formula before evaluation
        ok, reason = self._validate(formula)
        if not ok:
            return FormulaResult(formula=formula, signals=None,
                                 success=False, error=reason)

        try:
            # Build evaluation context with feature DataFrames
            context   = self._build_context()
            operators = self._build_operators()

            # Merge into evaluation namespace
            namespace = {**context, **operators}

            # Evaluate formula in restricted namespace
            result_df = eval(formula, {"__builtins__": {}}, namespace)  # nosec B307

            if not isinstance(result_df, pd.DataFrame):
                return FormulaResult(formula=formula, signals=None,
                                     success=False, error="Formula did not return a DataFrame")

            # Convert to per-asset Series dict
            signals = {
                col: result_df[col].dropna()
                for col in result_df.columns
            }

            n_obs = min(len(s) for s in signals.values()) if signals else 0
            if n_obs < 5:
                return FormulaResult(formula=formula, signals=None,
                                     success=False,
                                     error=f"Too few observations: {n_obs} (min 5)")

            return FormulaResult(formula=formula, signals=signals,
                                 success=True, n_obs=n_obs)

        except Exception as exc:
            return FormulaResult(formula=formula, signals=None,
                                 success=False,
                                 error=f"{type(exc).__name__}: {exc}")

    def validate_formula(self, formula: str) -> tuple[bool, str]:
        """
        Check if a formula is syntactically valid and safe before evaluation.

        Args:
            formula: Formula string to validate.

        Returns:
            (True, "") if valid, (False, reason) if not.
        """
        return self._validate(formula)

    def list_features(self) -> list[str]:
        """Return all available feature names."""
        return sorted(self.ALLOWED_FEATURES)

    def list_operators(self) -> list[str]:
        """Return all available operator names."""
        return sorted(self.ALLOWED_OPS)

    # ── Formula validation ────────────────────────────────────────────────────

    def _validate(self, formula: str) -> tuple[bool, str]:
        """Security validation before eval()."""
        if not formula:
            return False, "Empty formula"

        # Reject forbidden patterns
        forbidden = [
            "__", "import", "exec", "open", "eval",
            "os.", "sys.", "subprocess", "builtins",
            "globals", "locals", "vars",
        ]
        for pat in forbidden:
            if pat in formula:
                return False, f"Forbidden pattern: '{pat}'"

        # Check all function calls use whitelisted operators
        called_funcs = set(re.findall(r"([a-z_][a-z0-9_]*)\s*\(", formula))
        unknown_ops  = called_funcs - self.ALLOWED_OPS - self.ALLOWED_FEATURES
        if unknown_ops:
            return False, f"Unknown operators: {unknown_ops}"

        # Check feature names referenced (outside function calls)
        # Exclude keyword argument names (identifiers followed by '=')
        kwarg_names = set(re.findall(r"\b([a-z_][a-z0-9_]*)\s*=", formula))
        identifiers = set(re.findall(r"\b([a-z_][a-z0-9_]*)\b", formula))
        feature_refs = identifiers - self.ALLOWED_OPS - {"True", "False", "None"} - kwarg_names
        unknown_feats = feature_refs - self.ALLOWED_FEATURES
        if unknown_feats:
            return False, f"Unknown features: {unknown_feats}"

        return True, ""

    # ── Context builders ──────────────────────────────────────────────────────

    def _build_context(self) -> dict[str, pd.DataFrame]:
        """Load all allowed features from the feature store."""
        context = {}
        for feat in self.ALLOWED_FEATURES:
            try:
                df = self._fs.get(feat)
                if df is not None:
                    context[feat] = df
            except Exception:
                pass
        return context

    def _build_operators(self) -> dict:
        """Build the whitelisted operator namespace."""
        return {
            "zscore":     self._op_zscore,
            "rank":       self._op_rank,
            "decay":      self._op_decay,
            "neutralize": self._op_neutralize,
            "clip":       self._op_clip,
            "lag":        self._op_lag,
            "sign":       self._op_sign,
            "abs":        self._op_abs,
        }

    # ── Operator implementations ──────────────────────────────────────────────

    @staticmethod
    def _op_zscore(x: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional z-score: (x - mean) / std across assets."""
        mean = x.mean(axis=1)
        std  = x.std(axis=1).replace(0, np.nan)
        return x.subtract(mean, axis=0).divide(std, axis=0)

    @staticmethod
    def _op_rank(x: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional rank normalised to [0, 1]."""
        return x.rank(axis=1, pct=True)

    @staticmethod
    def _op_decay(x: pd.DataFrame, halflife: float = 10.0) -> pd.DataFrame:
        """Exponential weighted moving average with given half-life."""
        alpha = 1 - np.exp(-np.log(2) / halflife)
        return x.ewm(alpha=alpha, adjust=False).mean()

    @staticmethod
    def _op_neutralize(x: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional demeaning: subtract universe mean each day."""
        mean = x.mean(axis=1)
        return x.subtract(mean, axis=0)

    @staticmethod
    def _op_clip(x: pd.DataFrame, lo: float = -3.0, hi: float = 3.0) -> pd.DataFrame:
        """Clip values to [lo, hi]."""
        return x.clip(lower=lo, upper=hi)

    @staticmethod
    def _op_lag(x: pd.DataFrame, n: int = 1) -> pd.DataFrame:
        """Shift signal backwards by n periods (creates n-day lag)."""
        return x.shift(n)

    @staticmethod
    def _op_sign(x: pd.DataFrame) -> pd.DataFrame:
        """Element-wise sign: -1, 0, +1."""
        return np.sign(x)

    @staticmethod
    def _op_abs(x: pd.DataFrame) -> pd.DataFrame:
        """Element-wise absolute value."""
        return x.abs()
