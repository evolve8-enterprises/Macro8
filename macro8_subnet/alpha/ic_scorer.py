"""alpha/ic_scorer.py — Information Coefficient scoring (Sprint 9 rebuild)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

@dataclass
class ICResult:
    factor_name: str; mean_ic: float; ic_ir: float; ic_stability: float
    ic_series: pd.Series; ic_decay: list; n_periods: int; success: bool
    error: Optional[str] = None
    @property
    def passes_threshold(self): return self.mean_ic > 0.02 and self.ic_ir > 0.3
    def to_dict(self): return {"factor_name": self.factor_name, "mean_ic": round(self.mean_ic, 6), "success": self.success}

class ICScorer:
    def __init__(self, n_lags: int = 4, min_obs: int = 10, min_ic: float = 0.02, min_ir: float = 0.30):
        self.n_lags = n_lags; self.min_obs = min_obs; self.min_ic = min_ic; self.min_ir = min_ir

    def score(self, factor_name: str, signals: dict, forward_returns: pd.DataFrame) -> ICResult:
        try:
            signal_df = pd.DataFrame(signals)
            overlap   = signal_df.index.intersection(forward_returns.index)
            if len(overlap) < self.min_obs:
                return self._failed(factor_name, f"Only {len(overlap)} observations")
            sig  = signal_df.loc[overlap]
            rets = forward_returns.loc[overlap]
            ic_series = self._compute_ic(sig, rets, lag=1)
            if len(ic_series.dropna()) < self.min_obs:
                return self._failed(factor_name, "Insufficient IC observations")
            mean_ic  = float(ic_series.mean())
            std_ic   = float(ic_series.std())
            ic_ir    = mean_ic / std_ic if std_ic > 1e-8 else 0.0
            ic_stab  = float((ic_series > 0).mean())
            ic_decay = []
            for lag in range(1, self.n_lags + 1):
                ls = self._compute_ic(sig, rets, lag=lag)
                ic_decay.append(float(ls.mean()) if len(ls) > 0 else 0.0)
            return ICResult(factor_name=factor_name, mean_ic=mean_ic, ic_ir=ic_ir,
                            ic_stability=ic_stab, ic_series=ic_series, ic_decay=ic_decay,
                            n_periods=len(ic_series.dropna()), success=True)
        except Exception as exc:
            return self._failed(factor_name, f"{type(exc).__name__}: {exc}")

    def score_batch(self, factors: dict, forward_returns: pd.DataFrame) -> dict:
        return {name: self.score(name, signals, forward_returns) for name, signals in factors.items()}

    @staticmethod
    def _compute_ic(signals: pd.DataFrame, returns: pd.DataFrame, lag: int = 1) -> pd.Series:
        assets = list(set(signals.columns) & set(returns.columns))
        if not assets: return pd.Series(dtype=float)
        sig = signals[assets]; ret = returns[assets].shift(-lag)
        ic_values, dates = [], []
        for date in sig.index:
            if date not in ret.index: continue
            s_row = sig.loc[date].dropna(); r_row = ret.loc[date].dropna()
            common = s_row.index.intersection(r_row.index)
            if len(common) < 3: continue
            corr, _ = spearmanr(s_row[common].values, r_row[common].values)
            if not np.isnan(corr): ic_values.append(float(corr)); dates.append(date)
        return pd.Series(ic_values, index=dates, name="IC")

    @staticmethod
    def _failed(name: str, reason: str) -> ICResult:
        return ICResult(factor_name=name, mean_ic=0.0, ic_ir=0.0, ic_stability=0.0,
                        ic_series=pd.Series(dtype=float), ic_decay=[], n_periods=0,
                        success=False, error=reason)
