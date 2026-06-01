"""alpha/portfolio_optimizer.py — Portfolio optimization (Sprint 9 rebuild)."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize

class OptMethod(str, Enum):
    IC_WEIGHTED = "ic_weighted"; MEAN_VARIANCE = "mean_variance"; RISK_PARITY = "risk_parity"

@dataclass
class OptResult:
    weights: dict; method: OptMethod; expected_return: float
    expected_vol: float; sharpe_ratio: float; success: bool; message: str = ""
    def to_dict(self):
        return {"weights": {k: round(v,6) for k,v in self.weights.items()},
                "method": self.method.value, "sharpe_ratio": round(self.sharpe_ratio,6), "success": self.success}

class PortfolioOptimizer:
    def __init__(self, method: OptMethod = OptMethod.IC_WEIGHTED, max_weight: float = 0.40,
                 risk_aversion: float = 1.0, risk_free: float = 0.0):
        self.method=method; self.max_weight=max_weight; self.risk_aversion=risk_aversion; self.risk_free=risk_free

    def optimize(self, signals: dict, returns: pd.DataFrame) -> OptResult:
        assets = [a for a in signals if a in returns.columns]
        if not assets: return self._failed("No overlapping assets")
        sig_vec  = np.array([signals[a] for a in assets])
        ret_data = returns[assets].dropna()
        if len(ret_data) < 5: return self._failed("Insufficient data")
        try:
            if self.method == OptMethod.MEAN_VARIANCE: weights = self._mean_variance(sig_vec, ret_data, assets)
            elif self.method == OptMethod.RISK_PARITY: weights = self._risk_parity(ret_data, assets)
            else: weights = self._ic_weighted(sig_vec, assets)
            w_vec = np.array([weights[a] for a in assets])
            mu    = float(ret_data.mean() @ w_vec) * 252
            sigma = float(np.sqrt(max(w_vec @ ret_data.cov().values * 252 @ w_vec, 1e-12)))
            sharpe = (mu - self.risk_free) / sigma if sigma > 1e-8 else 0.0
            return OptResult(weights=weights, method=self.method, expected_return=mu,
                             expected_vol=sigma, sharpe_ratio=sharpe, success=True)
        except Exception as e: return self._failed(str(e))

    def _ic_weighted(self, signals: np.ndarray, assets: list) -> dict:
        pos = np.maximum(signals, 0.0)
        if pos.sum() < 1e-8:
            n = len(assets); return {a: 1.0/n for a in assets}
        exp_s = np.exp(pos - pos.max()); raw_w = exp_s / exp_s.sum()
        return self._apply_constraints(raw_w, assets)

    def _mean_variance(self, signals: np.ndarray, returns: pd.DataFrame, assets: list) -> dict:
        n = len(assets); cov = returns.cov().values * 252
        mu = signals / (np.abs(signals).max() + 1e-8) * 0.10
        def neg_utility(w): return -(mu @ w - self.risk_aversion * (w @ cov @ w))
        res = minimize(neg_utility, np.ones(n)/n, method="SLSQP",
                       bounds=[(0.0, self.max_weight)]*n,
                       constraints=[{"type":"eq","fun":lambda w: w.sum()-1.0}],
                       options={"maxiter":200,"ftol":1e-9})
        return self._apply_constraints(res.x if res.success else np.ones(n)/n, assets)

    def _risk_parity(self, returns: pd.DataFrame, assets: list) -> dict:
        n = len(assets); cov = returns.cov().values * 252
        def rc(w):
            pv = w @ cov @ w
            return float(np.sum((w * (cov@w)/max(pv,1e-12) - 1/n)**2)) if pv > 1e-12 else 1e6
        res = minimize(rc, np.ones(n)/n, method="SLSQP",
                       bounds=[(0.01, self.max_weight)]*n,
                       constraints=[{"type":"eq","fun":lambda w: w.sum()-1.0}],
                       options={"maxiter":300,"ftol":1e-10})
        return self._apply_constraints(res.x if res.success else np.ones(n)/n, assets)

    def _apply_constraints(self, weights: np.ndarray, assets: list) -> dict:
        w = np.clip(weights, 0.0, self.max_weight)
        for _ in range(10):
            t = w.sum()
            if t < 1e-8: w = np.ones(len(assets))/len(assets); break
            w = w/t; c = np.clip(w, 0.0, self.max_weight)
            if np.allclose(c, w, atol=1e-9): break
            w = c
        return {a: float(w[i]) for i, a in enumerate(assets)}

    @staticmethod
    def _failed(reason: str) -> OptResult:
        return OptResult(weights={}, method=OptMethod.IC_WEIGHTED,
                         expected_return=0.0, expected_vol=0.0, sharpe_ratio=0.0, success=False, message=reason)
