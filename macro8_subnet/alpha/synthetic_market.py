"""
simulation/synthetic_market.py
--------------------------------
Synthetic market simulation for alpha factor stress-testing.

Instead of testing alphas only on historical data, this module generates
parametric market scenarios — allowing stress-testing against market
conditions not yet observed in history.

This is extremely valuable because:
  - Historical stress periods are rare (few 2008-style crashes in the data)
  - Allows testing how alphas behave in regimes outside the sample
  - Generates unlimited training data for meta-alpha models
  - Tests alpha robustness against parameter uncertainty

Six simulation models
----------------------
    GBM            Geometric Brownian Motion (log-normal, baseline)
    JUMP_DIFFUSION  GBM + Poisson jumps (fat tails, crashes)
    MEAN_REVERT    Ornstein-Uhlenbeck (mean-reverting, range-bound markets)
    REGIME_SWITCH  Hidden Markov-like regime switching (bull/bear alternation)
    CORR_SHOCK     Correlated systemic shock (all assets fall together)
    INFLATION_SPIRAL Trending upward with increasing volatility

Each model returns:
    pd.DataFrame of daily prices (date × asset)
    SimulationMetadata with scenario parameters and realised statistics
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_MACRO8_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_MACRO8_ROOT) not in sys.path:
    sys.path.insert(0, str(_MACRO8_ROOT))


# ── Enums and Config ──────────────────────────────────────────────────────────

class SimModel(str, Enum):
    GBM             = "gbm"
    JUMP_DIFFUSION  = "jump_diffusion"
    MEAN_REVERT     = "mean_revert"
    REGIME_SWITCH   = "regime_switch"
    CORR_SHOCK      = "corr_shock"
    INFLATION_SPIRAL = "inflation_spiral"


@dataclass
class SimulationMetadata:
    """Parameters and realised statistics of a synthetic market run."""
    model:         SimModel
    n_days:        int
    n_assets:      int
    assets:        list[str]
    seed:          int
    # Realised statistics
    realised_vol:    Optional[float]   = None   # annualised portfolio vol
    realised_return: Optional[float]   = None   # total portfolio return
    max_drawdown:    Optional[float]   = None
    parameters:      dict              = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model":           self.model.value,
            "n_days":          self.n_days,
            "n_assets":        self.n_assets,
            "seed":            self.seed,
            "realised_vol":    round(self.realised_vol,    4) if self.realised_vol    else None,
            "realised_return": round(self.realised_return, 4) if self.realised_return else None,
            "max_drawdown":    round(self.max_drawdown,    4) if self.max_drawdown    else None,
            "parameters":      self.parameters,
        }


@dataclass
class SyntheticMarket:
    """Output of one synthetic market simulation."""
    prices:   pd.DataFrame
    returns:  pd.DataFrame
    metadata: SimulationMetadata

    def realised_stats(self, eq_weight: bool = True) -> dict:
        """Compute realised portfolio statistics."""
        port = self.returns.mean(axis=1)
        ann_ret = float((1 + port).prod() ** (252 / len(port)) - 1)
        ann_vol = float(port.std() * np.sqrt(252))
        cum     = (1 + port).cumprod()
        dd      = float(((cum - cum.cummax()) / cum.cummax()).min())
        return {
            "annualised_return": round(ann_ret, 4),
            "annualised_vol":    round(ann_vol, 4),
            "max_drawdown":      round(abs(dd), 4),
            "sharpe":            round((ann_ret) / ann_vol, 4) if ann_vol > 1e-8 else 0.0,
        }


# ── Simulator ─────────────────────────────────────────────────────────────────

class SyntheticMarketSimulator:
    """
    Generates synthetic price series under various market models.

    All models produce a pd.DataFrame of daily closing prices with
    the same interface, making them drop-in replacements for historical
    data in any Macro8 evaluation pipeline.
    """

    DEFAULT_ASSETS = ["SPY", "AAPL", "GLD"]

    def __init__(
        self,
        assets:    list[str] = None,
        n_days:    int       = 252,
        start_price: float   = 100.0,
        seed:      int       = 42,
    ):
        self.assets      = assets or self.DEFAULT_ASSETS
        self.n_days      = n_days
        self.start_price = start_price
        self.seed        = seed
        self._rng        = np.random.default_rng(seed)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        model:  SimModel    = SimModel.GBM,
        params: dict | None = None,
    ) -> SyntheticMarket:
        """
        Generate a synthetic market under the specified model.

        Args:
            model:  Which simulation model to use.
            params: Optional override parameters (model-specific).

        Returns:
            SyntheticMarket with prices, returns, and metadata.
        """
        p = params or {}
        self._rng = np.random.default_rng(self.seed)   # reset for reproducibility

        if model == SimModel.GBM:
            prices = self._gbm(
                mu=p.get("mu", 0.08),
                sigma=p.get("sigma", 0.16),
            )
        elif model == SimModel.JUMP_DIFFUSION:
            prices = self._jump_diffusion(
                mu=p.get("mu", 0.06),
                sigma=p.get("sigma", 0.18),
                jump_intensity=p.get("jump_intensity", 3.0),
                jump_mean=p.get("jump_mean", -0.03),
                jump_vol=p.get("jump_vol", 0.05),
            )
        elif model == SimModel.MEAN_REVERT:
            prices = self._mean_revert(
                theta=p.get("theta", 0.2),
                mu=p.get("mu", 100.0),
                sigma=p.get("sigma", 0.12),
            )
        elif model == SimModel.REGIME_SWITCH:
            prices = self._regime_switch(
                bull_mu=p.get("bull_mu", 0.15),
                bear_mu=p.get("bear_mu", -0.20),
                bull_vol=p.get("bull_vol", 0.12),
                bear_vol=p.get("bear_vol", 0.28),
                p_switch=p.get("p_switch", 0.02),
            )
        elif model == SimModel.CORR_SHOCK:
            prices = self._corr_shock(
                normal_vol=p.get("normal_vol", 0.15),
                shock_vol=p.get("shock_vol", 0.45),
                shock_corr=p.get("shock_corr", 0.85),
                shock_day=p.get("shock_day", self.n_days // 2),
                shock_dur=p.get("shock_dur", 30),
            )
        elif model == SimModel.INFLATION_SPIRAL:
            prices = self._inflation_spiral(
                base_mu=p.get("base_mu", 0.05),
                vol_drift=p.get("vol_drift", 0.002),
            )
        else:
            prices = self._gbm()

        returns  = prices.pct_change().dropna()
        metadata = self._compute_metadata(model, prices, returns, p)

        return SyntheticMarket(prices=prices, returns=returns, metadata=metadata)

    def generate_batch(
        self,
        models: list[SimModel] | None = None,
        n_per_model: int = 1,
    ) -> list[SyntheticMarket]:
        """
        Generate multiple synthetic markets (useful for bootstrap testing).

        Args:
            models:      List of models to generate. None = all six.
            n_per_model: How many runs per model (different seeds).

        Returns:
            List of SyntheticMarket objects.
        """
        models  = models or list(SimModel)
        results = []
        for model in models:
            for i in range(n_per_model):
                self.seed = i * 100 + hash(model) % 100
                results.append(self.generate(model))
        return results

    # ── Simulation models ─────────────────────────────────────────────────────

    def _gbm(self, mu: float = 0.08, sigma: float = 0.16) -> pd.DataFrame:
        """Geometric Brownian Motion — lognormal price process."""
        dt      = 1 / 252
        n       = self.n_days
        prices  = {}
        for i, asset in enumerate(self.assets):
            # Each asset gets slightly different drift/vol
            a_mu    = mu + self._rng.uniform(-0.02, 0.02)
            a_sigma = sigma * (0.8 + i * 0.15)
            W       = self._rng.standard_normal(n)
            r       = (a_mu - 0.5 * a_sigma ** 2) * dt + a_sigma * np.sqrt(dt) * W
            prices[asset] = self.start_price * np.exp(np.cumsum(r))

        return self._to_df(prices)

    def _jump_diffusion(
        self,
        mu: float = 0.06, sigma: float = 0.18,
        jump_intensity: float = 3.0,
        jump_mean: float = -0.03, jump_vol: float = 0.05,
    ) -> pd.DataFrame:
        """Merton jump-diffusion — GBM with Poisson-distributed jumps."""
        dt     = 1 / 252
        n      = self.n_days
        prices = {}
        for asset in self.assets:
            W     = self._rng.standard_normal(n)
            jumps = self._rng.poisson(jump_intensity * dt, n)
            J     = np.array([
                sum(self._rng.normal(jump_mean, jump_vol) for _ in range(j))
                for j in jumps
            ])
            r = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * W + J
            prices[asset] = self.start_price * np.exp(np.cumsum(r))

        return self._to_df(prices)

    def _mean_revert(
        self, theta: float = 0.2, mu: float = 100.0, sigma: float = 0.12,
    ) -> pd.DataFrame:
        """Ornstein-Uhlenbeck mean-reverting process."""
        dt     = 1 / 252
        n      = self.n_days
        prices = {}
        for asset in self.assets:
            p       = [self.start_price]
            a_mu    = mu * self._rng.uniform(0.9, 1.1)
            a_sigma = sigma * self._rng.uniform(0.8, 1.2)
            for _ in range(n - 1):
                dp = theta * (a_mu - p[-1]) * dt + a_sigma * p[-1] * np.sqrt(dt) * self._rng.standard_normal()
                p.append(max(p[-1] + dp, 1.0))
            prices[asset] = np.array(p)

        return self._to_df(prices)

    def _regime_switch(
        self,
        bull_mu: float = 0.15, bear_mu: float = -0.20,
        bull_vol: float = 0.12, bear_vol: float = 0.28,
        p_switch: float = 0.02,
    ) -> pd.DataFrame:
        """Two-state Markov regime-switching model."""
        dt     = 1 / 252
        n      = self.n_days
        # Generate regime path
        regime = np.zeros(n, dtype=int)  # 0 = bull, 1 = bear
        for t in range(1, n):
            if self._rng.random() < p_switch:
                regime[t] = 1 - regime[t - 1]
            else:
                regime[t] = regime[t - 1]

        mus   = np.where(regime == 0, bull_mu, bear_mu)
        vols  = np.where(regime == 0, bull_vol, bear_vol)
        W     = self._rng.standard_normal((n, len(self.assets)))

        prices = {}
        for i, asset in enumerate(self.assets):
            r = (mus - 0.5 * vols**2) * dt + vols * np.sqrt(dt) * W[:, i]
            prices[asset] = self.start_price * np.exp(np.cumsum(r))

        return self._to_df(prices)

    def _corr_shock(
        self,
        normal_vol:  float = 0.15,
        shock_vol:   float = 0.45,
        shock_corr:  float = 0.85,
        shock_day:   int   = 126,
        shock_dur:   int   = 30,
    ) -> pd.DataFrame:
        """Systemic correlation shock — assets become highly correlated during crisis."""
        n    = self.n_days
        na   = len(self.assets)
        dt   = 1 / 252

        def corr_matrix(rho: float) -> np.ndarray:
            C = np.full((na, na), rho)
            np.fill_diagonal(C, 1.0)
            return C

        returns_arr = np.zeros((n, na))
        normal_C    = corr_matrix(0.3)
        shock_C     = corr_matrix(shock_corr)

        normal_L = np.linalg.cholesky(normal_C)
        shock_L  = np.linalg.cholesky(shock_C)

        for t in range(n):
            in_shock = shock_day <= t < shock_day + shock_dur
            mu_t     = -0.001 if in_shock else 0.0003   # crisis drift
            vol_t    = shock_vol if in_shock else normal_vol
            L        = shock_L   if in_shock else normal_L
            z        = L @ self._rng.standard_normal(na)
            returns_arr[t] = mu_t * dt + vol_t * np.sqrt(dt) * z

        prices = {}
        for i, asset in enumerate(self.assets):
            prices[asset] = self.start_price * np.exp(np.cumsum(returns_arr[:, i]))

        return self._to_df(prices)

    def _inflation_spiral(
        self,
        base_mu: float = 0.05,
        vol_drift: float = 0.002,
    ) -> pd.DataFrame:
        """
        Inflation spiral — trending upward nominal returns but
        increasing volatility, commodity outperformance.
        """
        dt     = 1 / 252
        n      = self.n_days
        prices = {}

        for i, asset in enumerate(self.assets):
            is_commodity = asset in ("GLD", "OIL", "COMD")
            mu    = base_mu * (1.5 if is_commodity else 0.7)
            vols  = np.linspace(0.10, 0.10 + vol_drift * n, n)
            W     = self._rng.standard_normal(n)
            r     = (mu - 0.5 * vols**2) * dt + vols * np.sqrt(dt) * W
            prices[asset] = self.start_price * np.exp(np.cumsum(r))

        return self._to_df(prices)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _to_df(self, prices: dict[str, np.ndarray]) -> pd.DataFrame:
        """Convert price dict to date-indexed DataFrame."""
        dates = pd.date_range("2020-01-01", periods=self.n_days, freq="B")[:self.n_days]
        df    = pd.DataFrame(prices, index=dates[:len(next(iter(prices.values())))])
        return df.clip(lower=0.01)   # prices can't go negative

    def _compute_metadata(
        self,
        model:   SimModel,
        prices:  pd.DataFrame,
        returns: pd.DataFrame,
        params:  dict,
    ) -> SimulationMetadata:
        """Compute realised statistics for the simulation."""
        port    = returns.mean(axis=1)
        ann_ret = float((1 + port).prod() ** (252 / len(port)) - 1) if len(port) > 0 else 0.0
        ann_vol = float(port.std() * np.sqrt(252))
        cum     = (1 + port).cumprod()
        dd      = float(((cum - cum.cummax()) / cum.cummax()).min())

        meta = SimulationMetadata(
            model=model,
            n_days=self.n_days,
            n_assets=len(self.assets),
            assets=self.assets,
            seed=self.seed,
            realised_vol=ann_vol,
            realised_return=ann_ret,
            max_drawdown=abs(dd),
            parameters=params,
        )
        return meta
