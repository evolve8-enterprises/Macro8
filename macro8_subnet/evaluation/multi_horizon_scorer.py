"""
evaluation/multi_horizon_scorer.py
------------------------------------
Multi-Horizon Alpha Evaluation Engine for Macro8.

Replaces single 1-day IC with a full evaluation across:
    IC_1d, IC_5d, IC_21d, IC_63d   — cross-sectional Spearman IC
    turnover                         — mean daily rank change (scalability)
    stability                        — rank autocorrelation at lag 5
    portfolio simulation             — rank→weights→PnL at optimal rebalance

Composite score:
    mh_score = 0.40*IC_1d + 0.30*IC_5d + 0.20*IC_21d + 0.10*IC_63d
    tc_adj   = mh_score / (1 + 2.0 * turnover)
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

HORIZONS = [1, 5, 21, 63]
HORIZON_WEIGHTS = {1: 0.40, 5: 0.30, 21: 0.20, 63: 0.10}
DEFAULT_TC_BPS   = 10
TURNOVER_LAMBDA  = 2.0
MIN_OBS_RATIO    = 0.15


@dataclass
class HorizonIC:
    horizon: int
    ic:      float
    ic_ir:   float
    weight:  float


@dataclass
class PortfolioResult:
    rebal_freq:   int
    ann_return:   float
    ann_vol:      float
    sharpe:       float
    max_drawdown: float
    calmar:       float
    turnover_pa:  float
    n_days:       int

    def summary(self) -> str:
        return (
            f"Rebal={self.rebal_freq}d  "
            f"Ret={self.ann_return:.1%}  "
            f"Vol={self.ann_vol:.1%}  "
            f"Sharpe={self.sharpe:.2f}  "
            f"MaxDD={self.max_drawdown:.1%}  "
            f"TO={self.turnover_pa:.0f}x/yr"
        )


@dataclass
class MultiHorizonResult:
    formula_id:   str
    formula_string: str
    horizon_ics:  list[HorizonIC]
    mh_ic_score:  float
    tc_adj_score: float
    turnover:     float
    stability:    float
    best_horizon: int
    n_obs_1d:     int
    portfolio:    Optional[PortfolioResult] = None

    def ic_at(self, h: int) -> float:
        for hic in self.horizon_ics:
            if hic.horizon == h:
                return hic.ic
        return 0.0

    def summary_line(self) -> str:
        ics = "  ".join(f"{hic.horizon:2d}d={hic.ic:+.4f}" for hic in self.horizon_ics)
        return (
            f"{self.formula_string[:38]:<38}  "
            f"mh={self.mh_ic_score:+.4f}  "
            f"tc={self.tc_adj_score:+.4f}  "
            f"to={self.turnover:.3f}  [{ics}]"
        )

    def to_dict(self) -> dict:
        return {
            "formula_id":    self.formula_id,
            "formula_string": self.formula_string,
            "mh_ic_score":   round(self.mh_ic_score,  6),
            "tc_adj_score":  round(self.tc_adj_score,  6),
            "turnover":      round(self.turnover,      4),
            "stability":     round(self.stability,     4),
            "best_horizon":  self.best_horizon,
            "horizon_ics":   {hic.horizon: round(hic.ic, 6) for hic in self.horizon_ics},
        }


class MultiHorizonScorer:
    """
    Evaluates alpha signals across multiple lookforward horizons with
    turnover-adjusted composite scoring and optional portfolio simulation.

    Usage
    -----
        scorer = MultiHorizonScorer(prices)
        result = scorer.score_signal(signal_df, 'f1', 'rank(momentum_20d)')
        print(result.summary_line())
    """

    def __init__(
        self,
        prices:          pd.DataFrame,
        horizons:        list[int]  = None,
        horizon_weights: dict       = None,
        tc_bps:          float      = DEFAULT_TC_BPS,
        turnover_lambda: float      = TURNOVER_LAMBDA,
        min_obs_ratio:   float      = MIN_OBS_RATIO,
    ):
        self.prices          = prices.copy()
        self.horizons        = horizons        or HORIZONS
        self.horizon_wts     = horizon_weights or HORIZON_WEIGHTS
        self.tc_bps          = tc_bps
        self.turnover_lambda = turnover_lambda
        self.min_obs_ratio   = min_obs_ratio

        self._log_ret  = np.log(prices).diff()
        self._fwd_rets: dict[int, pd.DataFrame] = {}
        for h in self.horizons:
            self._fwd_rets[h] = self._log_ret.rolling(h).sum().shift(-h)

    # ── Public API ────────────────────────────────────────────────────────────

    def score_signal(
        self,
        signal:      pd.DataFrame,
        formula_id:  str,
        formula_str: str,
        run_pnl:     bool = False,
    ) -> MultiHorizonResult:
        sig_ranked  = signal.rank(axis=1, pct=True)
        horizon_ics = []
        for h in self.horizons:
            ic, ic_ir = self._ic_at(sig_ranked, h)
            horizon_ics.append(HorizonIC(h, ic, ic_ir, self.horizon_wts.get(h, 0)))

        tw     = sum(hic.weight for hic in horizon_ics)
        mh_ic  = (sum(hic.ic * hic.weight for hic in horizon_ics) / tw) if tw else 0.0
        to     = self._turnover(sig_ranked)
        stab   = self._stability(sig_ranked)
        best_h = self.horizons[int(np.argmax([abs(hic.ic) for hic in horizon_ics]))]
        tc_adj = mh_ic / (1 + self.turnover_lambda * to)

        # n_obs at 1d
        try:
            fwd1 = self._fwd_rets[1]
            common = sig_ranked.index.intersection(fwd1.index)
            n_obs = int((sig_ranked.loc[common].notna().any(axis=1) &
                         fwd1.loc[common].notna().any(axis=1)).sum())
        except Exception:
            n_obs = 0

        portfolio = None
        if run_pnl:
            portfolio = self._run_portfolio(sig_ranked, best_h)

        return MultiHorizonResult(
            formula_id=formula_id, formula_string=formula_str,
            horizon_ics=horizon_ics, mh_ic_score=float(mh_ic),
            tc_adj_score=float(tc_adj), turnover=float(to),
            stability=float(stab), best_horizon=best_h,
            n_obs_1d=n_obs, portfolio=portfolio,
        )

    def score_batch(
        self,
        triples:  list[tuple[pd.DataFrame, str, str]],
        run_pnl:  bool = False,
        verbose:  bool = False,
    ) -> list[MultiHorizonResult]:
        results = []
        for i, (sig, fid, fstr) in enumerate(triples):
            try:
                results.append(self.score_signal(sig, fid, fstr, run_pnl=run_pnl))
            except Exception:
                results.append(MultiHorizonResult(
                    formula_id=fid, formula_string=fstr, horizon_ics=[],
                    mh_ic_score=0.0, tc_adj_score=0.0, turnover=1.0,
                    stability=0.0, best_horizon=1, n_obs_1d=0,
                ))
            if verbose and (i + 1) % 5 == 0:
                print(f"  Scored {i+1}/{len(triples)}")
        results.sort(key=lambda r: r.tc_adj_score, reverse=True)
        return results

    # ── IC computation ────────────────────────────────────────────────────────

    def _ic_at(
        self, sig_ranked: pd.DataFrame, horizon: int
    ) -> tuple[float, float]:
        fwd = self._fwd_rets.get(horizon)
        if fwd is None or fwd.empty:
            return 0.0, 0.0
        common = sig_ranked.index.intersection(fwd.index)
        min_n  = max(10, int(len(sig_ranked) * self.min_obs_ratio))
        if len(common) < min_n:
            return 0.0, 0.0

        sig_a      = sig_ranked.loc[common]
        fwd_ranked = fwd.loc[common].rank(axis=1, pct=True)

        ic_series = []
        for t in common:
            s_t = sig_a.loc[t]
            r_t = fwd_ranked.loc[t]
            ok  = s_t.notna() & r_t.notna()
            if ok.sum() < 2:
                continue
            sv, rv = s_t[ok].values, r_t[ok].values
            sc, rc = sv - sv.mean(), rv - rv.mean()
            denom  = np.sqrt((sc**2).sum() * (rc**2).sum())
            if denom > 1e-8:
                ic_series.append(float(sc @ rc / denom))

        if len(ic_series) < 5:
            return 0.0, 0.0

        arr  = np.array(ic_series)
        mean = float(arr.mean())
        std  = float(arr.std())
        return mean, (mean / (std + 1e-8) if std > 0 else 0.0)

    # ── Signal characteristics ────────────────────────────────────────────────

    def _turnover(self, sig_ranked: pd.DataFrame) -> float:
        return float(sig_ranked.diff().abs().mean().mean())

    def _stability(self, sig_ranked: pd.DataFrame, lag: int = 5) -> float:
        try:
            corrs = [
                sig_ranked[c].dropna().autocorr(lag=lag)
                for c in sig_ranked.columns
                if len(sig_ranked[c].dropna()) > lag + 10
            ]
            corrs = [c for c in corrs if not np.isnan(c)]
            return float(np.mean(corrs)) if corrs else 0.0
        except Exception:
            return 0.0

    # ── Portfolio simulation ──────────────────────────────────────────────────

    def _run_portfolio(
        self, sig_ranked: pd.DataFrame, best_horizon: int
    ) -> Optional[PortfolioResult]:
        best_r, best_sharpe = None, -np.inf
        for freq in sorted({1, 5, 21, 63, best_horizon}):
            r = self._sim_at_freq(sig_ranked, freq)
            if r is not None and r.sharpe > best_sharpe:
                best_sharpe, best_r = r.sharpe, r
        return best_r

    def _sim_at_freq(
        self, sig_ranked: pd.DataFrame, freq: int
    ) -> Optional[PortfolioResult]:
        try:
            n      = sig_ranked.shape[1]
            fwd1d  = self._log_ret.shift(-1)
            prev_w = None
            w_rows = {}

            for i, date in enumerate(sig_ranked.index):
                if i % freq == 0:
                    r     = sig_ranked.loc[date]
                    w     = pd.Series(0.0, index=sig_ranked.columns)
                    w[r > 0.8] =  1.0 / n
                    w[r < 0.2] = -1.0 / n
                    prev_w = w
                w_rows[date] = prev_w if prev_w is not None else pd.Series(0.0, index=sig_ranked.columns)

            weights   = pd.DataFrame(w_rows).T
            pnl_gross = (weights.shift(1) * fwd1d).sum(axis=1)
            tc        = weights.diff().abs().sum(axis=1) * self.tc_bps / 10000
            pnl_net   = (pnl_gross - tc).dropna()

            if len(pnl_net) < 60:
                return None

            ann_ret  = float(pnl_net.mean() * 252)
            ann_vol  = float(pnl_net.std()  * np.sqrt(252))
            sharpe   = ann_ret / ann_vol if ann_vol > 1e-8 else 0.0
            cum      = (1 + pnl_net).cumprod()
            maxdd    = float(((cum - cum.cummax()) / cum.cummax()).min())
            calmar   = ann_ret / abs(maxdd) if abs(maxdd) > 1e-8 else 0.0
            to_pa    = float(weights.diff().abs().sum(axis=1).mean() * 252)

            return PortfolioResult(
                rebal_freq=freq, ann_return=ann_ret, ann_vol=ann_vol,
                sharpe=sharpe, max_drawdown=maxdd, calmar=calmar,
                turnover_pa=to_pa, n_days=len(pnl_net),
            )
        except Exception:
            return None


# ── Integration: score GP formula strings via FormulaEngine ──────────────────

def score_gp_batch(
    prices:      pd.DataFrame,
    formulas:    list[str],
    formula_ids: list[str],
    run_pnl:     bool = False,
    verbose:     bool = False,
) -> list[MultiHorizonResult]:
    """
    Score string formulas from the GP using FormulaEngine (true cross-sectional)
    then multi-horizon IC evaluation.
    """
    from macro8_subnet.alpha.formula_engine import FormulaEngine
    from macro8_subnet.alpha.feature_store   import FeatureStore

    eng    = FormulaEngine(FeatureStore(prices))
    scorer = MultiHorizonScorer(prices)
    triples = []

    for formula, fid in zip(formulas, formula_ids):
        try:
            r = eng.evaluate(formula)
            if r.success and r.signals:
                sig_df = pd.DataFrame(r.signals)
                triples.append((sig_df, fid, formula))
        except Exception:
            pass

    if verbose:
        print(f"  {len(triples)}/{len(formulas)} formulas evaluated")

    return scorer.score_batch(triples, run_pnl=run_pnl, verbose=verbose)
