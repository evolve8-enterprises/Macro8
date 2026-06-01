"""
evaluation/pr_tester.py
------------------------
Production-Readiness (PR) Tester — walk-forward validation for Macro8.

Answers the question: "does this strategy actually work out-of-sample?"

Three validation modes
----------------------

1. Simple train/test split
   Classic holdout. Fast. Good for initial screening.

2. Walk-forward validation
   Rolling windows: train on W days, test on T days, advance by S days.
   The gold standard for time-series strategies. Avoids look-ahead bias.

   Example (3yr train, quarterly test, quarterly step):
       Window 1:  train 2009-2012, test Q1 2012
       Window 2:  train 2009-2012 + Q1, test Q2 2012
       ...
       Window 48: train 2020-2023, test Q4 2023

3. Regime-conditional testing
   Splits test windows by market regime (bull/bear/crisis) and reports
   IC and Sharpe separately per regime. Answers:
   "does this signal work in downturns or only in bull markets?"

Outputs
-------
For each formula tested:

    PRResult:
        mean_ic_train     — mean IC across all training windows
        mean_ic_test      — mean IC across all test windows
        ic_stability      — std(test_IC) / mean(test_IC): lower = more stable
        sharpe_gross      — annualised gross Sharpe across test windows
        sharpe_net        — net of transaction costs
        max_drawdown      — worst peak-to-trough in test periods
        regime_ics        — {bull: IC, bear: IC, crisis: IC}
        n_windows         — number of walk-forward windows
        pass_rate         — fraction of test windows with positive IC

Key design decisions
--------------------
- Walk-forward uses BatchEvaluator (not PortfolioEvaluator) for IC scoring
  to keep each window evaluation fast (~150ms vs ~2s).
- Full portfolio simulation (PortfolioEvaluator) is only run on the
  final held-out period for surviving formulas.
- This gives the right budget: 48 windows × 5 formulas × 150ms = 36s total.

Usage
-----
    from macro8_subnet.evaluation.pr_tester import PRTester, WalkForwardConfig

    tester = PRTester(prices)

    # Simple split
    results = tester.simple_split(formulas, train_frac=0.67)

    # Walk-forward
    cfg     = WalkForwardConfig(train_days=756, test_days=63, step_days=21)
    results = tester.walk_forward(formulas, cfg)
    tester.print_results(results)

    # Regime-conditional
    results = tester.regime_split(formulas)
"""

from __future__ import annotations

import sys
import time
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


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class WalkForwardConfig:
    """
    Walk-forward validation configuration.

    Parameters
    ----------
    train_days : int  — Training window size in trading days.
    test_days  : int  — Test window size per fold.
    step_days  : int  — Step between consecutive folds (overlap if < test_days).
    min_windows: int  — Minimum windows required; raises if fewer available.
    """
    train_days:  int = 756    # 3 years
    test_days:   int = 63     # 1 quarter
    step_days:   int = 21     # monthly step
    min_windows: int = 6      # need at least 6 folds for meaningful estimate

    def __post_init__(self):
        assert self.train_days > 0
        assert self.test_days  > 0
        assert self.step_days  > 0

    @classmethod
    def quarterly_3yr(cls) -> "WalkForwardConfig":
        return cls(train_days=756, test_days=63, step_days=21)

    @classmethod
    def annual_5yr(cls) -> "WalkForwardConfig":
        return cls(train_days=1260, test_days=252, step_days=63)

    @classmethod
    def monthly_2yr(cls) -> "WalkForwardConfig":
        return cls(train_days=504, test_days=21, step_days=5)

    def n_windows(self, n_total: int) -> int:
        if n_total < self.train_days + self.test_days:
            return 0
        return (n_total - self.train_days - self.test_days) // self.step_days + 1


# ── Results ───────────────────────────────────────────────────────────────────

@dataclass
class WindowResult:
    """IC result for one walk-forward window."""
    window_idx:  int
    train_start: pd.Timestamp
    train_end:   pd.Timestamp
    test_start:  pd.Timestamp
    test_end:    pd.Timestamp
    train_ic:    float
    test_ic:     float
    regime:      str    # "bull" | "bear" | "crisis"


@dataclass
class PRResult:
    """
    Production-readiness test result for one formula.

    Core metrics
    ------------
    mean_ic_test   — Average IC across test windows (primary signal quality measure).
    ic_stability   — CV of test ICs: std/|mean|. Lower = more stable. Target < 1.0.
    pass_rate      — Fraction of test windows with positive IC. Target > 0.55.
    sharpe_net     — Net Sharpe on combined test periods (after transaction costs).

    Regime breakdown
    ----------------
    regime_ics    — IC per regime (bull/bear/crisis). A good signal works in all.
    worst_regime_ic — Minimum IC across regimes. Key for drawdown protection.
    """
    formula:         str
    n_windows:       int

    # IC statistics
    mean_ic_train:   float
    mean_ic_test:    float
    ic_stability:    float    # std(test_ics) / |mean_ic_test|
    pass_rate:       float    # fraction of windows with test_ic > 0
    ic_ir:           float    # mean_ic_test / std(test_ics)

    # Portfolio performance (from final OOS window)
    sharpe_gross:    float = 0.0
    sharpe_net:      float = 0.0
    max_drawdown:    float = 0.0
    daily_turnover:  float = 0.0
    cost_drag_bps:   float = 0.0

    # Regime breakdown
    regime_ics:          dict[str, float] = field(default_factory=dict)
    worst_regime_ic:     float = 0.0

    # Window-level detail
    window_results:  list[WindowResult] = field(default_factory=list)

    # Verdict
    passes:          bool = False
    fail_reasons:    list[str] = field(default_factory=list)

    def verdict_line(self) -> str:
        icon = "✓" if self.passes else "✗"
        return (
            f"{icon} {self.formula[:38]:<40} "
            f"IC_test={self.mean_ic_test:>+7.4f}  "
            f"stability={self.ic_stability:>5.2f}  "
            f"pass_rate={self.pass_rate:>4.0%}  "
            f"net_Sharpe={self.sharpe_net:>+6.3f}"
        )

    def regime_line(self) -> str:
        parts = []
        for regime in ["bull", "bear", "crisis"]:
            ic = self.regime_ics.get(regime, float("nan"))
            flag = "✓" if not np.isnan(ic) and ic > 0 else "✗"
            parts.append(f"{regime}={flag}{ic:>+.4f}" if not np.isnan(ic) else f"{regime}=N/A")
        return f"  Regimes: {' | '.join(parts)} | worst={self.worst_regime_ic:>+.4f}"


# ── PR Tester ─────────────────────────────────────────────────────────────────

class PRTester:
    """
    Production-readiness tester for Macro8 strategies.

    Runs walk-forward validation and regime-conditional tests on
    formula candidates before they are submitted to the validator.

    Parameters
    ----------
    prices:          pd.DataFrame — market prices (date × tickers).
    capital:         float       — portfolio size for cost calculation.
    min_ic_threshold: float      — minimum test IC to pass (default 0.005).
    min_pass_rate:   float       — minimum fraction of windows with IC>0 (default 0.55).
    max_ic_stability: float      — maximum IC CV (default 2.0; lower = more stable).
    verbose:         bool        — print progress.
    """

    def __init__(
        self,
        prices:            pd.DataFrame,
        capital:           float = 100_000,
        min_ic_threshold:  float = 0.005,
        min_pass_rate:     float = 0.55,
        max_ic_stability:  float = 2.0,
        verbose:           bool  = True,
    ):
        self.prices           = prices
        self.capital          = capital
        self.min_ic           = min_ic_threshold
        self.min_pass_rate    = min_pass_rate
        self.max_ic_stability = max_ic_stability
        self.verbose          = verbose
        self._n               = len(prices)

    # ── Public API ────────────────────────────────────────────────────────────

    def simple_split(
        self,
        formulas:   list[str],
        train_frac: float = 0.67,
    ) -> list[PRResult]:
        """
        Simple train/test holdout.

        Args:
            formulas:   Formula strings to test.
            train_frac: Fraction of data used for training.

        Returns:
            List of PRResult, one per formula.
        """
        split      = int(self._n * train_frac)
        train_p    = self.prices.iloc[:split]
        test_p     = self.prices.iloc[split:]

        train_ics  = self._batch_ic(formulas, train_p)
        test_ics   = self._batch_ic(formulas, test_p)

        results = []
        for f in formulas:
            t_ic = float(train_ics.get(f, 0.0))
            o_ic = float(test_ics.get(f, 0.0))
            # single-split: stability = |train-test| / |train|
            stab  = abs(t_ic - o_ic) / (abs(t_ic) + 1e-8)
            pr = PRResult(
                formula=f,
                n_windows=1,
                mean_ic_train=t_ic,
                mean_ic_test=o_ic,
                ic_stability=stab,
                pass_rate=1.0 if o_ic > 0 else 0.0,
                ic_ir=o_ic / (abs(o_ic) + 1e-8),
            )
            self._compute_portfolio_metrics(pr, test_p)
            self._compute_regime_ics(pr, test_p)
            self._set_verdict(pr)
            results.append(pr)

        return results

    def walk_forward(
        self,
        formulas: list[str],
        config:   Optional[WalkForwardConfig] = None,
    ) -> list[PRResult]:
        """
        Walk-forward validation across rolling windows.

        For each formula, computes IC on each test fold and aggregates.
        Full portfolio simulation (Sharpe, drawdown) runs only on the
        final held-out period to keep runtime manageable.

        Args:
            formulas: Formula strings to test.
            config:   Walk-forward configuration. Default: 3yr train, quarterly test.

        Returns:
            List of PRResult sorted by test IC descending.
        """
        cfg = config or WalkForwardConfig.quarterly_3yr()
        n_win = cfg.n_windows(self._n)

        if n_win < cfg.min_windows:
            if self.verbose:
                print(f"[PRTester] Only {n_win} windows available "
                      f"(need ≥ {cfg.min_windows}). Using simple split.")
            return self.simple_split(formulas)

        if self.verbose:
            print(f"[PRTester] Walk-forward: {n_win} windows | "
                  f"train={cfg.train_days}d test={cfg.test_days}d step={cfg.step_days}d")

        # Build window indices
        windows = []
        for i in range(n_win):
            t0 = i * cfg.step_days
            t1 = t0 + cfg.train_days
            t2 = min(t1 + cfg.test_days, self._n)
            if t2 > self._n:
                break
            windows.append((t0, t1, t2))

        # Per-formula accumulators
        formula_windows: dict[str, list[WindowResult]] = {f: [] for f in formulas}

        t_start = time.perf_counter()
        for wi, (t0, t1, t2) in enumerate(windows):
            train_p  = self.prices.iloc[t0:t1]
            test_p   = self.prices.iloc[t1:t2]
            regime   = self._regime_label(train_p)

            train_ics = self._batch_ic(formulas, train_p)
            test_ics  = self._batch_ic(formulas, test_p)

            for f in formulas:
                wr = WindowResult(
                    window_idx=wi,
                    train_start=train_p.index[0],
                    train_end=train_p.index[-1],
                    test_start=test_p.index[0],
                    test_end=test_p.index[-1],
                    train_ic=float(train_ics.get(f, 0.0)),
                    test_ic=float(test_ics.get(f, 0.0)),
                    regime=regime,
                )
                formula_windows[f].append(wr)

            if self.verbose and (wi + 1) % 10 == 0:
                elapsed = time.perf_counter() - t_start
                print(f"  Window {wi+1}/{n_win} | {elapsed:.1f}s elapsed | "
                      f"~{elapsed/(wi+1)*(n_win-wi-1):.0f}s remaining")

        # Assemble PRResult per formula
        results = []
        final_test = self.prices.iloc[windows[-1][1]:]  # last test window for portfolio sim

        for f in formulas:
            wrs       = formula_windows[f]
            test_ics  = [w.test_ic  for w in wrs]
            train_ics = [w.train_ic for w in wrs]

            mean_test  = float(np.mean(test_ics))
            mean_train = float(np.mean(train_ics))
            std_test   = float(np.std(test_ics)) + 1e-8
            ic_ir      = mean_test / std_test
            stability  = std_test / (abs(mean_test) + 1e-8)
            pass_rate  = float(np.mean([ic > 0 for ic in test_ics]))

            pr = PRResult(
                formula=f,
                n_windows=len(wrs),
                mean_ic_train=mean_train,
                mean_ic_test=mean_test,
                ic_stability=stability,
                pass_rate=pass_rate,
                ic_ir=ic_ir,
                window_results=wrs,
            )
            self._compute_portfolio_metrics(pr, final_test)
            self._compute_regime_ics(pr, wrs)
            self._set_verdict(pr)
            results.append(pr)

        results.sort(key=lambda r: r.mean_ic_test, reverse=True)
        return results

    def regime_split(
        self,
        formulas: list[str],
    ) -> list[PRResult]:
        """
        Regime-conditional testing.

        Splits the full price history into bull/bear/crisis periods
        (based on 252-day trailing return) and evaluates formula IC
        separately in each regime.

        Returns PRResult with regime_ics populated.
        """
        labels = self._label_regimes(self.prices)
        results = []

        for f in formulas:
            regime_ics = {}
            for regime in ["bull", "bear", "crisis"]:
                mask = labels == regime
                if mask.sum() < 100:
                    continue
                regime_prices = self.prices[mask]
                ics           = self._batch_ic([f], regime_prices)
                regime_ics[regime] = float(ics.get(f, 0.0))

            all_ics = list(regime_ics.values())
            worst   = min(all_ics) if all_ics else 0.0
            mean_test = float(np.mean(all_ics)) if all_ics else 0.0

            pr = PRResult(
                formula=f,
                n_windows=len(all_ics),
                mean_ic_train=mean_test,
                mean_ic_test=mean_test,
                ic_stability=float(np.std(all_ics) / (abs(mean_test) + 1e-8)),
                pass_rate=float(np.mean([ic > 0 for ic in all_ics])),
                ic_ir=mean_test / (float(np.std(all_ics)) + 1e-8),
                regime_ics=regime_ics,
                worst_regime_ic=worst,
            )
            self._compute_portfolio_metrics(pr, self.prices)
            self._set_verdict(pr)
            results.append(pr)

        results.sort(key=lambda r: r.mean_ic_test, reverse=True)
        return results

    def print_results(
        self,
        results:    list[PRResult],
        show_regime: bool = True,
        top_n:      int   = 20,
    ) -> None:
        """Print formatted walk-forward results."""
        n_pass = sum(1 for r in results if r.passes)
        print(f"\n  {'═'*90}")
        print(f"  WALK-FORWARD VALIDATION — {len(results)} formulas | {n_pass} pass")
        print(f"  {'═'*90}")
        print(f"  {'Formula':<42} {'IC_tst':>8} {'Stab':>6} {'Pass%':>6} "
              f"{'IC_IR':>6} {'ShrpN':>7} {'Verdict'}")
        print(f"  {'─'*90}")

        for r in results[:top_n]:
            stab_flag = "✓" if r.ic_stability < 1.0 else "~" if r.ic_stability < 2.0 else "✗"
            pass_flag = "✓" if r.pass_rate > 0.55 else "~" if r.pass_rate > 0.4 else "✗"
            verdict   = "PASS" if r.passes else f"FAIL({','.join(r.fail_reasons[:2])})"
            print(
                f"  {r.formula[:40]:<42} "
                f"{r.mean_ic_test:>+8.4f} "
                f"{stab_flag}{r.ic_stability:>5.2f} "
                f"{pass_flag}{r.pass_rate:>5.0%} "
                f"{r.ic_ir:>+6.2f} "
                f"{r.sharpe_net:>+7.3f}  "
                f"{verdict}"
            )
            if show_regime and r.regime_ics:
                print(r.regime_line())

        print(f"  {'═'*90}")
        if n_pass > 0:
            passing = [r for r in results if r.passes]
            avg_ic  = np.mean([r.mean_ic_test for r in passing])
            avg_ns  = np.mean([r.sharpe_net   for r in passing])
            print(f"  Passing: {n_pass}/{len(results)} | "
                  f"avg IC={avg_ic:.4f} | avg net Sharpe={avg_ns:.3f}")
        print()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _batch_ic(self, formulas: list[str], prices: pd.DataFrame) -> dict[str, float]:
        """
        Compute mean IC for each formula on a price window.
        Uses BatchEvaluator (fast: ~150ms for 5 formulas on 756 days).
        """
        if len(prices) < 60:
            return {}
        try:
            from macro8_subnet.alpha.batch_evaluator import BatchEvaluator
            ev  = BatchEvaluator(prices, min_ic=0.0)
            res = ev.evaluate(formulas)
            return dict(zip(res.formulas, res.mean_ics.tolist()))
        except Exception:
            return {}

    def _compute_portfolio_metrics(
        self,
        pr:     PRResult,
        prices: pd.DataFrame,
    ) -> None:
        """Run PortfolioEvaluator on the given price window and populate pr."""
        if len(prices) < 100:
            return
        try:
            from macro8_subnet.alpha.portfolio_evaluator import PortfolioEvaluator
            ev  = PortfolioEvaluator(prices, apply_costs=True,
                                     default_capital=self.capital)
            res = ev.evaluate([pr.formula])
            if res.n_formulas > 0:
                ps = res.portfolio_scores[0]
                pr.sharpe_gross   = ps.sharpe
                pr.sharpe_net     = ps.net_sharpe
                pr.max_drawdown   = ps.max_drawdown
                pr.daily_turnover = ps.daily_turnover
                pr.cost_drag_bps  = ps.cost_drag_annual
        except Exception:
            pass

    def _compute_regime_ics(
        self,
        pr:      PRResult,
        source,  # list[WindowResult] OR pd.DataFrame
    ) -> None:
        """Compute per-regime IC either from window results or raw prices."""
        if isinstance(source, list):
            # Aggregate from window results
            regime_ics: dict[str, list[float]] = {}
            for wr in source:
                regime_ics.setdefault(wr.regime, []).append(wr.test_ic)
            pr.regime_ics = {
                k: float(np.mean(v)) for k, v in regime_ics.items()
            }
        else:
            # Compute from raw prices split by regime
            prices   = source
            labels   = self._label_regimes(prices)
            for regime in ["bull", "bear", "crisis"]:
                mask = labels == regime
                if mask.sum() < 100:
                    continue
                ics = self._batch_ic([pr.formula], prices[mask])
                if pr.formula in ics:
                    pr.regime_ics[regime] = float(ics[pr.formula])

        if pr.regime_ics:
            pr.worst_regime_ic = min(pr.regime_ics.values())

    def _set_verdict(self, pr: PRResult) -> None:
        """Apply pass/fail criteria and populate pr.passes and pr.fail_reasons."""
        reasons = []
        if pr.mean_ic_test < self.min_ic:
            reasons.append(f"IC<{self.min_ic:.3f}")
        if pr.pass_rate < self.min_pass_rate:
            reasons.append(f"pass_rate<{self.min_pass_rate:.0%}")
        if pr.ic_stability > self.max_ic_stability:
            reasons.append(f"stability>{self.max_ic_stability:.1f}")
        if pr.sharpe_net < 0:
            reasons.append("net_Sharpe<0")
        pr.fail_reasons = reasons
        pr.passes       = len(reasons) == 0

    def _regime_label(self, prices: pd.DataFrame) -> str:
        """
        Classify a price window as bull / bear / crisis.
        Based on equal-weight portfolio 252-day trailing return and vol.
        """
        ret  = prices.pct_change().mean(axis=1)
        ann  = ret.mean() * 252
        vol  = ret.std() * np.sqrt(252)
        if vol > 0.35:
            return "crisis"
        if ann < -0.10:
            return "bear"
        return "bull"

    def _label_regimes(self, prices: pd.DataFrame) -> pd.Series:
        """
        Label each day of prices as bull / bear / crisis.
        Uses rolling 252-day annualised return on equal-weight portfolio.
        """
        ret      = prices.pct_change().mean(axis=1)
        roll_ret = ret.rolling(252).mean() * 252
        roll_vol = ret.rolling(252).std()  * np.sqrt(252)

        labels = pd.Series("bull", index=prices.index)
        labels[roll_ret < -0.05]   = "bear"
        labels[roll_vol > 0.30]    = "crisis"
        return labels
