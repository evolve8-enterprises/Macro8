"""
evaluation/scenario_engine.py
------------------------------
Scenario Engine for Macro8 — Sprint 26.

Answers: "how does each strategy survive possible futures?"

This is NOT prediction. It is:

    ranking strategies by how they survive possible futures

Which is far more powerful than prediction for portfolio construction.

Eight calibrated scenarios
--------------------------
Each scenario applies directional shocks to specific asset returns,
with increased volatility and a realistic duration. Scenarios are
calibrated from actual historical episodes:

    rates_up_200bps    — 2022 Fed hiking cycle (TLT −26%, bonds crash)
    rates_down_100bps  — 2019 / 2024 Fed pivot (bonds rally)
    equity_crash_30pct — 2008 GFC / 2020 COVID (SPY −35%)
    oil_spike_50pct    — 1973 / 2022 (DBC +40%, stagflation risk)
    china_crisis       — 2015 / 2021 (FXI −40%, EM contagion)
    soft_landing       — 1995 / 2024 (Goldilocks: growth + disinflation)
    stagflation        — 1970s (growth ↓ inflation ↑, no safe haven)
    ai_boom            — 2023-2024 (QQQ +55%, productivity surge)

Scenario simulation
-------------------
For each scenario × formula pair:

    1. Inject daily shocks into the price series over the scenario duration
       Shock = target_total_return / duration_days + noise × vol_multiplier

    2. Evaluate the formula's signal on the shocked prices

    3. Simulate the portfolio and compute Sharpe, drawdown, turnover

    4. Record survival metrics:
       - survived: net Sharpe > 0 on shocked prices
       - drawdown: worst peak-to-trough under scenario
       - relative_sharpe: Sharpe(shocked) / Sharpe(base)

ScenarioReport per formula
--------------------------
    scenario_results   : dict[scenario_name → ScenarioResult]
    n_survived         : how many scenarios it survives (net Sharpe > 0)
    worst_drawdown     : worst drawdown across all scenarios
    best_scenario      : scenario name where it performs best
    worst_scenario     : scenario name where it performs worst
    robustness_score   : fraction of scenarios survived (0-1)

Usage
-----
    engine = ScenarioEngine(prices)

    # Test one formula
    report = engine.run(['market_corr_60d', 'momentum_20d'])
    engine.print_report(report)

    # Test under one scenario only
    result = engine.run_one_scenario('equity_crash_30pct', formulas)

    # Get the most robust formula
    best   = max(report, key=lambda r: r.robustness_score)
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


# ── Scenario definitions ─────────────────────────────────────────────────────
# Calibrated from real historical episodes.
# shocks: {ticker: total_return_over_duration}  (fractional, e.g. -0.20 = -20%)
# vol_mult: multiplier on historical daily vol during the scenario
# duration_days: trading days the scenario unfolds over

SCENARIOS: dict[str, dict] = {
    "rates_up_200bps": {
        "description": "Fed hikes 200bps — bonds crash, value > growth",
        "episode":     "2022 rate hiking cycle",
        "shocks": {
            "TLT": -0.26, "IEF": -0.15, "HYG": -0.12,
            "QQQ": -0.18, "VNQ": -0.22, "GLD": -0.03,
        },
        "vol_mult":     1.6,
        "duration_days": 126,   # 6 months
    },
    "rates_down_100bps": {
        "description": "Fed cuts 100bps — bonds rally, growth outperforms",
        "episode":     "2019 / 2024 Fed pivot",
        "shocks": {
            "TLT": +0.12, "IEF": +0.07, "HYG": +0.06,
            "QQQ": +0.10, "SPY": +0.08, "GLD": +0.05,
        },
        "vol_mult":     0.85,
        "duration_days": 126,
    },
    "equity_crash_30pct": {
        "description": "Equity market −30% in 3 months (2008 / 2020 style)",
        "episode":     "2008 GFC / 2020 COVID crash",
        "shocks": {
            "SPY": -0.30, "QQQ": -0.35, "IWM": -0.38,
            "EEM": -0.40, "VNQ": -0.44, "FXI": -0.42,
            "HYG": -0.22, "DBC": -0.25,
            "TLT": +0.18, "GLD": +0.10,
        },
        "vol_mult":     3.5,
        "duration_days": 63,    # 3 months
    },
    "oil_spike_50pct": {
        "description": "Oil +50% in 6 weeks — stagflation risk",
        "episode":     "1973 oil embargo / 2022 Russia-Ukraine",
        "shocks": {
            "DBC": +0.35, "GLD": +0.12, "TLT": -0.10,
            "QQQ": -0.14, "EEM": -0.12, "SPY": -0.08,
        },
        "vol_mult":     2.0,
        "duration_days": 42,    # 6 weeks
    },
    "china_crisis": {
        "description": "China deleveraging — EM selloff, flight to safety",
        "episode":     "2015 devaluation / 2021 property crisis",
        "shocks": {
            "FXI": -0.38, "EEM": -0.27, "DBC": -0.18,
            "TLT": +0.10, "GLD": +0.14, "SPY": -0.10,
            "QQQ": -0.08,
        },
        "vol_mult":     2.2,
        "duration_days": 45,
    },
    "soft_landing": {
        "description": "Goldilocks — low inflation, moderate growth, no crash",
        "episode":     "1995 soft landing / 2024",
        "shocks": {
            "SPY": +0.12, "QQQ": +0.18, "IWM": +0.14,
            "TLT": +0.05, "HYG": +0.06, "EEM": +0.10,
        },
        "vol_mult":     0.70,
        "duration_days": 252,   # 1 year
    },
    "stagflation": {
        "description": "High inflation + low growth — 1970s replay",
        "episode":     "1973-1974 / partial 2022",
        "shocks": {
            "SPY": -0.20, "QQQ": -0.28, "TLT": -0.18,
            "GLD": +0.28, "DBC": +0.22,
            "HYG": -0.14, "IWM": -0.24,
        },
        "vol_mult":     2.0,
        "duration_days": 252,
    },
    "ai_boom": {
        "description": "AI productivity surge — tech outperforms all asset classes",
        "episode":     "2023-2024 AI rally",
        "shocks": {
            "QQQ": +0.35, "SPY": +0.22, "IWM": +0.14,
            "GLD": -0.03, "TLT": -0.04, "DBC": -0.08,
        },
        "vol_mult":     0.85,
        "duration_days": 252,
    },
}


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class ScenarioResult:
    """Performance of one formula under one scenario."""
    scenario_name:    str
    formula:          str
    description:      str

    # Shocked market performance
    sharpe_base:      float   # Sharpe on unshocked prices
    sharpe_shocked:   float   # Sharpe on shocked prices
    relative_sharpe:  float   # shocked / base (1.0 = no change; >1 = benefits)
    drawdown_shocked: float   # max drawdown under scenario (negative)
    ic_base:          float   # mean IC on unshocked prices
    ic_shocked:       float   # mean IC on shocked prices
    ic_change:        float   # ic_shocked - ic_base

    # Survival
    survived:         bool    # net_sharpe_shocked > 0

    def one_line(self) -> str:
        icon = "✓" if self.survived else "✗"
        rel  = f"{self.relative_sharpe:>+.2f}x"
        return (
            f"  {icon} {self.scenario_name:<24} "
            f"Sharpe: base={self.sharpe_base:>+.3f} → shocked={self.sharpe_shocked:>+.3f} "
            f"({rel})  DD={self.drawdown_shocked:.2f}  IC: {self.ic_base:>+.4f}→{self.ic_shocked:>+.4f}"
        )


@dataclass
class ScenarioReport:
    """Full scenario analysis for one formula."""
    formula:           str
    scenario_results:  dict[str, ScenarioResult] = field(default_factory=dict)

    # Aggregate metrics
    robustness_score:  float = 0.0   # fraction of scenarios survived
    n_survived:        int   = 0
    n_scenarios:       int   = 0
    worst_drawdown:    float = 0.0   # worst DD across all scenarios
    best_scenario:     str   = ""    # where it performs best (relative Sharpe)
    worst_scenario:    str   = ""    # where it performs worst
    mean_relative_sharpe: float = 1.0

    def __post_init__(self):
        if self.scenario_results:
            self._compute_aggregates()

    def _compute_aggregates(self) -> None:
        results = list(self.scenario_results.values())
        self.n_scenarios      = len(results)
        self.n_survived       = sum(1 for r in results if r.survived)
        self.robustness_score = self.n_survived / max(self.n_scenarios, 1)
        self.worst_drawdown   = min((r.drawdown_shocked for r in results), default=0.0)

        rel_sharpes = [(r.scenario_name, r.relative_sharpe) for r in results]
        if rel_sharpes:
            self.best_scenario  = max(rel_sharpes, key=lambda x: x[1])[0]
            self.worst_scenario = min(rel_sharpes, key=lambda x: x[1])[0]
            self.mean_relative_sharpe = float(np.mean([r.relative_sharpe for r in results]))

    def summary_line(self) -> str:
        icon = "✓" if self.robustness_score >= 0.75 else "~" if self.robustness_score >= 0.5 else "✗"
        return (
            f"{icon} {self.formula[:40]:<42} "
            f"robust={self.robustness_score:.0%}  "
            f"survived={self.n_survived}/{self.n_scenarios}  "
            f"worst_DD={self.worst_drawdown:>+.2f}  "
            f"best={self.best_scenario[:16]}  worst={self.worst_scenario[:16]}"
        )

    def print(self) -> None:
        print(f"\n  Scenario results for: {self.formula}")
        print(f"  Robustness: {self.robustness_score:.0%} ({self.n_survived}/{self.n_scenarios} survived)")
        for name, result in self.scenario_results.items():
            print(result.one_line())


# ── Scenario Engine ───────────────────────────────────────────────────────────

class ScenarioEngine:
    """
    Tests strategy robustness across calibrated economic scenarios.

    For each scenario, applies return shocks to the price series,
    then evaluates strategy performance on shocked vs baseline data.

    Parameters
    ----------
    prices:   pd.DataFrame — market prices (date × tickers).
    capital:  float        — portfolio size for cost simulation.
    seed:     int          — random seed for shock noise reproducibility.
    verbose:  bool         — print progress.
    """

    def __init__(
        self,
        prices:  pd.DataFrame,
        capital: float = 100_000,
        seed:    int   = 42,
        verbose: bool  = True,
    ):
        self.prices  = prices
        self.capital = capital
        self.seed    = seed
        self.verbose = verbose
        self._universe = list(prices.columns)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        formulas:  list[str],
        scenarios: Optional[list[str]] = None,
    ) -> list[ScenarioReport]:
        """
        Run all (or selected) scenarios for each formula.

        Args:
            formulas:  Formula strings to evaluate.
            scenarios: Scenario names to run. None = all 8 scenarios.

        Returns:
            List of ScenarioReport, sorted by robustness_score descending.
        """
        scenario_names = scenarios or list(SCENARIOS.keys())
        reports        = {f: ScenarioReport(formula=f) for f in formulas}

        t_start = time.perf_counter()
        for si, sname in enumerate(scenario_names):
            if sname not in SCENARIOS:
                if self.verbose:
                    print(f"[Scenario] Unknown scenario '{sname}', skipping")
                continue

            scfg = SCENARIOS[sname]
            if self.verbose:
                print(f"[Scenario] {si+1}/{len(scenario_names)}: {sname} — {scfg['description']}")

            results = self._run_scenario(sname, scfg, formulas)

            for f, sr in zip(formulas, results):
                reports[f].scenario_results[sname] = sr

        # Compute aggregates and sort
        for report in reports.values():
            report._compute_aggregates()

        sorted_reports = sorted(
            reports.values(),
            key=lambda r: r.robustness_score,
            reverse=True,
        )

        if self.verbose:
            elapsed = time.perf_counter() - t_start
            print(f"[Scenario] Complete in {elapsed:.1f}s — "
                  f"{len(formulas)} formulas × {len(scenario_names)} scenarios")

        return sorted_reports

    def run_one_scenario(
        self,
        scenario_name: str,
        formulas:      list[str],
    ) -> list[ScenarioResult]:
        """Run a single scenario against all formulas."""
        if scenario_name not in SCENARIOS:
            raise ValueError(f"Unknown scenario '{scenario_name}'. "
                             f"Available: {list(SCENARIOS.keys())}")
        scfg = SCENARIOS[scenario_name]
        return self._run_scenario(scenario_name, scfg, formulas)

    def print_report(
        self,
        reports: list[ScenarioReport],
        top_n:   int = 10,
    ) -> None:
        """Print formatted scenario analysis."""
        print(f"\n  {'═'*92}")
        print(f"  SCENARIO ENGINE — {len(reports)} formulas × {len(SCENARIOS)} scenarios")
        print(f"  {'═'*92}")
        print(f"  {'Formula':<44} {'Robust':>8} {'Surv':>8} {'WorstDD':>8} "
              f"{'Best scenario':<22} {'Worst scenario'}")
        print(f"  {'─'*92}")

        for r in reports[:top_n]:
            icon = "✓" if r.robustness_score >= 0.75 else "~" if r.robustness_score >= 0.5 else "✗"
            print(
                f"  {icon} {r.formula[:42]:<44} "
                f"{r.robustness_score:>8.0%} "
                f"{r.n_survived:>3}/{r.n_scenarios:<4} "
                f"{r.worst_drawdown:>+8.2f} "
                f"{r.best_scenario[:20]:<22} "
                f"{r.worst_scenario[:20]}"
            )

        print(f"  {'═'*92}")

        # Scenario-level survival rates
        print(f"\n  Per-scenario survival rates:")
        scenario_survival = {}
        for r in reports:
            for sname, sr in r.scenario_results.items():
                scenario_survival.setdefault(sname, []).append(sr.survived)

        for sname, survivals in sorted(scenario_survival.items()):
            rate = np.mean(survivals)
            bar  = "█" * int(rate * 20)
            desc = SCENARIOS.get(sname, {}).get("description", "")[:40]
            print(f"  {sname:<26} {rate:>5.0%} {bar:<20} {desc}")
        print()

    def available_scenarios(self) -> dict[str, str]:
        """Return dict of {scenario_name: description}."""
        return {k: v["description"] for k, v in SCENARIOS.items()}

    # ── Private: scenario simulation ──────────────────────────────────────────

    def _run_scenario(
        self,
        scenario_name: str,
        scfg:          dict,
        formulas:      list[str],
    ) -> list[ScenarioResult]:
        """Apply one scenario to all formulas and return ScenarioResult list."""
        shocked_prices = self._apply_shock(
            self.prices,
            shocks=scfg["shocks"],
            vol_mult=scfg["vol_mult"],
            duration_days=scfg["duration_days"],
        )

        # IC on base and shocked
        base_ics    = self._batch_ic(formulas, self.prices)
        shocked_ics = self._batch_ic(formulas, shocked_prices)

        # Portfolio metrics on base and shocked
        base_metrics    = self._portfolio_metrics(formulas, self.prices)
        shocked_metrics = self._portfolio_metrics(formulas, shocked_prices)

        results = []
        for f in formulas:
            base_ic  = float(base_ics.get(f, 0.0))
            shk_ic   = float(shocked_ics.get(f, 0.0))
            base_m   = base_metrics.get(f, {})
            shk_m    = shocked_metrics.get(f, {})

            base_sharpe = base_m.get("net_sharpe", 0.0)
            shk_sharpe  = shk_m.get("net_sharpe", 0.0)
            rel_sharpe  = (shk_sharpe / (abs(base_sharpe) + 1e-8)
                           if abs(base_sharpe) > 0.01 else 0.0)

            sr = ScenarioResult(
                scenario_name=scenario_name,
                formula=f,
                description=scfg["description"],
                sharpe_base=base_sharpe,
                sharpe_shocked=shk_sharpe,
                relative_sharpe=rel_sharpe,
                drawdown_shocked=shk_m.get("max_drawdown", 0.0),
                ic_base=base_ic,
                ic_shocked=shk_ic,
                ic_change=shk_ic - base_ic,
                survived=shk_sharpe > 0,
            )
            results.append(sr)

        return results

    def _apply_shock(
        self,
        prices:        pd.DataFrame,
        shocks:        dict[str, float],   # {ticker: total_return}
        vol_mult:      float,
        duration_days: int,
        seed:          int = None,
    ) -> pd.DataFrame:
        """
        Apply return shocks to a price series.

        Methodology:
        1. For each shocked ticker, compute a shocked daily return path
           over the duration window that achieves the target total return
           plus elevated noise.
        2. After the shock window, prices revert to base trajectory with
           the modified vol level (persistent vol regime).
        3. Unshocked tickers get elevated vol only (cross-asset contagion).

        Args:
            prices:        Base price DataFrame.
            shocks:        {ticker: total_return_over_duration}.
            vol_mult:      Volatility multiplier during shock.
            duration_days: Days the shock unfolds over.
            seed:          Random seed override.

        Returns:
            Shocked price DataFrame with same shape as input.
        """
        rng      = np.random.default_rng(seed or self.seed)
        shocked  = prices.copy().astype(float)
        n        = len(prices)
        dur      = min(duration_days, n)

        # Baseline daily log-returns
        log_rets = np.log(prices).diff().fillna(0).values  # [T × A]

        for col_i, ticker in enumerate(self._universe):
            col_log_rets = log_rets[:, col_i]
            base_vol     = col_log_rets.std() + 1e-8

            if ticker in shocks:
                total_shock  = shocks[ticker]
                daily_drift  = total_shock / dur
                shocked_vol  = base_vol * vol_mult
                noise        = rng.normal(0, shocked_vol, dur)
                daily_rets   = daily_drift + noise
            else:
                # Cross-asset contagion: slightly elevated vol, no directional shock
                extra_vol = max(0, base_vol * (vol_mult - 1.0))
                if extra_vol > 1e-8:
                    noise = rng.normal(0, extra_vol, dur)
                else:
                    noise = np.zeros(dur)
                daily_rets = col_log_rets[1:dur+1] + noise

            # Rebuild shocked price path
            col         = shocked.iloc[:, col_i].values.copy()
            shock_path  = np.zeros(dur + 1)
            shock_path[0] = col[0]
            for t in range(dur):
                shock_path[t + 1] = shock_path[t] * np.exp(daily_rets[t])

            # After shock window: continue from shocked level with original path scaled
            if dur < n - 1:
                orig_path_after  = col[dur + 1:] / (col[dur] + 1e-8)
                shocked_after    = shock_path[-1] * orig_path_after
                col[1:dur + 1]   = shock_path[1:]
                col[dur + 1:]    = shocked_after
            else:
                col[1:dur + 1] = shock_path[1:dur + 1]

            shocked.iloc[:, col_i] = col

        return shocked

    def _batch_ic(self, formulas: list[str], prices: pd.DataFrame) -> dict[str, float]:
        """Fast IC computation using BatchEvaluator."""
        if len(prices) < 60:
            return {}
        try:
            from macro8_subnet.alpha.batch_evaluator import BatchEvaluator
            ev  = BatchEvaluator(prices, min_ic=0.0)
            res = ev.evaluate(formulas)
            return dict(zip(res.formulas, res.mean_ics.tolist()))
        except Exception:
            return {}

    def _portfolio_metrics(
        self,
        formulas: list[str],
        prices:   pd.DataFrame,
    ) -> dict[str, dict]:
        """Run PortfolioEvaluator and return per-formula metric dicts."""
        if len(prices) < 100:
            return {f: {} for f in formulas}
        try:
            from macro8_subnet.alpha.portfolio_evaluator import PortfolioEvaluator
            ev  = PortfolioEvaluator(prices, apply_costs=True,
                                     default_capital=self.capital,
                                     horizons=[1, 7])
            res = ev.evaluate(formulas)
            return {
                ps.formula: {
                    "net_sharpe":   ps.net_sharpe,
                    "max_drawdown": ps.max_drawdown,
                    "daily_turnover": ps.daily_turnover,
                }
                for ps in res.portfolio_scores
            }
        except Exception:
            return {f: {} for f in formulas}
