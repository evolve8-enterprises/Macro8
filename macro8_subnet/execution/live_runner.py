"""
execution/live_runner.py
-------------------------
Sprint 31: Live Data Pipeline + Paper Trading Automation.

Turns the system from "works in a controlled environment" to
"survives reality" — daily automatic data refresh, paper trading
loop, failure logging, and the CLI entry point.

Four components
---------------

1. DataPipeline
   Fetches fresh market data every day via yfinance with:
   - Local disk cache (avoids re-fetching same data)
   - Fast connectivity probe before download attempt
   - Calibrated synthetic fallback if network unavailable
   - Data validation (staleness check, NaN detection, price sanity)

2. PaperTrader
   Runs the daily paper-trading loop:
   - Load cached data or fetch fresh
   - Run forecast pipeline (GP + ForecastedEnsemble)
   - Apply constraints + generate trades
   - Mark-to-market positions against next-day actual returns
   - Log everything to a JSON file

3. FailureLog
   Structured log of prediction failures for post-mortems:
   - High-confidence wrong regime predictions
   - Unexpected drawdowns (DD > threshold when confidence was HIGH)
   - Model retrain signals triggered
   - Data quality issues

4. Macro8Runner (CLI entry point)
   `python -m macro8_subnet.execution.live_runner`

   Modes:
     --mode backtest   Re-run on historical data (default)
     --mode paper      Forward paper-trading (uses today's data)
     --mode once       Single forecast + print (no loop)
     --mode retrain    Force GP retrain and save model

Usage
-----
    # One-shot forecast
    python -m macro8_subnet.execution.live_runner --mode once

    # Backtest 90 days
    python -m macro8_subnet.execution.live_runner --mode backtest --days 90

    # Paper trading (run daily via cron)
    python -m macro8_subnet.execution.live_runner --mode paper

    # Cron entry (run at 6pm daily after market close):
    # 0 18 * * 1-5 /usr/bin/python -m macro8_subnet.execution.live_runner --mode paper
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_SUBNET = Path(__file__).resolve().parent.parent
_ROOT   = _SUBNET.parent
for p in [str(_SUBNET), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

DEFAULT_TICKERS = ["SPY","QQQ","IWM","TLT","GLD","DBC","EEM","FXI","VNQ","HYG"]
DEFAULT_CACHE   = Path.home() / ".macro8" / "data"
DEFAULT_LOG     = Path.home() / ".macro8" / "paper_trading.json"
DEFAULT_FAIL    = Path.home() / ".macro8" / "failures.json"
DEFAULT_STATE   = Path.home() / ".macro8" / "state.json"


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DataStatus:
    source:       str       # "yfinance" | "cache" | "synthetic"
    n_assets:     int
    n_days:       int
    start_date:   str
    end_date:     str
    stale_days:   int       # how many days since last new row
    warnings:     list[str] = field(default_factory=list)
    valid:        bool      = True


class DataPipeline:
    """
    Fetches, caches, and validates daily market data.

    Priority:
        1. yfinance (live) — if connectivity probe succeeds
        2. Disk cache     — if yfinance fails or no connection
        3. Calibrated synthetic — final fallback

    Cache format: parquet files in ~/.macro8/data/{tickers_hash}.parquet

    Validation checks:
        - Staleness: last date within N trading days of today
        - NaN fraction: < 5% per column
        - Price sanity: no zero or negative prices
        - Asset count: all expected tickers present

    Parameters
    ----------
    tickers:     list[str] — tickers to fetch.
    start:       str       — historical start date "YYYY-MM-DD".
    cache_dir:   Path      — local cache directory.
    max_stale:   int       — max trading days since last update before refetch.
    verbose:     bool      — print progress.
    """

    def __init__(
        self,
        tickers:   list[str]        = None,
        start:     str              = "2010-01-01",
        cache_dir: Optional[Path]   = None,
        max_stale: int              = 3,
        verbose:   bool             = True,
    ):
        self.tickers   = tickers or DEFAULT_TICKERS
        self.start     = start
        self.cache_dir = cache_dir or DEFAULT_CACHE
        self.max_stale = max_stale
        self.verbose   = verbose
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self) -> tuple[pd.DataFrame, DataStatus]:
        """
        Fetch market data with fallback chain.

        Returns:
            (prices DataFrame, DataStatus) describing the source and quality.
        """
        cache_path = self._cache_path()

        # Try yfinance first (if online)
        if self._is_online():
            try:
                prices = self._fetch_yfinance()
                if prices is not None and len(prices) >= 100:
                    self._save_cache(prices, cache_path)
                    status = self._validate(prices, "yfinance")
                    if self.verbose:
                        print(f"[Data] yfinance: {status.n_assets} assets × "
                              f"{status.n_days} days → {status.end_date}")
                    return prices, status
            except Exception as e:
                if self.verbose:
                    print(f"[Data] yfinance failed: {str(e)[:60]}")

        # Try disk cache
        if cache_path.exists():
            try:
                prices = pd.read_parquet(cache_path)
                status = self._validate(prices, "cache")
                if status.stale_days <= self.max_stale * 3:  # tolerate longer stale for cache
                    if self.verbose:
                        print(f"[Data] Cache: {status.n_assets} assets × "
                              f"{status.n_days} days (stale {status.stale_days}d)")
                    return prices, status
            except Exception:
                pass

        # Synthetic fallback
        prices = self._synthetic_fallback()
        status = self._validate(prices, "synthetic")
        if self.verbose:
            print(f"[Data] Synthetic fallback: {status.n_assets} assets × {status.n_days} days")
        return prices, status

    def validate_fresh(self, prices: pd.DataFrame) -> list[str]:
        """
        Check a price DataFrame for data quality issues.

        Returns list of issue strings (empty = clean).
        """
        issues = []

        # NaN check
        nan_frac = prices.isna().mean()
        bad_cols  = nan_frac[nan_frac > 0.05].index.tolist()
        if bad_cols:
            issues.append(f"High NaN fraction in: {bad_cols}")

        # Zero / negative price check
        neg_cols = [c for c in prices.columns if (prices[c] <= 0).any()]
        if neg_cols:
            issues.append(f"Zero/negative prices in: {neg_cols}")

        # Staleness
        last_date = prices.index[-1].date()
        days_old  = (date.today() - last_date).days
        # Adjust for weekends: 3 calendar days max for last business day
        if days_old > 5:
            issues.append(f"Data is {days_old} calendar days old (last: {last_date})")

        # Asset count
        missing = [t for t in self.tickers if t not in prices.columns]
        if missing:
            issues.append(f"Missing tickers: {missing}")

        return issues

    # ── Private ───────────────────────────────────────────────────────────────

    def _is_online(self) -> bool:
        try:
            import socket
            socket.setdefaulttimeout(3)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(
                ("query1.finance.yahoo.com", 443)
            )
            return True
        except Exception:
            return False
        finally:
            import socket
            socket.setdefaulttimeout(None)

    def _fetch_yfinance(self) -> Optional[pd.DataFrame]:
        import yfinance as yf
        end   = pd.Timestamp.today().strftime("%Y-%m-%d")
        data  = yf.download(
            self.tickers, start=self.start, end=end,
            auto_adjust=True, progress=False, timeout=30,
        )
        if data.empty:
            return None
        prices = data["Close"] if "Close" in data else data
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=self.tickers[0])
        prices = prices.dropna(how="all").ffill().bfill()
        prices.index = pd.to_datetime(prices.index)
        return prices[prices.index >= self.start]

    def _save_cache(self, prices: pd.DataFrame, path: Path) -> None:
        try:
            prices.to_parquet(path)
        except Exception:
            pass

    def _cache_path(self) -> Path:
        key = "_".join(sorted(self.tickers)) + f"_{self.start}"
        import hashlib
        h = hashlib.md5(key.encode()).hexdigest()[:8]
        return self.cache_dir / f"prices_{h}.parquet"

    def _synthetic_fallback(self) -> pd.DataFrame:
        from macro8_subnet.data.market_data_fetcher import MarketDataFetcher
        fetcher = MarketDataFetcher(force_synthetic=True, verbose=False)
        result  = fetcher.fetch_prices(
            tickers=self.tickers, start=self.start, n_synthetic=3780
        )
        return result.prices

    def _validate(self, prices: pd.DataFrame, source: str) -> DataStatus:
        last_date   = prices.index[-1].date()
        days_old    = (date.today() - last_date).days
        issues      = self.validate_fresh(prices)
        return DataStatus(
            source=source,
            n_assets=len(prices.columns),
            n_days=len(prices),
            start_date=str(prices.index[0].date()),
            end_date=str(last_date),
            stale_days=days_old,
            warnings=issues,
            valid=len([i for i in issues if "Missing" not in i or len(prices.columns) >= 5]) == 0,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 2. FAILURE LOG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FailureRecord:
    """One failure event."""
    timestamp:    str
    failure_type: str       # "regime_wrong" | "unexpected_dd" | "retrain" | "data"
    date:         str
    description:  str
    predicted:    str       = ""
    actual:       str       = ""
    confidence:   float     = 0.0
    pnl:          float     = 0.0
    extra:        dict       = field(default_factory=dict)


class FailureLog:
    """
    Structured log of system failures for post-mortems.

    Captures:
        regime_wrong    — confident prediction that was wrong
        unexpected_dd   — drawdown when model was confident
        retrain         — retrain signal triggered (rolling Sharpe < floor)
        data            — data quality issues detected

    Persists to JSON at `path`.
    """

    def __init__(self, path: Path = DEFAULT_FAIL):
        self.path   = Path(path)
        self._log:  list[FailureRecord] = []
        self._load()

    def log_regime_failure(
        self,
        date_str:   str,
        predicted:  str,
        actual:     str,
        confidence: float,
        pnl:        float,
    ) -> None:
        self._append(FailureRecord(
            timestamp=datetime.now().isoformat(),
            failure_type="regime_wrong",
            date=date_str,
            description=f"Predicted {predicted!r} with conf={confidence:.2f}, actual was {actual!r}",
            predicted=predicted,
            actual=actual,
            confidence=confidence,
            pnl=pnl,
        ))

    def log_drawdown(self, date_str: str, drawdown: float, confidence: float) -> None:
        self._append(FailureRecord(
            timestamp=datetime.now().isoformat(),
            failure_type="unexpected_dd",
            date=date_str,
            description=f"DD={drawdown:.4f} occurred when confidence was HIGH ({confidence:.2f})",
            confidence=confidence,
            extra={"drawdown": drawdown},
        ))

    def log_retrain(self, date_str: str, rolling_sharpe: float) -> None:
        self._append(FailureRecord(
            timestamp=datetime.now().isoformat(),
            failure_type="retrain",
            date=date_str,
            description=f"Retrain signal triggered: rolling Sharpe={rolling_sharpe:.3f}",
            extra={"rolling_sharpe": rolling_sharpe},
        ))

    def log_data_issue(self, issues: list[str]) -> None:
        self._append(FailureRecord(
            timestamp=datetime.now().isoformat(),
            failure_type="data",
            date=str(date.today()),
            description="; ".join(issues),
        ))

    def recent(self, n: int = 10) -> list[FailureRecord]:
        return self._log[-n:]

    def summary(self) -> str:
        types = {}
        for r in self._log:
            types[r.failure_type] = types.get(r.failure_type, 0) + 1
        if not types:
            return "No failures logged."
        return "  ".join(f"{k}:{v}" for k, v in types.items())

    def print_recent(self, n: int = 5) -> None:
        recent = self.recent(n)
        if not recent:
            print("  [FailureLog] No failures recorded.")
            return
        print(f"  [FailureLog] {len(self._log)} total events (showing last {n}):")
        for r in recent:
            print(f"    {r.date}  {r.failure_type:<15}  {r.description[:65]}")

    def _append(self, record: FailureRecord) -> None:
        self._log.append(record)
        self._save()

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w") as f:
                json.dump([asdict(r) for r in self._log], f, indent=2)
        except Exception:
            pass

    def _load(self) -> None:
        if self.path.exists():
            try:
                with open(self.path) as f:
                    raw = json.load(f)
                self._log = [FailureRecord(**r) for r in raw]
            except Exception:
                self._log = []


# ══════════════════════════════════════════════════════════════════════════════
# 3. PAPER TRADER
# ══════════════════════════════════════════════════════════════════════════════

class PaperTrader:
    """
    Runs the daily paper-trading loop.

    Each call to `run_day()`:
        1. Fetch fresh data (DataPipeline)
        2. Forecast (ForecastedEnsemble)
        3. Apply constraints + generate trades
        4. Mark positions to market (next-day actual return)
        5. Record outcome (LiveTracker + FailureLog)
        6. Save state to disk

    Parameters
    ----------
    tickers:    list[str] — asset universe.
    start:      str       — historical data start.
    capital:    float     — paper portfolio size.
    gp_gens:    int       — GP generations per retrain.
    state_file: Path      — JSON file to persist holdings + history.
    verbose:    bool      — print progress.
    """

    def __init__(
        self,
        tickers:    list[str]      = None,
        start:      str            = "2010-01-01",
        capital:    float          = 100_000,
        gp_gens:    int            = 15,
        state_file: Optional[Path] = None,
        verbose:    bool           = True,
    ):
        self.tickers    = tickers or DEFAULT_TICKERS
        self.start      = start
        self.capital    = capital
        # Multi-horizon capital engine (initialised after first fit in initialise())
        self._cap_engine: Optional[MultiSignalCapitalEngine] = None
        self.gp_gens    = gp_gens
        self.state_file = state_file or DEFAULT_STATE
        self.verbose    = verbose

        self._pipeline  = DataPipeline(tickers=self.tickers, start=start, verbose=verbose)
        self._fail_log  = FailureLog()
        self._state     = self._load_state()
        self._fens      = None   # ForecastedEnsemble — lazy-trained

    # ── Public API ────────────────────────────────────────────────────────────

    def initialise(self, prices: pd.DataFrame) -> None:
        """Train GP + ForecastedEnsemble on provided price history. Call once."""
        from macro8_subnet.alpha.gp_miner import GPMiner
        from macro8_subnet.alpha.regime_prediction import ForecastedEnsemble

        if self.verbose:
            print(f"[PaperTrader] Training on {len(prices)} days...")

        t0 = time.perf_counter()
        gp = GPMiner(prices, pop_size=200, elite_n=20, seed=42, verbose=False)
        gp.run(n_epochs=self.gp_gens)
        formulas = gp.top_formulas(20)

        self._fens = ForecastedEnsemble(
            prices, formulas[:12], horizon=5, verbose=False
        )
        self._fens.fit()
        elapsed = time.perf_counter() - t0

        # Initialise multi-horizon capital engine for active signals
        rep_formulas = []
        try:
            ens = self._fens._ensemble if hasattr(self._fens, '_ensemble') else self._fens
            rep_formulas = list(ens._cluster_result.rep_formulas or [])
            if hasattr(ens, '_stress_rep_formula') and ens._stress_rep_formula:
                rep_formulas.append(ens._stress_rep_formula)
        except Exception:
            pass
        if rep_formulas:
            self._cap_engine = MultiSignalCapitalEngine(
                signal_names=list(dict.fromkeys(rep_formulas)),
                initial_capital=self.capital,
            )

        if self.verbose:
            print(f"[PaperTrader] Training complete ({elapsed:.1f}s) | "
                  f"{len(formulas)} formulas")

    def run_day(self, prices: pd.DataFrame, today: pd.Timestamp) -> dict:
        """
        Execute one day of paper trading.

        Args:
            prices: Full price history including today.
            today:  The date being processed.

        Returns:
            dict with keys: date, pnl, positions, forecast_summary, failures.
        """
        from macro8_subnet.execution.engine import (
            PortfolioConstraints, ConstraintSolver,
            TradeExecutor, LiveTracker, PredictionMarket,
        )

        if self._fens is None:
            raise RuntimeError("Call initialise() before run_day()")

        # 1. Forecast — use a recent window (last 500 days) for feature computation
        #    Passing the full growing history to forecast() causes FeatureStore to
        #    rebuild all 34 features on an ever-growing dataset (O(n) per day).
        #    The last 500 days is sufficient for all feature windows (max=252).
        recent_window = 500
        recent = prices.loc[:today].iloc[-recent_window:]
        result = self._fens.forecast(prices=recent, date=recent.index[-1])

        # 2. Constrain
        tracker   = self._get_tracker()
        dd_scale  = tracker.drawdown_guard.position_scale()
        conf_mult = tracker.confidence_multiplier()
        cs        = ConstraintSolver(PortfolioConstraints())
        positions = cs.apply(
            result.positions,
            p_stress=result.regime_forecast.stress,
            scale=dd_scale * conf_mult,
        )

        # 3. Trades
        executor  = TradeExecutor(capital=self.capital)
        plan      = executor.compute_trades(
            positions, self._state.get("holdings", {}), date=today
        )

        # 4. PnL (using previous holdings on today's return)
        holdings   = self._state.get("holdings", {})
        today_idx  = prices.index.get_loc(today) if today in prices.index else -1
        pnl        = 0.0
        if today_idx > 0:
            prev_idx = today_idx - 1
            log_ret  = np.log(prices.iloc[today_idx] / prices.iloc[prev_idx])
            pnl      = sum(
                holdings.get(t, 0) * float(log_ret.get(t, 0))
                for t in holdings
            )

        # 5. Track
        tracker.update(today, positions, plan, pnl, result)

        # 6. Failure detection
        pw = tracker.snapshot()
        failures_today = []

        # Check confident regime prediction resolving as wrong
        records = tracker._records
        if len(records) > 6:
            lag_record = records[-6]
            if (lag_record.confidence > 0.55 and
                    lag_record.regime_actual not in ("unknown", lag_record.regime_predicted)):
                self._fail_log.log_regime_failure(
                    str(lag_record.date.date()),
                    lag_record.regime_predicted,
                    lag_record.regime_actual,
                    lag_record.confidence,
                    lag_record.pnl,
                )
                failures_today.append("regime_wrong")

        # Check unexpected drawdown under confidence
        if (tracker.drawdown_guard.current_drawdown < -0.05 and
                result.confidence > 0.65):
            self._fail_log.log_drawdown(
                str(today.date()),
                tracker.drawdown_guard.current_drawdown,
                result.confidence,
            )
            failures_today.append("unexpected_dd")

        # Check retrain signal
        if tracker.retrain_signal():
            self._fail_log.log_retrain(str(today.date()), pw.sharpe_ann)
            failures_today.append("retrain_triggered")
            if self.verbose:
                print(f"  [PaperTrader] RETRAIN SIGNAL at {today.date()} "
                      f"(Sharpe={pw.sharpe_ann:.2f})")

        # 7. Persist state
        self._state["holdings"]  = dict(positions)
        self._state["last_date"] = str(today.date())
        self._state["cum_pnl"]   = float(tracker.drawdown_guard.cumulative_pnl)
        self._save_state()

        # ── Capital engine feedback ───────────────────────────────────────────
        # Feed today's realised PnL back into each signal bucket, then
        # (on reallocation days) run softmax to shift capital toward the
        # frequencies/signals currently compounding best.
        from macro8_subnet.execution.capital_engine import HORIZONS  # noqa
        if self._cap_engine is not None:
            # Apportion today's total PnL across signals equally (simplified;
            # a full implementation would compute per-signal PnL separately)
            rep_pnl_share = pnl / max(len(self._cap_engine.engines), 1)
            pnl_by_signal: dict[str, dict[str, float]] = {}
            for sig_name in self._cap_engine.engines:
                # Each horizon bucket gets the same daily PnL for now;
                # full per-horizon tracking requires per-signal position vectors
                pnl_by_signal[sig_name] = {h: rep_pnl_share for h in HORIZONS}
            self._cap_engine.record_pnl(str(today.date()), pnl_by_signal)

        cap_allocs = (
            self._cap_engine.engines[
                list(self._cap_engine.engines.keys())[0]
            ].get_weights()
            if self._cap_engine and self._cap_engine.engines
            else {h: 0.25 for h in HORIZONS}
        )

        day_result = {
            "date":      str(today.date()),
            "pnl":       round(pnl, 5),
            "cum_pnl":   round(self._state["cum_pnl"], 5),
            "sharpe":    round(pw.sharpe_ann, 3),
            "drawdown":  round(tracker.drawdown_guard.current_drawdown, 5),
            "cap_1d":    round(cap_allocs.get("1d",  0.25), 4),
            "cap_7d":    round(cap_allocs.get("7d",  0.25), 4),
            "cap_30d":   round(cap_allocs.get("30d", 0.25), 4),
            "cap_90d":   round(cap_allocs.get("90d", 0.25), 4),
            "dd_scale": round(dd_scale, 3),
            "confidence":      round(result.confidence, 3),
            "regime":          result.regime_current,
            "regime_forecast": result.regime_forecast.most_likely,
            "p_stress":        round(result.regime_forecast.stress, 3),
            "n_trades":        len(plan.orders),
            "turnover":        round(plan.total_turnover, 4),
            "failures":        failures_today,
        }
        self._append_log(day_result)

        if self.verbose:
            print(
                f"  [{today.date()}] pnl={pnl:+.4f} cum={self._state['cum_pnl']:+.4f} "
                f"sharpe={pw.sharpe_ann:+.2f} regime={result.regime_current} "
                f"conf={result.confidence:.2f} trades={len(plan.orders)}"
                + (f" FAILURES:{failures_today}" if failures_today else "")
            )

        return day_result

    def run_backtest(
        self,
        prices:   pd.DataFrame,
        n_days:   int = 90,
        train_frac: float = 0.70,
    ) -> pd.DataFrame:
        """
        Run backtest: train on first `train_frac` of prices, paper-trade remainder.

        Args:
            prices:     Full price history.
            n_days:     Number of OOS days to paper-trade.
            train_frac: Fraction used for training.

        Returns:
            DataFrame with one row per day.
        """
        split      = int(len(prices) * train_frac)
        train      = prices.iloc[:split]
        oos        = prices.iloc[split:]
        eval_days  = min(n_days, len(oos) - 1)

        # Reset state so stale disk holdings don't contaminate backtest
        self._state   = {"holdings": {}, "last_date": None, "cum_pnl": 0.0}
        self._tracker = None   # fresh LiveTracker per backtest

        self.initialise(train)

        if self.verbose:
            print(f"[PaperTrader] Backtest: {eval_days} OOS days from {oos.index[0].date()}")

        results = []
        for i in range(eval_days):
            today   = oos.index[i]
            # Use all data up to and including today
            prices_so_far = pd.concat([train, oos.iloc[:i+1]])
            try:
                r = self.run_day(prices_so_far, today)
                results.append(r)
            except Exception as e:
                if self.verbose:
                    print(f"  [{today.date()}] ERROR: {e}")
                results.append({"date": str(today.date()), "pnl": 0.0, "error": str(e)})

        df = pd.DataFrame(results)
        if "date" in df.columns:
            df = df.set_index("date")
        return df

    def run_paper(self, n_retrain_days: int = 60) -> dict:
        """
        Run one paper-trading day using today's live data.

        Retrains every `n_retrain_days` calendar days.
        Designed to be called daily from a cron job.

        Returns:
            Day result dict.
        """
        # Fetch data
        prices, status = self._pipeline.fetch()

        # Log data issues
        if status.warnings:
            self._fail_log.log_data_issue(status.warnings)

        # Determine if retrain needed
        last_train = self._state.get("last_train_date")
        days_since = (date.today() - date.fromisoformat(last_train)).days if last_train else 999
        if self._fens is None or days_since >= n_retrain_days:
            split = int(len(prices) * 0.80)
            self.initialise(prices.iloc[:split])
            self._state["last_train_date"] = str(date.today())
            self._save_state()

        today = prices.index[-1]
        return self.run_day(prices, today)

    # ── State persistence ─────────────────────────────────────────────────────

    def _get_tracker(self):
        """Return or create the LiveTracker (stored in state)."""
        from macro8_subnet.execution.engine import LiveTracker
        if "tracker" not in self.__dict__:
            self._tracker = LiveTracker()
        return self._tracker

    def _load_state(self) -> dict:
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"holdings": {}, "last_date": None, "cum_pnl": 0.0}

    def _save_state(self) -> None:
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump(self._state, f, indent=2, default=str)
        except Exception:
            pass

    def _append_log(self, record: dict) -> None:
        log = []
        if DEFAULT_LOG.exists():
            try:
                with open(DEFAULT_LOG) as f:
                    log = json.load(f)
            except Exception:
                pass
        log.append(record)
        try:
            DEFAULT_LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(DEFAULT_LOG, "w") as f:
                json.dump(log[-365:], f, indent=2, default=str)  # keep last 1yr
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# 4. CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _print_banner() -> None:
    print("=" * 60)
    print("  MACRO8 — Autonomous Macro Strategy Engine")
    print("  Sprint 31 · Live Runner")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Macro8 Live Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  once      Single forecast + print (no loop, no writes)
  backtest  Simulate on historical data (default: 90 OOS days)
  paper     Forward paper-trading using today's market data
  retrain   Force retrain and save model state

Examples:
  python -m macro8_subnet.execution.live_runner --mode once
  python -m macro8_subnet.execution.live_runner --mode backtest --days 90
  python -m macro8_subnet.execution.live_runner --mode paper

Cron (run after market close, weekdays):
  0 18 * * 1-5 python -m macro8_subnet.execution.live_runner --mode paper
        """,
    )
    parser.add_argument("--mode",    default="backtest",
                        choices=["once","backtest","paper","retrain"])
    parser.add_argument("--days",    type=int,   default=90,
                        help="OOS days for backtest mode")
    parser.add_argument("--capital", type=float, default=100_000,
                        help="Paper portfolio size")
    parser.add_argument("--tickers", nargs="+",  default=DEFAULT_TICKERS)
    parser.add_argument("--start",   default="2010-01-01")
    parser.add_argument("--quiet",   action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet
    _print_banner()

    # ── once: single forecast ─────────────────────────────────────────────────
    if args.mode == "once":
        print("\n[Mode] Single forecast")
        pipeline = DataPipeline(tickers=args.tickers, start=args.start, verbose=verbose)
        prices, status = pipeline.fetch()
        print(f"Data: {status.source} | {status.n_assets}×{status.n_days} days | "
              f"last={status.end_date}")
        if status.warnings:
            for w in status.warnings:
                print(f"  WARNING: {w}")

        from macro8_subnet.alpha.gp_miner import GPMiner
        from macro8_subnet.alpha.regime_prediction import ForecastedEnsemble
        from macro8_subnet.execution.engine import PortfolioConstraints, ConstraintSolver

        split = int(len(prices) * 0.80)
        gp    = GPMiner(prices.iloc[:split], pop_size=150, elite_n=20,
                        seed=42, verbose=False)
        gp.run(n_epochs=10)
        formulas = gp.top_formulas(12)

        fens = ForecastedEnsemble(prices.iloc[:split], formulas[:8],
                                  horizon=5, verbose=False)
        fens.fit()
        result = fens.forecast()

        cs        = ConstraintSolver(PortfolioConstraints())
        positions = cs.apply(result.positions,
                             p_stress=result.regime_forecast.stress)
        result.positions = positions  # replace with constrained

        result.print()

        print("\nTop 3 scenarios:")
        for name, prob in result.top_scenarios(3):
            print(f"  {name:<28} {prob:.1%}")
        return

    # ── backtest ─────────────────────────────────────────────────────────────
    if args.mode == "backtest":
        print(f"\n[Mode] Backtest ({args.days} OOS days)")
        pipeline = DataPipeline(tickers=args.tickers, start=args.start, verbose=verbose)
        prices, status = pipeline.fetch()
        print(f"Data: {status.source} | {status.n_assets}×{status.n_days} | {status.end_date}")

        trader = PaperTrader(tickers=args.tickers, start=args.start,
                              capital=args.capital, verbose=verbose)
        hist   = trader.run_backtest(prices, n_days=args.days)

        print("\n" + "=" * 60)
        print("  BACKTEST RESULTS")
        print("=" * 60)
        if len(hist) > 0:
            pnl  = hist["pnl"].dropna() if "pnl" in hist.columns else pd.Series([0])
            sh   = float(pnl.mean() / (pnl.std() + 1e-8) * np.sqrt(252))
            cum  = float(pnl.sum())
            cum_s= pnl.cumsum()
            mdd  = float((cum_s - cum_s.cummax()).min())
            turn = float(hist["turnover"].mean()) if "turnover" in hist.columns else 0
            n_f  = int(hist["failures"].apply(len).sum()) if "failures" in hist.columns else 0
            print(f"  Days:        {len(hist)}")
            print(f"  Cum PnL:     {cum:+.4f}  ({cum*100:.2f}%)")
            print(f"  Ann Sharpe:  {sh:+.3f}")
            print(f"  Max DD:      {mdd:.4f}  ({mdd*100:.2f}%)")
            print(f"  Avg turn/d:  {turn:.4f}")
            print(f"  Failures:    {n_f}")

        trader._fail_log.print_recent(5)
        return

    # ── paper ─────────────────────────────────────────────────────────────────
    if args.mode == "paper":
        print("\n[Mode] Paper trading — today's data")
        trader = PaperTrader(tickers=args.tickers, start=args.start,
                              capital=args.capital, verbose=verbose)
        result = trader.run_paper()
        print("\nToday's result:")
        for k, v in result.items():
            if k != "failures":
                print(f"  {k:<20}: {v}")
        if result.get("failures"):
            print(f"  failures            : {result['failures']}")
        trader._fail_log.print_recent(3)
        return

    # ── retrain ───────────────────────────────────────────────────────────────
    if args.mode == "retrain":
        print("\n[Mode] Force retrain")
        pipeline = DataPipeline(tickers=args.tickers, start=args.start, verbose=verbose)
        prices, status = pipeline.fetch()
        trader = PaperTrader(tickers=args.tickers, capital=args.capital, verbose=verbose)
        split  = int(len(prices) * 0.80)
        trader.initialise(prices.iloc[:split])
        trader._state["last_train_date"] = str(date.today())
        trader._save_state()
        print("Retrain complete.")
        return


if __name__ == "__main__":
    main()
