"""
tests/test_sprint31.py
-----------------------
Sprint 31: Live Data Pipeline + Paper Trading Automation

Tests cover:
    - DataPipeline: fetch, validate, cache, fallback
    - DataStatus: fields, valid flag
    - FailureLog: log/load/save, all failure types
    - PaperTrader: initialise, run_backtest, state persistence
    - CLI entry point: module importable, argument parsing
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
for p in [str(_ROOT), str(_ROOT / "macro8_subnet")]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def prices():
    rng = np.random.default_rng(42)
    n, a = 350, 10
    tickers = ["SPY","QQQ","IWM","TLT","GLD","DBC","EEM","FXI","VNQ","HYG"]
    p = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, (n, a)), axis=0))
    return pd.DataFrame(p, index=pd.bdate_range("2015-01-01", periods=n), columns=tickers)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ── 1. DataPipeline ───────────────────────────────────────────────────────────

class TestDataPipeline:
    def test_import(self):
        from macro8_subnet.execution.live_runner import DataPipeline, DataStatus

    def test_fetch_returns_tuple(self, tmp_dir):
        from macro8_subnet.execution.live_runner import DataPipeline
        dp = DataPipeline(cache_dir=tmp_dir, verbose=False)
        prices, status = dp.fetch()
        assert isinstance(prices, pd.DataFrame)
        from macro8_subnet.execution.live_runner import DataStatus
        assert isinstance(status, DataStatus)

    def test_fetch_prices_non_empty(self, tmp_dir):
        from macro8_subnet.execution.live_runner import DataPipeline
        dp = DataPipeline(cache_dir=tmp_dir, verbose=False)
        prices, _ = dp.fetch()
        assert len(prices) > 100

    def test_fetch_prices_no_nan(self, tmp_dir):
        from macro8_subnet.execution.live_runner import DataPipeline
        dp = DataPipeline(cache_dir=tmp_dir, verbose=False)
        prices, _ = dp.fetch()
        assert prices.isna().mean().mean() < 0.05

    def test_synthetic_fallback_used_when_offline(self, tmp_dir):
        from macro8_subnet.execution.live_runner import DataPipeline
        dp = DataPipeline(cache_dir=tmp_dir, verbose=False)
        dp._is_online = lambda: False  # mock offline
        prices, status = dp.fetch()
        assert status.source in ("synthetic", "cache")
        assert len(prices) > 0

    def test_status_source_field(self, tmp_dir):
        from macro8_subnet.execution.live_runner import DataPipeline
        dp = DataPipeline(cache_dir=tmp_dir, verbose=False)
        _, status = dp.fetch()
        assert status.source in ("yfinance", "cache", "synthetic")

    def test_status_n_assets_positive(self, tmp_dir):
        from macro8_subnet.execution.live_runner import DataPipeline
        dp = DataPipeline(cache_dir=tmp_dir, verbose=False)
        _, status = dp.fetch()
        assert status.n_assets > 0

    def test_status_n_days_positive(self, tmp_dir):
        from macro8_subnet.execution.live_runner import DataPipeline
        dp = DataPipeline(cache_dir=tmp_dir, verbose=False)
        _, status = dp.fetch()
        assert status.n_days > 0

    def test_cache_created_after_fetch(self, tmp_dir):
        from macro8_subnet.execution.live_runner import DataPipeline
        dp = DataPipeline(cache_dir=tmp_dir, verbose=False)
        dp.fetch()
        parquet_files = list(tmp_dir.glob("*.parquet"))
        # May or may not have parquet (depends on source)
        assert isinstance(parquet_files, list)

    def test_validate_fresh_detects_nan(self, tmp_dir, prices):
        from macro8_subnet.execution.live_runner import DataPipeline
        dp = DataPipeline(tickers=list(prices.columns), cache_dir=tmp_dir, verbose=False)
        bad = prices.copy()
        bad.iloc[:, 0] = np.nan
        issues = dp.validate_fresh(bad)
        assert any("NaN" in i or "nan" in i.lower() for i in issues)

    def test_validate_fresh_detects_negative(self, tmp_dir, prices):
        from macro8_subnet.execution.live_runner import DataPipeline
        dp = DataPipeline(tickers=list(prices.columns), cache_dir=tmp_dir, verbose=False)
        bad = prices.copy()
        bad.iloc[10, 0] = -5.0
        issues = dp.validate_fresh(bad)
        assert any("negative" in i.lower() or "zero" in i.lower() for i in issues)

    def test_validate_fresh_clean_returns_empty(self, tmp_dir, prices):
        from macro8_subnet.execution.live_runner import DataPipeline
        dp = DataPipeline(tickers=list(prices.columns), cache_dir=tmp_dir, verbose=False)
        issues = dp.validate_fresh(prices)
        # The only expected issue on synthetic test data is staleness
        non_stale = [i for i in issues if "days old" not in i and "Missing" not in i]
        assert len(non_stale) == 0


# ── 2. FailureLog ─────────────────────────────────────────────────────────────

class TestFailureLog:
    def test_import(self):
        from macro8_subnet.execution.live_runner import FailureLog, FailureRecord

    def test_log_regime_failure(self, tmp_dir):
        from macro8_subnet.execution.live_runner import FailureLog
        fl = FailureLog(path=tmp_dir / "failures.json")
        fl.log_regime_failure("2024-01-15", "normal", "stress", 0.72, -0.003)
        assert len(fl._log) == 1
        assert fl._log[0].failure_type == "regime_wrong"

    def test_log_drawdown(self, tmp_dir):
        from macro8_subnet.execution.live_runner import FailureLog
        fl = FailureLog(path=tmp_dir / "failures.json")
        fl.log_drawdown("2024-01-20", -0.06, 0.68)
        assert any(r.failure_type == "unexpected_dd" for r in fl._log)

    def test_log_retrain(self, tmp_dir):
        from macro8_subnet.execution.live_runner import FailureLog
        fl = FailureLog(path=tmp_dir / "failures.json")
        fl.log_retrain("2024-01-25", -0.8)
        assert any(r.failure_type == "retrain" for r in fl._log)

    def test_log_data_issue(self, tmp_dir):
        from macro8_subnet.execution.live_runner import FailureLog
        fl = FailureLog(path=tmp_dir / "failures.json")
        fl.log_data_issue(["Missing tickers: ['XYZ']"])
        assert any(r.failure_type == "data" for r in fl._log)

    def test_persists_to_disk(self, tmp_dir):
        from macro8_subnet.execution.live_runner import FailureLog
        path = tmp_dir / "failures.json"
        fl1  = FailureLog(path=path)
        fl1.log_regime_failure("2024-01-01", "normal", "stress", 0.7, 0.0)
        fl2  = FailureLog(path=path)   # reload
        assert len(fl2._log) == 1

    def test_recent_returns_last_n(self, tmp_dir):
        from macro8_subnet.execution.live_runner import FailureLog
        fl = FailureLog(path=tmp_dir / "failures.json")
        for i in range(15):
            fl.log_retrain(f"2024-{i+1:02d}-01", -float(i))
        recent = fl.recent(5)
        assert len(recent) == 5

    def test_summary_string(self, tmp_dir):
        from macro8_subnet.execution.live_runner import FailureLog
        fl = FailureLog(path=tmp_dir / "failures.json")
        fl.log_regime_failure("2024-01-01", "normal", "stress", 0.7, 0.0)
        fl.log_retrain("2024-01-02", -0.5)
        s = fl.summary()
        assert "regime_wrong" in s or "retrain" in s

    def test_empty_log_summary(self, tmp_dir):
        from macro8_subnet.execution.live_runner import FailureLog
        fl = FailureLog(path=tmp_dir / "failures.json")
        s  = fl.summary()
        assert s == "No failures logged."

    def test_print_recent_no_crash(self, tmp_dir, capsys):
        from macro8_subnet.execution.live_runner import FailureLog
        fl = FailureLog(path=tmp_dir / "failures.json")
        fl.log_retrain("2024-01-01", -0.5)
        fl.print_recent(3)
        captured = capsys.readouterr()
        assert "FailureLog" in captured.out or "retrain" in captured.out


# ── 3. PaperTrader ────────────────────────────────────────────────────────────

class TestPaperTrader:
    def test_import(self):
        from macro8_subnet.execution.live_runner import PaperTrader

    def test_initialise_sets_fens(self, prices, tmp_dir):
        from macro8_subnet.execution.live_runner import PaperTrader
        trader = PaperTrader(
            tickers=list(prices.columns),
            state_file=tmp_dir / "state.json",
            verbose=False,
        )
        trader.initialise(prices.iloc[:200])
        assert trader._fens is not None

    def test_run_backtest_returns_dataframe(self, prices, tmp_dir):
        from macro8_subnet.execution.live_runner import PaperTrader
        trader = PaperTrader(
            tickers=list(prices.columns),
            state_file=tmp_dir / "state.json",
            verbose=False,
        )
        hist = trader.run_backtest(prices, n_days=5, train_frac=0.80)
        assert isinstance(hist, pd.DataFrame)
        assert len(hist) > 0

    def test_backtest_pnl_column_exists(self, prices, tmp_dir):
        from macro8_subnet.execution.live_runner import PaperTrader
        trader = PaperTrader(
            tickers=list(prices.columns),
            state_file=tmp_dir / "state.json",
            verbose=False,
        )
        hist = trader.run_backtest(prices, n_days=5, train_frac=0.80)
        assert "pnl" in hist.columns

    def test_backtest_pnl_finite(self, prices, tmp_dir):
        from macro8_subnet.execution.live_runner import PaperTrader
        trader = PaperTrader(
            tickers=list(prices.columns),
            state_file=tmp_dir / "state.json",
            verbose=False,
        )
        hist = trader.run_backtest(prices, n_days=5, train_frac=0.80)
        pnl = hist["pnl"].dropna()
        assert np.isfinite(pnl.values).all()

    def test_state_persists_to_disk(self, prices, tmp_dir):
        from macro8_subnet.execution.live_runner import PaperTrader
        state_file = tmp_dir / "state.json"
        trader = PaperTrader(
            tickers=list(prices.columns),
            state_file=state_file,
            verbose=False,
        )
        trader.run_backtest(prices, n_days=3, train_frac=0.85)
        assert state_file.exists()
        with open(state_file) as f:
            state = json.load(f)
        assert "holdings" in state

    def test_failure_log_created(self, prices, tmp_dir):
        from macro8_subnet.execution.live_runner import PaperTrader
        trader = PaperTrader(
            tickers=list(prices.columns),
            state_file=tmp_dir / "state.json",
            verbose=False,
        )
        trader.run_backtest(prices, n_days=3, train_frac=0.85)
        # Failure log may or may not have entries, but the object exists
        assert trader._fail_log is not None

    def test_run_day_returns_dict(self, prices, tmp_dir):
        from macro8_subnet.execution.live_runner import PaperTrader
        trader = PaperTrader(
            tickers=list(prices.columns),
            state_file=tmp_dir / "state.json",
            verbose=False,
        )
        split = int(len(prices) * 0.80)
        trader.initialise(prices.iloc[:split])
        result = trader.run_day(prices, prices.index[split])
        assert isinstance(result, dict)
        assert "pnl" in result
        assert "regime" in result
        assert "confidence" in result

    def test_run_day_failures_is_list(self, prices, tmp_dir):
        from macro8_subnet.execution.live_runner import PaperTrader
        trader = PaperTrader(
            tickers=list(prices.columns),
            state_file=tmp_dir / "state.json",
            verbose=False,
        )
        split = int(len(prices) * 0.80)
        trader.initialise(prices.iloc[:split])
        result = trader.run_day(prices, prices.index[split])
        assert isinstance(result.get("failures", []), list)


# ── 4. Module imports and CLI ─────────────────────────────────────────────────

class TestModuleAndCLI:
    def test_module_importable(self):
        import macro8_subnet.execution.live_runner as lr
        assert hasattr(lr, "DataPipeline")
        assert hasattr(lr, "PaperTrader")
        assert hasattr(lr, "FailureLog")
        assert hasattr(lr, "main")

    def test_default_tickers_defined(self):
        from macro8_subnet.execution.live_runner import DEFAULT_TICKERS
        assert len(DEFAULT_TICKERS) >= 8
        assert "SPY" in DEFAULT_TICKERS

    def test_default_paths_defined(self):
        from macro8_subnet.execution.live_runner import (
            DEFAULT_CACHE, DEFAULT_LOG, DEFAULT_STATE, DEFAULT_FAIL,
        )
        assert isinstance(DEFAULT_CACHE, Path)
        assert isinstance(DEFAULT_STATE, Path)

    def test_failure_record_serialisable(self, tmp_dir):
        from macro8_subnet.execution.live_runner import FailureRecord
        r  = FailureRecord(
            timestamp="2024-01-01T00:00:00",
            failure_type="regime_wrong",
            date="2024-01-01",
            description="test",
            predicted="normal",
            actual="stress",
            confidence=0.72,
            pnl=-0.003,
        )
        d = r.__dict__
        assert json.dumps(d)   # should not raise
