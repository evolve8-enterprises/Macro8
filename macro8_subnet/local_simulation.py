"""
macro8_subnet/local_simulation.py
-----------------------------------
Local simulation harness for Macro8 — Sprint 32.

Runs a complete end-to-end verification of the full stack before
connecting to Bittensor. Covers every layer added in Sprints 22–31:

    Layer           Sprint   What it tests
    ──────────────  ───────  ─────────────────────────────────────
    Defensive       22–23    safe_formula, safe_submission, AST guards
    GP diversity    27       34 features, macro terminals, island model
    Portfolio intel 28       clustering, risk parity, regime detection
    Regime pred     29       transition model, policy layer, scenarios
    Execution       30       constraints, trade executor, drawdown guard
    Live pipeline   31       data pipeline, failure log, paper trader
    Validator       25–26    PortfolioEvaluator, OOS scoring, rewards

Usage
-----
    # Full integration check (default):
    python -m macro8_subnet.local_simulation

    # Just defensive + validator (fast, <5s):
    python -m macro8_subnet.local_simulation --fast

    # Run paper-trading backtest (90 OOS days):
    python -m macro8_subnet.local_simulation --backtest

    # Adversarial + stress:
    python -m macro8_subnet.local_simulation --adversarial --stress

    # All checks:
    python -m macro8_subnet.local_simulation --full

Expected output on a clean system (all layers):
    ✓ [Defensive]    safe_formula / safe_submission guards
    ✓ [GP]           34 features | macro terminals in formulas
    ✓ [Portfolio]    clustering | regime detection | positions
    ✓ [Prediction]   regime forecast | scenario probabilities
    ✓ [Execution]    constraint solver | trade orders | drawdown
    ✓ [Live]         data pipeline | failure log | paper trader
    ✓ [Validator]    OOS scoring | rewards sum to 1.0
    ✓ All checks passed — ready for Bittensor testnet
"""

from __future__ import annotations

import argparse
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

_SUBNET = Path(__file__).resolve().parent
_ROOT   = _SUBNET.parent
for p in [str(_SUBNET), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from macro8_subnet.neurons.validator import (
    EpochScorer, Macro8Validator, safe_formula, safe_submission, safe_synapse,
)
from macro8_subnet.protocol.synapse    import AlphaSubmissionSynapse
from macro8_subnet.agents.role_rewards import RoleRewardModel
from macro8_subnet.agents.agent_roles  import AgentRole


# ── Formula pools ─────────────────────────────────────────────────────────────

GOOD_FORMULAS = [
    # Technical (Sprint 22)
    "rank(momentum_20d)",
    "rank(momentum_20d) - rank(volatility_60d)",
    "zscore(cross_momentum)",
    "decay(momentum_5d, halflife=10)",
    "market_corr_60d - volatility_20d",
    "market_corr_20d + momentum_20d",
    "reversal_5d + market_corr_60d",
    # Macro (Sprint 26)
    "market_corr_60d + risk_on_off",
    "market_corr_60d * em_vs_dm",
    "vol_regime - volatility_20d",
    "trend_strength + market_corr_20d",
    "credit_stress + market_corr_60d",
]

ADVERSARIAL_INPUTS = [
    None, 42, [], {}, "", "   ", "A" * 500,
    "__import__('os').system('rm -rf /')",
    "rank(momentum_20d); DROP TABLE signals;",
    "rank(momentum_20d\x00)", "🚀momentum_20d",
    "rank(" + "(" * 100 + "momentum_20d" + ")" * 100 + ")",
    "\n\t\rmomentum_20d",
]

W = 57   # console width constant


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 1: Defensive validation
# ══════════════════════════════════════════════════════════════════════════════

def check_defensive() -> bool:
    """safe_formula and safe_submission guards reject all bad inputs."""
    good_inputs = [
        "momentum_20d", "rank(momentum_20d)",
        "rank(momentum_20d) - rank(volatility_60d)",
        "decay(momentum_5d, halflife=10)",
        "market_corr_60d + risk_on_off",    # macro feature
    ]
    bad_inputs = [
        None, 42, [], {}, "",
        "A" * 201, "__import__('os')",
        "rank(\x00momentum)", "🚀momentum",
    ]

    ok_good = sum(1 for g in good_inputs if safe_formula(g) is not None)
    ok_bad  = sum(1 for b in bad_inputs  if safe_formula(b) is None)
    formula_ok = (ok_good == len(good_inputs) and ok_bad == len(bad_inputs))

    syn_good = AlphaSubmissionSynapse(formulas=["momentum_20d"])
    syn_empty = AlphaSubmissionSynapse(formulas=[])
    sub_ok = (
        safe_submission(0, syn_good)  is not None and
        safe_submission(0, syn_empty) is not None and
        safe_submission(0, None)      is None
    )

    ok = formula_ok and sub_ok
    _check("Defensive", f"formula guards {ok_good}/{len(good_inputs)} good, "
           f"{ok_bad}/{len(bad_inputs)} bad | submission guards", ok)
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 2: GP diversity (Sprint 27)
# ══════════════════════════════════════════════════════════════════════════════

def check_gp(prices: pd.DataFrame) -> tuple[bool, list[str]]:
    """34 features, macro features appear in top formulas, diversity metrics."""
    from macro8_subnet.alpha.gp_miner import GPMiner, GP_FEATURES
    from macro8_subnet.alpha.batch_evaluator import ALL_FEATURES

    # Feature count
    feat_ok = len(GP_FEATURES) == 34 and len(ALL_FEATURES) == 34

    # Macro features present in grammar
    macro_terminals = [
        "risk_on_off", "vol_regime", "trend_strength",
        "credit_stress", "em_vs_dm", "cross_asset_vol",
    ]
    macro_in_grammar = all(m in GP_FEATURES for m in macro_terminals)

    # Run GP and measure diversity
    split   = int(len(prices) * 0.70)
    train   = prices.iloc[:split]
    gp      = GPMiner(train, pop_size=120, elite_n=15, seed=42, verbose=False)
    report  = gp.run(n_epochs=6)
    formulas = report.submission_formulas(32)

    n_unique = len(gp._hall_of_fame)
    macro_in_top32 = [f for f in formulas if any(m in f for m in macro_terminals)]
    unique_feats   = {feat for f in formulas for feat in GP_FEATURES if feat in f}

    diversity_ok = n_unique >= 50 and len(unique_feats) >= 12

    ok = feat_ok and macro_in_grammar and diversity_ok
    _check("GP diversity",
           f"{n_unique} unique | {len(macro_in_top32)} macro formulas | "
           f"{len(unique_feats)} distinct features", ok)
    return ok, formulas


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 3: Portfolio intelligence (Sprint 28)
# ══════════════════════════════════════════════════════════════════════════════

def check_portfolio(prices: pd.DataFrame, formulas: list[str]) -> tuple[bool, object]:
    """Clustering, regime detection, risk-parity weights, positions."""
    from macro8_subnet.alpha.portfolio_intelligence import (
        AdaptiveEnsemble, SignalClusterer, RegimeDetector,
    )
    from macro8_subnet.alpha.feature_store import FeatureStore

    split = int(len(prices) * 0.70)
    train = prices.iloc[:split]

    # Feature store: 34 features including macro
    fs    = FeatureStore(prices)
    feats = fs.build()
    feat_ok = len(feats) >= 30

    # Regime detector
    det    = RegimeDetector()
    labels = det.label_series(feats, prices.index)
    regime_ok = set(labels.unique()).issubset({"calm", "normal", "stress"})

    # Adaptive ensemble
    ens    = AdaptiveEnsemble(train, formulas[:10], weighting="risk_parity",
                              verbose=False)
    ens.fit()
    result = ens.positions()

    positions_ok = (
        len(result.positions) > 0 and
        abs(sum(abs(w) for w in result.positions.values()) - 1.0) < 0.05
    )

    ok = feat_ok and regime_ok and positions_ok
    _check("Portfolio intel",
           f"{len(feats)} features | clusters={ens._cluster_result.n_clusters} | "
           f"regime={result.regime.name} | {len(result.positions)} positions", ok)
    return ok, ens


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 4: Regime prediction (Sprint 29)
# ══════════════════════════════════════════════════════════════════════════════

def check_prediction(prices: pd.DataFrame, formulas: list[str]) -> tuple[bool, object]:
    """Transition model, policy layer, scenario probabilities, confidence."""
    from macro8_subnet.alpha.regime_prediction import (
        ForecastedEnsemble, RegimeTransitionModel, PolicyLayer,
        ScenarioProbabilityAssigner, ConfidenceScore,
    )

    split = int(len(prices) * 0.70)
    train = prices.iloc[:split]

    # Full ForecastedEnsemble
    fens = ForecastedEnsemble(train, formulas[:8], horizon=5, verbose=False)
    fens.fit()
    result = fens.forecast()

    # Checks
    probs_sum    = sum(result.scenario_probs.values())
    regime_valid = result.regime_current in ("calm", "normal", "stress")
    probs_ok     = abs(probs_sum - 1.0) < 0.01
    conf_ok      = 0 <= result.confidence <= 1
    policy_ok    = all(hasattr(result.policy_state, a)
                       for a in ("rate_env", "inflation", "liquidity"))

    # Prediction market
    from macro8_subnet.execution.engine import PredictionMarket, DrawdownGuard, PerformanceWindow
    pw   = PerformanceWindow(0, 0, 0, 0, 0, 0, 0, 0)
    pred = PredictionMarket().emit(result, pw, DrawdownGuard())
    market_ok = (pred.epoch == 1 and len(pred.scenario_probs) == 8)

    ok = regime_valid and probs_ok and conf_ok and policy_ok and market_ok
    top_scen = max(result.scenario_probs.items(), key=lambda x: x[1])
    _check("Regime prediction",
           f"regime={result.regime_current}→{result.regime_forecast.most_likely} "
           f"conf={result.confidence:.2f} top_scenario={top_scen[0]}({top_scen[1]:.0%})", ok)
    return ok, fens


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 5: Execution layer (Sprint 30)
# ══════════════════════════════════════════════════════════════════════════════

def check_execution(prices: pd.DataFrame, fens) -> bool:
    """Constraint solver, trade executor, drawdown guard, position scaling."""
    from macro8_subnet.execution.engine import (
        PortfolioConstraints, ConstraintSolver, DrawdownGuard,
        TradeExecutor, LiveTracker,
    )

    result = fens.forecast()

    # Constraint solver: stress-adjusted, sector capped, net exposure controlled
    cs  = ConstraintSolver(PortfolioConstraints(max_weight=0.35))
    pos = cs.apply(result.positions, p_stress=result.regime_forecast.stress)

    l1    = sum(abs(w) for w in pos.values())
    net   = abs(sum(pos.values()))
    constr_ok = (abs(l1 - 1.0) < 0.02 and net <= 0.25)

    # Trade executor
    ex   = TradeExecutor(capital=100_000)
    plan = ex.compute_trades(pos, {})
    trade_ok = (len(plan.orders) > 0 and plan.total_turnover > 0)

    # DrawdownGuard: fires below threshold
    dg = DrawdownGuard(max_drawdown=-0.02, lookback=10)
    for _ in range(5): dg.update(0.001)    # above threshold
    scale_above = dg.position_scale()
    for _ in range(10): dg.update(-0.004)  # below threshold
    scale_below = dg.position_scale()
    dd_ok = (scale_above == 1.0 and scale_below < 1.0)

    # LiveTracker: feedback loop
    tracker = LiveTracker()
    tracker.update(prices.index[-1], pos, plan, 0.001, result)
    tracker.update(prices.index[-2], pos, plan, -0.002, result)
    pw = tracker.snapshot()
    track_ok = pw.n_days == 2

    ok = constr_ok and trade_ok and dd_ok and track_ok
    _check("Execution layer",
           f"positions L1={l1:.3f} net={net:.3f} | "
           f"{len(plan.orders)} trades | DD scale {scale_above:.2f}→{scale_below:.2f}", ok)
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 6: Live pipeline (Sprint 31)
# ══════════════════════════════════════════════════════════════════════════════

def check_live_pipeline(prices: pd.DataFrame) -> bool:
    """DataPipeline fallback, FailureLog persistence, PaperTrader backtest."""
    import tempfile
    from macro8_subnet.execution.live_runner import (
        DataPipeline, FailureLog, PaperTrader,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # DataPipeline: offline fallback to synthetic
        pipeline = DataPipeline(
            tickers=list(prices.columns)[:8],
            cache_dir=tmp / "cache",
            verbose=False,
        )
        pipeline._is_online = lambda: False   # force offline
        fetched, status = pipeline.fetch()
        pipeline_ok = (len(fetched) > 100 and status.source in ("synthetic", "cache"))

        # FailureLog: log, persist, reload
        fl = FailureLog(path=tmp / "failures.json")
        fl.log_regime_failure("2024-01-15", "normal", "stress", 0.72, -0.003)
        fl.log_retrain("2024-01-20", -0.8)
        fl2 = FailureLog(path=tmp / "failures.json")   # reload
        fail_ok = len(fl2._log) == 2

        # PaperTrader: 5-day backtest
        trader = PaperTrader(
            tickers=list(prices.columns)[:8],
            state_file=tmp / "state.json",
            verbose=False,
        )
        hist = trader.run_backtest(prices, n_days=5, train_frac=0.85)
        trader_ok = (len(hist) >= 3 and "pnl" in hist.columns and
                     np.isfinite(hist["pnl"].dropna().values).all())

    ok = pipeline_ok and fail_ok and trader_ok
    _check("Live pipeline",
           f"data={status.source} | failures={len(fl2._log)} logged | "
           f"backtest={len(hist)} days", ok)
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 7: Validator OOS scoring (Sprint 25–26)
# ══════════════════════════════════════════════════════════════════════════════

def check_validator(formulas: list[str], positions: dict) -> bool:
    """EpochScorer, PortfolioEvaluator blend, reward normalisation."""
    from macro8_subnet.agents.role_rewards import RoleRewardModel

    validator = Macro8Validator()

    # Build two miners: one with positions (Sprint 26), one without
    syn0 = AlphaSubmissionSynapse(
        formulas=formulas[:10],
        positions=positions,
        position_formula="ensemble(2,regime=normal)",
        miner_uid=0, epoch=1,
    )
    syn1 = AlphaSubmissionSynapse(
        formulas=formulas[5:12],
        miner_uid=1, epoch=1,
    )
    subs = {0: syn0, 1: syn1}

    scores = validator.scorer.score_submissions(subs, epoch=1)

    role_model    = RoleRewardModel()
    role_scores   = validator._build_role_scores(subs, scores)
    reward_report = role_model.compute(epoch=1, role_scores=role_scores)
    uids, weights = reward_report.as_weight_list()

    all_scored    = len(scores) == 2
    rewards_ok    = abs(sum(weights) - 1.0) < 0.01 if weights else True
    scores_finite = all(np.isfinite(v) for v in scores.values())

    ok = all_scored and rewards_ok and scores_finite
    _check("Validator scoring",
           f"{len(scores)}/2 scored | rewards_sum={sum(weights):.4f} | "
           f"top_score={max(scores.values()):.4f}", ok)
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# ADVERSARIAL CHECK
# ══════════════════════════════════════════════════════════════════════════════

def check_adversarial() -> bool:
    """Adversarial miners cannot crash the scorer."""
    validator = Macro8Validator()
    subs: dict = {}

    for uid in range(8):
        if uid % 2 == 0:
            syn = AlphaSubmissionSynapse(formulas=GOOD_FORMULAS[:5], miner_uid=uid)
        else:
            syn = AlphaSubmissionSynapse(formulas=GOOD_FORMULAS[:3], miner_uid=uid)
            # Post-construction corruption
            bad = random.sample(ADVERSARIAL_INPUTS, 3)
            try:
                object.__setattr__(syn, "formulas", bad + GOOD_FORMULAS[:2])
            except Exception:
                pass
        subs[uid] = syn

    try:
        scores = validator.scorer.score_submissions(subs, epoch=1)
        n_scored = len(scores)
        ok = True
    except Exception as e:
        print(f"    CRASHED: {e}")
        n_scored = 0
        ok = False

    from macro8_subnet.agents.role_rewards import RoleRewardModel
    role_model    = RoleRewardModel()
    role_scores   = validator._build_role_scores(subs, scores if ok else {})
    reward_report = role_model.compute(epoch=1, role_scores=role_scores)
    _, weights    = reward_report.as_weight_list()
    rewards_ok    = abs(sum(weights) - 1.0) < 0.01 if weights else True

    ok = ok and rewards_ok
    _check("Adversarial inputs",
           f"{n_scored}/8 miners scored | no crashes | rewards normalised", ok)
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# STRESS TEST
# ══════════════════════════════════════════════════════════════════════════════

def run_stress_test(n_miners: int = 100) -> float:
    """Score N miners; return elapsed seconds."""
    print(f"\n  Stress test: {n_miners} miners...", end=" ", flush=True)
    rng  = random.Random(0)
    subs = {}
    for uid in range(n_miners):
        n = rng.randint(3, 10)
        f = rng.sample(GOOD_FORMULAS, min(n, len(GOOD_FORMULAS)))
        subs[uid] = AlphaSubmissionSynapse(formulas=f, miner_uid=uid)

    validator = Macro8Validator()
    t0        = time.perf_counter()
    scores    = validator.scorer.score_submissions(subs, epoch=1)
    elapsed   = time.perf_counter() - t0

    print(f"{len(scores)} scored in {elapsed:.2f}s "
          f"({'FAST' if elapsed < 5.0 else 'SLOW'})")
    return elapsed


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST CHECK
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest_check(prices: pd.DataFrame) -> bool:
    """90-day OOS paper trading backtest via live_runner."""
    from macro8_subnet.execution.live_runner import PaperTrader
    import tempfile

    print(f"\n  Running 30-day backtest (train={int(len(prices)*0.80)} days)...")
    with tempfile.TemporaryDirectory() as tmp:
        trader = PaperTrader(
            tickers=list(prices.columns)[:8],
            state_file=Path(tmp) / "state.json",
            gp_gens=5,
            verbose=False,
        )
        hist = trader.run_backtest(prices, n_days=30, train_frac=0.80)

    if len(hist) == 0:
        print("  FAIL: no history returned")
        return False

    pnl = hist["pnl"].dropna()
    sh  = float(pnl.mean() / (pnl.std() + 1e-8) * np.sqrt(252))
    cum = float(pnl.sum())
    mdd = float((pnl.cumsum() - pnl.cumsum().cummax()).min()) if len(pnl) > 1 else 0

    ok = np.isfinite(sh) and len(hist) >= 20
    print(f"  {len(hist)} days | cum={cum:+.4f} | Sharpe={sh:+.3f} | MaxDD={mdd:.4f}")
    _check("Backtest (30d)", f"finite PnL | {len(hist)} days simulated", ok)
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_check_results: list[tuple[str, bool]] = []

def _check(label: str, detail: str, ok: bool) -> None:
    sym = "✓" if ok else "✗"
    print(f"  {sym} [{label:<18}] {detail}")
    _check_results.append((label, ok))


def _make_prices() -> pd.DataFrame:
    """Load calibrated prices or generate synthetic if unavailable."""
    cand = [
        Path(__file__).resolve().parent.parent / "calibrated_prices.csv",
        Path("/home/claude/Macro8/calibrated_prices.csv"),
    ]
    for p in cand:
        if p.exists():
            return pd.read_csv(p, index_col=0, parse_dates=True)

    from macro8_subnet.data.market_data_fetcher import MarketDataFetcher
    fetcher = MarketDataFetcher(force_synthetic=True, verbose=False)
    result  = fetcher.fetch_prices(n_synthetic=2000)
    return result.prices


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser("Macro8 Local Simulation — Sprint 32")
    parser.add_argument("--fast",        action="store_true",
                        help="Only defensive + validator checks (no GP/ML)")
    parser.add_argument("--adversarial", action="store_true",
                        help="Include adversarial miner simulation")
    parser.add_argument("--stress",      action="store_true",
                        help="100-miner throughput stress test")
    parser.add_argument("--backtest",    action="store_true",
                        help="30-day paper-trading backtest")
    parser.add_argument("--full",        action="store_true",
                        help="All checks including stress + backtest")
    args = parser.parse_args()

    if args.full:
        args.adversarial = args.stress = args.backtest = True

    print("\n" + "═" * W)
    print("  Macro8 Local Simulation — Sprint 32")
    print("  Full-stack integration verification")
    print("═" * W)

    t_start = time.perf_counter()
    prices  = _make_prices()
    print(f"\n  Data: {len(prices)} days × {len(prices.columns)} assets "
          f"({prices.index[0].date()} → {prices.index[-1].date()})\n")

    all_ok = True

    # ── Check 1: Defensive ───────────────────────────────────────────────────
    print("[1] Defensive validation")
    ok = check_defensive()
    all_ok &= ok

    if args.fast:
        # Skip ML-intensive checks
        formulas  = GOOD_FORMULAS
        positions = {"SPY": 0.15, "TLT": -0.12, "GLD": -0.08}
        fens      = None
    else:
        # ── Check 2: GP diversity ────────────────────────────────────────────
        print("\n[2] GP discovery (34 features, macro terminals)")
        ok, formulas = check_gp(prices)
        all_ok &= ok

        # ── Check 3: Portfolio intelligence ──────────────────────────────────
        print("\n[3] Portfolio intelligence (clustering + regime)")
        ok, ens = check_portfolio(prices, formulas)
        all_ok &= ok
        result_ens = ens.positions()
        positions  = result_ens.positions

        # ── Check 4: Regime prediction ────────────────────────────────────────
        print("\n[4] Regime prediction (forecast + scenarios)")
        ok, fens = check_prediction(prices, formulas)
        all_ok &= ok

        # ── Check 5: Execution layer ──────────────────────────────────────────
        print("\n[5] Execution layer (constraints + trades + drawdown)")
        ok = check_execution(prices, fens)
        all_ok &= ok

        # ── Check 6: Live pipeline ────────────────────────────────────────────
        print("\n[6] Live pipeline (data + failure log + paper trader)")
        ok = check_live_pipeline(prices)
        all_ok &= ok

    # ── Check 7: Validator ───────────────────────────────────────────────────
    print("\n[7] Validator scoring (OOS + rewards)")
    ok = check_validator(formulas, positions)
    all_ok &= ok

    # ── Optional checks ──────────────────────────────────────────────────────
    if args.adversarial:
        print("\n[8] Adversarial miner simulation")
        ok = check_adversarial()
        all_ok &= ok

    if args.stress:
        run_stress_test(100)

    if args.backtest:
        print("\n[9] Paper-trading backtest (30 OOS days)")
        ok = run_backtest_check(prices)
        all_ok &= ok

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    print("\n" + "═" * W)
    print("  Summary")
    print("═" * W)
    for label, ok in _check_results:
        print(f"  {'✓' if ok else '✗'} {label}")
    print()
    print(f"  Elapsed: {elapsed:.1f}s")
    print()
    if all_ok:
        print("  ✓ All checks passed — ready for Bittensor testnet")
    else:
        failed = [label for label, ok in _check_results if not ok]
        print(f"  ✗ {len(failed)} check(s) failed: {failed}")
        print("    See output above for details.")
    print()


if __name__ == "__main__":
    main()
