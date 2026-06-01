"""
neurons/miner.py
-----------------
Macro8 Miner Neuron — Sprint 26.

The miner is the research agent. Each Bittensor epoch it:
    1. Runs GP to evolve alpha formulas (composite fitness = IC + net Sharpe + scalability)
    2. Evaluates top formulas through walk-forward PR testing (optional, every N epochs)
    3. Computes actual daily positions from the best formula (signal → weights)
    4. Submits formulas + positions to the validator

Signal layer (NEW Sprint 26)
----------------------------
The miner now outputs ACTUAL POSITIONS, not just formula strings.

    best_formula → FeatureStore → signal[T × A] → rank → weights[A]

Positions are the latest-day cross-sectional weights from the top formula.
These are the trades the miner recommends as of the current epoch.

    positions = {
        "SPY": +0.18,    # long 18% of portfolio
        "TLT": −0.12,    # short 12%
        ...
    }

Roles:
    signal    — alpha formula discovery + position output (default)
    strategy  — weighted combination of library signals
    risk      — covariance model parameters
    portfolio — constraint set submission
    meta      — IC prediction for library signals

Running:
    python -m macro8_subnet.neurons.miner --role signal
    python -m macro8_subnet.neurons.miner --role signal --n_formulas 200 --pr_test_interval 5

Environment variables:
    FRED_API_KEY         — free FRED API key for macro data
    MACRO8_N_FORMULAS    — formulas per submission (default 200)
    MACRO8_DATA_START    — historical start date (default 2010-01-01)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

_SUBNET = Path(__file__).resolve().parent.parent
_ROOT   = _SUBNET.parent
for p in [str(_SUBNET), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import bittensor as bt
    _BT_AVAILABLE = True
except ImportError:
    _BT_AVAILABLE = False
    print("[Miner] bittensor not installed — running in dry-run mode")

from macro8_subnet.protocol.synapse        import AlphaSubmissionSynapse
from macro8_subnet.alpha.hypothesis_engine import (
    HypothesisLibrary, HypothesisCategory, HypothesisEvolution,
)
from macro8_subnet.alpha.macro_session     import _seed_hypotheses
from macro8_subnet.data.market_data_fetcher import MarketDataFetcher


# ── Configuration ─────────────────────────────────────────────────────────────

def get_config() -> argparse.Namespace:
    cfg = argparse.Namespace(
        netuid=1,
        role="signal",
        n_formulas=200,
        n_hypotheses=5,
        data_start=os.environ.get("MACRO8_DATA_START", "2010-01-01"),
        n_assets=10,
        pr_test_interval=10,   # run walk-forward PR test every N epochs
        gp_generations=3,      # GP generations per Bittensor epoch
        capital=100_000,
        wallet_name="default",
        wallet_hotkey="default",
        subtensor_network="finney",
    )
    return cfg


# ── Miner ─────────────────────────────────────────────────────────────────────

class Macro8Miner:
    """
    Macro8 miner neuron.

    Research pipeline per epoch:
        1. GPMiner.step() × 3 — evolve formula population (fitness = composite)
        2. Extract top formulas from hall of fame
        3. Compute signal → positions from best formula
        4. Optionally run PR tester (walk-forward) and filter by survival
        5. Submit formulas + positions to validator
    """

    def __init__(self, config: argparse.Namespace = None):
        self.config     = config or get_config()
        self.role       = self.config.role
        self.n_formulas = int(os.environ.get("MACRO8_N_FORMULAS",
                                              self.config.n_formulas))
        self.epoch      = 0

        # ── Bittensor ─────────────────────────────────────────────────────────
        self.wallet = self.subtensor = self.metagraph = self.axon = None
        if _BT_AVAILABLE:
            try:
                self.wallet    = bt.Wallet(
                    name=getattr(self.config, "wallet_name",   "default"),
                    hotkey=getattr(self.config, "wallet_hotkey", "default"),
                )
                self.subtensor = bt.Subtensor(
                    network=getattr(self.config, "subtensor_network", "finney")
                )
                self.metagraph = self.subtensor.metagraph(self.config.netuid)
                self.axon      = bt.Axon(wallet=self.wallet)
            except Exception as e:
                print(f"[Miner] Bittensor network unavailable: {e}")

        self._initialise_research_engine()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _initialise_research_engine(self) -> None:
        """Fetch market data, build FeatureStore, GPMiner, PR tester."""
        tickers = [
            "SPY", "QQQ", "IWM", "TLT", "GLD",
            "DBC", "EEM", "FXI", "VNQ", "HYG",
        ][:self.config.n_assets]

        fetcher     = MarketDataFetcher(
            fred_api_key=os.environ.get("FRED_API_KEY", ""),
            verbose=False,
        )
        data_result = fetcher.fetch_prices(
            tickers=tickers,
            start=self.config.data_start,
            n_synthetic=3780,
        )
        self.prices     = data_result.prices
        self.returns    = self.prices.pct_change().dropna()
        self.data_source = data_result.source

        # Hypothesis library
        self.hyp_lib = HypothesisLibrary()
        _seed_hypotheses(self.hyp_lib, n=self.config.n_hypotheses)
        self.hyp_evo = HypothesisEvolution(self.hyp_lib)

        # GPMiner — uses PortfolioEvaluator internally (Sprint 25)
        from macro8_subnet.alpha.gp_miner import GPMiner
        self.gp_miner = GPMiner(
            prices=self.prices,
            pop_size=min(self.n_formulas, 200),
            elite_n=20,
            seed=42,
            verbose=False,
        )

        # FeatureStore — for signal → position computation
        from macro8_subnet.alpha.feature_store import FeatureStore
        self.feature_store = FeatureStore(self.prices)
        self._features     = None   # lazy-built

        # Best formula tracking
        self.best_formulas:  list[str] = []
        self.pr_passed:      list[str] = []   # formulas that passed PR testing
        self.last_positions: dict[str, float] = {}

        print(
            f"[Miner] Engine ready: {len(self.prices)} days × "
            f"{len(self.prices.columns)} assets | "
            f"source={self.data_source} | role={self.role}"
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, synapse: AlphaSubmissionSynapse) -> AlphaSubmissionSynapse:
        """Handle validator request: run GP, compute positions, submit."""
        self.epoch += 1
        t0 = time.perf_counter()
        try:
            if self.role == "signal":
                synapse = self._forward_signal(synapse)
            elif self.role == "strategy":
                synapse = self._forward_strategy(synapse)
            elif self.role == "risk":
                synapse = self._forward_risk(synapse)
            elif self.role == "portfolio":
                synapse = self._forward_portfolio(synapse)
            elif self.role == "meta":
                synapse = self._forward_meta(synapse)

            synapse.miner_uid    = 0
            synapse.epoch        = self.epoch
            synapse.eval_success = True

        except Exception as e:
            synapse.eval_error   = str(e)[:200]
            synapse.eval_success = False
            traceback.print_exc()

        elapsed = time.perf_counter() - t0
        n_pos   = len(synapse.positions) if synapse.positions else 0
        print(
            f"[Miner] Epoch {self.epoch} | {len(synapse.formulas)} formulas | "
            f"{n_pos} positions | {elapsed:.1f}s"
        )
        return synapse

    # ── Signal role ───────────────────────────────────────────────────────────

    def _forward_signal(self, syn: AlphaSubmissionSynapse) -> AlphaSubmissionSynapse:
        """
        Full signal pipeline:
            1. GP evolution (n_gens generations)
            2. Extract top formulas (composite fitness)
            3. Optional walk-forward PR test
            4. Build adaptive ensemble (cluster + regime-conditional weights)
            5. Compute positions from ensemble
            6. Submit formulas + positions
        """
        # ── Step 1: GP evolution ──────────────────────────────────────────────
        seed_formulas = self.hyp_evo.seed_formulas(n=20)
        if seed_formulas:
            self.gp_miner.add_hypothesis_seeds(seed_formulas)

        n_gens = getattr(self.config, "gp_generations", 3)
        for _ in range(n_gens):
            self.gp_miner.step()

        top_formulas       = self.gp_miner.top_formulas(n=32)
        self.best_formulas = top_formulas

        # ── Step 2: Walk-forward PR test (periodic) ───────────────────────────
        pr_interval = getattr(self.config, "pr_test_interval", 10)
        if self.epoch % pr_interval == 0 and len(top_formulas) >= 3:
            try:
                from macro8_subnet.evaluation.pr_tester import (
                    PRTester, WalkForwardConfig,
                )
                tester = PRTester(
                    self.prices,
                    capital=getattr(self.config, "capital", 100_000),
                    min_ic_threshold=0.003,
                    min_pass_rate=0.50,
                    max_ic_stability=3.0,
                    verbose=False,
                )
                results = tester.walk_forward(
                    top_formulas[:15],
                    WalkForwardConfig(train_days=504, test_days=63, step_days=21),
                )
                self.pr_passed = [r.formula for r in results if r.passes]
                print(f"[Miner] PR test: {len(self.pr_passed)}/{len(top_formulas[:15])} passed")
            except Exception as e:
                print(f"[Miner] PR test skipped: {e}")
                self.pr_passed = top_formulas[:10]

        # ── Step 3: Build submission list ─────────────────────────────────────
        submission = list(self.pr_passed[:20]) if self.pr_passed else []
        for f in top_formulas:
            if f not in submission:
                submission.append(f)
            if len(submission) >= 32:
                break
        syn.formulas = submission or seed_formulas[:10]

        # ── Step 4: Adaptive ensemble (rebuild every N epochs) ───────────────
        rebuild_interval = getattr(self.config, "ensemble_rebuild_interval", 5)
        if (not hasattr(self, "_ensemble") or self._ensemble is None
                or self.epoch % rebuild_interval == 1):
            try:
                from macro8_subnet.alpha.portfolio_intelligence import AdaptiveEnsemble
                self._ensemble = AdaptiveEnsemble(
                    self.prices,
                    syn.formulas,
                    n_clusters=None,
                    weighting="risk_parity",
                    capital=getattr(self.config, "capital", 100_000),
                    verbose=False,
                )
                robustness = getattr(self, "_scenario_robustness", {})
                self._ensemble.fit(robustness_scores=robustness or None)
                cr = self._ensemble._cluster_result
                if cr:
                    print(f"[Miner] Ensemble: {cr.n_clusters} clusters | "
                          f"sizes={cr.cluster_sizes}")
            except Exception as e:
                print(f"[Miner] Ensemble build failed: {e}")
                self._ensemble = None

        # ── Step 5: Positions from ensemble (fallback: single formula) ────────
        if self._ensemble is not None and self._ensemble._cluster_result is not None:
            try:
                ens_result = self._ensemble.positions()
                if ens_result.positions:
                    syn.positions        = ens_result.positions
                    self.last_positions  = ens_result.positions
                    syn.position_formula = (
                        f"ensemble({len(ens_result.active_formulas)}"
                        f",regime={ens_result.regime.name})"
                    )
            except Exception as e:
                print(f"[Miner] Ensemble positions failed: {e}")
                self._fallback_positions(syn, top_formulas)
        else:
            self._fallback_positions(syn, top_formulas)

        # ── Step 6: Submit hypotheses every 5 epochs ──────────────────────────
        if self.epoch % 5 == 0:
            syn.hypothesis_statements = [
                rec.statement for rec in self.hyp_lib.rank_by_confidence(3)
            ]
            syn.hypothesis_categories = [
                rec.category.value for rec in self.hyp_lib.rank_by_confidence(3)
            ]

        return syn

    def _fallback_positions(
        self, syn: AlphaSubmissionSynapse, top_formulas: list[str]
    ) -> None:
        """Single-formula fallback when ensemble is unavailable."""
        best = (self.pr_passed[0] if self.pr_passed
                else top_formulas[0] if top_formulas else None)
        if best:
            positions = self._formula_to_positions(best)
            if positions:
                syn.positions        = positions
                self.last_positions  = positions
                syn.position_formula = best

    # ── Signal → Positions ────────────────────────────────────────────────────

    def _formula_to_positions(
        self,
        formula: str,
        lookback: int = 120,
    ) -> dict[str, float]:
        """
        Convert a formula string to a cross-sectional long/short portfolio.

        Steps:
            1. Evaluate formula to get signal DataFrame [T × A]
            2. Take the latest row (today's signal)
            3. Cross-sectional rank → centre → L1-normalise → weights

        Returns:
            {ticker: weight} where weights sum to 0 in absolute value = 1.
            Positive = long, negative = short.
        """
        try:
            from macro8_subnet.alpha.batch_evaluator import FormulaEncoder, FeatureTensor
            from macro8_subnet.alpha.feature_store import FeatureStore

            # Use recent data only for position computation
            recent = self.prices.iloc[-lookback:]
            fs     = FeatureStore(recent)
            feats  = fs.build()

            if not feats:
                return {}

            ft  = FeatureTensor.from_feature_dict(feats)
            enc = FormulaEncoder(ft.feature_names)

            if not enc.can_encode(formula):
                return {}

            W = enc.encode_batch([formula])           # [n_feat × 1]
            S = np.einsum("taf,fn->tan", ft.tensor, W, optimize=True)  # [T × A × 1]

            latest_signal = S[-1, :, 0]   # [A] — today's cross-sectional signal

            # Guard: all-zero or NaN signal
            if not np.isfinite(latest_signal).all() or np.all(latest_signal == 0):
                return {}

            # Cross-sectional rank → L/S weights
            from scipy.stats import rankdata
            ranks   = rankdata(latest_signal).astype(float)
            ranks  -= ranks.mean()
            l1_norm = np.abs(ranks).sum()
            if l1_norm < 1e-8:
                return {}
            weights = ranks / l1_norm

            tickers = list(self.prices.columns)
            positions = {
                ticker: float(round(w, 6))
                for ticker, w in zip(tickers, weights)
                if abs(w) > 1e-4   # drop near-zero positions
            }
            return positions

        except Exception as e:
            print(f"[Miner] Position computation failed: {e}")
            return {}

    def get_current_positions(self) -> dict[str, float]:
        """Return the latest computed positions (for inspection / dry-run)."""
        return dict(self.last_positions)

    def get_position_table(self) -> pd.DataFrame:
        """Return positions as a formatted DataFrame."""
        pos = self.last_positions
        if not pos:
            return pd.DataFrame(columns=["ticker", "weight", "direction"])
        rows = [
            {
                "ticker":    t,
                "weight":    round(w, 4),
                "direction": "LONG" if w > 0 else "SHORT",
            }
            for t, w in sorted(pos.items(), key=lambda x: x[1], reverse=True)
        ]
        return pd.DataFrame(rows)

    # ── Other roles ───────────────────────────────────────────────────────────

    def _forward_strategy(self, syn: AlphaSubmissionSynapse) -> AlphaSubmissionSynapse:
        """Submit signal combination weights."""
        n = 5
        syn.signal_weights = {f"signal_{i}": 1.0 / n for i in range(n)}
        syn.agent_role     = "strategy"
        return syn

    def _forward_risk(self, syn: AlphaSubmissionSynapse) -> AlphaSubmissionSynapse:
        """Submit covariance model parameters."""
        syn.risk_payload = {
            "shrinkage":   0.3,
            "n_factors":   3,
            "model_type":  "ledoit_wolf",
        }
        syn.agent_role = "risk"
        return syn

    def _forward_portfolio(self, syn: AlphaSubmissionSynapse) -> AlphaSubmissionSynapse:
        """Submit portfolio constraint set."""
        syn.portfolio_payload = {
            "max_weight":   0.35,
            "min_weight":   0.02,
            "max_turnover": 0.5,
            "method":       "ic_weighted",
        }
        syn.agent_role = "portfolio"
        return syn

    def _forward_meta(self, syn: AlphaSubmissionSynapse) -> AlphaSubmissionSynapse:
        """Submit IC predictions for library signals."""
        preds = {
            f"hyp_{rec.hypothesis_id[:8]}": rec.mean_ic * rec.confidence_score
            for rec in self.hyp_lib.rank_by_confidence(10)
        }
        syn.ic_predictions = preds
        syn.agent_role     = "meta"
        return syn

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Main miner loop."""
        if not _BT_AVAILABLE or self.wallet is None:
            print("[Miner] Running in dry-run mode")
            self._dry_run()
            return

        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            print(f"[Miner] NOT registered on netuid {self.config.netuid}.")
            print("  Run: btcli s register --netuid <UID> --wallet.name miner")
            return

        self.axon.attach(forward_fn=self.forward).start()
        print(f"[Miner] Serving | hotkey={self.wallet.hotkey.ss58_address}")
        print("[Miner] Press Ctrl+C to stop")

        try:
            while True:
                self.metagraph.sync(subtensor=self.subtensor)
                time.sleep(60)
        except KeyboardInterrupt:
            print("[Miner] Stopping...")
            self.axon.stop()

    def _dry_run(self, n_epochs: int = 3) -> None:
        """Local test without Bittensor."""
        print(f"[Miner] Dry-run: {n_epochs} epochs")
        for epoch in range(1, n_epochs + 1):
            syn    = AlphaSubmissionSynapse(epoch=epoch)
            result = self.forward(syn)
            print(f"  Epoch {epoch}: {len(result.formulas)} formulas | "
                  f"{len(result.positions or {})} positions")
            if result.positions:
                tbl = self.get_position_table()
                print(tbl.to_string(index=False))
            print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    config = get_config()
    miner  = Macro8Miner(config)
    miner.run()


if __name__ == "__main__":
    main()
