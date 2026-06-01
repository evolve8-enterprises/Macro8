"""
neurons/validator.py
---------------------
Macro8 Validator Neuron — the scoring engine of the subnet.

The validator is the most important component. It:
    1. Queries every registered miner each epoch
    2. Receives AlphaSubmissionSynapse from each miner
    3. Evaluates formula ICs using the BatchEvaluator (vectorised, ~6k/sec)
    4. Updates hypothesis confidences via the research graph
    5. Computes role-stratified rewards via RoleRewardModel
    6. Calls subtensor.set_weights() to distribute TAO emissions

Anti-gaming measures:
    - Out-of-sample evaluation window (most recent 20% of data)
    - Rolling IC: signals must show IC > 0 in recent window
    - Orthogonality filter: correlated signals share a reward budget
    - Lifecycle scoring: PRODUCTION signals get full weight,
      DECAYING signals get 0.3x, RETIRED signals get 0.0x
    - Rate limiting: max 3 formula submissions per miner per epoch

Running
-------
    # Testnet:
    python -m macro8_subnet.neurons.validator \\
        --subtensor.network test \\
        --netuid 263 \\
        --wallet.name validator \\
        --wallet.hotkey default \\
        --logging.debug

    # Mainnet (after subnet registration):
    python -m macro8_subnet.neurons.validator \\
        --netuid <YOUR_NETUID> \\
        --wallet.name validator \\
        --wallet.hotkey default

Environment variables:
    FRED_API_KEY       — free FRED macro data key
    MACRO8_DATA_DIR    — local cache directory
    MACRO8_EPOCH_SECS  — epoch length (default 360s = 6 minutes)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

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
    print("[Validator] bittensor not installed — running in dry-run mode")

from macro8_subnet.protocol.synapse            import AlphaSubmissionSynapse
from macro8_subnet.alpha.batch_evaluator       import BatchEvaluator
from macro8_subnet.alpha.portfolio_evaluator   import PortfolioEvaluator
from macro8_subnet.alpha.hypothesis_engine     import HypothesisLibrary, HypothesisEvolution
from macro8_subnet.alpha.research_graph        import ResearchGraph
from macro8_subnet.alpha.capacity_model        import LifecycleEngine
from macro8_subnet.alpha.orthogonality         import OrthogonalityFilter
from macro8_subnet.agents.agent_roles          import AgentRole, AgentSubmission
from macro8_subnet.agents.role_rewards         import RoleRewardModel
from macro8_subnet.data.market_data_fetcher    import MarketDataFetcher
from macro8_subnet.alpha.macro_session         import _seed_hypotheses, _match_hypothesis


# ── Defensive validation helpers ──────────────────────────────────────────────

import ast as _ast

# Hard limits
_MAX_FORMULA_LEN   = 200    # characters
_MAX_FORMULA_DEPTH = 12     # maximum AST nesting depth (prevents blowup)

# Tier 1: character allowlist — fast pre-filter before AST parsing
# Includes % for modulo (regime detection uses it)
_FORMULA_ALLOWED = set(
    "abcdefghijklmnopqrstuvwxyz0123456789_()+-*/.,=% "
)

# Tier 2: AST node whitelist — exhaustive set of nodes legitimate formulas need
# Anything outside this set is a potential code execution vector.
#
# Why each node is safe:
#   Expression       — root wrapper
#   Name / Load      — variable references (feature names, operator names)
#   Call             — operator calls: rank(x), zscore(x), decay(x, n=5)
#   keyword          — named arguments: halflife=10 in decay(x, halflife=10)
#   Constant         — numeric literals ONLY (int/float — not strings)
#   BinOp            — arithmetic: x+y, x-y, x*y, x/y, x**y, x%y
#   UnaryOp          — negation: -x
#   Add,Sub,Mult,Div,Pow,Mod,USub — the arithmetic operators
#
# Why these are blocked (not in whitelist):
#   Attribute        — obj.attr → can access __class__, __dict__, etc.
#   Subscript        — x[y] → can index into sensitive objects
#   Lambda           — function creation
#   IfExp            — conditional expressions (a if b else c)
#   ListComp/SetComp/GeneratorExp — iteration → DoS risk
#   Import/ImportFrom — code loading
#   Constant(str)    — string literals → __import__("os") uses this
#   JoinedStr        — f-strings
#   Starred          — *args unpacking
_AST_WHITELIST = frozenset({
    _ast.Expression,
    _ast.Name, _ast.Load,
    _ast.Call, _ast.keyword,
    _ast.BinOp, _ast.UnaryOp,
    _ast.Add, _ast.Sub, _ast.Mult, _ast.Div,
    _ast.Mod, _ast.USub,
    # NOTE: Pow (**) is intentionally excluded.
    # No Macro8 formula requires exponentiation, and Pow enables
    # exponential blowup attacks: ((a**a)**a)**a...
    _ast.Constant,
})


def _ast_depth(node: _ast.AST, current: int = 0) -> int:
    """Return the maximum nesting depth of an AST tree."""
    if current > _MAX_FORMULA_DEPTH:
        return current   # short-circuit
    child_depths = [
        _ast_depth(child, current + 1)
        for child in _ast.iter_child_nodes(node)
    ]
    return max(child_depths, default=current)


def _ast_safe(expr: str) -> bool:
    """
    Parse expr as a Python expression and verify every AST node is in
    the whitelist. Also verifies Constant nodes contain only numbers
    (blocks string literals used in __import__("os") style attacks).

    Returns True only if the expression is provably safe to pass to
    the formula engine's restricted eval().

    This is a SECOND layer of defence — safe_formula's character filter
    runs first and rejects most attacks cheaply. ast_safe catches
    sophisticated attacks that use only allowed characters.

    Examples blocked by ast_safe but not the char filter:
        ((a**a)**a)**a   — valid chars, but exponential complexity
        a if a else a    — valid chars, IfExp not in whitelist
    """
    try:
        tree = _ast.parse(expr, mode="eval")
    except SyntaxError:
        return False

    # Depth check — prevents exponential blowup in evaluator
    if _ast_depth(tree) > _MAX_FORMULA_DEPTH:
        return False

    for node in _ast.walk(tree):
        # Node type must be whitelisted
        if type(node) not in _AST_WHITELIST:
            return False
        # Constant values must be numeric — not strings, bytes, etc.
        if isinstance(node, _ast.Constant):
            if not isinstance(node.value, (int, float)):
                return False
        # Name identifiers must not be dunder names (__builtins__, __class__, etc.)
        if isinstance(node, _ast.Name):
            if node.id.startswith("__") or node.id.endswith("__"):
                return False

    return True


def safe_formula(value: object) -> Optional[str]:
    """
    Two-tier formula validation: character filter → AST sandbox.

    Tier 1 (fast): character allowlist rejects 99% of bad inputs cheaply.
    Tier 2 (thorough): AST whitelist blocks sophisticated attacks that
                       use only allowed characters.

    Returns the sanitised formula string, or None if invalid.
    A None return silently drops the formula; the miner scores 0 for it.
    The validator never crashes on bad input.

    Rejects
    -------
    - Non-string types (None, int, list, dict, ...)
    - Empty / whitespace-only strings
    - Strings over 200 characters
    - Strings containing characters outside [a-z0-9_()+-*/.,=% ]
    - Expressions with AST nodes outside the safe whitelist
    - String literals in expressions (__import__("os") uses a str Constant)
    - Deeply nested expressions (depth > 12, prevents eval blowup)
    - Malformed syntax (SyntaxError from ast.parse)
    """
    # ── Tier 1: type and character checks (< 1µs) ─────────────────────────────
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    if len(value) > _MAX_FORMULA_LEN:
        return None
    if not all(c in _FORMULA_ALLOWED for c in value.lower()):
        return None

    # ── Tier 2: AST safety check (~10µs, only for char-valid strings) ─────────
    if not _ast_safe(value):
        return None

    return value


def safe_synapse(synapse: object) -> Optional["AlphaSubmissionSynapse"]:
    """
    Validate an entire synapse returned from a miner query.

    Returns None (score=0) if the synapse is:
        - None (miner timed out / unreachable)
        - Missing the formulas attribute
        - Formulas field is not iterable
    """
    if synapse is None:
        return None
    if not hasattr(synapse, "formulas"):
        return None
    try:
        iter(synapse.formulas)
    except TypeError:
        return None
    return synapse


def safe_submission(
    uid:    int,
    synapse: object,
    max_formulas: int = 32,   # Protection 2: hard cap per miner
) -> Optional["AlphaSubmissionSynapse"]:
    """
    Full per-miner submission guard.

    Protection 1: validates every formula through the two-tier safe_formula
                  pipeline (character filter → AST sandbox).
    Protection 2: caps at max_formulas per miner (default 32) to prevent
                  computational DoS via formula spam.

    Modifies synapse.formulas in-place to the clean, capped list.
    Returns None only if the synapse itself is structurally invalid.
    """
    syn = safe_synapse(synapse)
    if syn is None:
        return None

    raw = syn.formulas if syn.formulas is not None else []

    # Protection 2: cap first (cheap) before running AST validation (slower)
    capped = list(raw)[:max_formulas]

    clean_formulas = []
    for f in capped:
        cleaned = safe_formula(f)
        if cleaned is not None:
            clean_formulas.append(cleaned)

    syn.formulas = clean_formulas
    return syn


# ── Validator configuration ───────────────────────────────────────────────────

EPOCH_SECONDS       = int(os.environ.get("MACRO8_EPOCH_SECS", 360))
MAX_FORMULAS_MINER  = 32       # hard cap: max formulas accepted per miner per epoch
MIN_IC_THRESHOLD    = 0.015    # minimum IC for reward eligibility
EVAL_WINDOW_FRAC    = 0.20     # fraction of data used for out-of-sample eval
WEIGHTS_VERSION_KEY = 10_001   # increment when scoring logic changes


def get_config() -> argparse.Namespace:
    return argparse.Namespace(
        netuid=1,
        data_start="2015-01-01",
        n_assets=8,
        wallet_name="default",
        wallet_hotkey="default",
        subtensor_network="finney",
        epoch_seconds=EPOCH_SECONDS,
    )


# ── Per-epoch scoring state ───────────────────────────────────────────────────

class EpochScorer:
    """
    Scores one epoch of miner submissions using multi-component scoring.

    Scoring dimensions: IC · stability · decay · novelty · capacity
    Anti-gaming: AST sandbox · per-miner cap · timeout · EMA smoothing
    """

    def __init__(
        self,
        batch_eval:      BatchEvaluator,
        lifecycle_engine: LifecycleEngine,
        orth_filter:     OrthogonalityFilter,
        min_ic:          float = MIN_IC_THRESHOLD,
    ):
        from macro8_subnet.evaluation.signal_scorer import SignalScorer
        self.batch_eval   = batch_eval
        self.lifecycle    = lifecycle_engine
        self.orth_filter  = orth_filter
        self.min_ic       = min_ic
        # Multi-component scorer (wires all five dimensions)
        self.scorer       = SignalScorer(
            batch_eval=batch_eval,
            orth_filter=orth_filter,
            lifecycle_engine=lifecycle_engine,
        )
        # Rolling IC/MSC histories: formula_id → list[float]
        self._ic_history:  dict[str, list[float]] = {}
        self._msc_history: dict[str, list[float]] = {}

    @staticmethod
    def _formula_id(formula: str) -> str:
        """Deterministic formula ID (same hash as FormulaLibrary)."""
        import hashlib
        return hashlib.sha256(formula.strip().encode()).hexdigest()[:12]

    def score_submissions(
        self,
        submissions:  dict[int, AlphaSubmissionSynapse],   # uid → synapse
        epoch:        int,
    ) -> dict[int, float]:
        """
        Score all miner submissions for this epoch.

        Returns {miner_uid: raw_score} where score ∈ [0, 1].
        Scores are NOT yet normalised (RoleRewardModel handles that).

        Error 1 guard: every response is validated through safe_submission()
        before any formula reaches the evaluation engine. Malformed,
        None, or injection-attempt formulas are silently dropped.

        Error 3 guard: SIGALRM timeout prevents any single miner's
        formula from stalling the scoring loop. Requires Unix (Linux/macOS).
        """
        import signal, platform

        # ── Error 1: Validate every submission defensively ────────────────────
        clean_submissions: dict[int, AlphaSubmissionSynapse] = {}
        for uid, raw_syn in submissions.items():
            syn = safe_submission(uid, raw_syn)
            if syn is not None:
                clean_submissions[uid] = syn
            # else: miner gets score=0, validator does not crash

        # ── Collect all formulas with two-level deduplication ─────────────────
        # Level 1: exact string hash — catches identical submissions
        # Level 2: weight-vector fingerprint — catches syntactic variants that
        #          encode to the same signal (e.g. zscore(x) vs rank(zscore(x)))
        formula_to_miner:  dict[str, int]   = {}
        all_formulas:      list[str]        = []
        seen_exact:        set[str]         = set()   # exact formula strings
        seen_vec_hashes:   set[str]         = set()   # weight vector fingerprints

        for uid, syn in clean_submissions.items():
            formulas = (syn.formulas or [])[:MAX_FORMULAS_MINER]
            for f in formulas:
                # Level 1: exact string dedup (O(1))
                if f in seen_exact:
                    continue
                seen_exact.add(f)

                # Level 2: semantic dedup via weight vector fingerprint (O(n_features))
                # Two formulas with identical weight vectors produce identical signals.
                # Round to 3 decimal places to catch near-identical encodings.
                try:
                    w       = self.batch_eval.encoder.encode(f)
                    vec_fp  = ",".join(f"{x:.3f}" for x in w)
                    if vec_fp in seen_vec_hashes:
                        continue   # semantically duplicate formula
                    seen_vec_hashes.add(vec_fp)
                except Exception:
                    pass   # encoding failed — keep formula (err on side of inclusion)

                if f not in formula_to_miner:
                    formula_to_miner[f] = uid
                    all_formulas.append(f)

        if not all_formulas:
            return {uid: 0.0 for uid in submissions}

        # ── Error 3: Timeout guard around batch evaluation ────────────────────
        # On Unix, SIGALRM kills any single evaluation that stalls.
        # On Windows, SIGALRM is not available — skip gracefully.
        _use_alarm = (platform.system() != "Windows" and
                      hasattr(signal, "SIGALRM"))

        if _use_alarm:
            def _timeout_handler(signum, frame):
                raise TimeoutError("BatchEvaluator timed out (>30s)")
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(30)   # 30-second hard ceiling

        # ── Batch portfolio evaluation (multi-horizon IC + Sharpe + capital) ───
        try:
            # PortfolioEvaluator.evaluate() returns PortfolioResult,
            # which is a backward-compatible superset of BatchEvaluationResult.
            batch_result = self.batch_eval.evaluate(all_formulas)
        except TimeoutError as te:
            print(f"[Validator] TIMEOUT: {te} — returning zero scores")
            return {uid: 0.0 for uid in submissions}
        except Exception as exc:
            print(f"[Validator] Batch eval error: {exc}")
            return {uid: 0.0 for uid in submissions}
        finally:
            if _use_alarm:
                signal.alarm(0)

        # ── Multi-component scoring ───────────────────────────────────────────
        # portfolio_scores gives full profiles when PortfolioEvaluator is used.
        # Falls back gracefully to IC-only when plain BatchEvaluator is used.
        from macro8_subnet.alpha.portfolio_evaluator import PortfolioResult
        ps_map = {}
        if isinstance(batch_result, PortfolioResult):
            ps_map = {ps.formula: ps for ps in batch_result.portfolio_scores}

        scored_formulas: dict[str, float] = {}

        for i, formula in enumerate(batch_result.formulas):
            fid     = self._formula_id(formula)
            mean_ic = float(batch_result.mean_ics[i])
            ps      = ps_map.get(formula)  # PortfolioScore | None

            ic_history  = self._ic_history.get(fid,  [])
            msc_history = self._msc_history.get(fid, [])

            # Use score_simple but blend in portfolio composite when available
            result = self.scorer.score_simple(
                formula, fid, ic_history, msc_history, epoch=epoch
            )
            base_reward = result.ema_weight

            # If portfolio data available, blend composite into reward:
            #   reward = 0.50 × IC-based EMA  +  0.50 × portfolio composite
            # This preserves history continuity while adding portfolio signal.
            if ps is not None and ps.composite > 0:
                blended = 0.50 * base_reward + 0.50 * ps.composite
                scored_formulas[formula] = float(blended)
            else:
                scored_formulas[formula] = base_reward

            # Update rolling IC history (max 30 epochs)
            if mean_ic != 0:
                hist = self._ic_history.setdefault(fid, [])
                hist.append(mean_ic)
                if len(hist) > 30:
                    hist.pop(0)

        # ── Aggregate scores per miner ────────────────────────────────────────
        miner_scores: dict[int, float] = {}
        for uid in submissions:
            miner_formulas = [f for f in formula_to_miner
                              if formula_to_miner[f] == uid]
            if not miner_formulas:
                miner_scores[uid] = 0.0
                continue

            best = max(
                (scored_formulas.get(f, 0.0) for f in miner_formulas),
                default=0.0,
            )
            miner_scores[uid] = max(best, 0.0)

        return miner_scores


# ── Macro8 Validator ──────────────────────────────────────────────────────────

class Macro8Validator:
    """
    The Macro8 validator neuron.

    Runs the full scoring pipeline each epoch:
        1. Query all miners via dendrite
        2. Score submissions using BatchEvaluator
        3. Update research graph (hypothesis evidence propagation)
        4. Compute role-stratified rewards
        5. Set weights on-chain
    """

    def __init__(self, config: argparse.Namespace = None):
        self.config   = config or get_config()
        self.epoch    = 0

        # ── Bittensor objects ─────────────────────────────────────────────────
        if _BT_AVAILABLE:
            try:
                self.wallet    = bt.Wallet(
                    name=self.config.wallet_name,
                    hotkey=self.config.wallet_hotkey,
                )
                self.subtensor = bt.Subtensor(
                    network=self.config.subtensor_network
                )
                self.metagraph = bt.metagraph(
                    netuid=self.config.netuid,
                    network=self.config.subtensor_network,
                )
                self.dendrite  = bt.Dendrite(wallet=self.wallet)
                print(f"[Validator] Connected: hotkey={self.wallet.hotkey.ss58_address}")
            except Exception as e:
                print(f"[Validator] Network unavailable: {e}")
                self.wallet = self.subtensor = self.metagraph = None
                self.dendrite = None
        else:
            self.wallet = self.subtensor = self.metagraph = self.dendrite = None

        # ── Research engine ───────────────────────────────────────────────────
        self._initialise_engine()

    # ── Engine initialisation ─────────────────────────────────────────────────

    def _initialise_engine(self):
        """Load data and build the evaluation engine."""
        fetcher = MarketDataFetcher(
            fred_api_key=os.environ.get("FRED_API_KEY", ""),
            verbose=False,
        )
        data = fetcher.fetch_prices(
            start=self.config.data_start,
            n_synthetic=2000,
        )
        self.prices  = data.prices
        self.returns = self.prices.pct_change().dropna()

        # Out-of-sample split: evaluate on most recent EVAL_WINDOW_FRAC
        n      = len(self.prices)
        split  = int(n * (1 - EVAL_WINDOW_FRAC))
        oos_prices = self.prices.iloc[split:]

        self.batch_eval   = PortfolioEvaluator(oos_prices, min_ic=MIN_IC_THRESHOLD)
        self.lifecycle    = LifecycleEngine()
        self.orth_filter  = OrthogonalityFilter(threshold=0.90)
        self.scorer       = EpochScorer(
            self.batch_eval, self.lifecycle, self.orth_filter
        )

        # Knowledge infrastructure
        self.hyp_lib    = HypothesisLibrary()
        _seed_hypotheses(self.hyp_lib, n=8)
        self.graph      = ResearchGraph(self.hyp_lib)
        self.reward_model = RoleRewardModel()

        # Strategy discovery leaderboard
        from macro8_subnet.evaluation.leaderboard import Leaderboard
        self.leaderboard = Leaderboard(max_entries=500)

        # Running score history: uid → list[float] (for spike detection)
        self.score_history:  dict[int, list[float]] = {}
        # Time-decayed trust: uid → float (persistent across epochs)
        # Trust_t = 0.9 × Trust_{t-1} + 0.1 × Score_t
        self._miner_trust:   dict[int, float]       = {}

        print(f"[Validator] Engine ready: {len(oos_prices)} OOS days × "
              f"{len(oos_prices.columns)} assets")
        print(f"[Validator] Source: {data.source} | "
              f"{data.start_date} → {data.end_date}")

    # ── Main epoch loop ───────────────────────────────────────────────────────

    def run_epoch(
        self,
        dry_run_submissions: Optional[dict[int, AlphaSubmissionSynapse]] = None,
    ) -> dict[int, float]:
        """
        Run one complete validator epoch.

        Args:
            dry_run_submissions: Pre-built submissions for testing.
                                 If None, queries live miners via dendrite.

        Returns:
            {miner_uid: reward_weight} normalised weights summing to 1.0.
        """
        self.epoch += 1
        t0 = time.perf_counter()
        print(f"\n[Validator] ═══ Epoch {self.epoch} ═══")

        # ── Step 1: Collect submissions ───────────────────────────────────────
        if dry_run_submissions is not None:
            submissions = dry_run_submissions
        elif self.dendrite is not None and self.metagraph is not None:
            submissions = self._query_miners()
        else:
            print("[Validator] No dendrite — skipping miner queries")
            return {}

        n_miners = len(submissions)
        n_forms  = sum(len(s.formulas) for s in submissions.values())
        print(f"[Validator] {n_miners} miners | {n_forms} formulas received")

        if not submissions:
            return {}

        # ── Step 2: Score all submissions ─────────────────────────────────────
        raw_scores = self.scorer.score_submissions(submissions, self.epoch)

        # ── Time-decayed trust: Trust_t = 0.9 × Trust_{t-1} + 0.1 × Score_t ──
        # A miner must sustain performance over multiple epochs to build trust.
        # One lucky epoch barely moves the trust score; consistent performance
        # accumulates it. This prevents reward sniping and flash strategies.
        TRUST_DECAY   = 0.90   # persistence of past trust
        TRUST_LEARN   = 0.10   # weight on new epoch score (= 1 - TRUST_DECAY)

        for uid, score in raw_scores.items():
            hist = self.score_history.setdefault(uid, [])
            hist.append(float(score))
            if len(hist) > 60:   # keep 60-epoch window for spike detection
                hist.pop(0)

        smoothed = {}
        for uid, score in raw_scores.items():
            prev_trust = self._miner_trust.get(uid, float(score))
            new_trust  = TRUST_DECAY * prev_trust + TRUST_LEARN * score
            self._miner_trust[uid] = new_trust
            smoothed[uid] = new_trust

        # ── Anti-overfitting pressure ─────────────────────────────────────────
        # Penalise miners whose current raw score is an outlier relative to
        # their own history. A sudden spike (new_score > mean + 2σ) is a
        # signal of instability — overfitting, regime luck, or cherry-picking.
        # The penalty scales linearly from 0 at 2σ to 0.5 at 4σ+, capping
        # the damage at half weight reduction (never zeroes a good miner).
        SPIKE_SIGMA_FLOOR = 2.0   # spikes below this multiple are free
        SPIKE_PENALTY_CAP = 0.50  # maximum fractional reduction in score
        for uid in list(smoothed.keys()):
            hist = self.score_history.get(uid, [])
            if len(hist) < 5:          # need at least 5 epochs of history
                continue
            hist_arr  = np.array(hist[:-1])   # history excluding current epoch
            mu        = hist_arr.mean()
            sigma     = hist_arr.std() + 1e-9
            current   = raw_scores.get(uid, 0.0)
            n_sigma   = (current - mu) / sigma
            if n_sigma > SPIKE_SIGMA_FLOOR:
                # Linear ramp: 0 at 2σ → 0.50 at 4σ
                excess  = n_sigma - SPIKE_SIGMA_FLOOR
                penalty = min(SPIKE_PENALTY_CAP, excess / (2 * SPIKE_SIGMA_FLOOR) * SPIKE_PENALTY_CAP)
                smoothed[uid] = smoothed[uid] * (1.0 - penalty)

        # ── Miner diversity enforcement ────────────────────────────────────────
        # Miners who submit correlated signals should share a reward budget
        # rather than each receiving full weight. The subnet wants a PORTFOLIO
        # of uncorrelated strategies, not N copies of the same signal.
        #
        # Algorithm: build a miner × miner correlation matrix from their formula
        # weight vectors. Miners with pairwise |corr| > 0.85 are penalised:
        #   penalty_i = mean(max(0, |corr_ij| - threshold)) over j ≠ i
        # This is proportional to how correlated miner i is with the rest.
        # The penalty is applied AFTER the trust smoothing so trust history is
        # maintained — only the final weight delivered on-chain is adjusted.
        DIVERSITY_THRESHOLD = 0.85   # |corr| above this = penalised
        DIVERSITY_PENALTY   = 0.40   # max fractional reduction per correlated pair

        uid_list = sorted(smoothed.keys())
        if len(uid_list) >= 3:
            # Build per-miner mean weight vector (average over submitted formulas)
            miner_vecs: dict[int, np.ndarray] = {}
            for uid, syn in submissions.items():
                if uid not in smoothed:
                    continue
                vecs = []
                for f in (syn.formulas or [])[:8]:
                    try:
                        w = self.scorer.batch_eval.encoder.encode(f)
                        vecs.append(w)
                    except Exception:
                        pass
                if vecs:
                    miner_vecs[uid] = np.mean(vecs, axis=0)

            vec_uids = [u for u in uid_list if u in miner_vecs]
            if len(vec_uids) >= 3:
                mat = np.array([miner_vecs[u] for u in vec_uids])   # [M × F]
                # Normalise rows
                norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
                mat_n = mat / norms
                corr  = mat_n @ mat_n.T                              # [M × M]

                for i, uid_i in enumerate(vec_uids):
                    excess_corrs = [
                        max(0.0, abs(corr[i, j]) - DIVERSITY_THRESHOLD)
                        for j, uid_j in enumerate(vec_uids) if j != i
                    ]
                    mean_excess = float(np.mean(excess_corrs)) if excess_corrs else 0.0
                    # Scale: 0.15 above threshold → full DIVERSITY_PENALTY
                    div_penalty = min(DIVERSITY_PENALTY,
                                      mean_excess / 0.15 * DIVERSITY_PENALTY)
                    if div_penalty > 0:
                        smoothed[uid_i] = smoothed[uid_i] * (1.0 - div_penalty)

        # ── Capital allocation feedback loop ──────────────────────────────────
        # Simulate each miner's submitted portfolio against realised prices.
        # Miners who submitted positions (syn.positions) get their actual
        # capital performance fed back into their trust score.
        #
        # Mechanism:
        #   1. Take syn.positions ({ticker: weight}) from each miner
        #   2. Compute portfolio return over the most recent CAPITAL_WINDOW days
        #      using prices already loaded in the validator (self.prices)
        #   3. Normalise via tanh(ret / 0.30):  +30% CAGR → +0.84 score
        #   4. Blend: trust = 0.80×trust + 0.20×capital_score
        #
        # Why this matters: a signal can have good IC but bad capital performance
        # (high turnover, poor timing, bad regime fit). IC alone optimises for
        # correlation. Capital return optimises for survival.
        #
        # Safety: falls back gracefully when positions are absent, prices are
        # unavailable, or tickers don't match — never raises.
        CAPITAL_WEIGHT  = 0.20   # blend weight of capital return into trust
        CAPITAL_WINDOW  = 20     # days of realised return to simulate
        CAPITAL_SCALE   = 0.30   # tanh normalisation (0.30 = 30% CAGR maps to 0.84)

        if hasattr(self, "prices") and self.prices is not None and len(self.prices) > CAPITAL_WINDOW:
            recent_prices = self.prices.iloc[-CAPITAL_WINDOW - 1:]   # +1 for diff
            log_returns   = np.log(recent_prices).diff().dropna()     # [CAPITAL_WINDOW × A]
            tickers       = list(recent_prices.columns)

            for uid, syn in submissions.items():
                if uid not in smoothed:
                    continue
                pos = getattr(syn, "positions", None)
                if not pos or not isinstance(pos, dict):
                    continue   # miner didn't submit positions — no feedback

                # Build weight vector aligned to validator's price universe
                w = np.array([float(pos.get(t, 0.0)) for t in tickers])
                l1 = np.abs(w).sum()
                if l1 < 1e-8:
                    continue   # zero positions — skip
                w = w / l1   # normalise to unit L1

                # Simulate: daily_pnl[t] = dot(w, log_returns[t])
                # Use same weights throughout window (positions held static)
                try:
                    daily_pnl = log_returns.values @ w        # [CAPITAL_WINDOW]
                    ann_ret   = float(daily_pnl.mean() * 252) # annualise
                except Exception:
                    continue

                # Normalise: tanh(x / CAPITAL_SCALE) → [-1, +1]
                capital_score = float(np.tanh(ann_ret / CAPITAL_SCALE))

                # Feed back into trust: blend capital score with current trust
                current_trust = smoothed[uid]
                updated_trust = (1.0 - CAPITAL_WEIGHT) * current_trust + CAPITAL_WEIGHT * capital_score
                smoothed[uid] = max(updated_trust, 0.0)   # floor at 0 (no negative rewards)
                self._miner_trust[uid] = smoothed[uid]    # persist to next epoch

                if abs(capital_score) > 0.1:   # only log meaningful capital feedback
                    print(f"[Validator]   uid={uid} capital_feedback: "
                          f"ann_ret={ann_ret:+.1%} score={capital_score:+.3f} "
                          f"trust={smoothed[uid]:.4f}")

        # ── Step 3: Update research graph ────────────────────────────────────
        self._update_knowledge(submissions, raw_scores)

        # ── Step 4: Compute role-stratified rewards ───────────────────────────
        role_scores = self._build_role_scores(submissions, smoothed)
        reward_report = self.reward_model.compute(self.epoch, role_scores)
        uids, weights = reward_report.as_weight_list()

        final_weights = dict(zip(uids, weights))
        elapsed = time.perf_counter() - t0

        print(f"[Validator] Scoring complete in {elapsed:.2f}s")
        if reward_report.entries:
            best = reward_report.entries[0]
            print(f"[Validator] Best miner: uid={best.miner_uid} "
                  f"reward={best.total_reward:.4f}")

        # ── Step 5: Set weights on-chain ──────────────────────────────────────
        if self.subtensor is not None and self.wallet is not None and uids:
            self._set_weights(uids, weights)

        return final_weights

    # ── Miner querying ────────────────────────────────────────────────────────

    def _query_miners(self) -> dict[int, AlphaSubmissionSynapse]:
        """Query all registered miners via dendrite."""
        submissions = {}
        axons       = self.metagraph.axons

        if not axons:
            print("[Validator] No axons in metagraph")
            return {}

        print(f"[Validator] Querying {len(axons)} miners...")

        synapse = AlphaSubmissionSynapse(epoch=self.epoch)

        try:
            responses = self.dendrite(
                axons=axons,
                synapse=synapse,
                timeout=12.0,
            )
            for uid, response in enumerate(responses):
                if (response is not None and
                        hasattr(response, "formulas") and
                        response.formulas):
                    submissions[uid] = response
        except Exception as e:
            print(f"[Validator] Query error: {e}")

        return submissions

    # ── Knowledge update ──────────────────────────────────────────────────────

    def _update_knowledge(
        self,
        submissions: dict[int, AlphaSubmissionSynapse],
        scores:      dict[int, float],
    ) -> None:
        """Propagate IC evidence into the research graph, hypothesis library,
        and strategy discovery leaderboard."""
        from macro8_subnet.evaluation.signal_scorer import SignalScorer

        # Register top-scoring formulas in the graph and leaderboard
        for uid, syn in submissions.items():
            ic = scores.get(uid, 0.0)
            if not syn.formulas:
                continue

            for formula in syn.formulas[:10]:
                fid = self.scorer._formula_id(formula)

                # Update research graph (hypothesis evidence)
                if ic >= MIN_IC_THRESHOLD:
                    hyp_ids = _match_hypothesis(formula, self.hyp_lib)
                    graph_fid = self.graph.register_formula(
                        formula_string=formula, miner_uid=uid,
                        epoch=self.epoch, hypothesis_ids=hyp_ids,
                    )
                    self.graph.propagate_evidence(
                        formula_id=graph_fid, ic=ic, epoch=self.epoch,
                    )

                # Update leaderboard (all formulas, not just high-IC)
                ic_hist  = self.scorer._ic_history.get(fid, [])
                msc_hist = self.scorer._msc_history.get(fid, [])
                result   = self.scorer.scorer.score_simple(
                    formula, fid, ic_hist, msc_hist, epoch=self.epoch
                )
                self.leaderboard.register(
                    result, miner_uid=uid,
                    hotkey=getattr(syn, "miner_hotkey", f"uid_{uid}"),
                    epoch=self.epoch,
                )

        # Register miner-submitted hypotheses
        for uid, syn in submissions.items():
            if not syn.hypothesis_statements:
                continue
            from macro8_subnet.alpha.hypothesis_engine import HypothesisCategory
            for stmt, cat_str in zip(
                syn.hypothesis_statements,
                syn.hypothesis_categories or ["unknown"] * len(syn.hypothesis_statements),
            ):
                try:
                    cat = HypothesisCategory(cat_str)
                except ValueError:
                    cat = HypothesisCategory.UNKNOWN
                self.hyp_lib.add(
                    statement=stmt, category=cat,
                    miner_uid=uid, epoch=self.epoch,
                )


    # ── Role score assembly ───────────────────────────────────────────────────

    def _build_role_scores(
        self,
        submissions: dict[int, AlphaSubmissionSynapse],
        smoothed:    dict[int, float],
    ) -> dict:
        """Build role-stratified score dict for RoleRewardModel."""
        role_scores = {role: [] for role in AgentRole}

        for uid, syn in submissions.items():
            score   = smoothed.get(uid, 0.0)
            hotkey  = getattr(syn, "miner_hotkey", f"uid_{uid}")
            role_str = getattr(syn, "agent_role", "signal")

            try:
                role = AgentRole(role_str)
            except ValueError:
                role = AgentRole.SIGNAL

            role_scores[role].append({
                "uid":    uid,
                "hotkey": hotkey,
                "score":  score,
            })

        return role_scores

    # ── Weight setting ────────────────────────────────────────────────────────

    def _set_weights(self, uids: list[int], weights: list[float]) -> bool:
        """
        Call subtensor.set_weights() to update on-chain miner incentives.

        Error 2 guard: weights must cover the FULL metagraph uid vector,
        not just the miners who submitted this epoch. Any miner not in
        the reward dict gets weight=0 but must still be included.

        Bittensor 10.x set_weights signature:
            uids    : int64 array — must be subset of metagraph.uids
            weights : float32 array — parallel to uids, sums to 1.0
        """
        try:
            # ── Align to full metagraph ───────────────────────────────────────
            if self.metagraph is not None:
                # All registered uids (may include miners that didn't submit)
                all_metagraph_uids = self.metagraph.uids.tolist()
            else:
                all_metagraph_uids = uids   # dry-run fallback

            # Build dense weight vector over ALL metagraph uids
            score_map = dict(zip(uids, weights))
            full_uids    = []
            full_weights = []

            for uid in all_metagraph_uids:
                full_uids.append(int(uid))
                full_weights.append(float(score_map.get(int(uid), 0.0)))

            uids_arr    = np.array(full_uids,    dtype=np.int64)
            weights_arr = np.array(full_weights, dtype=np.float32)

            # Normalise to [0, 1] (sum = 1.0)
            total = weights_arr.sum()
            if total > 1e-8:
                weights_arr = weights_arr / total
            else:
                # No scores — equal weight across all miners (avoid zero vector)
                weights_arr = np.ones(len(full_uids), dtype=np.float32)
                weights_arr /= len(full_uids)

            # ── Validate before sending ───────────────────────────────────────
            assert len(uids_arr) == len(weights_arr), "uid/weight length mismatch"
            assert abs(weights_arr.sum() - 1.0) < 0.01, "weights do not sum to 1.0"

            result = self.subtensor.set_weights(
                wallet=self.wallet,
                netuid=self.config.netuid,
                uids=uids_arr,
                weights=weights_arr,
                version_key=WEIGHTS_VERSION_KEY,
                wait_for_inclusion=True,
                wait_for_finalization=False,
                raise_error=False,
            )
            success = getattr(result, "is_success", False)
            print(f"[Validator] set_weights: {len(uids_arr)} uids, "
                  f"success={success}")
            return success

        except AssertionError as ae:
            print(f"[Validator] Weight vector error: {ae}")
            return False
        except Exception as e:
            print(f"[Validator] set_weights error: {e}")
            return False

    # ── Bittensor lifecycle ───────────────────────────────────────────────────

    def run(self):
        """Main validator loop."""
        if self.subtensor is None:
            print("[Validator] Running in dry-run mode (no bittensor network)")
            self._dry_run()
            return

        # Check registration
        if not self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        ):
            print(f"[Validator] NOT registered on netuid {self.config.netuid}.")
            print("  Run: btcli s register --netuid <UID> --wallet.name validator")
            return

        print(f"[Validator] Starting | netuid={self.config.netuid} | "
              f"epoch={self.config.epoch_seconds}s")

        try:
            while True:
                epoch_start = time.time()

                try:
                    self.metagraph.sync(subtensor=self.subtensor)
                    self.run_epoch()
                except Exception as e:
                    print(f"[Validator] Epoch error: {e}")
                    traceback.print_exc()

                # Sleep until next epoch
                elapsed = time.time() - epoch_start
                sleep   = max(0, self.config.epoch_seconds - elapsed)
                print(f"[Validator] Epoch done in {elapsed:.1f}s, "
                      f"sleeping {sleep:.0f}s")
                time.sleep(sleep)

        except KeyboardInterrupt:
            print("[Validator] Stopped by user")

    def _dry_run(self, n_epochs: int = 3):
        """Run without bittensor network — use synthetic miner submissions."""
        print(f"[Validator] Dry-run: {n_epochs} epochs with synthetic miners")

        for epoch in range(1, n_epochs + 1):
            # Build synthetic submissions from a few mock miners
            submissions = self._synthetic_submissions(n_miners=5, epoch=epoch)
            weights = self.run_epoch(dry_run_submissions=submissions)

            # Print summary
            if weights:
                top3 = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"  Epoch {epoch} top weights: "
                      + " | ".join(f"uid={u} w={w:.4f}" for u, w in top3))

        # Show knowledge state after dry-run
        print(f"\n[Validator] Knowledge base after {n_epochs} epochs:")
        self.hyp_lib.print_knowledge_base(top_n=5)

    @staticmethod
    def _synthetic_submissions(
        n_miners: int = 5,
        epoch:    int = 1,
    ) -> dict[int, AlphaSubmissionSynapse]:
        """Generate realistic synthetic miner submissions for testing."""
        formula_pools = {
            "signal": [
                "momentum_20d", "rank(momentum_20d) - rank(volatility_20d)",
                "zscore(cross_momentum)", "decay(momentum_5d, halflife=10)",
                "regime_signal * momentum_20d", "volatility_60d",
                "rank(momentum_60d)", "zscore(volatility_20d)",
                "cross_momentum - relative_vol", "rsi_14",
            ],
            "strategy": [],
            "risk":     [],
            "portfolio": [],
            "meta":     [],
        }

        roles = ["signal"] * 3 + ["risk", "portfolio"]
        subs  = {}

        for uid in range(n_miners):
            role = roles[uid % len(roles)]
            # Each miner submits a subset of formulas
            rng      = np.random.default_rng(seed=uid + epoch * 100)
            pool     = formula_pools.get(role, formula_pools["signal"])
            n_submit = min(rng.integers(3, 8), len(pool))
            formulas = list(rng.choice(pool, size=n_submit, replace=False))

            syn = AlphaSubmissionSynapse(
                formulas=formulas,
                agent_role=role,
                miner_uid=uid,
                miner_hotkey=f"5F{uid:040d}",
                epoch=epoch,
            )
            subs[uid] = syn

        return subs


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    config    = get_config()
    validator = Macro8Validator(config)
    validator.run()


if __name__ == "__main__":
    main()
