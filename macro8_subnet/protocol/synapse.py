"""
protocol/synapse.py
--------------------
Bittensor Synapse definitions for the Macro8 subnet.

The Synapse is the wire protocol — the typed message that miners and
validators exchange over the Bittensor network. Every subnet defines
its own Synapse subclass that carries the domain-specific payload.

Macro8 defines three synapses:

1. AlphaSubmissionSynapse
   Miner → Validator
   Carries: formula strings, hypothesis submissions, agent role
   Validator evaluates IC, updates hypothesis confidence, issues rewards

2. MarketDataSynapse
   Validator → Miner (optional)
   Carries: current prices, feature tensor summary
   Allows validators to push market context to miners

3. RewardSynapse
   Validator → Miner
   Carries: reward weights from the epoch
   Not strictly needed (weights go on-chain), but useful for feedback

Bittensor 10.x uses pydantic v2 for Synapse field validation.
All fields must be JSON-serialisable (str, int, float, list, dict).

Usage (miner side)
------------------
    import bittensor as bt
    from macro8_subnet.protocol.synapse import AlphaSubmissionSynapse

    synapse = AlphaSubmissionSynapse(
        formulas=["rank(momentum_20d)", "zscore(cross_momentum)"],
        agent_role="signal",
        miner_uid=0,
    )
    # Send via dendrite
    response = await dendrite(axon, synapse, deserialize=True)

Usage (validator side)
----------------------
    def forward(synapse: AlphaSubmissionSynapse) -> AlphaSubmissionSynapse:
        # Evaluate formulas
        ics = evaluate_batch(synapse.formulas)
        synapse.ic_scores = ics
        synapse.reward_signal = compute_reward(ics)
        return synapse
    axon.attach(forward_fn=forward, blacklist_fn=blacklist)
"""

from __future__ import annotations

from typing import Optional

try:
    import bittensor as bt
    _BT_AVAILABLE = True
except ImportError:
    _BT_AVAILABLE = False


# ── Alpha Submission Synapse ──────────────────────────────────────────────────

if _BT_AVAILABLE:
    class AlphaSubmissionSynapse(bt.Synapse):
        """
        Miner → Validator: submit alpha formulas and/or hypotheses.

        This is the primary synapse. Miners use it to:
            - Submit formula strings for IC evaluation
            - Submit hypothesis statements for registration
            - Indicate their agent role for role-stratified rewards
            - Submit prediction market positions

        The validator fills in ic_scores and reward_signal before
        returning the synapse to the miner.
        """
        # ── Miner fills these ─────────────────────────────────────────────────
        formulas:        list[str]           = []
        agent_role:      str                 = "signal"   # signal|strategy|risk|portfolio|meta
        miner_uid:       int                 = 0
        miner_hotkey:    str                 = ""
        epoch:           int                 = 0

        # Hypothesis submissions (optional)
        hypothesis_statements: list[str]     = []
        hypothesis_categories: list[str]     = []

        # Signal weights for STRATEGY role
        signal_weights:  dict[str, float]    = {}

        # Covariance model for RISK role
        risk_payload:    dict                = {}

        # Portfolio constraints for PORTFOLIO role
        portfolio_payload: dict              = {}

        # IC predictions for META role
        ic_predictions:  dict[str, float]    = {}

        # Prediction market positions
        market_positions: list[dict]         = []   # [{signal, direction, stake, confidence}]

        # ── Signal layer (Sprint 26) — actual trade positions ─────────────────
        # {ticker: weight} from the best formula's latest cross-sectional signal.
        # Positive = long, negative = short. L1 norm = 1.0.
        # Example: {"SPY": +0.18, "TLT": -0.12, "GLD": +0.08, ...}
        positions:       dict[str, float]    = {}
        position_formula: str               = ""   # formula that generated positions

        # ── Validator fills these ─────────────────────────────────────────────
        ic_scores:       list[float]         = []   # one per formula
        ic_irs:          list[float]         = []   # IC information ratios
        reward_signal:   float               = 0.0  # normalised reward weight
        eval_success:    bool                = False
        eval_error:      str                 = ""

        def deserialize(self) -> "AlphaSubmissionSynapse":
            return self

    class MarketDataSynapse(bt.Synapse):
        """
        Validator → Miner: push current market context.

        Allows validators to optionally inform miners of the current
        market state (feature summary, library state) so miners can
        generate better formulas.

        This synapse is optional — miners can operate with no context.
        """
        # Current epoch info
        epoch:           int                  = 0
        n_active_signals: int                 = 0
        library_top_ics: list[float]          = []   # top 10 ICs in library

        # Feature summary (no raw data, just statistics)
        feature_means:   dict[str, float]     = {}   # feature → cross-asset mean
        regime_label:    str                  = ""   # current regime

        # Hypothesis confidence summary
        top_hypotheses:  list[dict]           = []   # [{statement, confidence}]

        def deserialize(self) -> "MarketDataSynapse":
            return self

    class RewardSynapse(bt.Synapse):
        """
        Validator → Miner: communicate reward allocation.

        Not strictly necessary (weights go on-chain via set_weights),
        but useful so miners can observe their reward in real-time
        and adjust their submission strategy.
        """
        epoch:           int                  = 0
        miner_uid:       int                  = 0
        ic_reward:       float                = 0.0
        msc_reward:      float                = 0.0
        total_reward:    float                = 0.0
        rank:            int                  = 0
        n_miners:        int                  = 0

        def deserialize(self) -> "RewardSynapse":
            return self

else:
    # Fallback shims when bittensor is not installed
    class AlphaSubmissionSynapse:
        def __init__(self, **kwargs):
            self.formulas         = kwargs.get("formulas", [])
            self.agent_role       = kwargs.get("agent_role", "signal")
            self.miner_uid        = kwargs.get("miner_uid", 0)
            self.miner_hotkey     = kwargs.get("miner_hotkey", "")
            self.epoch            = kwargs.get("epoch", 0)
            self.hypothesis_statements = kwargs.get("hypothesis_statements", [])
            self.hypothesis_categories = kwargs.get("hypothesis_categories", [])
            self.signal_weights   = kwargs.get("signal_weights", {})
            self.risk_payload     = kwargs.get("risk_payload", {})
            self.portfolio_payload = kwargs.get("portfolio_payload", {})
            self.ic_predictions   = kwargs.get("ic_predictions", {})
            self.market_positions = kwargs.get("market_positions", [])
            self.ic_scores        = kwargs.get("ic_scores", [])
            self.ic_irs           = kwargs.get("ic_irs", [])
            self.reward_signal    = kwargs.get("reward_signal", 0.0)
            self.eval_success     = kwargs.get("eval_success", False)
            self.eval_error       = kwargs.get("eval_error", "")

    class MarketDataSynapse:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class RewardSynapse:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


__all__ = [
    "AlphaSubmissionSynapse",
    "MarketDataSynapse",
    "RewardSynapse",
]
