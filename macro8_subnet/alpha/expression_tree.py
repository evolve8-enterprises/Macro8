"""
alpha/expression_tree.py
-------------------------
Expression Trees for True Symbolic Regression in Macro8.

The problem with string formulas
---------------------------------
String formulas like "rank(momentum_20d - volatility_20d)" look nested,
but the FormulaEncoder maps them to flat linear weight vectors. The rank()
wrapper is completely ignored — rank(x) and x get the same weight vector.

This means the GP can only discover linear combinations of base features.
Cross-sectional operations (rank, zscore applied to sub-expressions) are
decorative, not functional.

Expression trees fix this
--------------------------
An expression tree represents a formula as a computation graph.
Each node has a type and knows how to evaluate itself on time-series data:

    Tree for: rank(momentum_20d - volatility_20d)

        Rank
          │
        BinOp(Sub)
        /          \\
    Feature        Feature
    momentum_20d   volatility_20d

Evaluation traverses the tree bottom-up:
    1. Load momentum_20d DataFrame [T × A]
    2. Load volatility_20d DataFrame [T × A]
    3. Subtract element-wise → spread [T × A]
    4. Rank cross-sectionally at each time step → ranked_spread [T × A]

This produces a genuinely different signal from:
    rank(momentum_20d) - rank(volatility_20d)

    BinOp(Sub)
    /          \\
  Rank         Rank
   │             │
 Feature      Feature
 momentum_20d  volatility_20d

    1. Load and rank momentum_20d separately
    2. Load and rank volatility_20d separately
    3. Subtract the two ranked series

The cross-sectional rank of a spread is a different signal from
the difference of two cross-sectional ranks. This is what makes
true symbolic regression more powerful than string encoding.

Tree nodes
-----------
    FeatureNode     — leaf: returns a feature DataFrame
    UnaryNode       — rank(), zscore(), decay()
    BinaryNode      — +, -, *, /
    ConstantNode    — numeric literal (for decay halflife, etc.)

Tree evaluation
----------------
Every node's eval() method returns a pd.DataFrame [time × assets].
The tree evaluates bottom-up (children before parents).

Tree → string conversion
--------------------------
Trees serialise to the same formula string format used by the validator.
String → tree parsing (optional — trees are primarily created by the GP).

GP integration
---------------
Trees replace string formulas as the unit of evolution.
Mutation and crossover operate on tree nodes directly, producing
structurally valid trees rather than string manipulations.

Tree → validator compatibility
--------------------------------
TreeEvaluator.to_formula_string() produces an AST-safe string that
passes safe_formula() and can be sent in AlphaSubmissionSynapse.
The validator scores trees by evaluating their string representations
(which still go through BatchEvaluator for speed).

The advantage is that the GP now explores a space of tree structures
that includes operations the linear encoder cannot represent, then
submits the winning formulas as strings for validator scoring.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd

_SUBNET = Path(__file__).resolve().parent.parent
_ROOT   = _SUBNET.parent
for p in [str(_SUBNET), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ── Node types ────────────────────────────────────────────────────────────────

class ExprNode:
    """Base class for all expression tree nodes."""

    def eval(self, context: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Evaluate this node given a feature context. Returns [T × A] DataFrame."""
        raise NotImplementedError

    def to_string(self) -> str:
        """Serialise to formula string (AST-safe, validator-compatible)."""
        raise NotImplementedError

    def depth(self) -> int:
        """Maximum depth of tree rooted at this node."""
        raise NotImplementedError

    def size(self) -> int:
        """Number of nodes in tree rooted at this node."""
        raise NotImplementedError

    def clone(self) -> "ExprNode":
        """Deep copy of this node."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.to_string()})"


@dataclass
class FeatureNode(ExprNode):
    """Leaf node: loads a feature from the context."""
    name: str

    def eval(self, context: dict[str, pd.DataFrame]) -> pd.DataFrame:
        df = context.get(self.name)
        if df is None:
            raise ValueError(f"Feature '{self.name}' not in context")
        return df.copy()

    def to_string(self) -> str:
        return self.name

    def depth(self) -> int:
        return 0

    def size(self) -> int:
        return 1

    def clone(self) -> "FeatureNode":
        return FeatureNode(self.name)


@dataclass
class ConstantNode(ExprNode):
    """Leaf node: a numeric constant."""
    value: float

    def eval(self, context: dict[str, pd.DataFrame]) -> pd.DataFrame:
        # Return a scalar — wrapping as DataFrame done by parent node
        return self.value  # type: ignore

    def to_string(self) -> str:
        return str(int(self.value)) if self.value == int(self.value) else str(self.value)

    def depth(self) -> int:
        return 0

    def size(self) -> int:
        return 1

    def clone(self) -> "ConstantNode":
        return ConstantNode(self.value)


@dataclass
class UnaryNode(ExprNode):
    """
    Unary operator node: rank, zscore, decay.

    rank():  cross-sectional rank at each time step (0 to 1)
    zscore(): cross-sectional z-score at each time step
    decay():  exponential weighted moving average (halflife parameter)
    """
    op:      str         # "rank" | "zscore" | "decay"
    child:   ExprNode
    halflife: float = 10.0   # used only for decay

    def eval(self, context: dict[str, pd.DataFrame]) -> pd.DataFrame:
        child_val = self.child.eval(context)
        if not isinstance(child_val, pd.DataFrame):
            return child_val

        if self.op == "rank":
            return self._rank_cs(child_val)
        if self.op == "zscore":
            return self._zscore_cs(child_val)
        if self.op == "decay":
            return child_val.ewm(halflife=self.halflife).mean()
        raise ValueError(f"Unknown unary op: {self.op}")

    @staticmethod
    def _rank_cs(df: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional rank at each time step: values in (0, 1]."""
        return df.rank(axis=1, pct=True)

    @staticmethod
    def _zscore_cs(df: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional z-score at each time step."""
        mean = df.mean(axis=1)
        std  = df.std(axis=1).replace(0, np.nan)
        return df.subtract(mean, axis=0).divide(std, axis=0)

    def to_string(self) -> str:
        if self.op == "decay":
            hl = int(self.halflife) if self.halflife == int(self.halflife) else self.halflife
            return f"decay({self.child.to_string()}, halflife={hl})"
        return f"{self.op}({self.child.to_string()})"

    def depth(self) -> int:
        return 1 + self.child.depth()

    def size(self) -> int:
        return 1 + self.child.size()

    def clone(self) -> "UnaryNode":
        return UnaryNode(self.op, self.child.clone(), self.halflife)


@dataclass
class BinaryNode(ExprNode):
    """
    Binary operator node: +, -, *, /.

    Operations are element-wise on [T × A] DataFrames.
    Division guards against zero denominators.
    """
    op:    str       # "+" | "-" | "*" | "/"
    left:  ExprNode
    right: ExprNode

    def eval(self, context: dict[str, pd.DataFrame]) -> pd.DataFrame:
        lval = self.left.eval(context)
        rval = self.right.eval(context)

        # Ensure both are DataFrames (constants become scalar)
        if isinstance(lval, pd.DataFrame) and isinstance(rval, pd.DataFrame):
            if self.op == "+":  return lval + rval
            if self.op == "-":  return lval - rval
            if self.op == "*":  return lval * rval
            if self.op == "/":  return lval / rval.replace(0, np.nan)
        raise ValueError(f"Cannot apply {self.op} to {type(lval)}, {type(rval)}")

    def to_string(self) -> str:
        ls = self.left.to_string()
        rs = self.right.to_string()
        # Wrap sub-expressions in parens if they contain operators
        if isinstance(self.left, BinaryNode):
            ls = f"({ls})"
        if isinstance(self.right, BinaryNode):
            rs = f"({rs})"
        return f"{ls} {self.op} {rs}"

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())

    def size(self) -> int:
        return 1 + self.left.size() + self.right.size()

    def clone(self) -> "BinaryNode":
        return BinaryNode(self.op, self.left.clone(), self.right.clone())


# ── Tree Evaluator ────────────────────────────────────────────────────────────

class TreeEvaluator:
    """
    Evaluates expression trees against market data.

    Provides IC computation for individual trees and batch evaluation
    for populations of trees.
    """

    def __init__(
        self,
        prices:      pd.DataFrame,
        min_obs:     int   = 10,
        lag:         int   = 1,
    ):
        from macro8_subnet.alpha.feature_store import FeatureStore
        self.prices    = prices
        self.returns   = prices.pct_change().dropna()
        self.min_obs   = min_obs
        self.lag       = lag

        # Build feature context once
        fs = FeatureStore(prices)
        self._context: dict[str, pd.DataFrame] = fs.build()

    def eval_tree(self, tree: ExprNode) -> Optional[pd.DataFrame]:
        """
        Evaluate a tree and return the signal DataFrame [T × A].
        Returns None if evaluation fails.
        """
        try:
            result = tree.eval(self._context)
            if not isinstance(result, pd.DataFrame):
                return None
            return result
        except Exception:
            return None

    def ic_of_tree(self, tree: ExprNode) -> float:
        """
        Compute mean cross-sectional IC for a tree.

        Uses Spearman rank correlation between signal at t and return at t+lag.
        This is the same metric the validator uses, ensuring GP optimises
        exactly what is scored.
        """
        signal = self.eval_tree(tree)
        if signal is None:
            return 0.0
        return self._compute_ic(signal)

    def _compute_ic(self, signal: pd.DataFrame) -> float:
        """Mean cross-sectional Spearman IC."""
        try:
            # Align signal and forward returns
            fwd_ret = self.returns.shift(-self.lag)
            common  = signal.index.intersection(fwd_ret.index)
            if len(common) < self.min_obs:
                return 0.0

            sig_aligned = signal.loc[common].dropna()
            ret_aligned = fwd_ret.loc[sig_aligned.index].dropna()
            common2     = sig_aligned.index.intersection(ret_aligned.index)

            if len(common2) < self.min_obs:
                return 0.0

            s = sig_aligned.loc[common2]
            r = ret_aligned.loc[common2]

            ic_series = []
            for t in common2:
                s_t = s.loc[t]
                r_t = r.loc[t]
                valid = s_t.notna() & r_t.notna()
                if valid.sum() < 2:
                    continue
                # Spearman: rank both then correlate
                s_rank = s_t[valid].rank()
                r_rank = r_t[valid].rank()
                n      = valid.sum()
                # Pearson on ranks = Spearman
                s_c = s_rank - s_rank.mean()
                r_c = r_rank - r_rank.mean()
                num = (s_c * r_c).sum()
                den = np.sqrt((s_c**2).sum() * (r_c**2).sum())
                if den > 1e-8:
                    ic_series.append(num / den)

            return float(np.mean(ic_series)) if ic_series else 0.0
        except Exception:
            return 0.0

    def batch_ic(self, trees: list[ExprNode]) -> list[float]:
        """
        Compute IC for a list of trees.
        Returns list of float, one per tree.
        """
        return [self.ic_of_tree(t) for t in trees]

    def to_formula_string(self, tree: ExprNode) -> Optional[str]:
        """
        Convert tree to validator-compatible formula string.

        Returns None if the string would fail safe_formula().
        """
        try:
            formula = tree.to_string()
            from macro8_subnet.neurons.validator import safe_formula
            return safe_formula(formula)
        except Exception:
            return None


# ── Tree Builder (for GP) ─────────────────────────────────────────────────────

class TreeBuilder:
    """
    Builds random expression trees for GP initialisation.

    Trees are constructed to be:
    - Valid (all leaves are known features)
    - Safe (depth ≤ MAX_TREE_DEPTH)
    - AST-safe when serialised to strings
    """

    MAX_TREE_DEPTH = 4
    UNARY_OPS      = ["rank", "zscore"]
    DECAY_HALFLIVES = [5, 10, 15, 20, 30]
    BINARY_OPS     = ["+", "-", "*"]   # / excluded (zero-denominator risk)

    def __init__(
        self,
        features:     list[str],
        seed:         Optional[int] = None,
    ):
        from macro8_subnet.alpha.gp_miner import GP_FEATURES
        self.features  = features or GP_FEATURES
        import random
        self._rng      = random.Random(seed)

    def random_tree(self, depth: int = 2) -> ExprNode:
        """Generate a random valid expression tree."""
        return self._build(depth)

    def _build(self, depth: int) -> ExprNode:
        if depth == 0:
            return self._leaf()

        r = self._rng.random()

        if r < 0.30:
            # Leaf
            return self._leaf()

        if r < 0.50:
            # Unary node — key for cross-sectional ops
            op    = self._rng.choice(self.UNARY_OPS)
            child = self._build(depth - 1)
            return UnaryNode(op, child)

        if r < 0.60:
            # Decay
            hl    = float(self._rng.choice(self.DECAY_HALFLIVES))
            child = self._build(max(depth - 2, 0))
            return UnaryNode("decay", child, halflife=hl)

        # Binary node
        op    = self._rng.choice(self.BINARY_OPS)
        left  = self._build(depth - 1)
        right = self._build(depth - 1)
        return BinaryNode(op, left, right)

    def _leaf(self) -> FeatureNode:
        return FeatureNode(self._rng.choice(self.features))

    def initial_population(
        self,
        n: int,
        seed_trees: list[ExprNode] = (),
    ) -> list[ExprNode]:
        """Build initial GP population of n trees."""
        pop = list(seed_trees[:n])
        depths = [1] * (n // 3) + [2] * (n // 3) + [3] * (n // 3 + 1)
        self._rng.shuffle(depths)
        while len(pop) < n:
            d = depths[len(pop) % len(depths)]
            pop.append(self.random_tree(d))
        return pop[:n]


# ── Tree Genetic Operators ────────────────────────────────────────────────────

class TreeGeneticOps:
    """
    Mutation and crossover for expression trees.

    Tree-level operations are semantically meaningful:
    - Subtree mutation replaces a random subtree with a fresh one
    - Point mutation changes a leaf feature or an operator
    - Subtree crossover swaps subtrees between two parents

    These produce valid trees by construction — no string parsing needed.
    """

    def __init__(self, builder: TreeBuilder):
        self._builder = builder
        self._rng     = builder._rng

    def mutate(self, tree: ExprNode) -> ExprNode:
        """Apply one of three mutations, returning a new valid tree."""
        r = self._rng.random()
        if r < 0.25:
            # Full replacement
            return self._builder.random_tree(self._rng.randint(1, 3))
        if r < 0.60:
            # Point mutation: change a leaf or operator
            return self._point_mutate(tree.clone())
        # Subtree mutation: replace random subtree
        return self._subtree_mutate(tree.clone())

    def crossover(self, t1: ExprNode, t2: ExprNode) -> ExprNode:
        """
        Subtree crossover: take t1 and replace a random subtree with
        a random subtree from t2.

        Preserves tree validity (both parents are valid by construction).
        Falls back to a binary combination if depth limit would be exceeded.
        """
        child  = t1.clone()
        donor  = t2.clone()

        # Simple crossover: wrap in binary op if both are small enough
        if child.depth() + donor.depth() <= TreeBuilder.MAX_TREE_DEPTH + 1:
            op = self._rng.choice(TreeBuilder.BINARY_OPS)
            return BinaryNode(op, child, donor)

        # Fall back: replace a leaf in child with a leaf from donor
        return self._leaf_crossover(child, donor)

    def _point_mutate(self, tree: ExprNode) -> ExprNode:
        """Change a randomly selected leaf feature."""
        leaves = self._collect_features(tree)
        if not leaves:
            return tree
        # Pick a random feature leaf and change its name
        idx  = self._rng.randrange(len(leaves))
        leaves[idx].name = self._rng.choice(self._builder.features)
        return tree

    def _subtree_mutate(self, tree: ExprNode) -> ExprNode:
        """Replace a random subtree with a fresh random tree."""
        # Simple version: if tree is binary, replace one child
        if isinstance(tree, BinaryNode):
            if self._rng.random() < 0.5:
                tree.left  = self._builder.random_tree(max(tree.left.depth(), 1))
            else:
                tree.right = self._builder.random_tree(max(tree.right.depth(), 1))
        elif isinstance(tree, UnaryNode):
            tree.child = self._builder.random_tree(max(tree.child.depth(), 1))
        else:
            # Leaf — full replacement
            return self._builder.random_tree(1)
        return tree

    def _leaf_crossover(self, tree: ExprNode, donor: ExprNode) -> ExprNode:
        """Replace a random leaf in tree with a random leaf from donor."""
        donor_leaves = self._collect_features(donor)
        if not donor_leaves:
            return tree
        donor_leaf = self._rng.choice(donor_leaves)

        my_leaves  = self._collect_features(tree)
        if not my_leaves:
            return tree
        target = self._rng.choice(my_leaves)
        target.name = donor_leaf.name
        return tree

    @staticmethod
    def _collect_features(tree: ExprNode) -> list[FeatureNode]:
        """Collect all FeatureNode leaves in a tree."""
        if isinstance(tree, FeatureNode):
            return [tree]
        if isinstance(tree, ConstantNode):
            return []
        if isinstance(tree, UnaryNode):
            return TreeGeneticOps._collect_features(tree.child)
        if isinstance(tree, BinaryNode):
            return (TreeGeneticOps._collect_features(tree.left) +
                    TreeGeneticOps._collect_features(tree.right))
        return []


# ── Tree GP Miner ─────────────────────────────────────────────────────────────

class TreeGPMiner:
    """
    GP miner using expression trees instead of string formulas.

    Advantages over string-based GP:
    1. Cross-sectional operations are evaluated correctly
       rank(a - b) ≠ rank(a) - rank(b)
    2. Mutations are structurally valid by construction
       No invalid formula strings produced
    3. Genetic operators work on tree topology, not string tokens
       Subtree crossover preserves semantic structure
    4. Depth control prevents exponential blowup at the tree level

    Compatible with validator: trees are serialised to formula strings
    for submission (via TreeEvaluator.to_formula_string()).
    """

    def __init__(
        self,
        prices:    pd.DataFrame,
        pop_size:  int   = 100,
        elite_n:   int   = 20,
        seed:      int   = 42,
        verbose:   bool  = False,
    ):
        from macro8_subnet.alpha.gp_miner import GP_FEATURES
        self.prices   = prices
        self.pop_size = pop_size
        self.elite_n  = elite_n
        self.verbose  = verbose

        self._evaluator = TreeEvaluator(prices)
        self._builder   = TreeBuilder(GP_FEATURES, seed=seed)
        self._ops       = TreeGeneticOps(self._builder)

        self._population: list[ExprNode] = self._builder.initial_population(pop_size)
        self._hall_of_fame: dict[str, tuple[ExprNode, float]] = {}  # str → (tree, ic)
        self._generation  = 0

    def step(self) -> list[tuple[ExprNode, float]]:
        """
        Run one GP generation.
        Returns top (tree, ic) pairs from hall of fame.
        """
        # Evaluate
        ics    = self._evaluator.batch_ic(self._population)
        scored = sorted(zip(self._population, ics),
                        key=lambda x: x[1], reverse=True)

        # Update hall of fame
        for tree, ic in scored:
            formula = tree.to_string()
            existing_ic = self._hall_of_fame.get(formula, (None, -1))[1]
            if ic > existing_ic:
                self._hall_of_fame[formula] = (tree, ic)

        if self.verbose:
            best_ic = scored[0][1] if scored else 0.0
            mean_ic = float(np.mean([ic for _, ic in scored if ic > 0])) if scored else 0.0
            print(f"  TreeGP gen {self._generation+1:3d} | "
                  f"best={best_ic:.4f} | mean={mean_ic:.4f} | "
                  f"hof={len(self._hall_of_fame)}")

        # Evolve
        elites = [tree for tree, _ in scored[:self.elite_n]]
        new_pop = list(elites)

        while len(new_pop) < self.pop_size:
            r = self._builder._rng.random()
            if r < 0.10:
                child = self._builder.random_tree(self._builder._rng.randint(1, 3))
            elif r < 0.65:
                parent = self._builder._rng.choice(elites)
                child  = self._ops.mutate(parent)
            else:
                p1, p2 = self._builder._rng.sample(elites, min(2, len(elites)))
                child  = self._ops.crossover(p1, p2)

            if child.depth() <= TreeBuilder.MAX_TREE_DEPTH:
                new_pop.append(child)

        self._population = new_pop[:self.pop_size]
        self._generation += 1

        return sorted(self._hall_of_fame.values(), key=lambda x: x[1], reverse=True)

    def top_formula_strings(self, n: int = 32) -> list[str]:
        """
        Return top N formulas as validator-compatible strings.
        Filters through safe_formula() to ensure submission safety.
        """
        top = sorted(self._hall_of_fame.values(),
                     key=lambda x: x[1], reverse=True)
        results = []
        seen    = set()
        for tree, ic in top:
            formula = self._evaluator.to_formula_string(tree)
            if formula and formula not in seen:
                seen.add(formula)
                results.append(formula)
            if len(results) >= n:
                break
        return results

    def run(self, n_epochs: int = 5) -> list[str]:
        """Run N generations and return top formula strings."""
        for _ in range(n_epochs):
            self.step()
        return self.top_formula_strings()
