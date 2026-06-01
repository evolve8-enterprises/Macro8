"""alpha_schema.py — Core data types (Sprint 9 rebuild)."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import pandas as pd

class AlphaCategory(str, Enum):
    MOMENTUM = "momentum"; VOLATILITY = "volatility"; MACRO = "macro"
    SENTIMENT = "sentiment"; CROSS_ASSET = "cross_asset"
    MEAN_REVERSION = "mean_reversion"; UNKNOWN = "unknown"

@dataclass
class AlphaFactor:
    name: str; miner_uid: int; miner_hotkey: str
    signals: dict; category: AlphaCategory = AlphaCategory.UNKNOWN
    description: str = ""; submitted: datetime = field(default_factory=datetime.utcnow)
    @property
    def assets(self): return list(self.signals.keys())
    @property
    def n_observations(self):
        return min(len(s) for s in self.signals.values()) if self.signals else 0
    def is_valid(self):
        if not self.name: return False, "empty name"
        if not self.signals: return False, "empty signals"
        return True, ""
    def to_dict(self): return {"name": self.name, "miner_uid": self.miner_uid}

@dataclass
class AlphaRecord:
    name: str; miner_uid: int; category: AlphaCategory; birth_epoch: int
    ic_history: list = field(default_factory=list)
    ir_history: list = field(default_factory=list)
    current_ic: float = 0.0; current_ir: float = 0.0
    decay_rate: float = 0.0; capacity: float = 1.0
    retired: bool = False; retire_epoch: Optional[int] = None; epochs_alive: int = 0
    @property
    def mean_ic(self): return float(sum(self.ic_history)/len(self.ic_history)) if self.ic_history else 0.0
    @property
    def ic_stability(self):
        if not self.ic_history: return 0.0
        return sum(1 for x in self.ic_history if x > 0) / len(self.ic_history)
    def update(self, new_ic, new_ir, epoch):
        import numpy as np
        self.ic_history.append(new_ic); self.ir_history.append(new_ir)
        self.current_ic = new_ic; self.current_ir = new_ir
        self.epochs_alive = epoch - self.birth_epoch
        if len(self.ic_history) >= 4:
            x = np.arange(len(self.ic_history[-8:]), dtype=float)
            y = np.array(self.ic_history[-8:])
            self.decay_rate = float(np.polyfit(x, y, 1)[0])
    def should_retire(self, min_ic=0.02, min_epochs=5, max_decay=-0.01):
        if self.epochs_alive < min_epochs: return False
        if self.mean_ic < min_ic: return True
        if len(self.ic_history) >= 8 and self.decay_rate < max_decay: return True
        return False
    def to_dict(self): return {"name": self.name, "mean_ic": self.mean_ic, "retired": self.retired}

@dataclass
class AlphaEvaluation:
    factor_name: str; miner_uid: int
    mean_ic: Optional[float] = None; ic_ir: Optional[float] = None
    ic_decay: Optional[float] = None; max_corr: Optional[float] = None
    is_duplicate: bool = False; passes_ic_threshold: bool = False
    reward_score: Optional[float] = None; success: bool = False; error: Optional[str] = None
    def to_dict(self): return {"factor_name": self.factor_name, "mean_ic": self.mean_ic, "success": self.success}
