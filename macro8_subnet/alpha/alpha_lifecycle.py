"""alpha/alpha_lifecycle.py — Lifecycle management (Sprint 9 rebuild)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class LifecycleAction:
    factor_name: str; action: str; reason: str; capacity_adj: float
    current_ic: float; decay_rate: float
    @property
    def should_retire(self): return self.action == "retire"

@dataclass
class LifecycleReport:
    epoch: int; actions: list; n_kept: int; n_warned: int; n_retired: int
    def retired_names(self): return [a.factor_name for a in self.actions if a.should_retire]
    def summary(self): return f"LifecycleReport epoch={self.epoch}: {self.n_kept} kept | {self.n_warned} warned | {self.n_retired} retired"

class AlphaLifecycleManager:
    def __init__(self, min_ic=0.02, max_decay=-0.008, min_stability=0.35, min_epochs=5):
        self.min_ic=min_ic; self.max_decay=max_decay
        self.min_stability=min_stability; self.min_epochs=min_epochs

    def assess(self, records: list, epoch: int) -> LifecycleReport:
        actions = [self._assess_single(r, epoch) for r in records]
        return LifecycleReport(epoch=epoch, actions=actions,
            n_kept=sum(1 for a in actions if a.action=="keep"),
            n_warned=sum(1 for a in actions if a.action=="warn"),
            n_retired=sum(1 for a in actions if a.action=="retire"))

    def compute_capacity(self, record) -> float:
        if not record.ic_history: return 0.5
        ic = record.mean_ic; ir = record.current_ir; stab = record.ic_stability
        decay_pen = max(1.0 + record.decay_rate * 10, 0.1)
        cap = min(ic/0.10,1.0) * min(ir/1.0,1.0) * stab * decay_pen
        return float(np.clip(cap, 0.05, 1.0))

    def _assess_single(self, record, epoch: int) -> LifecycleAction:
        if record.epochs_alive < self.min_epochs:
            return LifecycleAction(record.name,"keep",f"Warmup",0.5,record.current_ic,record.decay_rate)
        reasons = []
        if record.mean_ic < self.min_ic: reasons.append(f"mean_IC {record.mean_ic:.4f}")
        if record.decay_rate < self.max_decay: reasons.append(f"decay {record.decay_rate:.4f}")
        if record.ic_stability < self.min_stability: reasons.append(f"stability {record.ic_stability:.2f}")
        if len(reasons) >= 2:
            return LifecycleAction(record.name,"retire","; ".join(reasons),0.0,record.current_ic,record.decay_rate)
        elif len(reasons) == 1:
            return LifecycleAction(record.name,"warn",reasons[0],0.3,record.current_ic,record.decay_rate)
        return LifecycleAction(record.name,"keep","Healthy",self.compute_capacity(record),record.current_ic,record.decay_rate)
