"""alpha/alpha_library.py — Alpha library store (Sprint 9 rebuild)."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
from macro8_subnet.alpha.alpha_schema import AlphaFactor, AlphaRecord, AlphaCategory, AlphaEvaluation

@dataclass
class AlphaLibrary:
    _factors:  dict = field(default_factory=dict)
    _records:  dict = field(default_factory=dict)
    _cat_idx:  dict = field(default_factory=lambda: defaultdict(list))

    def add_factor(self, factor: AlphaFactor, evaluation: AlphaEvaluation,
                   epoch: int, min_ic: float = 0.02) -> tuple:
        if factor.name in self._factors: return False, "already in library"
        if evaluation.is_duplicate: return False, "duplicate"
        if evaluation.mean_ic is None or evaluation.mean_ic < min_ic:
            return False, f"IC {evaluation.mean_ic} < {min_ic}"
        self._factors[factor.name] = factor
        self._records[factor.name] = AlphaRecord(
            name=factor.name, miner_uid=factor.miner_uid,
            category=factor.category, birth_epoch=epoch,
            ic_history=[evaluation.mean_ic], ir_history=[evaluation.ic_ir or 0.0],
            current_ic=evaluation.mean_ic, current_ir=evaluation.ic_ir or 0.0,
        )
        self._cat_idx[factor.category.value].append(factor.name)
        return True, ""

    def get_factor(self, name: str) -> Optional[AlphaFactor]: return self._factors.get(name)
    def get_record(self, name: str) -> Optional[AlphaRecord]: return self._records.get(name)
    def all_active_factors(self) -> list:
        return [f for n, f in self._factors.items()
                if not self._records.get(n, AlphaRecord("",0,AlphaCategory.UNKNOWN,0)).retired]
    def all_signals(self) -> dict:
        return {f.name: f.signals for f in self.all_active_factors()}
    def top_by_ic(self, n: int = 10) -> list:
        active = [r for r in self._records.values() if not r.retired]
        return sorted(active, key=lambda r: r.mean_ic, reverse=True)[:n]
    def update_ic(self, name: str, new_ic: float, new_ir: float, epoch: int):
        if name in self._records: self._records[name].update(new_ic, new_ir, epoch)
    def retire(self, name: str, epoch: int):
        if name in self._records:
            self._records[name].retired = True; self._records[name].retire_epoch = epoch
    @property
    def size(self) -> int: return len(self._factors)
    @property
    def n_active(self) -> int: return len(self.all_active_factors())
    @property
    def n_retired(self) -> int: return sum(1 for r in self._records.values() if r.retired)
    def summary(self): return f"AlphaLibrary: {self.size} total | {self.n_active} active"
