"""alpha/orthogonality.py — Signal decorrelation (Sprint 9 rebuild)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

@dataclass
class OrthogonalityReport:
    factor_names: list; corr_matrix: pd.DataFrame; duplicate_pairs: list
    unique_factors: list; rejected_factors: list; orthogonality_scores: dict

class OrthogonalityFilter:
    def __init__(self, threshold: float = 0.90, penalty: float = 0.5):
        if not 0.0 < threshold <= 1.0: raise ValueError(f"threshold must be in (0,1], got {threshold}")
        if not 0.0 <= penalty <= 1.0: raise ValueError(f"penalty must be in [0,1], got {penalty}")
        self.threshold = threshold; self.penalty = penalty

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return self._cosine_similarity_static(a, b)

    def pairwise_similarity(self, wa: dict, wb: dict) -> float:
        universe = sorted(set(wa) | set(wb))
        return self._cosine_similarity_static(
            np.array([wa.get(k, 0.0) for k in universe]),
            np.array([wb.get(k, 0.0) for k in universe]),
        )

    def correlate_with_library(self, new_name: str, new_signals: dict, library: dict) -> tuple:
        if not library: return 0.0, ""
        new_vec = self._flatten(new_signals); max_c = 0.0; max_n = ""
        for name, sigs in library.items():
            lib_vec = self._flatten(sigs)
            c = self._vector_corr(new_vec, lib_vec)
            if c > max_c: max_c = c; max_n = name
        return float(max_c), max_n

    def analyse(self, factors: dict, ic_scores: Optional[dict] = None) -> OrthogonalityReport:
        names   = list(factors.keys())
        vectors = {n: self._flatten(s) for n, s in factors.items()}
        n       = len(names)
        matrix  = np.eye(n)
        for i in range(n):
            for j in range(i+1, n):
                c = self._vector_corr(vectors[names[i]], vectors[names[j]])
                matrix[i,j] = c; matrix[j,i] = c
        corr_df = pd.DataFrame(matrix, index=names, columns=names)
        sorted_names = sorted(names, key=lambda n: (ic_scores or {}).get(n, 0.0), reverse=True)
        accepted, rejected, dups = [], [], []
        for name in sorted_names:
            is_dup = any(abs(corr_df.loc[name, k]) > self.threshold for k in accepted)
            (rejected if is_dup else accepted).append(name)
        orth = {n: float(1 - max((abs(corr_df.loc[n, a]) for a in accepted if a != n), default=0.0)) for n in names}
        return OrthogonalityReport(names, corr_df, dups, accepted, rejected, orth)

    @staticmethod
    def _flatten(signals: dict) -> np.ndarray:
        df = pd.DataFrame(signals).dropna(how="all")
        df_z = (df - df.mean()) / df.std().replace(0, 1.0)
        return df_z.values.flatten()

    @staticmethod
    def _vector_corr(a: np.ndarray, b: np.ndarray) -> float:
        m = min(len(a), len(b))
        if m < 3: return 0.0
        a, b = a[:m], b[:m]; mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 3: return 0.0
        try:
            c = float(np.corrcoef(a[mask], b[mask])[0,1])
            return 0.0 if np.isnan(c) else abs(c)
        except: return 0.0

    @staticmethod
    def _cosine_similarity_static(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12: return 0.0
        return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))
