"""Structural feature extraction.

Builds S^str (Eq. 2 of methodology.tex, matching Mo2oM's own structural
dependency matrix) and the structural-neighborhood function N_str used to
seed the capability-descriptor prompts.
"""

from typing import Dict, List, Tuple

import numpy as np

from .parsing.java_parser import ClassInfo


def build_structural_matrix(
    classes: Dict[str, ClassInfo]
) -> Tuple[List[str], np.ndarray]:
    """S^str[i, j] = normalized, symmetrized call frequency between
    class i and class j."""
    names = sorted(classes.keys())
    idx = {n: i for i, n in enumerate(names)}
    y = len(names)
    raw = np.zeros((y, y))

    for name, info in classes.items():
        i = idx[name]
        for callee in info.calls:
            if callee in idx:
                j = idx[callee]
                raw[i, j] += 1
                raw[j, i] += 1

    max_val = raw.max() if raw.max() > 0 else 1.0
    s_str = raw / max_val
    np.fill_diagonal(s_str, 0.0)
    return names, s_str


def structural_neighborhood(
    class_idx: int, s_str: np.ndarray, delta: float
) -> List[int]:
    """N_str(C_k) = {C_j : S^str(k, j) > delta}   (Eq. 2)."""
    return [j for j in range(s_str.shape[0]) if s_str[class_idx, j] > delta]
