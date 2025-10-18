from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

def kshot_indices(labels: List[int], num_classes: int, k: int, seed: int) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)
    labels = np.array(labels)
    support: List[int] = []
    rest: List[int] = []
    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        k_c = min(k, len(idx))
        support.extend(idx[:k_c].tolist())
        rest.extend(idx[k_c:].tolist())

    rest = sorted(rest)
    return support, rest