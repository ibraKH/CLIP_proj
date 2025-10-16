from __future__ import annotations
from collections import defaultdict
from typing import Iterable, List, Sequence, Dict
import random
import torch

def balanced_subset_indices(
    labels: Sequence[int],
    limit_per_class: int,
    class_limit: int | None = None,
    seed: int = 42,
) -> List[int]:
    rng = random.Random(seed)
    by_class: Dict[int, List[int]] = defaultdict(list)
    for idx, y in enumerate(labels):
        by_class[int(y)].append(idx)

    classes = sorted(by_class.keys())
    if class_limit is not None:
        classes = classes[:int(class_limit)]

    picked: List[int] = []
    for c in classes:
        idxs = by_class[c]
        rng.shuffle(idxs)
        picked.extend(sorted(idxs[:limit_per_class]))
    return picked