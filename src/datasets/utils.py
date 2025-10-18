from __future__ import annotations
from typing import Sequence, List
import numpy as np
import torch
from torch.utils.data import Subset, Dataset

def get_targets(ds: Dataset) -> List[int]:
    """Return label list for any Dataset or Subset(Dataset)."""
    # torchvision datasets usually expose .targets or .labels
    if hasattr(ds, "targets"):
        t = getattr(ds, "targets")
        return list(t) if not isinstance(t, list) else t
    if hasattr(ds, "labels"):
        l = getattr(ds, "labels")
        return list(l) if not isinstance(l, list) else l
    if isinstance(ds, Subset):
        base_t = get_targets(ds.dataset)
        return [base_t[i] for i in ds.indices]  # map to subset view
    raise AttributeError("Dataset has no .targets/.labels and is not a Subset")

def build_fewshot_subset(train_subset: Dataset, k: int, seed: int = 0) -> Subset:
    """
    Pick exactly k examples per class from *train_subset* using LOCAL indices.
    Works whether train_subset is a Dataset or a Subset(Dataset).
    """
    rng = np.random.RandomState(seed)
    targets = np.array(get_targets(train_subset))
    classes = np.unique(targets)
    local_indices: List[int] = []
    for c in classes:
        cls_idx = np.where(targets == c)[0]  # <-- LOCAL positions
        assert len(cls_idx) >= k, f"class {c} has only {len(cls_idx)} samples (< {k})"
        pick = rng.choice(cls_idx, size=k, replace=False)
        local_indices.extend(pick.tolist())
    # sanity: indices must be within the subset's local range
    assert max(local_indices) < len(targets)
    return Subset(train_subset, local_indices)