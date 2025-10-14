from __future__ import annotations
from typing import Tuple, List
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision.datasets import Caltech101

from .transforms import clip_train, clip_test
from ..common.registry import DATASETS

@DATASETS.register("caltech101")
def get_caltech101_loaders(
    root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    root_p = Path(root)
    root_p.mkdir(parents=True, exist_ok=True)

    train_tf = clip_train(img_size)
    test_tf = clip_test(img_size)

    # Caltech101 doesn't ship official val/test; we split deterministically
    ds = Caltech101(root=str(root_p), download=True, target_type="category", transform=train_tf)
    classes = ds.categories

    # deterministic splits: 60% train, 20% val, 20% test
    from torch.utils.data import random_split
    import torch
    n = len(ds)
    n_tr = int(0.6 * n)
    n_val = int(0.2 * n)
    n_te = n - n_tr - n_val
    g = torch.Generator().manual_seed(0)
    ds_tr, ds_val, ds_te = random_split(ds, [n_tr, n_val, n_te], generator=g)
    # swap transforms for val/test
    ds_val.dataset.transform = test_tf
    ds_te.dataset.transform = test_tf

    def to_loader(d, shuffle=False):
        return DataLoader(d, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    return to_loader(ds_tr, shuffle=True), to_loader(ds_val), to_loader(ds_te), classes