from __future__ import annotations
from typing import Tuple, List
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10

from .transforms import clip_train, clip_test
from ..common.registry import DATASETS

@DATASETS.register("cifar10")
def get_cifar10_loaders(
    root: str,
    batch_size: int = 64,
    num_workers: int = 4,
    img_size: int = 224,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    root_p = Path(root)
    root_p.mkdir(parents=True, exist_ok=True)

    train_tf = clip_train(img_size)
    test_tf = clip_test(img_size)

    ds_train_full = CIFAR10(root=str(root_p), train=True, download=True, transform=train_tf)
    ds_test = CIFAR10(root=str(root_p), train=False, download=True, transform=test_tf)

    # Deterministic 45k/5k split from the 50k train set
    g = torch.Generator()
    g.manual_seed(0)
    ds_train, ds_val = random_split(ds_train_full, [45_000, 5_000], generator=g)

    # Prefer instance attribute to avoid API changes
    classnames: List[str] = list(ds_train_full.classes)

    def to_loader(ds, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    return to_loader(ds_train, shuffle=True), to_loader(ds_val), to_loader(ds_test), classnames