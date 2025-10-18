from __future__ import annotations
from typing import Tuple, List
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102
import torchvision.transforms as T

from .transforms import clip_train, clip_test
from ..common.registry import DATASETS

@DATASETS.register("flowers102")
def get_flowers102_loaders(
    root: str,
    batch_size: int = 64,
    num_workers: int = 4,
    img_size: int = 224,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Uses torchvision's official splits:
    split='train', 'val', 'test'
    """
    root_p = Path(root)
    root_p.mkdir(parents=True, exist_ok=True)

    train_tf = clip_train(img_size)
    test_tf = clip_test(img_size)

    ds_train = Flowers102(root=str(root_p), split="train", download=True, transform=train_tf)
    ds_val = Flowers102(root=str(root_p), split="val", download=True, transform=test_tf)
    ds_test = Flowers102(root=str(root_p), split="test", download=True, transform=test_tf)

    classes = ds_train.classes  # 102 class names

    def to_loader(ds, shuffle=False):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    return to_loader(ds_train, shuffle=True), to_loader(ds_val), to_loader(ds_test), classes