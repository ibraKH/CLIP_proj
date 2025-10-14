from __future__ import annotations
from typing import Tuple, List
from pathlib import Path

from torch.utils.data import DataLoader
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
    train_tf = clip_train(img_size)
    test_tf = clip_test(img_size)
    root_p = Path(root)
    root_p.mkdir(parents=True, exist_ok=True)

    ds_train = CIFAR10(root=str(root_p), train=True, download=True, transform=train_tf)
    ds_test = CIFAR10(root=str(root_p), train=False, download=True, transform=test_tf)
    # split small validation from train (deterministic)
    # 45k train / 5k val
    from torch.utils.data import random_split
    g = torch.Generator().manual_seed(0)
    ds_train, ds_val = random_split(ds_train, [45000, 5000], generator=g)

    classes = CIFAR10.classes

    def to_loader(ds, shuffle=False):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    return to_loader(ds_train, shuffle=True), to_loader(ds_val), to_loader(ds_test), classes
