from __future__ import annotations
from typing import Tuple, Sequence
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import Flowers102
from .transforms import clip_train_transforms, clip_test_transforms
from .sampler import balanced_subset_indices

CLASSNAMES = [  # (existing list in your file)
    # ...
]

def get_flowers102_loaders(
    root: str,
    batch_size: int = 64,
    num_workers: int = 4,
    img_size: int = 224,
    limit_per_class: int | None = None,
    class_limit: int | None = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Sequence[str]]:
    tr = Flowers102(root=root, split="train", download=True, transform=clip_train_transforms(img_size))
    va = Flowers102(root=root, split="val",   download=True, transform=clip_test_transforms(img_size))
    te = Flowers102(root=root, split="test",  download=True, transform=clip_test_transforms(img_size))

    def maybe_subset(ds):
        if limit_per_class is None:
            return ds
        idxs = balanced_subset_indices(ds._labels, limit_per_class, class_limit, seed)
        return Subset(ds, idxs)

    tr = maybe_subset(tr)
    va = maybe_subset(va)  # small val makes temp-scaling fast
    te = maybe_subset(te)

    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(tr, shuffle=True,  **kwargs),
        DataLoader(va, shuffle=False, **kwargs),
        DataLoader(te, shuffle=False, **kwargs),
        CLASSNAMES,
    )
