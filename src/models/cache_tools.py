from __future__ import annotations
from pathlib import Path
from typing import Tuple
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm


def build_cache(
    dataloader: DataLoader,
    clip_wrapper,
    num_classes: int,
    device: torch.device | None = None,
) -> Tuple[Tensor, Tensor]:
    """
    Build cache (keys, values) from support set.

    Args:
        dataloader: DataLoader over support set
        clip_wrapper: CLIPWrapper instance with encode_image()
        num_classes: number of classes
        device: device to use; if None, use clip_wrapper.device

    Returns:
        keys: [Ns, D] L2-normalized image features
        values: [Ns, C] one-hot labels
    """
    if device is None:
        device = clip_wrapper.device

    keys_list = []
    values_list = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Build cache", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            # Encode images
            img_f = clip_wrapper.model.encode_image(images)
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)  # L2-normalize

            # One-hot encode labels
            vals = F.one_hot(labels, num_classes=num_classes).float()

            keys_list.append(img_f.cpu())
            values_list.append(vals.cpu())

    keys = torch.cat(keys_list, dim=0)  # [Ns, D]
    values = torch.cat(values_list, dim=0)  # [Ns, C]

    return keys, values


def save_cache(path: str | Path, keys: Tensor, values: Tensor) -> None:
    """
    Save cache to disk.

    Args:
        path: save path
        keys: [Ns, D]
        values: [Ns, C]
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"keys": keys, "values": values}, path)


def load_cache(path: str | Path) -> Tuple[Tensor, Tensor]:
    """
    Load cache from disk.

    Args:
        path: load path

    Returns:
        keys: [Ns, D]
        values: [Ns, C]
    """
    path = Path(path)
    data = torch.load(path)
    return data["keys"], data["values"]
