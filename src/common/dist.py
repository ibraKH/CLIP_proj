from __future__ import annotations
import torch

def get_device(dev_str: str | None = None) -> torch.device:
    if dev_str is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev_str)