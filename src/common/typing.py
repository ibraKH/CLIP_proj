from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Protocol, TypedDict

import torch
from torch import Tensor

class Batch(TypedDict):
    images: Tensor  # (B,3,H,W), normalized
    labels: Tensor  # (B,)

class EvalResult(TypedDict):
    top1: float
    macro_f1: float
    ece: float

class AttackCurve(TypedDict):
    severities: List[int]
    acc: List[float]
    delta_acc: List[float]

class HasLogits(Protocol):
    def logits(self, images: Tensor) -> Tensor:  # (B, num_classes)
        ...