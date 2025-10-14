from __future__ import annotations
from typing import Tuple
import torchvision.transforms as T

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

def clip_train(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CLIP_MEAN, CLIP_STD),
    ])

def clip_test(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(CLIP_MEAN, CLIP_STD),
    ])

def denormalize() -> T.Normalize:
    mean = CLIP_MEAN
    std = CLIP_STD
    inv_std = [1/s for s in std]
    inv_mean = [-m/s for m, s in zip(mean, std)]
    return T.Normalize(mean=inv_mean, std=inv_std)