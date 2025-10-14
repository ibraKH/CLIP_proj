from __future__ import annotations
from typing import Literal, Tuple
import torch
from torch import Tensor
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageFilter 

from ..datasets.transforms import CLIP_MEAN, CLIP_STD, denormalize

def _tensor_to_pil(x: Tensor) -> Image.Image:
    x = x.detach().cpu()
    inv = denormalize()
    x = inv(x)
    x = torch.clamp(x, 0, 1)
    return F.to_pil_image(x)

def _pil_to_tensor(img: Image.Image) -> Tensor:
    t = F.to_tensor(img)
    return F.normalize(t, CLIP_MEAN, CLIP_STD)

def _severity_to_params(name: str, s: int) -> Tuple:
    s = int(s)
    if name == "gaussian_noise":
        sigma = [0.0, 0.02, 0.05, 0.08, 0.12, 0.18][s]
        return (sigma,)
    if name == "gaussian_blur":
        k = [0, 1, 2, 3, 4, 6][s]
        rad = max(0.0, float(k))
        return (rad,)
    if name == "lowres":
        factor = [1, 2, 3, 4, 6, 8][s]
        return (factor,)
    if name == "text_overlay":
        scale = [0.0, 0.5, 0.7, 0.9, 1.1, 1.3][s]
        op = [0.0, 0.25, 0.35, 0.45, 0.6, 0.75][s]
        return (scale, op)
    return ()

def _apply_text_overlay(img: Image.Image, text: str, severity: int, position: str = "br") -> Image.Image:
    W, H = img.size
    scale, opacity = _severity_to_params("text_overlay", severity)
    if scale == 0:
        return img
    try:
        font = ImageFont.truetype("arial.ttf", size=int(0.08 * scale * min(W, H)))
    except Exception:
        font = ImageFont.load_default()
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    margin = int(0.02 * min(W, H))
    positions = {
        "tl": (margin, margin),
        "tr": (W - tw - margin, margin),
        "bl": (margin, H - th - margin),
        "br": (W - tw - margin, H - th - margin),
        "center": ((W - tw) // 2, (H - th) // 2),
    }
    xy = positions.get(position.lower(), positions["br"])
    alpha = int(opacity * 255)
    draw.text(xy, text, fill=(255, 255, 255, alpha), font=font, stroke_width=1, stroke_fill=(0, 0, 0, alpha))
    out = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    return out

def apply_attacks(
    images: Tensor,
    attack_name: Literal["none", "text_overlay", "gaussian_noise", "gaussian_blur", "lowres"] = "none",
    severity: int = 0,
    text: str | None = None,
    position: str = "br",
) -> Tensor:
    """Batch-wise application; returns tensor normalized to CLIP stats."""
    if attack_name == "none" or severity == 0:
        return images

    out = []
    for i in range(images.size(0)):
        img = _tensor_to_pil(images[i])
        if attack_name == "text_overlay":
            t = text or "sample"
            img = _apply_text_overlay(img, t, severity=severity, position=position)
        elif attack_name == "gaussian_noise":
            (sigma,) = _severity_to_params("gaussian_noise", severity)
            t = F.to_tensor(img)
            noise = torch.randn_like(t) * sigma
            t = torch.clamp(t + noise, 0, 1)
            img = F.to_pil_image(t)
        elif attack_name == "gaussian_blur":
            (rad,) = _severity_to_params("gaussian_blur", severity)
            if rad > 0:
                img = img.filter(ImageFilter.GaussianBlur(radius=rad)) 
        elif attack_name == "lowres":
            (factor,) = _severity_to_params("lowres", severity)
            w, h = img.size
            dw, dh = max(1, w // factor), max(1, h // factor)
            img = img.resize((dw, dh), resample=Image.BILINEAR).resize((w, h), resample=Image.BILINEAR)
        out.append(_pil_to_tensor(img))
    return torch.stack(out, dim=0)