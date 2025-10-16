from __future__ import annotations
from typing import Sequence, Tuple
import torch
from torch import nn, Tensor
import open_clip
import torch.nn.functional as F


class TipAdapter(nn.Module):
    """
    Build a key-value cache from support set features (image features + one-hot labels).
    Fuse cache logits with CLIP zero-shot logits using alpha, with temperature beta.
    Optional 'fine' flag to learn a small linear layer on top of image features (Tip-Adapter-F-lite).
    """
    def __init__(
        self,
        model_name: str,
        pretrained: str,
        prompt_templates: Sequence[str],
        classnames: Sequence[str],
        device: str = "cpu",
        alpha: float = 0.5,
        beta: float = 5.0,
        fine: bool = False,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.alpha = alpha
        self.beta = beta
        self.fine = fine

        self.clip, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=self.device)
        self.clip.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.classnames = list(classnames)
        self.prompt_templates = list(prompt_templates)

        with torch.no_grad():
            self.text_features = self._encode_text().to(self.device)  # (C,D)
            dummy = self.clip.encode_image(torch.zeros(1, 3, 224, 224, device=self.device))
            self.dim = int(dummy.shape[-1])

        self.adapter = nn.Identity() if not fine else nn.Linear(self.dim, self.dim, bias=False)

        self.register_buffer("keys", torch.empty(0, self.dim))
        self.register_buffer("values", torch.empty(0, len(classnames)))  # one-hot

    @torch.no_grad()
    def _encode_text(self) -> Tensor:
        feats = []
        for cname in self.classnames:
            tokens = self.tokenizer([t.format_map({"class": cname.replace("_", " ")}) for t in self.prompt_templates]).to(self.device)
            f = self.clip.encode_text(tokens)
            f = f / f.norm(dim=-1, keepdim=True)
            f = f.mean(dim=0)
            feats.append(f)
        F = torch.stack(feats, 0)
        return F / F.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def build_cache(self, images: Tensor, labels: Tensor) -> None:
        img_f = self.clip.encode_image(images.to(self.device))
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        keys = self.adapter(img_f)
        C = len(self.classnames)
        vals = F.one_hot(labels.to(self.device), num_classes=C).float()
        if self.keys.numel() == 0:
            self.keys = keys
            self.values = vals
        else:
            self.keys = torch.cat([self.keys, keys], dim=0)
            self.values = torch.cat([self.values, vals], dim=0)

    def _clip_logits(self, images: Tensor) -> Tensor:
        with torch.no_grad():
            img_f = self.clip.encode_image(images.to(self.device))
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            scale = self.clip.logit_scale.exp()
            return scale * img_f @ self.text_features.t()

    def _cache_logits(self, images: Tensor) -> Tensor:
        with torch.no_grad():
            img_f = self.clip.encode_image(images.to(self.device))
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            q = self.adapter(img_f)  # (B,D)
        # cosine sim → distance kernel
        sim = q @ self.keys.t()  # (B,N)
        # convert to affinity with temperature beta
        aff = torch.exp(self.beta * sim)
        logits = aff @ self.values  # (B,C)
        return logits

    def logits(self, images: Tensor) -> Tensor:
        z = self._clip_logits(images)
        c = self._cache_logits(images)
        return (1 - self.alpha) * z + self.alpha * c
