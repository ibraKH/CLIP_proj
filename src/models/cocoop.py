from __future__ import annotations
from typing import Sequence
import torch
from torch import nn, Tensor
import open_clip

class CoCoOpHead(nn.Module):
    """
    Minimal CoCoOp-style conditioning:
    A small MLP predicts a delta vector from image features and adds it to the class text features.
    """
    def __init__(
        self,
        model_name: str,
        pretrained: str,
        prompt_templates: Sequence[str],
        classnames: Sequence[str],
        device: str = "cpu",
        hidden: int = 512,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.clip, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=self.device)
        self.clip.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.classnames = list(classnames)
        self.prompt_templates = list(prompt_templates)
        with torch.no_grad():
            self.base_text = self._encode_text().to(self.device)  # (C,D)
            dummy = self.clip.encode_image(torch.zeros(1, 3, 224, 224, device=self.device))
            self.dim = int(dummy.shape[-1])
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.dim),
        )

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

    def logits(self, images: Tensor) -> Tensor:
        img_f = self.clip.encode_image(images.to(self.device))
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        delta = self.mlp(img_f)  # (B,D)
        # condition text features per-image: (B,C,D)
        txt = self.base_text.unsqueeze(0) + delta.unsqueeze(1)
        txt = txt / txt.norm(dim=-1, keepdim=True)
        scale = self.clip.logit_scale.exp()
        return scale * torch.einsum("bd,bcd->bc", img_f, txt)