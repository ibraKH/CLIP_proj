from __future__ import annotations
from typing import Sequence
import torch
from torch import nn, Tensor
import open_clip


class CoOpHead(nn.Module):
    """
    Minimal CoOp-style prompt tuning:
    Learn a global prompt delta vector in text feature space added to per-class text features.
    Trains with few-shot CE while CLIP backbone stays frozen.
    """
    def __init__(
        self,
        model_name: str,
        pretrained: str,
        prompt_templates: Sequence[str],
        classnames: Sequence[str],
        device: str = "cpu",
        n_ctx: int = 16,  # kept for API symmetry; not used directly in this simplified variant
        ctx_init: str | None = None,
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
        self.delta = nn.Parameter(torch.zeros_like(self.base_text[0]))  # (D,)

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

    def _text_features(self) -> Tensor:
        tf = self.base_text + self.delta  # broadcast add (C,D)
        return tf / tf.norm(dim=-1, keepdim=True)

    def logits(self, images: Tensor) -> Tensor:
        with torch.no_grad():
            img_f = self.clip.encode_image(images.to(self.device))
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        scale = self.clip.logit_scale.exp()
        return scale * img_f @ self._text_features().t()