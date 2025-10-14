from __future__ import annotations
from typing import List, Sequence, Tuple
import torch
from torch import nn, Tensor
import open_clip


class CLIPWrapper(nn.Module):
    """
    Loads an open_clip model and produces zero-shot logits.
    """
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2B-s34B-b79K",
        prompt_templates: Sequence[str] = ("a photo of a {class}.",),
        classnames: Sequence[str] = (),
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=self.device)
        self.model = model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.prompt_templates = list(prompt_templates)
        self.classnames = list(classnames)
        with torch.no_grad():
            self.text_features = self._build_text_features().to(self.device)  # (C, D)

    @torch.no_grad()
    def _build_text_features(self) -> Tensor:
        token_batches = []
        for cname in self.classnames:
            prompts = [
                t.format_map({"class": cname.replace("_", " ")})
                for t in self.prompt_templates
            ]
            tokens = self.tokenizer(prompts)
            token_batches.append(tokens)

        feats = []
        for tok in token_batches:
            tok = tok.to(self.device)
            f = self.model.encode_text(tok)  # (P,D)
            f = f / f.norm(dim=-1, keepdim=True)
            f = f.mean(dim=0)
            feats.append(f)

        text_features = torch.stack(feats, dim=0)  # (C,D)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    @torch.no_grad()
    def logits(self, images: Tensor) -> Tensor:
        # images expected already normalized to CLIP
        img_f = self.model.encode_image(images.to(self.device))  # (B,D)
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        scale = self.model.logit_scale.exp()
        return scale * img_f @ self.text_features.t()  # (B,C)