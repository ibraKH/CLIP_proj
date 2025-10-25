from __future__ import annotations
from typing import Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class GateMLP(nn.Module):
    """
    Small MLP that predicts per-image alpha (and optionally beta) for gated fusion.
    Alpha is in (0,1) via sigmoid; beta is >0 via softplus.
    """
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 2)  # [alpha_logit, beta_logit]

    def forward(self, q: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            q: [B, D] - normalized image features
        Returns:
            alpha_hat: [B, 1] in (0, 1)
            beta_hat: [B, 1] > 0
        """
        x = F.relu(self.fc1(q))  # [B, hidden]
        x = self.fc2(x)  # [B, 2]
        alpha_logit = x[:, 0:1]
        beta_logit = x[:, 1:2]

        alpha_hat = torch.sigmoid(alpha_logit)  # (0, 1)
        beta_hat = F.softplus(beta_logit) + 1e-3  # > 0

        return alpha_hat, beta_hat


class TipAdapterPlus(nn.Module):
    """
    Tip-Adapter++ with gated fusion:
    - Predicts per-image alpha (and optionally beta) via a small MLP
    - Fuses zero-shot CLIP logits with cache-based logits
    - Keeps CLIP backbone frozen; only trains the gate
    """
    def __init__(
        self,
        clip_wrapper,
        keys: Tensor,
        values: Tensor,
        alpha0: float = 0.5,
        beta0: float = 5.0,
        gate_hidden: int = 256,
        reg: float = 0.05,
        learn_beta: bool = True,
        learn_scale: bool = False,
    ):
        """
        Args:
            clip_wrapper: CLIPWrapper instance with encode_image() and logits() methods
            keys: [Ns, D] L2-normalized support features
            values: [Ns, C] one-hot support labels
            alpha0: default/target alpha for regularization
            beta0: default/target beta for regularization
            gate_hidden: hidden dim for gate MLP
            reg: regularization strength
            learn_beta: if True, learn beta; else use beta0
            learn_scale: if True, add learnable post-fusion scale
        """
        super().__init__()
        self.clip = clip_wrapper
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.reg = reg
        self.learn_beta = learn_beta

        # Buffers for cache
        self.register_buffer("keys", keys)
        self.register_buffer("values", values)

        # Gate network
        D = keys.shape[1]
        self.gate = GateMLP(D, gate_hidden)

        # Optional post-fusion scale
        if learn_scale:
            self.post_scale = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("post_scale", torch.ones(1))

    def forward(self, images: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            images: [B, 3, 224, 224]
        Returns:
            logits: [B, C]
            alpha_hat: [B, 1]
            beta_hat: [B, 1]
        """
        # Zero-shot logits from CLIP
        z = self.clip.logits(images)  # [B, C]

        # Encode images and normalize
        with torch.no_grad():
            img_f = self.clip.model.encode_image(images.to(self.clip.device))
            q = img_f / img_f.norm(dim=-1, keepdim=True)  # [B, D]

        # Predict alpha and beta
        alpha_hat, beta_hat = self.gate(q)  # [B, 1], [B, 1]

        # Use predicted beta or fixed beta0
        if self.learn_beta:
            beta = beta_hat
        else:
            beta = torch.full_like(beta_hat, self.beta0)

        # Cache logits
        sim = q @ self.keys.T  # [B, Ns]
        affinity = torch.exp(beta * sim)  # [B, Ns]
        c = affinity @ self.values  # [B, C]

        # Gated fusion
        logits = (1 - alpha_hat) * z + alpha_hat * c  # [B, C]

        # Optional post-fusion scale
        logits = logits * self.post_scale

        return logits, alpha_hat, beta

    def reg_loss(self, alpha_hat: Tensor, beta_hat: Tensor) -> Tensor:
        """
        Regularization to keep alpha and beta near their defaults.
        """
        loss_alpha = ((alpha_hat - self.alpha0) ** 2).mean()
        if self.learn_beta:
            loss_beta = ((beta_hat - self.beta0) ** 2).mean()
            return self.reg * (loss_alpha + loss_beta)
        else:
            return self.reg * loss_alpha

    def logits(self, images: Tensor) -> Tensor:
        """
        For compatibility with eval loops that expect model.logits(images).
        """
        out, _, _ = self.forward(images)
        return out
