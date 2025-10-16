from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import torch
from torch import Tensor
from sklearn.metrics import f1_score

def top1_from_logits(logits: Tensor, labels: Tensor) -> float:
    pred = logits.argmax(dim=1).cpu().numpy()
    y = labels.cpu().numpy()
    return float((pred == y).mean())

def macro_f1_from_logits(logits: Tensor, labels: Tensor) -> float:
    pred = logits.argmax(dim=1).cpu().numpy()
    y = labels.cpu().numpy()
    return float(f1_score(y, pred, average="macro"))

class TemperatureScaler(torch.nn.Module):
    def __init__(self, init_T: float = 1.0):
        super().__init__()
        self.log_T = torch.nn.Parameter(torch.tensor(np.log(init_T), dtype=torch.float32))

    def forward(self, logits: Tensor) -> Tensor:
        T = torch.exp(self.log_T)
        return logits / T

    def fit(self, logits: Tensor, labels: Tensor, lr: float = 0.01, steps: int = 200) -> float:
        logits = logits.detach()
        labels = labels.detach().long()

        self.train()
        opt = torch.optim.LBFGS([self.log_T], lr=lr, max_iter=steps)
        nll = torch.nn.CrossEntropyLoss()

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = nll(self.forward(logits), labels)
            loss.backward()
            return loss

        opt.step(closure)
        return float(torch.exp(self.log_T).item())

def expected_calibration_error(logits: Tensor, labels: Tensor, n_bins: int = 15) -> float:
    """
    ECE on softmax probabilities.
    """
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
    preds = probs.argmax(axis=1)
    confs = probs.max(axis=1)
    y = labels.detach().cpu().numpy()

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confs > lo) & (confs <= hi)
        if not np.any(mask):
            continue
        acc = (preds[mask] == y[mask]).mean()
        conf = confs[mask].mean()
        ece += np.abs(acc - conf) * (mask.mean())
    return float(ece)
