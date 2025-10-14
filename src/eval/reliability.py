from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

def save_reliability_diagram(
    logits: torch.Tensor,
    labels: torch.Tensor,
    out_path: str | Path,
    n_bins: int = 15,
) -> None:
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
    preds = probs.argmax(axis=1)
    confs = probs.max(axis=1)
    y = labels.detach().cpu().numpy()

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    accs = []
    conf_means = []
    widths = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (confs > lo) & (confs <= hi)
        if m.sum() == 0:
            accs.append(0)
            conf_means.append(0)
            widths.append(hi - lo)
            continue
        accs.append((preds[m] == y[m]).mean())
        conf_means.append(confs[m].mean())
        widths.append(hi - lo)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.bar(bin_centers, accs, width=widths, edgecolor="black", alpha=0.7)
    plt.plot([0, 1], [0, 1])
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()