from __future__ import annotations
from typing import Dict, List, Tuple
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import top1_from_logits, macro_f1_from_logits, TemperatureScaler, expected_calibration_error
from .attacks import apply_attacks
from ..common.typing import EvalResult
from ..datasets.transforms import denormalize


@torch.no_grad()
def _collect_logits_labels(model, loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Collect logits and labels. Also track alpha/beta statistics if model returns them.

    Returns:
        logits: [N, C]
        labels: [N]
        stats: dict with optional 'mean_alpha', 'std_alpha', 'mean_beta', 'std_beta'
    """
    logits_all, labels_all = [], []
    alpha_all, beta_all = [], []

    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device)
        out = model.logits(images)

        # Handle models that return (logits, alpha, beta) vs just logits
        if isinstance(out, tuple):
            logits, alpha, beta = out
            alpha_all.append(alpha.cpu())
            beta_all.append(beta.cpu())
        else:
            logits = out

        logits_all.append(logits.cpu())
        labels_all.append(labels.cpu())

    stats = {}
    if alpha_all:
        alpha_tensor = torch.cat(alpha_all)
        beta_tensor = torch.cat(beta_all)
        stats['mean_alpha'] = float(alpha_tensor.mean().item())
        stats['std_alpha'] = float(alpha_tensor.std().item())
        stats['mean_beta'] = float(beta_tensor.mean().item())
        stats['std_beta'] = float(beta_tensor.std().item())

    return torch.cat(logits_all), torch.cat(labels_all), stats


def clean_eval(model, loader: DataLoader, device: torch.device) -> EvalResult:
    logits, labels, stats = _collect_logits_labels(model, loader, device)
    # Fit temperature on the same loader (or pass a separate val if available)
    logits = logits.detach()
    scaler = TemperatureScaler()
    scaler.fit(logits, labels)
    logits_t = scaler(logits)
    res: EvalResult = {
        "top1": top1_from_logits(logits_t, labels),
        "macro_f1": macro_f1_from_logits(logits_t, labels),
        "ece": expected_calibration_error(logits_t, labels),
    }
    # Add alpha/beta stats if available
    res.update(stats)  # type: ignore
    return res


def attack_eval(model, loader: DataLoader, device: torch.device, attack_name: str, severities: List[int]) -> Dict[int, EvalResult]:
    results: Dict[int, EvalResult] = {}
    # First get clean for ΔAcc
    clean = clean_eval(model, loader, device)
    clean_acc = clean["top1"]

    for s in severities:
        logits_all, labels_all = [], []
        alpha_all, beta_all = [], []
        with torch.no_grad():
            for images, labels in tqdm(loader, desc=f"Attack={attack_name} s={s}", leave=False):
                images = apply_attacks(images, attack_name=attack_name, severity=s)
                images = images.to(device)
                out = model.logits(images)

                # Handle models that return (logits, alpha, beta) vs just logits
                if isinstance(out, tuple):
                    logits, alpha, beta = out
                    alpha_all.append(alpha.cpu())
                    beta_all.append(beta.cpu())
                else:
                    logits = out

                logits_all.append(logits.cpu())
                labels_all.append(labels.cpu())

        logits = torch.cat(logits_all).detach()
        labels = torch.cat(labels_all)
        scaler = TemperatureScaler()
        scaler.fit(logits, labels)
        logits_t = scaler(logits)
        acc = top1_from_logits(logits_t, labels)
        results[s] = {
            "top1": acc,
            "macro_f1": macro_f1_from_logits(logits_t, labels),
            "ece": expected_calibration_error(logits_t, labels),
        }
        results[s]["delta_acc"] = clean_acc - acc  # type: ignore

        # Add alpha/beta stats if available
        if alpha_all:
            alpha_tensor = torch.cat(alpha_all)
            beta_tensor = torch.cat(beta_all)
            results[s]['mean_alpha'] = float(alpha_tensor.mean().item())  # type: ignore
            results[s]['std_alpha'] = float(alpha_tensor.std().item())  # type: ignore
            results[s]['mean_beta'] = float(beta_tensor.mean().item())  # type: ignore
            results[s]['std_beta'] = float(beta_tensor.std().item())  # type: ignore

    return results
