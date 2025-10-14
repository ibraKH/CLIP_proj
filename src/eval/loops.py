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
def _collect_logits_labels(model, loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    logits_all, labels_all = [], []
    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device)
        logits = model.logits(images)
        logits_all.append(logits.cpu())
        labels_all.append(labels.cpu())
    return torch.cat(logits_all), torch.cat(labels_all)


def clean_eval(model, loader: DataLoader, device: torch.device) -> EvalResult:
    logits, labels = _collect_logits_labels(model, loader, device)
    # Fit temperature on the same loader (or pass a separate val if available)
    scaler = TemperatureScaler()
    scaler.fit(logits, labels)
    logits_t = scaler(logits)
    res: EvalResult = {
        "top1": top1_from_logits(logits_t, labels),
        "macro_f1": macro_f1_from_logits(logits_t, labels),
        "ece": expected_calibration_error(logits_t, labels),
    }
    return res


def attack_eval(model, loader: DataLoader, device: torch.device, attack_name: str, severities: List[int]) -> Dict[int, EvalResult]:
    results: Dict[int, EvalResult] = {}
    # First get clean for ΔAcc
    clean = clean_eval(model, loader, device)
    clean_acc = clean["top1"]

    for s in severities:
        logits_all, labels_all = [], []
        for images, labels in tqdm(loader, desc=f"Attack={attack_name} s={s}", leave=False):
            images = apply_attacks(images, attack_name=attack_name, severity=s)
            images = images.to(device)
            logits = model.logits(images)
            logits_all.append(logits.cpu())
            labels_all.append(labels.cpu())
        logits = torch.cat(logits_all)
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
    return results
