from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .metrics import top1_from_logits, macro_f1_from_logits, TemperatureScaler, expected_calibration_error
from .attacks import apply_attacks
from ..common.typing import EvalResult

@torch.no_grad()
def _collect_logits_labels(model, loader: DataLoader, device: torch.device, max_batches: Optional[int]=None) -> Tuple[torch.Tensor, torch.Tensor]:
    logits_all, labels_all = [], []
    for bi, (images, labels) in enumerate(tqdm(loader, desc="Eval", leave=False)):
        if max_batches is not None and bi >= max_batches:
            break
        images = images.to(device)
        logits = model.logits(images)
        logits_all.append(logits.cpu())
        labels_all.append(labels.cpu())
    return torch.cat(logits_all), torch.cat(labels_all)

def clean_eval(model, loader: DataLoader, device: torch.device, max_batches: Optional[int]=None) -> EvalResult:
    logits, labels = _collect_logits_labels(model, loader, device, max_batches=max_batches)
    logits = logits.detach()
    scaler = TemperatureScaler()
    scaler.fit(logits, labels)
    logits_t = scaler(logits)
    return {
        "top1": top1_from_logits(logits_t, labels),
        "macro_f1": macro_f1_from_logits(logits_t, labels),
        "ece": expected_calibration_error(logits_t, labels),
    }

def attack_eval(model, loader: DataLoader, device: torch.device, attack_name: str, severities: List[int], max_batches: Optional[int]=None) -> Dict[int, EvalResult]:
    results: Dict[int, EvalResult] = {}
    clean = clean_eval(model, loader, device, max_batches=max_batches)
    clean_acc = clean["top1"]
    for s in severities:
        logits_all, labels_all = [], []
        with torch.no_grad():
            for bi, (images, labels) in enumerate(tqdm(loader, desc=f"Attack={attack_name} s={s}", leave=False)):
                if max_batches is not None and bi >= max_batches:
                    break
                images = apply_attacks(images, attack_name=attack_name, severity=s)
                images = images.to(device)
                logits = model.logits(images)
                logits_all.append(logits.cpu()); labels_all.append(labels.cpu())
        logits = torch.cat(logits_all).detach(); labels = torch.cat(labels_all)
        scaler = TemperatureScaler(); scaler.fit(logits, labels)
        logits_t = scaler(logits)
        acc = top1_from_logits(logits_t, labels)
        results[s] = {
            "top1": acc,
            "macro_f1": macro_f1_from_logits(logits_t, labels),
            "ece": expected_calibration_error(logits_t, labels),
            "delta_acc": clean_acc - acc,
        }
    return results