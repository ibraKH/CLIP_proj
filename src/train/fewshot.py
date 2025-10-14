from __future__ import annotations
import argparse, yaml
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ..common.seed import set_seed
from ..common.dist import get_device
from ..common.io import write_csv
from ..common.registry import DATASETS, METHODS
from ..datasets.sampler import kshot_indices


def build_support_loader(loader: DataLoader, k: int, seed: int, num_classes: int) -> Tuple[DataLoader, DataLoader]:
    # materialize labels
    labels: List[int] = []
    idx_map: List[int] = []
    ds = loader.dataset
    # attempt to read targets
    if hasattr(ds, "targets"):
        labels = list(map(int, ds.targets))
    elif hasattr(ds, "dataset") and hasattr(ds.dataset, "targets"):
        # for Subset
        base = ds.dataset
        labels = [int(base.targets[i]) for i in ds.indices]  # type: ignore
    else:
        # fallback via one pass
        for i in range(len(ds)):
            idx_map.append(i)
            _, y = ds[i]
            labels.append(int(y))
    support_idx, rest_idx = kshot_indices(labels, num_classes, k, seed)
    if hasattr(ds, "indices"):
        # map to original Subset indices
        support_idx = [ds.indices[i] for i in support_idx]  # type: ignore
        rest_idx = [ds.indices[i] for i in rest_idx]  # type: ignore

    sup = DataLoader(Subset(ds, support_idx), batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    rem = DataLoader(Subset(ds, rest_idx), batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    return sup, rem


def train_coop_like(model, support: DataLoader, val: DataLoader, epochs: int = 50, lr: float = 1e-2, patience: int = 5, device: torch.device = torch.device("cpu")):
    optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    ce = nn.CrossEntropyLoss()
    best_acc = 0.0
    wait = 0
    best_state = None
    for ep in range(1, epochs + 1):
        model.train()
        for images, labels in tqdm(support, desc=f"Train ep={ep}", leave=False):
            images, labels = images.to(device), labels.to(device)
            logits = model.logits(images)
            loss = ce(logits, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()
        # val
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val:
                images, labels = images.to(device), labels.to(device)
                logits = model.logits(images)
                correct += (logits.argmax(1) == labels).sum().item()
                total += labels.numel()
        acc = correct / max(1, total)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()


def main(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run, data_cfg, model_cfg = cfg["run"], cfg["data"], cfg["model"]
    method_cfg, eval_cfg = cfg["method"], cfg["eval"]

    set_seed(run["seed"])
    device = get_device(run.get("device", None))
    use_wandb = bool(run.get("use_wandb", False))

    # datasets
    ds_fn = DATASETS.get(data_cfg["dataset"])
    tr, va, te, classnames = ds_fn(
        root=data_cfg["root"],
        batch_size=int(data_cfg.get("batch_size", 64)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        img_size=int(data_cfg.get("img_size", 224)),
    )
    num_classes = len(classnames)
    # support from train
    support, rem_train = build_support_loader(tr, k=int(method_cfg.get("k_shot", 4)), seed=run["seed"], num_classes=num_classes)

    # method
    method_name = method_cfg["name"].lower()
    if method_name == "coop":
        from ..models.coop import CoOpHead
        model = CoOpHead(
            model_name=model_cfg["name"],
            pretrained=model_cfg["pretrained"],
            prompt_templates=model_cfg.get("prompt_templates", ["a photo of a {class}."]),
            classnames=classnames,
            device=str(device),
            n_ctx=int(method_cfg.get("n_ctx", 16)),
        )
        for p in model.clip.parameters():
            p.requires_grad = False
        train_coop_like(model, support, va, epochs=int(method_cfg.get("epochs", 50)), lr=float(method_cfg.get("lr", 1e-2)), patience=int(method_cfg.get("patience", 5)), device=device)
    elif method_name == "cocoop":
        from ..models.cocoop import CoCoOpHead
        model = CoCoOpHead(
            model_name=model_cfg["name"],
            pretrained=model_cfg["pretrained"],
            prompt_templates=model_cfg.get("prompt_templates", ["a photo of a {class}."]),
            classnames=classnames,
            device=str(device),
            hidden=int(method_cfg.get("hidden", 512)),
        )
        for p in model.clip.parameters():
            p.requires_grad = False
        train_coop_like(model, support, va, epochs=int(method_cfg.get("epochs", 50)), lr=float(method_cfg.get("lr", 1e-3)), patience=int(method_cfg.get("patience", 5)), device=device)
    elif method_name == "tipadapter":
        from ..models.tip_adapter import TipAdapter
        model = TipAdapter(
            model_name=model_cfg["name"],
            pretrained=model_cfg["pretrained"],
            prompt_templates=model_cfg.get("prompt_templates", ["a photo of a {class}."]),
            classnames=classnames,
            device=str(device),
            alpha=float(method_cfg.get("alpha", 0.5)),
            beta=float(method_cfg.get("beta", 5.0)),
            fine=bool(method_cfg.get("fine", False)),
        )
        # build cache
        with torch.no_grad():
            for images, labels in tqdm(support, desc="Build cache", leave=False):
                model.build_cache(images.to(device), labels.to(device))
    else:
        raise ValueError(f"Unknown method: {method_name}")

    # eval loops
    from ..eval.loops import clean_eval, attack_eval
    from ..common.io import write_csv

    clean = clean_eval(model, te, device)
    print(f"[CLEAN] top1={clean['top1']:.4f} macro_f1={clean['macro_f1']:.4f} ece={clean['ece']:.4f}")

    rows = [{"attack": "none", "severity": 0, **clean}]
    for atk in eval_cfg.get("attacks", []):
        if atk == "none":
            continue
        res = attack_eval(model, te, device, atk, severities=eval_cfg.get("severities", [0,1,2,3,4,5]))
        for s, m in res.items():
            rows.append({"attack": atk, "severity": s, **m})

    if eval_cfg.get("save_tables", True):
        write_csv(Path("reports/tables") / f"{run['exp_name']}_results.csv", rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    main(args.config)
