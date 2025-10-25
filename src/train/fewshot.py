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
from src.datasets.utils import build_fewshot_subset

def build_support_loader(loader: DataLoader, k: int, seed: int, num_classes: int) -> Tuple[DataLoader, DataLoader]:
    """
    Build few-shot support and the complementary remainder *using local indices* for the given loader.dataset,
    which may be a Dataset or a Subset(Dataset). Avoids mapping to base indices entirely.
    """
    ds = loader.dataset  # may be Dataset or Subset

    # Build support subset using local indices (works for Dataset or Subset)
    support_subset = build_fewshot_subset(ds, k=k, seed=seed)

    # Build complement (remainder) subset in local index space
    if hasattr(ds, "indices"):
        # Subset case: local index space is range(len(ds))
        local_all = set(range(len(ds)))
        local_support = set(support_subset.indices)  # type: ignore[attr-defined]
        local_rest = sorted(list(local_all - local_support))
        rest_subset = Subset(ds, local_rest)
    else:
        # Plain Dataset case
        local_all = set(range(len(ds)))
        # support_subset is Subset(ds, support_local_indices)
        local_support = set(support_subset.indices)  # type: ignore[attr-defined]
        local_rest = sorted(list(local_all - local_support))
        rest_subset = Subset(ds, local_rest)

    # Use sane loader params; keep batch size/num_workers from the original loader when possible
    bs = getattr(loader, "batch_size", 64) or 64
    nw = getattr(loader, "num_workers", 2)

    sup = DataLoader(support_subset, batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=True)
    rem = DataLoader(rest_subset,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
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
    elif method_name == "tipadapter_plus":
        from ..models.clip_wrapper import CLIPWrapper
        from ..models.cache_tools import build_cache, save_cache, load_cache
        from ..models.tip_adapter_plus import TipAdapterPlus
        from pathlib import Path
        import random

        # Get gate config
        gate_cfg = method_cfg.get("gate", {})
        alpha0 = float(gate_cfg.get("alpha0", 0.5))
        beta0 = float(gate_cfg.get("beta0", 5.0))
        gate_hidden = int(gate_cfg.get("hidden", 256))
        reg = float(gate_cfg.get("reg", 0.05))
        learn_beta = bool(gate_cfg.get("learn_beta", True))
        learn_scale = bool(gate_cfg.get("learn_scale", False))

        # Training config
        train_cfg = method_cfg.get("train", {})
        epochs = int(train_cfg.get("epochs", 30))
        lr = float(train_cfg.get("lr", 5e-4))
        wd = float(train_cfg.get("weight_decay", 1e-4))
        patience = int(train_cfg.get("patience", 5))

        # Robustness augmentation config
        robust_cfg = method_cfg.get("robust_augment", {})
        robust_enable = bool(robust_cfg.get("enable", False))
        robust_p = float(robust_cfg.get("p", 0.3))
        robust_attacks = robust_cfg.get("attacks", ["gaussian_blur", "gaussian_noise"])
        robust_sev_range = robust_cfg.get("severity_range", [1, 2])

        # Build CLIP wrapper
        clip_wrapper = CLIPWrapper(
            model_name=model_cfg["name"],
            pretrained=model_cfg["pretrained"],
            prompt_templates=model_cfg.get("prompt_templates", ["a photo of a {class}."]),
            classnames=classnames,
            device=str(device),
        )

        # Build or load cache
        cache_dir = Path(method_cfg.get("cache_dir", "project_results/cache"))
        cache_path = cache_dir / f"cache_{data_cfg['dataset']}_k{method_cfg.get('k_shot', 4)}_s{run['seed']}.pt"

        if cache_path.exists():
            print(f"Loading cache from {cache_path}")
            keys, values = load_cache(cache_path)
            keys = keys.to(device)
            values = values.to(device)
        else:
            print(f"Building cache and saving to {cache_path}")
            keys, values = build_cache(support, clip_wrapper, num_classes, device)
            keys = keys.to(device)
            values = values.to(device)
            save_cache(cache_path, keys.cpu(), values.cpu())

        # Instantiate TipAdapterPlus
        model = TipAdapterPlus(
            clip_wrapper=clip_wrapper,
            keys=keys,
            values=values,
            alpha0=alpha0,
            beta0=beta0,
            gate_hidden=gate_hidden,
            reg=reg,
            learn_beta=learn_beta,
            learn_scale=learn_scale,
        ).to(device)

        # Freeze CLIP
        for p in clip_wrapper.model.parameters():
            p.requires_grad = False

        # Count trainable params
        gate_params = sum(p.numel() for p in model.gate.parameters() if p.requires_grad)
        scale_params = sum(p.numel() for p in [model.post_scale] if p.requires_grad)
        total_params = gate_params + scale_params
        print(f"[TipAdapter++] Trainable params: gate={gate_params}, scale={scale_params}, total={total_params}")

        # Optimizer
        optim = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=wd,
        )

        # Training loop
        ce = nn.CrossEntropyLoss()
        best_acc = 0.0
        wait = 0
        best_state = None

        for ep in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            epoch_ce = 0.0
            epoch_reg = 0.0
            alpha_vals = []
            beta_vals = []

            for images, labels in tqdm(support, desc=f"Train ep={ep}/{epochs}", leave=False):
                images, labels = images.to(device), labels.to(device)

                # Optional robustness-aware augmentation
                if robust_enable and random.random() < robust_p:
                    from ..eval.attacks import apply_attacks
                    atk = random.choice(robust_attacks)
                    sev = random.randint(robust_sev_range[0], robust_sev_range[1])
                    images = apply_attacks(images, attack_name=atk, severity=sev)

                logits, alpha_hat, beta_hat = model(images)
                loss_ce = ce(logits, labels)
                loss_reg = model.reg_loss(alpha_hat, beta_hat)
                loss = loss_ce + loss_reg

                optim.zero_grad()
                loss.backward()
                optim.step()

                epoch_loss += loss.item()
                epoch_ce += loss_ce.item()
                epoch_reg += loss_reg.item()
                alpha_vals.append(alpha_hat.detach().mean().item())
                beta_vals.append(beta_hat.detach().mean().item())

            # Validation
            model.eval()
            correct = 0
            total = 0
            val_alpha = []
            val_beta = []

            with torch.no_grad():
                for images, labels in va:
                    images, labels = images.to(device), labels.to(device)
                    logits, alpha_hat, beta_hat = model(images)
                    correct += (logits.argmax(1) == labels).sum().item()
                    total += labels.numel()
                    val_alpha.append(alpha_hat.mean().item())
                    val_beta.append(beta_hat.mean().item())

            acc = correct / max(1, total)

            print(f"[Epoch {ep:02d}] loss={epoch_loss / len(support):.4f} (ce={epoch_ce / len(support):.4f}, reg={epoch_reg / len(support):.4f}) "
                  f"val_acc={acc:.4f} α_train={sum(alpha_vals)/len(alpha_vals):.3f} β_train={sum(beta_vals)/len(beta_vals):.2f} "
                  f"α_val={sum(val_alpha)/len(val_alpha):.3f} β_val={sum(val_beta)/len(val_beta):.2f}")

            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1

            if wait >= patience:
                print(f"[Early stop] at epoch {ep}, best_val_acc={best_acc:.4f}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
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
