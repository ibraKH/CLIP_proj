from __future__ import annotations
import argparse, yaml
from pathlib import Path
import torch
from tqdm import tqdm

from ..common.seed import set_seed
from ..common.dist import get_device
from ..common.io import write_csv
from ..common.registry import DATASETS
from ..models.clip_wrapper import CLIPWrapper
from ..eval.loops import clean_eval, attack_eval
from ..eval.reliability import save_reliability_diagram
import src.datasets


def main(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run = cfg["run"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    eval_cfg = cfg["eval"]

    set_seed(run["seed"])
    device = get_device(run.get("device", None))
    use_wandb = bool(run.get("use_wandb", False))

    # dataset
    ds_fn = DATASETS.get(data_cfg["dataset"])
    tr, va, te, classnames = ds_fn(
        root=data_cfg["root"],
        batch_size=int(data_cfg.get("batch_size", 64)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        img_size=int(data_cfg.get("img_size", 224)),
    )

    model = CLIPWrapper(
        model_name=model_cfg["name"],
        pretrained=model_cfg["pretrained"],
        prompt_templates=model_cfg.get("prompt_templates", ["a photo of a {class}."]),
        classnames=classnames,
        device=str(device),
    )  

    # clean eval (use validation for calibration and report on test)
    print("Running clean eval on test...")
    clean = clean_eval(model, te, device)
    print(f"[CLEAN] top1={clean['top1']:.4f} macro_f1={clean['macro_f1']:.4f} ece={clean['ece']:.4f}")

    # save reliability diagram on test
    # collect logits/labels again for the plot with calibrated logits
    # quick pass:
    from ..eval.metrics import TemperatureScaler
    logits_all, labels_all = [], []
    with torch.no_grad():
        for images, labels in tqdm(te, desc="Collect for reliability", leave=False):
            images = images.to(device)
            logits_all.append(model.logits(images).cpu())
            labels_all.append(labels.cpu())
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    scaler = TemperatureScaler()
    scaler.fit(logits, labels)
    save_reliability_diagram(
        scaler(logits),
        labels,
        out_path=Path("reports/figs") / f"{run['exp_name']}_reliability.png",
    )

    rows = [{"attack": "none", "severity": 0, **clean}]
    # attacks
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