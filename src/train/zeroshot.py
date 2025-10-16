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
    
    limit_per_class = int(data_cfg.get("limit_per_class", 0)) or None
    class_limit = int(data_cfg.get("class_limit", 0)) or None
    run = cfg["run"]; data_cfg = cfg["data"]; model_cfg = cfg["model"]; eval_cfg = cfg["eval"]
    set_seed(run["seed"])
    device = get_device(run.get("device", None))
    verbose = bool(run.get("verbose", False))

    if verbose:
        print(f"[config] dataset={data_cfg['dataset']} root={data_cfg['root']} bs={data_cfg.get('batch_size',64)} "
              f"workers={data_cfg.get('num_workers',4)} img={data_cfg.get('img_size',224)}")

    # dataset
    ds_fn = DATASETS.get(data_cfg["dataset"])
    tr, va, te, classnames = ds_fn(
        root=data_cfg["root"],
        batch_size=int(data_cfg.get("batch_size", 64)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        img_size=int(data_cfg.get("img_size", 224)),
        limit_per_class=limit_per_class,
        class_limit=class_limit,
        seed=int(run.get("seed", 42)),
    )
    if verbose:
        def _len(loader): 
            try: return len(loader.dataset)
            except Exception: return None
        print(f"[data] train={_len(tr)} val={_len(va)} test={_len(te)} classes={len(classnames)}")

    # model
    if verbose:
        print(f"[model] loading open_clip {model_cfg['name']} / {model_cfg['pretrained']}")
        print(f"[model] prompts={model_cfg.get('prompt_templates', ['a photo of a {class}.'])}")
    model = CLIPWrapper(
        model_name=model_cfg["name"],
        pretrained=model_cfg["pretrained"],
        prompt_templates=model_cfg.get("prompt_templates", ["a photo of a {class}."]),
        classnames=classnames,
        device=str(device),
    )
    if verbose:
        D = int(model.text_features.shape[1])
        print(f"[model] text features ready: shape={tuple(model.text_features.shape)} (dim={D})")

    # clean eval
    if verbose: print("[eval] clean pass (with temperature scaling)")
    clean = clean_eval(model, te, device, max_batches=max_batches)
    print(f"[CLEAN] top1={clean['top1']:.4f} macro_f1={clean['macro_f1']:.4f} ece={clean['ece']:.4f}")

    # reliability
    if bool(eval_cfg.get("save_figs", True)):
        if verbose: print("[eval] saving reliability diagram")
        from ..eval.metrics import TemperatureScaler
        logits_all, labels_all = [], []
        with torch.no_grad():
            for images, labels in tqdm(te, desc="Collect for reliability", leave=False):
                images = images.to(device)
                logits_all.append(model.logits(images).cpu())
                labels_all.append(labels.cpu())
        logits = torch.cat(logits_all); labels = torch.cat(labels_all)
        scaler = TemperatureScaler(); scaler.fit(logits, labels)
        out_fig = Path("reports/figs") / f"{run['exp_name']}_reliability.png"
        save_reliability_diagram(scaler(logits), labels, out_path=out_fig)
        if verbose: print(f"[eval] saved {out_fig}")

    rows = [{"attack": "none", "severity": 0, "delta_acc": 0.0, **clean}]

    # attacks
    attacks = eval_cfg.get("attacks", [])
    severities = eval_cfg.get("severities", [0,1,2,3,4,5])
    max_batches = int(eval_cfg.get("max_batches", 0)) or None
    for atk in attacks:
        if atk == "none": 
            continue
        if verbose: print(f"[eval] attack='{atk}' severities={severities} max_batches={max_batches}")
        res = attack_eval(model, te, device, atk, severities=severities, max_batches=max_batches)
        for s, m in res.items():
            rows.append({"attack": atk, "severity": s, **m})

    if eval_cfg.get("save_tables", True):
        out_csv = Path("reports/tables") / f"{run['exp_name']}_results.csv"
        write_csv(out_csv, rows)
        if verbose: print(f"[save] wrote {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    main(args.config)