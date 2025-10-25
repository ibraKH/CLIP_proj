from __future__ import annotations
import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Any
import itertools
import copy

from .fewshot import main as run_fewshot
from ..common.io import write_csv


def run_ablation_grid(
    base_config_path: str,
    output_dir: str = "reports/tables",
    grid_params: Dict[str, List[Any]] | None = None,
) -> None:
    """
    Run ablation experiments for Tip-Adapter++ over a grid of hyperparameters.

    Args:
        base_config_path: path to base config file
        output_dir: directory to save results
        grid_params: dict of parameter names to lists of values to sweep
    """
    with open(base_config_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    # Default grid if not provided
    if grid_params is None:
        grid_params = {
            "gate.hidden": [64, 256],
            "gate.reg": [0.0, 0.05, 0.1],
            "gate.learn_beta": [False, True],
            "robust_augment.enable": [False, True],
        }

    # Generate all combinations
    param_names = list(grid_params.keys())
    param_values = list(grid_params.values())
    combinations = list(itertools.product(*param_values))

    print(f"Running {len(combinations)} ablation experiments...")

    all_results = []

    for i, values in enumerate(combinations):
        print(f"\n[Ablation {i+1}/{len(combinations)}]")

        # Create config variant
        cfg = copy.deepcopy(base_cfg)
        variant_name_parts = []

        for param_name, param_value in zip(param_names, values):
            # Set nested parameter
            keys = param_name.split(".")
            d = cfg["method"]
            for k in keys[:-1]:
                if k not in d:
                    d[k] = {}
                d = d[k]
            d[keys[-1]] = param_value

            # Build variant name
            short_name = param_name.replace("gate.", "").replace("robust_augment.", "ra_")
            variant_name_parts.append(f"{short_name}={param_value}")

        variant_name = "_".join(variant_name_parts)
        cfg["run"]["exp_name"] = f"tipadapter_plus_{variant_name}"

        print(f"Variant: {variant_name}")
        for pn, pv in zip(param_names, values):
            print(f"  {pn}: {pv}")

        # Save temp config
        temp_cfg_path = Path("configs") / f"temp_ablation_{i}.yaml"
        with open(temp_cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)

        # Run experiment
        try:
            run_fewshot(str(temp_cfg_path))

            # Collect results (assume they were saved to the standard location)
            results_path = Path("reports/tables") / f"{cfg['run']['exp_name']}_results.csv"
            if results_path.exists():
                import csv
                with open(results_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        row["variant"] = variant_name
                        for pn, pv in zip(param_names, values):
                            row[pn] = str(pv)
                        all_results.append(row)
        except Exception as e:
            print(f"ERROR in variant {variant_name}: {e}")
        finally:
            # Clean up temp config
            if temp_cfg_path.exists():
                temp_cfg_path.unlink()

    # Save aggregated results
    output_path = Path(output_dir) / "tipadapter_plus_ablation.csv"
    write_csv(output_path, all_results)
    print(f"\nAblation results saved to {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Base config file")
    ap.add_argument("--output", type=str, default="reports/tables", help="Output directory")
    args = ap.parse_args()

    run_ablation_grid(args.config, args.output)
