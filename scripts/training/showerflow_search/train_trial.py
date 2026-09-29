"""
Train and evaluate ONE trial of the shower-flow hyperparameter search.

Looks up its (af_dim, num_blocks) from <search_root>/trials.json by --trial_id,
trains via the real scripts/training/ShowerFlow.py::main (so results match
normal, non-search runs), then scores the resulting checkpoint by Wasserstein
distance between sampled and truth num_points / visible_energy (see common.py
for why NLL loss alone isn't a reliable score here).

Usage:
    cd /eos/user/m/mamozzan/CaloClouds-3
    python scripts/training/showerflow_search/train_trial.py \
        --search_root /path/to/search_root --trial_id 3 --epochs 800
"""

import argparse
import json
import os
import traceback

import torch

import stable_log1
from common import (
    apply_trial_overrides,
    import_module_from_path,
    link_cache_files,
    load_config,
    trial_dir,
    wasserstein_metrics,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search_root", required=True)
    parser.add_argument("--trial_id", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=2048)
    args = parser.parse_args()

    manifest_path = os.path.join(args.search_root, "trials.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    trial = next(t for t in manifest["trials"] if t["trial_id"] == args.trial_id)
    version = trial.get("shower_flow_version", "alt1")
    print(
        f"Trial {args.trial_id}: af_dim={trial['af_dim']} "
        f"num_blocks={trial['num_blocks']} shower_flow_version={version}"
    )

    config = load_config(manifest["base_config"])
    config = apply_trial_overrides(
        config,
        trial["af_dim"],
        trial["num_blocks"],
        version,
        args.search_root,
        args.trial_id,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.device = device

    from pointcloud.utils import showerflow_utils

    showerflow_dir = showerflow_utils.get_showerflow_dir(config)
    link_cache_files(manifest["cache_dir"], showerflow_dir)

    if version == "log1_stable":
        loc, scale = stable_log1.register(manifest["cache_dir"], device)
        print(
            f"Registered log1_stable: log-space loc range "
            f"[{loc.min():.2f}, {loc.max():.2f}], scale range "
            f"[{scale.min():.2f}, {scale.max():.2f}]"
        )

    showerflow_script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ShowerFlow.py"
    )
    showerflow_mod = import_module_from_path("_search_showerflow_main", showerflow_script)

    out_dir = trial_dir(args.search_root, args.trial_id)
    base_metrics = {
        "trial_id": args.trial_id,
        "af_dim": trial["af_dim"],
        "num_blocks": trial["num_blocks"],
        "shower_flow_version": version,
        "epochs": args.epochs,
    }

    try:
        best_model_path, best_data_path, history_data_path = showerflow_mod.main(
            config, batch_size=args.batch_size, total_epochs=args.epochs
        )

        with open(best_data_path) as f:
            best_loss, best_epoch = f.read().strip().split()

        metrics = wasserstein_metrics(config, best_model_path, device)
        metrics.update(base_metrics)
        metrics.update(
            {
                "best_val_nll_loss": float(best_loss),
                "best_epoch": int(best_epoch),
                "best_model_path": best_model_path,
                "failed": False,
            }
        )
    except Exception as exc:
        # A trial failing (e.g. NaN weights) shouldn't take the whole search
        # down - record it and let run_all_sequential.sh / pipeline.sh move
        # on to the next trial.
        print(f"Trial {args.trial_id} FAILED: {exc}")
        traceback.print_exc()
        metrics = dict(base_metrics)
        metrics.update({"failed": True, "error": f"{type(exc).__name__}: {exc}"})

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Trial {args.trial_id} done: {metrics}")


if __name__ == "__main__":
    main()
