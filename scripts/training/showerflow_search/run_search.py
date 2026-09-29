"""
Generate a random-search manifest (trials.json) of (af_dim, num_blocks,
shower_flow_version) combinations to try for a given shower-flow base config.

shower_flow_version choices:
  "alt1"        - current default, models clusters_per_layer/energy_per_layer
                  directly on their (linearly rescaled) scale.
  "log1"        - the repo's built-in log-scale variant (compile_HybridTanH_log1
                  / SafeExpTransform in pointcloud/models/shower_flow.py).
                  DO NOT USE - numerically broken for this data: most of the
                  60 per-layer inputs are exact zeros, so log(x+eps) puts them
                  ~15 units away from the nonzero ones with nothing to
                  renormalize that before the coupling stack, which reliably
                  produces NaN weights within the first epoch (confirmed: see
                  trial 2 in this search's first run).
  "log1_stable" - the log-vs-linear comparison that actually works: same
                  idea as "log1", but with z-score standardization of the
                  log-space inputs (stats computed from the cached data) and
                  a clamp before exp() as a safety net. Implemented in
                  pointcloud/models/stable_log1.py - registered into
                  shower_flow.versions_dict at runtime by
                  showerflow_utils.ensure_version_registered.

Usage:
    cd /eos/user/m/mamozzan/CaloClouds-3
    python scripts/training/showerflow_search/run_search.py \
        --base_config pointcloud/config_varients/caloclouds_3_S2P_hdbscan_ms3_mcs10.py \
        --cache_dir /eos/user/m/mamozzan/point-cloud-diffusion-data/showerFlow/input_cc3_hdbscan_ms3_mcs10 \
        --search_root /eos/user/m/mamozzan/point-cloud-diffusion-data/showerFlow_search/hdbscan_ms3_mcs10 \
        --n_trials 16 --af_dim_min 4 --af_dim_max 16 \
        --num_blocks_min 2 --num_blocks_max 6 \
        --shower_flow_versions alt1,log1_stable --seed 0

This only writes trials.json (and a couple of always-baseline trials) - it
does not train anything. Run train_trial.py for each trial afterwards
(see run_all_sequential.sh in this directory).
"""

import argparse
import json
import os
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_config", required=True, help="Path to a config_varients/*.py file")
    parser.add_argument(
        "--cache_dir",
        required=True,
        help="Existing showerFlow/<dataset> dir for this dataset, used as the "
        "shared read-only cache so each trial doesn't recompute clusters/energy "
        "per layer, cog, etc. from scratch.",
    )
    parser.add_argument("--search_root", required=True, help="Output dir for the search")
    parser.add_argument("--n_trials", type=int, default=16)
    parser.add_argument("--af_dim_min", type=int, default=4)
    parser.add_argument("--af_dim_max", type=int, default=16)
    parser.add_argument("--num_blocks_min", type=int, default=2)
    parser.add_argument("--num_blocks_max", type=int, default=6)
    parser.add_argument(
        "--shower_flow_versions",
        default="alt1,log1_stable",
        help="Comma-separated shower_flow_version choices to search over. "
        "'alt1' and 'log1_stable' are supported out of the box; 'log1' "
        "(the original, unfixed log-scale variant) is numerically broken "
        "for this data and shouldn't be included.",
    )
    parser.add_argument(
        "--include_baselines",
        action="store_true",
        default=True,
        help="Always include (af_dim=6, nb=2, alt1), (af_dim=10, nb=2, alt1) "
        "and (af_dim=6, nb=2, log1_stable) as trials 0/1/2, so the search is "
        "directly comparable to the runs already on disk and isolates the "
        "log-vs-linear-scale question with everything else held fixed.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    versions = [v.strip() for v in args.shower_flow_versions.split(",") if v.strip()]

    trials = []
    if args.include_baselines:
        trials.append(
            {"af_dim": 6, "num_blocks": 2, "shower_flow_version": "alt1", "note": "baseline_lessparams"}
        )
        trials.append(
            {"af_dim": 10, "num_blocks": 2, "shower_flow_version": "alt1", "note": "baseline_current"}
        )
        if "log1_stable" in versions:
            trials.append(
                {"af_dim": 6, "num_blocks": 2, "shower_flow_version": "log1_stable", "note": "baseline_log_scale"}
            )

    seen = {(t["af_dim"], t["num_blocks"], t["shower_flow_version"]) for t in trials}
    while len(trials) < args.n_trials:
        af_dim = rng.randint(args.af_dim_min, args.af_dim_max)
        num_blocks = rng.randint(args.num_blocks_min, args.num_blocks_max)
        version = rng.choice(versions)
        if (af_dim, num_blocks, version) in seen:
            continue
        seen.add((af_dim, num_blocks, version))
        trials.append(
            {"af_dim": af_dim, "num_blocks": num_blocks, "shower_flow_version": version, "note": ""}
        )

    for i, t in enumerate(trials):
        t["trial_id"] = i

    os.makedirs(args.search_root, exist_ok=True)
    manifest = {
        "base_config": os.path.abspath(args.base_config),
        "cache_dir": os.path.abspath(args.cache_dir),
        "trials": trials,
    }
    manifest_path = os.path.join(args.search_root, "trials.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(trials)} trials to {manifest_path}")
    for t in trials:
        print(
            f"  trial {t['trial_id']:2d}: af_dim={t['af_dim']:2d} "
            f"num_blocks={t['num_blocks']} version={t['shower_flow_version']:>4}  {t['note']}"
        )

    runner_path = os.path.join(args.search_root, "run_all_sequential.sh")
    with open(runner_path, "w") as f:
        f.write(
            "#!/bin/bash\n"
            "# Runs every trial in trials.json one after another on this machine.\n"
            "# Intended for an interactive GPU session - adapt into a Slurm array\n"
            "# or HTCondor submit file if you want them run in parallel/queued.\n"
            "set -e\n"
            f"SEARCH_ROOT={args.search_root}\n"
            f"N_TRIALS={len(trials)}\n"
            "cd /eos/user/m/mamozzan/CaloClouds-3\n"
            "source caloclouds3/bin/activate\n"
            'for ((i=0; i<N_TRIALS; i++)); do\n'
            '    echo "=== trial $i / $((N_TRIALS-1)) ==="\n'
            "    python scripts/training/showerflow_search/train_trial.py "
            '--search_root "$SEARCH_ROOT" --trial_id "$i" --epochs 800\n'
            "done\n"
            "python scripts/training/showerflow_search/summarize_results.py "
            '--search_root "$SEARCH_ROOT"\n'
        )
    os.chmod(runner_path, 0o755)
    print(f"Wrote {runner_path}")


if __name__ == "__main__":
    main()
