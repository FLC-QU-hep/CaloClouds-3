#!/usr/bin/env python3
"""
Fit a polynomial occupancy-correction curve for a CC3 variant: raw
(uncalibrated) ShowerFlow-predicted n_points -> real (step2point ground
truth) n_points, for the same conditioning-matched showers.

This is the CC3-repo equivalent of scripts/training/calculate_coef.ipynb_mirror.py's
real_coeff/fake_coeff polynomial fit. That notebook fits G4-detector-cell counts
(after projecting onto a regular grid with detector_map.get_projections + a MIP
cut) against a model's raw output, because CC2 conditions its diffusion model
directly on n_points and needs to correct for grid-quantisation effects.

The current CC3 pipeline (hdbscan/withincell/subcell variants) has no such
projection step: the step2point cc3input_<variant> files are already
pre-clustered, so "real n_points" is just the ground-truth hit count stored in
each file, and "model n_points" is just the count of above-threshold hits the
diffusion+ShowerFlow pipeline actually produces. So the fit here is a single,
direct polynomial: real_coeff = polyfit(raw_model_n, real_n, degree).

This replaces generate_showers.py's flat scale factor (_OCCUPANCY_SCALE_N,
scale_n = mean(real)/mean(raw)) with a count-dependent correction curve,
consumed the same way via gen_utils.get_scale_factor(num_clusters, coef_real,
coef_fake=None, n_splines=None).

Usage example
-------------
  python scripts/training/calculate_coef.py \\
      --log_dir log_dir/CD_S2P_RegularSubcell_2026_07_20__15_35_38 \\
      --n_showers 5000 \\
      --degree 3
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from matplotlib import pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(SCRIPTS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pointcloud.data.conditioning import read_raw_regaxes_withcond
from pointcloud.models.load import load_diffusion_model, load_flow_model
from pointcloud.utils.gen_utils import gen_cond_showers_batch, get_scale_factor
from pointcloud.utils.metadata import Metadata

from scripts.generate_showers import (  # noqa: E402
    _CONFIG_CLS,
    _SHOWER_FLOW,
    _detect_variant,
    _latest_checkpoint,
)
try:
    from scripts.generate_showers import ENERGY_THRESHOLD_GEV
except ImportError:
    ENERGY_THRESHOLD_GEV = 0.0

def _raw_model_n_points(model, distribution, cond_batch, config, chunk_size):
    """Generate showers with occupancy scaling disabled and count valid hits per shower."""
    n_total = cond_batch.shape[0]
    raw_n = np.empty(n_total, dtype=np.int64)
    start = 0
    while start < n_total:
        end = min(start + chunk_size, n_total)
        with torch.no_grad():
            batch_showers = gen_cond_showers_batch(
                model,
                distribution,
                cond_batch[start:end].to(config.device),
                bs=min(chunk_size, end - start),
                config=config,
            )
        # batch_showers col 3 is in MeV (see generate_showers._to_cc3_events docstring)
        raw_n[start:end] = (batch_showers[:, :, 3] > ENERGY_THRESHOLD_GEV * 1000.0).sum(axis=1)
        start = end
        print(f"  {end:>8} / {n_total} matched showers generated", flush=True)
    return raw_n


def main():
    parser = argparse.ArgumentParser(
        description="Fit a polynomial occupancy-correction curve for a CC3 variant.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log_dir",
        required=True,
        help="Path to the log_dir subdirectory containing checkpoints "
        "(e.g. log_dir/CD_S2P_RegularSubcell_2026_...).",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Specific checkpoint .pt file. Default: latest checkpoint in --log_dir.",
    )
    parser.add_argument(
        "--n_showers",
        type=int,
        default=5000,
        help="Number of conditioning-matched real showers to fit against.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Showers generated per call to gen_cond_showers_batch (controls memory usage).",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=3,
        help="Polynomial degree for the real_coeff fit (3 matches the CC2 notebook).",
    )
    parser.add_argument(
        "--cond_file",
        default=None,
        help="Path to a specific input_cc3 .h5 file to read real conditioning/n_points from. "
        "Default: the variant config's own dataset_path.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device (e.g. 'cpu', 'cuda', 'cuda:0').",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to write occupancy_coef_<variant>.npz/.png to. "
        "Default: the variant's showerFlow directory (next to input_norms.npz).",
    )
    args = parser.parse_args()

    log_dir = os.path.abspath(args.log_dir)
    variant = _detect_variant(log_dir)
    model_path = args.model_path or _latest_checkpoint(log_dir)

    print(f"Variant       : {variant}")
    print(f"Checkpoint    : {model_path}")
    print(f"N showers     : {args.n_showers}")
    print(f"Degree        : {args.degree}")

    # --- config and metadata ---
    config = _CONFIG_CLS[variant]()
    config.device = args.device
    if args.cond_file is not None:
        config.dataset_path = args.cond_file
    meta = Metadata(config)
    layer_centers = (
        meta.layer_bottom_pos_global + meta.cell_thickness_global / 2
    ).astype(np.float32)
    n_layers = len(layer_centers)

    sf = _SHOWER_FLOW[variant]
    output_dir = args.output_dir or sf["dir"]
    os.makedirs(output_dir, exist_ok=True)

    # --- load shower flow (raw, uncalibrated: no occupancy scaling) ---
    print("\nLoading shower flow...")
    _, distribution, _ = load_flow_model(config, sf["pth"])
    norms = np.load(os.path.join(sf["dir"], "input_norms.npz"))
    config.shower_flow_clusters_per_layer_norm = float(norms["clusters_per_layer_norm"])
    config.shower_flow_energy_per_layer_norm = float(norms["energy_per_layer_norm"])
    config.shower_flow_n_scaling = False

    # --- load diffusion model ---
    print("Loading diffusion model...")
    model, _, _, _ = load_diffusion_model(config, "Diffusion", model_path)
    print("  Done.")

    # --- real conditioning + ground-truth events, matched pairs ---
    print(f"\nReading {args.n_showers} real conditioning-matched showers...")
    conds_real, events_real = read_raw_regaxes_withcond(
        config, total_size=args.n_showers, for_model="diffusion"
    )
    events_real = np.asarray(events_real)
    real_n_points = (events_real[:, n_layers:, 3] > ENERGY_THRESHOLD_GEV).sum(axis=1)

    cond_batch = torch.tensor(np.array(conds_real)).float()
    if cond_batch.dim() == 1:
        cond_batch = cond_batch.unsqueeze(-1)

    # --- raw (uncalibrated) model occupancy for the same conditioning ---
    print(f"\nGenerating {cond_batch.shape[0]} matched showers (occupancy scaling disabled)...")
    raw_n_points = _raw_model_n_points(model, distribution, cond_batch, config, args.chunk_size)

    # --- fit ---
    scale_n_flat = real_n_points.mean() / max(raw_n_points.mean(), 1e-8)
    coef_real = np.polyfit(raw_n_points.astype(float), real_n_points.astype(float), args.degree)

    scale_factor = get_scale_factor(
        raw_n_points.astype(float), coef_real, None, None
    ).flatten()
    corrected_n_points = raw_n_points * scale_factor

    print("\nResults")
    print(f"  flat scale_n (old, mean-ratio)   : {scale_n_flat:.4f}")
    print(f"  polynomial coef_real (deg={args.degree}) : {coef_real}")
    print(
        f"  mean|raw - real| before: {np.abs(raw_n_points - real_n_points).mean():.1f}, "
        f"after: {np.abs(corrected_n_points - real_n_points).mean():.1f}"
    )

    # --- save ---
    out_path = os.path.join(output_dir, f"occupancy_coef_{variant}.npz")
    np.savez(
        out_path,
        coef_real=coef_real,
        degree=args.degree,
        scale_n_flat=scale_n_flat,
        variant=variant,
        n_showers=args.n_showers,
        raw_n_points=raw_n_points,
        real_n_points=real_n_points,
    )
    print(f"\nSaved coefficients to:\n  {out_path}")

    # incident energy per shower, for the energy-binned before/after panel
    # (diffusion cond_features_names always starts with "energy")
    incident_energy = cond_batch[:, 0].numpy()

    # --- diagnostic plot ---
    fig, axarr = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(f"Occupancy calibration — {variant} (before = raw model, after = poly-fit corrected)")

    ax = axarr[0]
    ax.scatter(raw_n_points, real_n_points, s=4, alpha=0.3, label="matched showers")
    xs = np.linspace(raw_n_points.min(), raw_n_points.max(), 200)
    ax.plot(xs, xs, color="gray", ls=":", label="before (y = x, uncalibrated)")
    ax.plot(xs, np.poly1d(coef_real)(xs), color="C1", label=f"after (degree-{args.degree} fit)")
    ax.plot(xs, scale_n_flat * xs, color="C2", ls="--", label="old flat scale_n")
    ax.set_xlabel("Raw model n_points (uncalibrated)")
    ax.set_ylabel("Real n_points")
    ax.legend(fontsize=8)

    ax = axarr[1]
    bins = np.linspace(0.5, 1.5, 60)
    ax.hist(raw_n_points / real_n_points, bins=bins, histtype="step", label="before (raw / real)")
    ax.hist(corrected_n_points / real_n_points, bins=bins, histtype="step", label="after (poly-fit / real)")
    ax.axvline(1.0, color="gray", ls=":")
    ax.set_xlabel("Gen / Real n_points ratio")
    ax.set_ylabel("Showers")
    ax.legend(fontsize=8)

    # energy-binned ratio, same layout as the "N hits per shower" Gen/Real
    # panel in compare_generated_vs_input_cc3.py's shower_level_observables plot
    ax = axarr[2]
    n_e_bins = 12
    e_edges = np.quantile(incident_energy, np.linspace(0, 1, n_e_bins + 1))
    e_edges[0] -= 1e-6
    e_centers = 0.5 * (e_edges[:-1] + e_edges[1:])
    bin_idx = np.digitize(incident_energy, e_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_e_bins - 1)
    before_ratio = np.full(n_e_bins, np.nan)
    after_ratio = np.full(n_e_bins, np.nan)
    for b in range(n_e_bins):
        sel = bin_idx == b
        if not np.any(sel):
            continue
        before_ratio[b] = (raw_n_points[sel] / real_n_points[sel]).mean()
        after_ratio[b] = (corrected_n_points[sel] / real_n_points[sel]).mean()
    ax.plot(e_centers, before_ratio, "o-", color="gray", label="before (raw / real)")
    ax.plot(e_centers, after_ratio, "o-", color="C1", label="after (poly-fit / real)")
    ax.axhline(1.0, color="gray", ls=":")
    ax.set_xlabel("Incident energy [GeV]")
    ax.set_ylabel("Mean Gen / Real n_points ratio")
    ax.legend(fontsize=8)

    fig.tight_layout()
    plot_path = os.path.join(output_dir, f"occupancy_coef_{variant}.png")
    fig.savefig(plot_path, dpi=150)
    print(f"Saved diagnostic plot to:\n  {plot_path}")


if __name__ == "__main__":
    main()
