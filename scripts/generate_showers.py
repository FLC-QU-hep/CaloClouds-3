#!/usr/bin/env python3
"""
Generate a large number of CaloClouds-3 showers and save them to an HDF5 file
in the same format as the CC3 training input (input_cc3_file_*.h5).

Output layout
-------------
generated_showers/<log_dir_name>/<output_file>   (default: generated_showers.h5)

HDF5 datasets  (identical layout to input_cc3_file_*.h5 from step2point)
-------------
  energy        : float32  (n_showers, 1)              — incident energy [GeV]
  events        : float32  (n_showers, max_points, 4)  — CC3 local coordinates:
                                                          col 0 = local_x = z_global_aligned [mm]
                                                          col 1 = local_y = x_global_aligned [mm]
                                                          col 2 = layer_idx + fuzz  (float in [0, n_layers))
                                                          col 3 = hit energy [GeV]; padding rows are all-zero
  n_points      : int64    (n_showers,)                — number of valid (non-padded) hits per shower
  p_norm_local  : float64  (n_showers, 3)              — local momentum unit vector [px, py, pz]
  p_norm_global : float32  (n_showers, 3)              — global momentum unit vector [py_l, pz_l, px_l]
  theta_local   : float64  (n_showers,)                — polar angle from local z-axis [degrees]
  theta_global  : float32  (n_showers,)                — polar angle from global z-axis [degrees]
  phi_local     : float64  (n_showers,)                — azimuthal angle in local frame [degrees]
  phi_global    : float32  (n_showers,)                — azimuthal angle in global frame [degrees]

Coordinate notes
----------------
The CC3 generation pipeline already outputs showers in the aligned local coordinate
space (col 0/1 in mm, col 2 in global mm).  The only post-processing applied here is:
  * col 2: z_global_mm  →  argmin-to-layer-center + uniform fuzz in [0, 1)
  * col 3: MeV          →  GeV  (divide by 1000)

Usage example
-------------
  python scripts/generate_showers.py \\
      --log_dir log_dir/CD_S2P_HDBScan_2026_06_10__15_07_21 \\
      --n_showers 100000 \\
      --chunk_size 2000 \\
      --gen_batch_size 64
"""

import argparse
import os
import re
import sys

import h5py
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pointcloud.config_varients import (
    caloclouds_3_S2P_hdbscan,
    caloclouds_3_S2P_steps,
    caloclouds_3_S2P_withincell,
)
from pointcloud.data.conditioning import read_raw_regaxes_withcond
from pointcloud.models.load import load_diffusion_model, load_flow_model
from pointcloud.utils.gen_utils import gen_cond_showers_batch
from pointcloud.utils.metadata import Metadata  # noqa: E402

_SHOWER_FLOW = {
    "hdbscan": {
        "pth": "/eos/user/m/mamozzan/point-cloud-diffusion-data/showerFlow/input_cc3_hdbscan/ShowerFlow_alt1_nb2_inputs1152921504606846975_fnorms_best.pth",
        "dir": "/eos/user/m/mamozzan/point-cloud-diffusion-data/showerFlow/input_cc3_hdbscan",
    },
    "withincell": {
        "pth": "/eos/user/m/mamozzan/point-cloud-diffusion-data/showerFlow/input_cc3_withincell/ShowerFlow_alt1_nb2_inputs1152921504606846975_fnorms_best.pth",
        "dir": "/eos/user/m/mamozzan/point-cloud-diffusion-data/showerFlow/input_cc3_withincell",
    },
    "steps": {
        "pth": "/eos/user/m/mamozzan/point-cloud-diffusion-data/showerFlow/input_cc3_identity/ShowerFlow_alt1_nb2_inputs1152921504606846975_fnorms_best.pth",
        "dir": "/eos/user/m/mamozzan/point-cloud-diffusion-data/showerFlow/input_cc3_identity",
    },
}

_CONFIG_CLS = {
    "hdbscan": caloclouds_3_S2P_hdbscan.Configs,
    "withincell": caloclouds_3_S2P_withincell.Configs,
    "steps": caloclouds_3_S2P_steps.Configs,
}


def _detect_variant(log_dir: str) -> str:
    name = os.path.basename(os.path.normpath(log_dir)).lower()
    if "hdbscan" in name:
        return "hdbscan"
    if "withincell" in name:
        return "withincell"
    if "steps" in name:
        return "steps"
    raise ValueError(
        f"Cannot infer variant from '{os.path.basename(log_dir)}'. "
        "Folder name must contain 'HDBScan', 'WithinCell', or 'Steps'."
    )


def _latest_checkpoint(log_dir: str) -> str:
    pattern = re.compile(r"ckpt_[\d.]+_(\d+)\.pt$")
    best_step, best_file = -1, None
    for fname in os.listdir(log_dir):
        m = pattern.match(fname)
        if m:
            step = int(m.group(1))
            if step > best_step:
                best_step, best_file = step, os.path.join(log_dir, fname)
    if best_file is None:
        raise FileNotFoundError(f"No checkpoint found in {log_dir}")
    return best_file


def _to_cc3_events(batch_showers: np.ndarray, layer_centers: np.ndarray) -> np.ndarray:
    """Convert gen_cond_showers_batch output to input_cc3_file format.

    Parameters
    ----------
    batch_showers : (B, max_points, 4)
        Generated showers; col 0/1 already in aligned local mm, col 2 in global mm, col 3 in MeV.
    layer_centers : (n_layers,)
        Global y positions of layer centres (layer_bottom_pos_global + cell_thickness/2).

    Returns
    -------
    events : (B, max_points, 4)  float32
        CC3 input format: col 0/1 = mm (unchanged), col 2 = layer_idx + fuzz, col 3 = GeV.
    """
    events = batch_showers.copy()
    valid_mask = events[:, :, 3] > 0  # (B, max_points)

    # col 2: z_global_mm → layer_idx + uniform fuzz in [0, 1)
    z_mm = events[:, :, 2]  # (B, max_points)
    layer_idx = np.abs(z_mm[:, :, None] - layer_centers[None, None, :]).argmin(
        axis=2
    )  # (B, max_points)
    fuzz = np.random.uniform(0.0, 1.0, size=z_mm.shape).astype(np.float32)
    events[:, :, 2] = np.where(valid_mask, layer_idx.astype(np.float32) + fuzz, 0.0)

    # col 3: MeV → GeV
    events[:, :, 3] = np.where(valid_mask, events[:, :, 3] / 1000.0, 0.0)

    return events.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Generate CaloClouds-3 showers and save to HDF5 (input_cc3 format).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log_dir",
        required=True,
        help="Path to the log_dir subdirectory containing checkpoints "
        "(e.g. log_dir/CD_S2P_HDBScan_2026_...).",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Specific checkpoint .pt file. Default: latest checkpoint in --log_dir.",
    )
    parser.add_argument(
        "--n_showers",
        type=int,
        default=100_000,
        help="Total number of showers to generate.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=2_000,
        help="Showers generated per call to gen_cond_showers_batch (controls memory usage).",
    )
    parser.add_argument(
        "--gen_batch_size",
        type=int,
        default=64,
        help="Internal diffusion batch size passed to gen_cond_showers_batch.",
    )
    parser.add_argument(
        "--cond_pool_size",
        type=int,
        default=5_000,
        help="Number of real conditioning vectors loaded as the sampling pool. "
        "Cycled if n_showers > cond_pool_size.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device (e.g. 'cpu', 'cuda', 'cuda:0').",
    )
    parser.add_argument(
        "--output_base",
        default=None,
        help="Base directory for output. Default: <repo_root>/generated_showers.",
    )
    parser.add_argument(
        "--output_file",
        default="generated_showers.h5",
        help="HDF5 filename inside the output subdirectory.",
    )
    args = parser.parse_args()

    log_dir = os.path.abspath(args.log_dir)
    variant = _detect_variant(log_dir)
    model_path = args.model_path or _latest_checkpoint(log_dir)
    log_dir_name = os.path.basename(log_dir)
    output_base = args.output_base or os.path.join(
        REPO_ROOT, "generated_showers"
    )
    out_dir = os.path.join(output_base, log_dir_name, f"generated_showers_{args.n_showers}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.output_file)

    print(f"Variant       : {variant}")
    print(f"Checkpoint    : {model_path}")
    print(f"N showers     : {args.n_showers}")
    print(f"Chunk size    : {args.chunk_size}")
    print(f"Gen batch size: {args.gen_batch_size}")
    print(f"Output        : {out_path}")

    # --- config and metadata ---
    config = _CONFIG_CLS[variant]()
    config.device = args.device
    meta = Metadata(config)
    layer_centers = (
        meta.layer_bottom_pos_global + meta.cell_thickness_global / 2
    ).astype(np.float32)

    sf = _SHOWER_FLOW[variant]

    # --- load shower flow ---
    print("\nLoading shower flow...")
    flow, distribution, transforms = load_flow_model(config, sf["pth"])
    norms = np.load(os.path.join(sf["dir"], "input_norms.npz"))
    config.shower_flow_clusters_per_layer_norm = float(norms["clusters_per_layer_norm"])
    config.shower_flow_energy_per_layer_norm = float(norms["energy_per_layer_norm"])
    config.shower_flow_n_scaling = False
    print(
        f"  clusters_norm={config.shower_flow_clusters_per_layer_norm:.4f}  "
        f"energy_norm={config.shower_flow_energy_per_layer_norm:.4f}"
    )

    # --- load diffusion model ---
    print("Loading diffusion model...")
    model, _, _, _ = load_diffusion_model(config, "Diffusion", model_path)
    print("  Done.")

    # --- load conditioning pool ---
    print(f"\nLoading {args.cond_pool_size} real conditioning vectors...")
    conds_real, _ = read_raw_regaxes_withcond(
        config, total_size=args.cond_pool_size, for_model="diffusion"
    )
    cond_pool = torch.tensor(np.array(conds_real)).float()
    if cond_pool.dim() == 1:
        cond_pool = cond_pool.unsqueeze(-1)
    print(f"  Pool shape: {cond_pool.shape}")

    # Build full conditioning array by cycling the pool
    n_total = args.n_showers
    pool_size = len(cond_pool)
    indices = np.tile(np.arange(pool_size), (n_total // pool_size) + 1)[:n_total]
    full_cond = cond_pool[indices]  # (n_total, cond_dim)

    # --- generate and write incrementally ---
    print(f"\nGenerating {n_total} showers -> {out_path}")
    chunk_events = (min(args.chunk_size, n_total), config.max_points, 4)

    with h5py.File(out_path, "w", locking=False) as f:
        energy_ds = f.create_dataset("energy", shape=(n_total, 1), dtype=np.float32)
        events_ds = f.create_dataset(
            "events",
            shape=(n_total, config.max_points, 4),
            dtype=np.float32,
            chunks=chunk_events,
        )
        n_points_ds = f.create_dataset("n_points", shape=(n_total,), dtype=np.int64)
        p_norm_local_ds = f.create_dataset(
            "p_norm_local", shape=(n_total, 3), dtype=np.float64
        )
        p_norm_global_ds = f.create_dataset(
            "p_norm_global", shape=(n_total, 3), dtype=np.float32
        )
        theta_local_ds = f.create_dataset(
            "theta_local", shape=(n_total,), dtype=np.float64
        )
        theta_global_ds = f.create_dataset(
            "theta_global", shape=(n_total,), dtype=np.float32
        )
        phi_local_ds = f.create_dataset("phi_local", shape=(n_total,), dtype=np.float64)
        phi_global_ds = f.create_dataset(
            "phi_global", shape=(n_total,), dtype=np.float32
        )

        start = 0
        while start < n_total:
            end = min(start + args.chunk_size, n_total)
            batch_cond = full_cond[start:end].to(config.device)

            with torch.no_grad():
                batch_showers = gen_cond_showers_batch(
                    model,
                    distribution,
                    batch_cond,
                    bs=args.gen_batch_size,
                    config=config,
                )

            # --- convert to input_cc3 format ---
            events = _to_cc3_events(batch_showers, layer_centers)

            # conditioning vectors: [energy_GeV, px_local, py_local, pz_local]
            cond_np = full_cond[start:end].numpy()
            inc_energy = cond_np[:, 0:1].astype(np.float32)  # (chunk, 1)
            p_norm_local = cond_np[:, 1:4].astype(np.float64)  # (chunk, 3)
            p_norm_global = p_norm_local[:, [1, 2, 0]].astype(
                np.float32
            )  # local→global: (py,pz,px)

            theta_local = np.degrees(np.arccos(np.clip(p_norm_local[:, 2], -1.0, 1.0)))
            phi_local = np.degrees(np.arctan2(p_norm_local[:, 1], p_norm_local[:, 0]))
            theta_global = np.degrees(
                np.arccos(np.clip(p_norm_global[:, 2], -1.0, 1.0))
            ).astype(np.float32)
            phi_global = np.degrees(
                np.arctan2(p_norm_global[:, 1], p_norm_global[:, 0])
            ).astype(np.float32)

            n_points = (batch_showers[:, :, 3] > 0).sum(axis=1).astype(np.int64)

            energy_ds[start:end] = inc_energy
            events_ds[start:end] = events
            n_points_ds[start:end] = n_points
            p_norm_local_ds[start:end] = p_norm_local
            p_norm_global_ds[start:end] = p_norm_global
            theta_local_ds[start:end] = theta_local
            theta_global_ds[start:end] = theta_global
            phi_local_ds[start:end] = phi_local
            phi_global_ds[start:end] = phi_global

            start = end
            print(f"  {end:>8} / {n_total} showers written", flush=True)

    print(f"\nDone. {n_total} showers saved to:\n  {out_path}")
    print("\nHDF5 layout:")
    print(f"  energy        : float32  ({n_total}, 1)                      [GeV]")
    print(
        f"  events        : float32  ({n_total}, {config.max_points}, 4) [local_x mm, local_y mm, layer_idx+fuzz, E_GeV]"
    )
    print(f"  n_points      : int64    ({n_total},)")
    print(f"  p_norm_local  : float64  ({n_total}, 3)")
    print(f"  p_norm_global : float32  ({n_total}, 3)")
    print(f"  theta/phi_local  : float64  ({n_total},)  [degrees]")
    print(f"  theta/phi_global : float32  ({n_total},)  [degrees]")


if __name__ == "__main__":
    main()
