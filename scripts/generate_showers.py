#!/usr/bin/env python3
"""
Generate a large number of CaloClouds-3 showers and save them to an HDF5 file
in the same format as convert_to_cc3_format_layercounts_test.py (step2point):
per-layer hit counts prepended to the 'events' array, layer stored as a plain
integer (no fuzz). Coordinates are already in local space, so no coordinate
transformation happens here beyond what gen_cond_showers_batch outputs.

Output layout
-------------
generated_showers/<log_dir_name>/<output_file>   (default: generated_showers.h5)

HDF5 datasets
-------------
  energy        : float32  (n_showers, 1)                      — incident energy [GeV]
  events        : float32  (n_showers, n_layers + max_points, 4)
                                                                 — events[:, :n_layers, :] = per-layer hit
                                                                   counts, repeated across all 4 columns
                                                                 — events[:, n_layers:, :] = actual hits,
                                                                   CC3 local coordinates:
                                                                   col 0 = local_x = z_global_aligned [mm]
                                                                   col 1 = local_y = x_global_aligned [mm]
                                                                   col 2 = layer_idx (int, no fuzz)
                                                                   col 3 = hit energy [GeV or MeV, see
                                                                           --energy_units / attrs["hit_energy_units"]];
                                                                           padding rows all-zero
  layer_counts  : float32  (n_showers, n_layers)                — same per-layer hit counts, standalone
  n_points      : int64    (n_showers,)                         — number of valid (non-padded) hits per shower
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
  * col 2: z_global_mm  →  argmin-to-layer-center  →  plain int layer index
  * col 3: MeV          →  GeV (default) or left as MeV, see --energy_units

--energy_units MeV is for feeding DDML's LoadHdf5 loader directly: its C++ side
(CaloCloudsTwoAngleModel.cc convertOutput) applies the hit energy value as a raw
G4/DD4hep native unit (MeV) with no GeV->MeV conversion, so writing GeV there silently
undervalues every hit by 1000x.

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

XY_HALF_SIZE_MM = 250.0  # local x/y clamp range, matching the training data's 500x500mm box cut

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pointcloud.config_varients import (
    caloclouds_3_S2P_hdbscan_ms8_mcs40,
    caloclouds_3_S2P_hdbscan_ms12_mcs12,
    caloclouds_3_S2P_hdbscan_ms3_mcs10,
    caloclouds_3_S2P_hdbscan_ms3_mcs3,
    caloclouds_3_S2P_hdbscan_ms7_mcs7,
    caloclouds_3_S2P_steps,
    caloclouds_3_S2P_subcell,
    caloclouds_3_S2P_withincell,
    caloclouds_3_S2P_subcell_6kcut,
)
from pointcloud.data.conditioning import read_raw_regaxes_withcond
from pointcloud.models.load import load_diffusion_model, load_flow_model
from pointcloud.utils import showerflow_utils
from pointcloud.utils.gen_utils import gen_cond_showers_batch
from pointcloud.utils.metadata import Metadata  # noqa: E402

_SHOWER_FLOW = {
    "hdbscan_ms8_mcs40": {
        "pth": "/eos/user/m/mamozzan/point-cloud-diffusion-data/showerFlow/input_cc3_hdbscan_ms8_mcs40/ShowerFlow_log1_stable_nb4_inputs1152921504606846975_fnorms_best.pth",
        "dir": "/eos/user/m/mamozzan/point-cloud-diffusion-data/showerFlow/input_cc3_hdbscan_ms8_mcs40",
    },
    # log1_stable af_dim=14 / num_blocks=4, the winner of the 1500-epoch
    # af_dim/num_blocks/version search (combined Wasserstein 13.12 vs 17.67 for
    # the best alt1 there), retrained here as a full run - best val NLL -135.68
    # at epoch 2983. The old alt1_nb2 checkpoint this used to point at now lives
    # in .../showerFlow/input_cc3_hdbscan_ms3_mcs10_old/.
    # log1_stable is registered at runtime from this dir's cached per-layer
    # arrays (see pointcloud/models/stable_log1.py), so "dir" is not just where
    # input_norms.npz lives - it also fixes the flow's log-space standardization.
    "hdbscan_ms3_mcs10": {
        "pth": "/eos/user/m/mamozzan/point-cloud-diffusion-data/showerFlow/input_cc3_hdbscan_ms3_mcs10/ShowerFlow_log1_stable_nb4_inputs1152921504606846975_fnorms_best.pth",
        "dir": "/eos/user/m/mamozzan/point-cloud-diffusion-data/showerFlow/input_cc3_hdbscan_ms3_mcs10",
    },
    # Same log1_stable af_dim=14 / num_blocks=4 setup as hdbscan_ms3_mcs10 above,
    # trained 2026-09-10/11. As there, "dir" also fixes the flow's log-space
    # standardization stats.
    "hdbscan_ms3_mcs3": {
        "pth": "/eos/user/m/mamozzan/point-cloud-diffusion-data/showerFlow/input_cc3_hdbscan_ms3_mcs3/ShowerFlow_log1_stable_nb4_inputs1152921504606846975_fnorms_best.pth",
        "dir": "/eos/user/m/mamozzan/point-cloud-diffusion-data/showerFlow/input_cc3_hdbscan_ms3_mcs3",
    },
    # Same log1_stable af_dim=14 / num_blocks=4 setup as hdbscan_ms3_mcs10 above
    # (best val NLL written to ..._fnorms_best_data.txt), trained 2026-09-09.
    # As there, "dir" also fixes the flow's log-space standardization stats.
    "hdbscan_ms7_mcs7": {
        "pth": "/eos/user/m/mamozzan/point-cloud-diffusion-data/showerFlow/input_cc3_hdbscan_ms7_mcs7/ShowerFlow_log1_stable_nb4_inputs1152921504606846975_fnorms_best.pth",
        "dir": "/eos/user/m/mamozzan/point-cloud-diffusion-data/showerFlow/input_cc3_hdbscan_ms7_mcs7",
    },
    "hdbscan_ms12_mcs12": {
        "pth": "/eos/user/m/mamozzan/point-cloud-diffusion-data/showerFlow/input_cc3_hdbscan_ms12_mcs12/ShowerFlow_log1_stable_nb4_inputs1152921504606846975_fnorms_best.pth",
        "dir": "/eos/user/m/mamozzan/point-cloud-diffusion-data/showerFlow/input_cc3_hdbscan_ms12_mcs12",
    },
    "withincell": {
        "pth": "/eos/user/m/mamozzan/point-cloud-diffusion-data/showerFlow/input_cc3_merge_within_cell/ShowerFlow_log1_stable_nb4_inputs1152921504606846975_fnorms_best.pth",
        "dir": "/eos/user/m/mamozzan/point-cloud-diffusion-data/showerFlow/input_cc3_merge_within_cell",
    },
    "steps": {
        "pth": "/eos/user/m/mamozzan/point-cloud-diffusion-data/showerFlow/input_cc3_identity/ShowerFlow_alt1_nb2_inputs1152921504606846975_fnorms_best.pth",
        "dir": "/eos/user/m/mamozzan/point-cloud-diffusion-data/showerFlow/input_cc3_identity",
    },
    "subcell": {
        "pth": "/eos/user/m/mamozzan/point-cloud-diffusion-data/showerFlow/input_cc3_merge_within_regular_subcell/ShowerFlow_log1_stable_nb4_inputs1152921504606846975_fnorms_best.pth",
        "dir": "/eos/user/m/mamozzan/point-cloud-diffusion-data/showerFlow/input_cc3_merge_within_regular_subcell",
    },
    "subcell_6kcut": {
        "pth": "/eos/user/m/mamozzan/point-cloud-diffusion-data/showerFlow/input_cc3_merge_within_regular_subcell_6kcut/ShowerFlow_log1_stable_nb4_inputs1152921504606846975_fnorms_best.pth",
        "dir": "/eos/user/m/mamozzan/point-cloud-diffusion-data/showerFlow/input_cc3_merge_within_regular_subcell_6kcut",
    },
}

# Occupancy bias correction (CC3 paper, section 3.3): a flat factor on
# ShowerFlow's per-layer cluster counts, applied in gen_cond_showers_batch.
# scale_n = mean(real n_points) / mean(raw generated n_points) over the first 5000
# conditioning-matched showers: real from
# /eos/project/f/fast/input_cc3/cc3input_<variant>/input_cc3_file_0.h5, generated
# by this script with the correction disabled.
#
# TODO(re-measure): STALE - all four values were measured with the old alt1_nb2
# flows; the variants now use log1_stable_nb4. Only the base (non-poly)
# generated_showers.h5 depends on them; the poly run (--occupancy_fit poly) uses
# calculate_coef.py's per-variant fit instead.
_OCCUPANCY_SCALE_N = {
    "hdbscan_ms3_mcs10": 1.0350,
    "hdbscan_ms8_mcs40": 1.0408,
    "withincell": 1.0438,
    "subcell": 1.0324,
}

_CONFIG_CLS = {
    "hdbscan_ms8_mcs40": caloclouds_3_S2P_hdbscan_ms8_mcs40.Configs,
    "hdbscan_ms3_mcs10": caloclouds_3_S2P_hdbscan_ms3_mcs10.Configs,
    "hdbscan_ms3_mcs3": caloclouds_3_S2P_hdbscan_ms3_mcs3.Configs,
    "hdbscan_ms7_mcs7": caloclouds_3_S2P_hdbscan_ms7_mcs7.Configs,
    "hdbscan_ms12_mcs12": caloclouds_3_S2P_hdbscan_ms12_mcs12.Configs,
    "withincell": caloclouds_3_S2P_withincell.Configs,
    "steps": caloclouds_3_S2P_steps.Configs,
    "subcell": caloclouds_3_S2P_subcell.Configs,
    "subcell_6kcut": caloclouds_3_S2P_subcell_6kcut.Configs,
}


def _detect_variant(log_dir: str) -> str:
    name = os.path.basename(os.path.normpath(log_dir)).lower()
    if "hdbscan_ms8_mcs40" in name:
        return "hdbscan_ms8_mcs40"
    if "hdbscan_ms3_mcs10" in name:
        return "hdbscan_ms3_mcs10"
    # Order-independent of the ms3_mcs10 test above: neither string is a
    # substring of the other ("ms3_mcs3" vs "ms3_mcs10"), so no log_dir can
    # match both.
    if "hdbscan_ms3_mcs3" in name:
        return "hdbscan_ms3_mcs3"
    if "hdbscan_ms7_mcs7" in name:
        return "hdbscan_ms7_mcs7"
    if "hdbscan_ms12_mcs12" in name:
        return "hdbscan_ms12_mcs12"
    if "subcell_6kcut" in name:
        return "subcell_6kcut"
    if "subcell" in name:
        return "subcell"
    if "within_cell" in name:
        return "withincell"
    if "steps" in name:
        return "steps"
    raise ValueError(
        f"Cannot infer variant from '{os.path.basename(log_dir)}'. "
        "Folder name must contain 'hdbscan', 'within_cell', 'subcell', 'subcell_6kcut', or 'steps'."
    )


def _fmt_num(x: float) -> str:
    """Compact, filename-safe number formatting: 20.0 -> '20', -12.5 -> 'm12p5'."""
    s = f"{x:g}"
    return s.replace("-", "m").replace(".", "p")


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


def _check_flow_matches_config(config, flow_pth: str, flow_dir: str) -> None:
    """Fail early, and readably, when the shower-flow checkpoint and the config disagree.

    load_flow_model builds the flow from the config's shower_flow_version /
    shower_flow_num_blocks / af_dim and only then loads the weights, so a config
    that has moved on from the checkpoint _SHOWER_FLOW names (as happened when
    hdbscan_ms3_mcs10 switched from alt1_nb2 to log1_stable_nb4) otherwise shows
    up as a load_state_dict key/shape dump. The checkpoint name encodes version,
    num_blocks and the input mask, so compare those directly. af_dim is *not* in
    the name - a mismatch there is still only caught by the weight shapes.

    log1_stable additionally derives its log-space standardization from the
    cached per-layer arrays of whichever showerflow dir the config resolves to
    (pointcloud/models/stable_log1.py). That dir follows config.dataset_path, so
    a --cond_file from a different dataset would decode the flow with the wrong
    loc/scale and silently distort occupancies - check it points at the same dir
    the checkpoint came from.
    """
    cut_inputs = np.where(~showerflow_utils.get_input_mask(config))[0]
    _, expected_best, _ = showerflow_utils.model_save_paths(
        config, config.shower_flow_version, config.shower_flow_num_blocks, cut_inputs
    )
    name_base = os.path.basename(expected_best)[: -len("_best.pth")]
    if not os.path.basename(flow_pth).startswith(name_base):
        raise ValueError(
            f"Shower-flow checkpoint does not match the config for this variant.\n"
            f"  checkpoint : {os.path.basename(flow_pth)}\n"
            f"  config     : version={config.shower_flow_version} "
            f"num_blocks={config.shower_flow_num_blocks} "
            f"(expects a checkpoint named {name_base}_*.pth)\n"
            f"Update _SHOWER_FLOW (or the config) so both name the same flow."
        )
    if config.shower_flow_version in showerflow_utils.RUNTIME_REGISTERED_VERSIONS:
        stats_dir = showerflow_utils.get_showerflow_dir(config)
        if os.path.realpath(stats_dir) != os.path.realpath(flow_dir):
            raise ValueError(
                f"shower_flow_version={config.shower_flow_version!r} takes its log-space "
                f"standardization stats from the dataset's showerflow dir, but that dir "
                f"is not where this checkpoint was trained:\n"
                f"  stats dir (from config.dataset_path): {stats_dir}\n"
                f"  checkpoint dir (_SHOWER_FLOW)       : {flow_dir}\n"
                f"Pass a --cond_file from the dataset this flow was trained on."
            )


def _to_cc3_events(batch_showers: np.ndarray, layer_centers: np.ndarray, energy_units: str = "GeV") -> np.ndarray:
    """Convert gen_cond_showers_batch output to input_cc3_file format.

    Parameters
    ----------
    batch_showers : (B, max_points, 4)
        Generated showers; col 0/1 already in aligned local mm, col 2 in global mm, col 3 in MeV.
    layer_centers : (n_layers,)
        Global y positions of layer centres (layer_bottom_pos_global + cell_thickness/2).
    energy_units : {"GeV", "MeV"}
        Units to write column 3 in. "GeV" (default) matches the input_cc3 convention used by
        training/comparison scripts. "MeV" leaves the model's native energy value unconverted —
        use this when feeding DDML's LoadHdf5 loader, whose C++ side (CaloCloudsTwoAngleModel.cc)
        applies hit energies directly as G4/DD4hep native units (MeV) with no GeV->MeV conversion,
        which otherwise undervalues every hit by 1000x.

    Returns
    -------
    events : (B, max_points, 4)  float32
        CC3 input format: col 0/1 = mm (unchanged), col 2 = layer_idx (int, no fuzz), col 3 = energy_units.
    """
    if energy_units not in ("GeV", "MeV"):
        raise ValueError(f"energy_units must be 'GeV' or 'MeV', got {energy_units!r}")

    events = batch_showers.copy()
    valid_mask = events[:, :, 3] > 0  # (B, max_points)

    # col 0/1: clamp local x/y to the +/-250mm (500x500mm) box used for training data
    events[:, :, 0] = np.clip(events[:, :, 0], -XY_HALF_SIZE_MM, XY_HALF_SIZE_MM)
    events[:, :, 1] = np.clip(events[:, :, 1], -XY_HALF_SIZE_MM, XY_HALF_SIZE_MM)

    # col 2: z_global_mm → plain int layer index (no fuzz)
    z_mm = events[:, :, 2]  # (B, max_points)
    layer_idx = np.abs(z_mm[:, :, None] - layer_centers[None, None, :]).argmin(
        axis=2
    )  # (B, max_points)
    events[:, :, 2] = np.where(valid_mask, layer_idx.astype(np.float32), 0.0)

    # col 3: MeV → requested energy_units
    if energy_units == "GeV":
        events[:, :, 3] = np.where(valid_mask, events[:, :, 3] / 1000.0, 0.0)

    return events.astype(np.float32)


def _prepend_layer_counts(events: np.ndarray, n_layers: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepend per-layer hit counts as the first n_layers "points" of events,
    matching convert_to_cc3_format_layercounts_test.py's format. Each
    prepended point repeats its count across all 4 columns.

    Parameters
    ----------
    events : (B, max_points, 4)
        CC3-format events with col 2 = plain int layer index, col 3 = GeV.
    n_layers : int

    Returns
    -------
    events_with_counts : (B, n_layers + max_points, 4)  float32
    layer_counts : (B, n_layers)  float32
    """
    layer_ids = events[:, :, 2].astype(int)
    valid = events[:, :, 3] > 0
    layer_counts = np.zeros((events.shape[0], n_layers), dtype=np.float32)
    for i in range(events.shape[0]):
        ids = layer_ids[i][valid[i]]
        ids = ids[(ids >= 0) & (ids < n_layers)]
        layer_counts[i] = np.bincount(ids, minlength=n_layers)[:n_layers]

    n_shown, max_hits, n_feat = events.shape
    events_with_counts = np.zeros((n_shown, n_layers + max_hits, n_feat), dtype=np.float32)
    events_with_counts[:, :n_layers, :] = layer_counts[:, :, None]
    events_with_counts[:, n_layers:, :] = events
    return events_with_counts, layer_counts


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
        default=None,
        help="Number of real conditioning vectors loaded as the sampling pool. "
        "Cycled if n_showers > cond_pool_size. Default: same as --n_showers, "
        "so every generated shower gets its own distinct real conditioning "
        "vector instead of repeating a smaller pool.",
    )
    parser.add_argument(
        "--cond_file",
        default=None,
        help="Path to a specific input_cc3 .h5 file to read conditioning vectors from "
        "(e.g. /eos/.../cc3input_merge_within_cell/input_cc3_file_0.h5). Overrides the "
        "variant config's default dataset_path. Useful for pinning generation to the same "
        "conditioning used by a specific reference/comparison file. Default: the variant "
        "config's own dataset_path.",
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
    parser.add_argument(
        "--shower_flow_pth",
        default=None,
        help="Specific ShowerFlow .pth to generate with. Default: the variant's entry in "
        "_SHOWER_FLOW. Its version/num_blocks must match the config (a search-trial "
        "checkpoint of the same architecture is fine); input_norms.npz and, for "
        "log1_stable, the log-standardization stats still come from the variant's dir.",
    )
    parser.add_argument(
        "--occupancy_fit",
        choices=["flat", "poly"],
        default="flat",
        help="Occupancy-scaling method. 'flat' (default): the single mean-ratio "
        "scale factor in _OCCUPANCY_SCALE_N. 'poly': the count-dependent "
        "polynomial fit from scripts/training/calculate_coef.py, loaded from "
        "occupancy_coef_<variant>.npz in the variant's showerFlow directory "
        "(run calculate_coef.py first to produce it).",
    )
    parser.add_argument(
        "--energy_units",
        choices=["GeV", "MeV"],
        default="MeV",
        help="Units for events[:, :, 3] (hit energy). 'GeV' (default) matches the input_cc3 "
        "convention used by training/comparison scripts. Use 'MeV' when the output feeds "
        "DDML's LoadHdf5 loader, which applies hit energies as raw G4/DD4hep native units "
        "(MeV) with no further conversion.",
    )
    parser.add_argument(
        "--fixed_energy_gev",
        type=float,
        default=None,
        help="Override the incident-energy conditioning column with this fixed value [GeV] "
        "for every shower, instead of the per-shower energies read from --cond_file/the "
        "variant's dataset. Direction (theta/phi) is left as-is, i.e. still drawn from the "
        "real conditioning pool. Combinable with --fixed_theta_deg/--fixed_phi_deg to fix both.",
    )
    parser.add_argument(
        "--fixed_theta_deg",
        type=float,
        default=None,
        help="Override the local polar angle conditioning (theta_local) with this fixed value "
        "[degrees] for every shower. Independent of --fixed_phi_deg: if phi isn't also fixed, "
        "each shower keeps its own real phi_local from the conditioning pool (only theta is "
        "overridden). Energy is likewise left as-is (real pool) unless --fixed_energy_gev is set.",
    )
    parser.add_argument(
        "--fixed_phi_deg",
        type=float,
        default=None,
        help="Override the local azimuthal angle conditioning (phi_local) with this fixed value "
        "[degrees] for every shower. Independent of --fixed_theta_deg: if theta isn't also fixed, "
        "each shower keeps its own real theta_local from the conditioning pool.",
    )
    args = parser.parse_args()
    if args.cond_pool_size is None:
        args.cond_pool_size = args.n_showers

    # Auto-suffix --output_file when fixing conditioning columns, so a fixed-energy
    # or fixed-angle run can't silently overwrite a normal run's file of the same
    # name in the same out_dir (log_dir + n_showers are otherwise identical).
    # theta/phi can be fixed independently (the other stays random, drawn per-shower
    # from the real conditioning pool), so each gets its own suffix piece.
    suffix_parts = []
    if args.fixed_energy_gev is not None:
        suffix_parts.append(f"E{_fmt_num(args.fixed_energy_gev)}GeV")
    if args.fixed_theta_deg is not None:
        suffix_parts.append(f"theta{_fmt_num(args.fixed_theta_deg)}")
    if args.fixed_phi_deg is not None:
        suffix_parts.append(f"phi{_fmt_num(args.fixed_phi_deg)}")
    if suffix_parts:
        base, ext = os.path.splitext(args.output_file)
        args.output_file = f"{base}_{'_'.join(suffix_parts)}{ext}"

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
    print(f"Energy units  : {args.energy_units}")
    print(f"Output        : {out_path}")

    # --- config and metadata ---
    config = _CONFIG_CLS[variant]()
    config.device = args.device
    if args.cond_file is not None:
        config.dataset_path = args.cond_file
        print(f"Cond file     : {args.cond_file} (first {args.n_showers} conditionings, in order)")
    meta = Metadata(config)
    layer_centers = (
        meta.layer_bottom_pos_global + meta.cell_thickness_global / 2
    ).astype(np.float32)
    n_layers = len(layer_centers)

    sf = _SHOWER_FLOW[variant]
    flow_pth = args.shower_flow_pth or sf["pth"]

    # --- load shower flow ---
    print("\nLoading shower flow...")
    print(f"  checkpoint: {flow_pth}")
    print(
        f"  version={config.shower_flow_version} af_dim={config.af_dim} "
        f"num_blocks={config.shower_flow_num_blocks}"
    )
    _check_flow_matches_config(config, flow_pth, sf["dir"])
    flow, distribution, transforms = load_flow_model(config, flow_pth)
    norms = np.load(os.path.join(sf["dir"], "input_norms.npz"))
    config.shower_flow_clusters_per_layer_norm = float(norms["clusters_per_layer_norm"])
    config.shower_flow_energy_per_layer_norm = float(norms["energy_per_layer_norm"])
    if args.occupancy_fit == "poly":
        coef_path = os.path.join(sf["dir"], f"occupancy_coef_{variant}.npz")
        if not os.path.exists(coef_path):
            raise FileNotFoundError(
                f"--occupancy_fit poly requested but {coef_path} does not exist. "
                "Run scripts/training/calculate_coef.py for this variant first."
            )
        coef_data = np.load(coef_path)
        config.shower_flow_n_scaling = True
        config.shower_flow_coef_real = coef_data["coef_real"]
        config.shower_flow_coef_fake = None
        print(f"  occupancy poly coef_real={config.shower_flow_coef_real} (variant={variant})")
    else:
        scale_n = _OCCUPANCY_SCALE_N.get(variant)
        if scale_n is not None:
            config.shower_flow_n_scaling = True
            config.shower_flow_coef_real = np.array([scale_n, 0.0])
            config.shower_flow_coef_fake = None
            print(f"  occupancy scale_n={scale_n:.4f} (variant={variant})")
        else:
            config.shower_flow_n_scaling = False
            print(f"  no occupancy scale factor for variant={variant!r}, generating uncalibrated")
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

    # Conditioning columns are [energy_GeV, px_local, py_local, pz_local] (see
    # cond_np unpacking below). Override one or both while leaving the other
    # as-is (i.e. still drawn from the real conditioning pool above).
    if args.fixed_energy_gev is not None:
        full_cond[:, 0] = args.fixed_energy_gev
        print(f"\nFixed energy  : {args.fixed_energy_gev} GeV (angles from real cond pool)")
    if args.fixed_theta_deg is not None or args.fixed_phi_deg is not None:
        # Decompose each pool row's real direction into its own theta/phi so
        # only the fixed component is overridden - the other one keeps its
        # real, per-shower paired value (not a single global fixed value).
        px, py, pz = full_cond[:, 1].numpy(), full_cond[:, 2].numpy(), full_cond[:, 3].numpy()
        theta_rad = np.arccos(np.clip(pz, -1.0, 1.0))
        phi_rad = np.arctan2(py, px)
        if args.fixed_theta_deg is not None:
            theta_rad = np.full_like(theta_rad, np.radians(args.fixed_theta_deg))
        if args.fixed_phi_deg is not None:
            phi_rad = np.full_like(phi_rad, np.radians(args.fixed_phi_deg))
        full_cond[:, 1] = torch.from_numpy(np.sin(theta_rad) * np.cos(phi_rad)).float()  # px_local
        full_cond[:, 2] = torch.from_numpy(np.sin(theta_rad) * np.sin(phi_rad)).float()  # py_local
        full_cond[:, 3] = torch.from_numpy(np.cos(theta_rad)).float()  # pz_local
        theta_label = f"{args.fixed_theta_deg} deg" if args.fixed_theta_deg is not None else "random (real cond pool)"
        phi_label = f"{args.fixed_phi_deg} deg" if args.fixed_phi_deg is not None else "random (real cond pool)"
        print(f"\nFixed angle   : theta_local={theta_label}, phi_local={phi_label}")

    # --- generate and write incrementally ---
    print(f"\nGenerating {n_total} showers -> {out_path}")
    total_points = n_layers + config.max_points
    chunk_events = (min(args.chunk_size, n_total), total_points, 4)

    with h5py.File(out_path, "w", locking=False) as f:
        f.attrs["hit_energy_units"] = args.energy_units
        energy_ds = f.create_dataset("energy", shape=(n_total, 1), dtype=np.float32)
        events_ds = f.create_dataset(
            "events",
            shape=(n_total, total_points, 4),
            dtype=np.float32,
            chunks=chunk_events,
        )
        layer_counts_ds = f.create_dataset(
            "layer_counts", shape=(n_total, n_layers), dtype=np.float32
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

            # --- convert to input_cc3 format, then prepend per-layer hit counts ---
            events = _to_cc3_events(batch_showers, layer_centers, energy_units=args.energy_units)
            events, layer_counts = _prepend_layer_counts(events, n_layers)

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

            n_points = (events[:, n_layers:, 3] > 0).sum(axis=1).astype(np.int64)

            energy_ds[start:end] = inc_energy
            events_ds[start:end] = events
            layer_counts_ds[start:end] = layer_counts
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
        f"  events        : float32  ({n_total}, {total_points}, 4)"
        f"  events[:, :{n_layers}, :] = per-layer hit counts"
        f"  events[:, {n_layers}:, :] = [local_x mm, local_y mm, layer_idx (int), E_{args.energy_units}]"
    )
    print(f"  attrs['hit_energy_units'] = {args.energy_units!r}")
    print(f"  layer_counts  : float32  ({n_total}, {n_layers})")
    print(f"  n_points      : int64    ({n_total},)")
    print(f"  p_norm_local  : float64  ({n_total}, 3)")
    print(f"  p_norm_global : float32  ({n_total}, 3)")
    print(f"  theta/phi_local  : float64  ({n_total},)  [degrees]")
    print(f"  theta/phi_global : float32  ({n_total},)  [degrees]")


if __name__ == "__main__":
    main()
