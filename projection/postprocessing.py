"""
Post-process a DDML-format h5 (events/energy/layer_counts/n_points/phi_global/
theta_global/...): apply a box cut, undo the per-shower per-layer alignment
shift, and convert local (aligned) coordinates back to global (x, y, z).

This mirrors, in order:
  1. Transform_pointcloud.apply_transformations()'s box_selection() in
     convert_to_cc3_format.py (same +-250mm box, applied in the *shifted*
     local frame - same frame the original box cut used).
  2. The alignment-shift removal this script used to do on its own (see
     get_alignment_shifts() below) - working purely from what's already
     stored in the file, no raw edm4hep.root or step2point re-run needed.
  3. The local -> global coordinate reconstruction convert_from_cc3_format.py
     does (minus its reference-cell KD-tree lookup - this script only
     reconstructs plain continuous global (x, y, z), no cell_id).
  4. Grid projection (merge_all/merge_shower below): snap the same
     global-frame hits to the nearest multiple of --cell-size-mm (the same
     origin-anchored round(x / cell_size) convention convert_from_cc3_format.py's
     own snap-to-grid uses - NOT an arbitrary/shifted bin grid, which would
     misalign with the true detector cells and cause spurious splitting),
     merge hits landing in the same cell, and write the sparse per-cell
     schema (hits/n_points) to --output, so downstream plot_*_from_grid.py
     scripts can consume it directly. Disable with --no-project to instead
     write the classic per-hit postprocessed output (no grid projection) to
     --output.

Why steps 2/3 are possible from the file alone: the shift is computed once
per (shower, layer) from phi_global/theta_global (already stored per shower)
via get_alignment_shifts(), then subtracted from the shower's raw global X/Z
*before* the rotation into local coordinates:
    event[:, 0] -= x_shift[layer]      # raw global X
    event[:, 2] -= z_shift[layer]      # raw global Z
    ... global_to_local_points: (X, Y, Z) -> (Z, X, Y) ...
so in the final stored local frame:
    x_local_stored = raw_Z - z_shift  =>  raw_Z = x_local_stored + z_shift
    y_local_stored = raw_X - x_shift  =>  raw_X = y_local_stored + x_shift
Undoing the shift and reading off the global X/Z is the same addition;
global Y (radial depth) comes from the hit's own layer_id via
layer_bottom_pos_global + cell_thickness/2.

With --no-project, output 'events' column layout changes from local (x,
y=layer+fuzz, z=unused, energy) to global (x, y=radial depth, z, energy).
layer_counts (embedded prefix rows and, if present, the standalone dataset)
and n_points (if present) are recomputed to reflect hits removed by the box
cut. Without --no-project, the output is the grid-projected 'hits' schema
instead (see step 4 above).

Usage:
    python postprocessing.py input_cc3_file_0_ddml_identity.h5
    # -> writes the grid-projected input_cc3_file_0_ddml_identity_postprocessed.h5

    # classic per-hit output instead (no grid projection):
    python postprocessing.py input_cc3_file_0_ddml_identity.h5 --no-project

    # or give an explicit output path:
    python postprocessing.py input_cc3_file_0_ddml_identity.h5 my_output.h5
"""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np
from tqdm import tqdm

sys.path.insert(0, "/eos/user/m/mamozzan/step2point")
from martina_test.metadata import Metadata  # noqa: E402

BOX_HALF_SIZE_MM = 250.0
CELL_SIZE_MM = 5.0883331298828125  # ECAL transverse cell pitch (regular CartesianGridXY readout)
HALF_MIP_MEV = 0.1  # detector readout threshold (MIP peak 0.2 MeV - detector_map.py, plotting.py):
# a real cell whose *total* deposited energy falls below this is never read out at all


def merge_shower(hit, cell_size_mm=CELL_SIZE_MM, energy_threshold_mev=HALF_MIP_MEV):
    """hit: (capacity, 4) = (x, z, layer_id, energy_MeV) for one shower.
    Snaps each hit's transverse position to the nearest multiple of
    cell_size_mm - the same origin-anchored convention
    convert_from_cc3_format.py's snap-to-grid uses (x_bin = round(x /
    cell_size)) - then sums the energy of hits landing in the same (x,
    layer, z) cell, dropping cells whose summed energy doesn't clear
    energy_threshold_mev (a real detector channel below that never fires,
    so never gets read out). Returns one (x_center, z_center, layer_id,
    energy) row per surviving cell."""
    valid = hit[:, 3] > 0
    x, z, layer, e = hit[valid, 0], hit[valid, 1], hit[valid, 2].astype(np.int64), hit[valid, 3]

    x_idx = np.round(x / cell_size_mm).astype(np.int64)
    z_idx = np.round(z / cell_size_mm).astype(np.int64)

    keys = np.stack([x_idx, layer, z_idx], axis=1)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    merged_e = np.zeros(len(unique_keys), dtype=np.float64)
    np.add.at(merged_e, inverse, e)

    fires = merged_e > energy_threshold_mev
    merged_x = unique_keys[fires, 0] * cell_size_mm
    merged_layer = unique_keys[fires, 1].astype(np.float32)
    merged_z = unique_keys[fires, 2] * cell_size_mm
    merged_e = merged_e[fires]

    return np.stack([merged_x, merged_z, merged_layer, merged_e.astype(np.float32)], axis=1)


def merge_all(hits, cell_size_mm=CELL_SIZE_MM, energy_threshold_mev=HALF_MIP_MEV):
    """hits: (n_showers, capacity, 4) = (x, z, layer_id, energy_MeV). Returns
    (merged, n_points): merged cells per shower zero-padded to the largest
    per-shower occupied-cell count, and the real (non-padding) count per
    shower."""
    n_showers = hits.shape[0]
    merged_list = [
        merge_shower(hits[i], cell_size_mm, energy_threshold_mev)
        for i in tqdm(range(n_showers), mininterval=5, desc="Merging showers")
    ]
    n_points = np.array([m.shape[0] for m in merged_list], dtype=np.int64)
    capacity = int(n_points.max()) if n_showers else 0
    merged = np.zeros((n_showers, capacity, 4), dtype=np.float32)
    for i, m in enumerate(merged_list):
        merged[i, : m.shape[0]] = m
    return merged, n_points


def degrees_to_radians(deg):
    return deg * np.pi / 180.0


def get_alignment_shifts(metadata, phi_global, theta_global):
    """Exact copy of Transform_pointcloud.get_alignment_shifts() (convert_to_cc3_format.py),
    reimplemented standalone here so this script doesn't need a Transform_pointcloud instance."""
    dist_to_layers = (
        metadata.layer_bottom_pos_global - metadata.gun_xyz_pos_global[1] + metadata.cell_thickness_global / 2
    )
    phi = degrees_to_radians(phi_global.reshape((-1, 1)))
    theta = degrees_to_radians(theta_global.reshape((-1, 1)))
    r = (dist_to_layers / np.sin(phi)) / np.sin(theta)
    x_shift = r * np.cos(phi) * np.sin(theta) - metadata.gun_xyz_pos_global[0]
    z_shift = r * np.cos(theta) + metadata.gun_xyz_pos_global[2]
    return x_shift, z_shift  # each (n_showers, n_layers)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Path to a DDML-format h5 (events/energy/layer_counts/n_points/phi_global/theta_global/...)")
    parser.add_argument(
        "output", nargs="?", default=None,
        help="Output path for the post-processed copy. Defaults to <input>_postprocessed.h5 next to the "
        "input file. By default this is the grid-projected result; with --no-project it is the classic "
        "per-hit output instead.",
    )
    parser.add_argument(
        "--no-project", action="store_true",
        help="Skip the grid-projection step (bin/merge onto the regular ECAL readout grid) and write the "
        "classic per-hit output to --output instead.",
    )
    parser.add_argument("--cell-size-mm", type=float, default=CELL_SIZE_MM)
    parser.add_argument(
        "--energy-threshold-mev", type=float, default=HALF_MIP_MEV,
        help="Detector readout threshold for the grid-projection step: a merged cell whose summed "
        "energy doesn't clear this is dropped (never read out). Default: half a MIP (0.1 MeV).",
    )
    parser.add_argument(
        "--n-showers", type=int, default=None,
        help="Only process the first N showers from the input file (e.g. to match a smaller "
        "generated sample for a fair comparison plot). Default: all showers in the file.",
    )
    args = parser.parse_args()
    output = args.output
    if output is None:
        base, ext = os.path.splitext(args.input)
        output = f"{base}_postprocessed{ext}"
    args.output = output

    metadata = Metadata()
    n_layers = len(metadata.layer_bottom_pos_global)
    layer_center = (metadata.layer_bottom_pos_global + metadata.cell_thickness_global / 2).astype(np.float32)

    sl = slice(None, args.n_showers)
    with h5py.File(args.input, "r") as f:
        events = f["events"][sl].astype(np.float32)
        phi_global = f["phi_global"][sl].astype(np.float64)
        theta_global = f["theta_global"][sl].astype(np.float64)
        passthrough = {k: f[k][sl] for k in f.keys() if k not in ("events", "layer_counts", "n_points")}
        has_layer_counts = "layer_counts" in f

    n_showers = events.shape[0]
    print(f"n_showers={n_showers}, n_layers={n_layers}, events shape={events.shape}")

    if not has_layer_counts:
        # Files that haven't been through convert_to_DDML_format.py (which adds
        # layer_counts) still have their energy column in GeV, not MeV - convert
        # it here so downstream consumers (the grid projection below, the
        # plot_*_from_grid.py scripts) see the same MeV convention regardless
        # of input stage.
        events[:, n_layers:, 3] *= 1000.0  # GeV -> MeV

    x_shift, z_shift = get_alignment_shifts(metadata, phi_global, theta_global)  # (n_showers, n_layers)

    hits = events[:, n_layers:, :]
    shower_idx = np.arange(n_showers)[:, None] * np.ones(hits.shape[1], dtype=int)[None, :]

    # ── 1. Box cut in the (still-shifted) local frame ──────────────────────
    valid = hits[:, :, 3] > 0
    in_box = (
        valid
        & (hits[:, :, 0] > -BOX_HALF_SIZE_MM) & (hits[:, :, 0] < BOX_HALF_SIZE_MM)
        & (hits[:, :, 1] > -BOX_HALF_SIZE_MM) & (hits[:, :, 1] < BOX_HALF_SIZE_MM)
    )
    n_before, n_after = valid.sum(), in_box.sum()
    print(f"Box cut (+-{BOX_HALF_SIZE_MM:.0f}mm): {n_before} -> {n_after} hits ({n_before - n_after} removed)")
    hits = np.where(in_box[:, :, None], hits, 0.0)

    # ── 2. Undo the alignment shift ─────────────────────────────────────────
    layer_id = np.clip(hits[:, :, 2].astype(int), 0, n_layers - 1)
    z_shift_per_hit = z_shift[shower_idx, layer_id]
    x_shift_per_hit = x_shift[shower_idx, layer_id]
    raw_global_z = np.where(in_box, hits[:, :, 0] + z_shift_per_hit, 0.0)
    raw_global_x = np.where(in_box, hits[:, :, 1] + x_shift_per_hit, 0.0)

    # ── 3. Local -> global: radial depth comes from the hit's own layer_id ──
    global_y = np.where(in_box, layer_center[layer_id], 0.0)

    new_hits = np.stack([raw_global_x, global_y, raw_global_z, hits[:, :, 3]], axis=-1)

    # ── Recompute per-layer hit counts / n_points to reflect the box cut ───
    counts = np.zeros((n_showers, n_layers), dtype=np.float32)
    for li in range(n_layers):
        counts[:, li] = (in_box & (layer_id == li)).sum(axis=1)
    prefix = np.repeat(counts[:, :, None], 4, axis=-1)

    events_out = np.concatenate([prefix, new_hits], axis=1).astype(np.float32)

    if args.no_project:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with h5py.File(args.output, "w") as hf:
            hf.create_dataset("events", data=events_out)
            hf.create_dataset("layer_counts", data=counts)
            hf.create_dataset("n_points", data=counts.sum(axis=1).astype(np.int64))
            # events columns are now (global_x, global_y=radial depth, global_z,
            # energy), not the (x_local, z_local, layer_id, energy) layout of the
            # input - downstream consumers need this to know layer_id must be
            # recovered via a layer_center lookup, not read straight from column 2.
            hf.attrs["frame"] = "global"
            for k, v in passthrough.items():
                hf.create_dataset(k, data=v)
        print(f"Wrote {args.output}")
        return

    # ── 4. Grid projection on the same hits ─────────────────────────────────
    # merge_shower expects (x, z, layer_id, energy); build that directly from
    # what's already computed above instead of round-tripping through the
    # on-disk "global" frame + layer_id-recovery load_ecal_hits() does for a
    # file it didn't just produce itself.
    grid_hits_in = np.stack(
        [raw_global_x, raw_global_z, layer_id.astype(np.float32), new_hits[:, :, 3]], axis=-1
    ).astype(np.float32)

    n_hits_before = int((grid_hits_in[:, :, 3] > 0).sum())
    print(
        f"{grid_hits_in.shape[0]} showers, {n_layers} ECAL layers, {n_hits_before:,} real hits, "
        f"snap-to-grid cell size {args.cell_size_mm:g} mm, energy threshold {args.energy_threshold_mev:g} MeV"
    )

    grid_out, grid_n_points = merge_all(grid_hits_in, args.cell_size_mm, args.energy_threshold_mev)
    print(f"Merged to {int(grid_n_points.sum()):,} cells (capacity {grid_out.shape[1]} per shower)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with h5py.File(args.output, "w") as hf:
        hf.create_dataset("hits", data=grid_out, compression="gzip")
        hf.create_dataset("n_points", data=grid_n_points)
        # Per-(shower, layer) global X/Z of the true shower axis (gun position +
        # that shower's own entrance angle) - the same shift get_alignment_shifts()
        # uses to build the local, axis-centred training frame. Downstream
        # plot_*_from_grid.py scripts need this to compute radial distance from
        # the actual (depth-drifting) axis rather than a single per-shower
        # centroid, which would average the drift away and smear the core.
        hf.create_dataset("axis_x_global", data=x_shift.astype(np.float32))
        hf.create_dataset("axis_z_global", data=z_shift.astype(np.float32))
        hf.attrs["cell_size_mm"] = args.cell_size_mm
        hf.attrs["energy_threshold_mev"] = args.energy_threshold_mev
        hf.attrs["n_layers"] = n_layers
    print(f"Saved {grid_out.shape} -> {args.output}")


if __name__ == "__main__":
    main()
