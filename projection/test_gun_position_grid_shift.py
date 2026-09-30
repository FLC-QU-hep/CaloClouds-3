"""
Test: gun position at the EDGE of an ECAL cell vs. at the CENTER (the
default), for postprocessing.py's grid-projection step.

postprocessing.py's grid projection (merge_shower/merge_all) snaps each
hit's transverse (x, z) position to the nearest multiple of cell_size_mm -
a grid whose vertices sit at 0, +-cell_size, +-2*cell_size, ... Since the
gun's transverse position is metadata.gun_xyz_pos_global[0] = 0, and that
grid is anchored at global x=0/z=0, the gun sits exactly on a grid vertex -
i.e. a cell CENTER under the "round to nearest multiple" convention that
postprocessing.py always uses.

This script reruns the same box-cut / undo-alignment-shift steps as
postprocessing.py (steps 1-2, reused unmodified via import of
get_alignment_shifts/BOX_HALF_SIZE_MM from postprocessing.py - no logic
duplicated), then grid-projects the resulting global-frame hits twice:

  * "gridcenter": merge_all() as-is - the normal postprocessing.py
    convention, gun position lands on a cell center.
  * "gridedge": the same hits, but shifted by +cell_size/2 in x and z
    before merge_all() and shifted back by -cell_size/2 afterwards. This
    moves every cell boundary by half a cell relative to the (fixed) gun
    position, without changing any hit's physical location - equivalent to
    moving the gun by half a cell relative to a fixed grid, so the gun now
    lands exactly on a cell boundary instead of a cell center.

Both variants are written in the same schema postprocessing.py's grid
projection produces (hits/n_points/axis_x_global/axis_z_global), into
directories named so the existing plot_distributions_from_grid.py /
plot_shower_profiles_from_grid.py derive_label() picks up "gridcenter" /
"gridedge" as the legend label automatically. This script then calls those
two plotting scripts unmodified (via subprocess) to produce the comparison
figures.

Does NOT modify postprocessing.py, plot_distributions_from_grid.py, or
plot_shower_profiles_from_grid.py.

Usage:
    python test_gun_position_grid_shift.py [input.h5] [--n-showers N]
    # default input: the same generated_showers_poly.h5 run.sh uses
"""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.metadata import Metadata  # noqa: E402

from postprocessing import (  # noqa: E402
    BOX_HALF_SIZE_MM,
    CELL_SIZE_MM,
    HALF_MIP_MEV,
    get_alignment_shifts,
    merge_all,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = (
    "/eos/user/m/mamozzan/CaloClouds-3/generated_showers/"
    "merge_within_regular_subcell_6kcut_2026_08_19__11_13_07/"
    "generated_showers_5000/generated_showers_poly.h5"
)


def project_grid(input_path, n_showers=None, random_sample=None, seed=12345):
    """Steps 1-2 of postprocessing.py (box cut + undo alignment shift),
    reimplemented here purely from imported pure functions/constants so the
    two grid variants below can share the same global-frame hits without
    postprocessing.py needing to expose them itself.

    n_showers takes the FIRST N showers; random_sample instead draws N at
    random without replacement. Use random_sample when the file is much larger
    than the sample you want (the Geant4 reference holds 35000 showers against
    the generated files' 10000) so the subset is not just the head of the file.
    Both grid variants are built from the same drawn showers, so gridcenter and
    gridedge stay a like-for-like pair whichever is used."""
    metadata = Metadata()
    n_layers = len(metadata.layer_bottom_pos_global)

    with h5py.File(input_path, "r") as f:
        total = f["events"].shape[0]
        if random_sample is not None and random_sample < total:
            rng = np.random.default_rng(seed)
            # Sorted because h5py needs an increasing selection; sorting a
            # without-replacement draw does not bias which showers are picked.
            sel = np.sort(rng.choice(total, size=random_sample, replace=False))
            print(f"Randomly sampling {random_sample} of {total} showers (seed={seed})")
        else:
            sel = slice(None, n_showers)
        events = f["events"][sel].astype(np.float32)
        phi_global = f["phi_global"][sel].astype(np.float64)
        theta_global = f["theta_global"][sel].astype(np.float64)
        has_layer_counts = "layer_counts" in f

    n_showers = events.shape[0]
    print(f"n_showers={n_showers}, n_layers={n_layers}, events shape={events.shape}")

    if not has_layer_counts:
        events[:, n_layers:, 3] *= 1000.0  # GeV -> MeV

    x_shift, z_shift = get_alignment_shifts(metadata, phi_global, theta_global)

    hits = events[:, n_layers:, :]
    shower_idx = np.arange(n_showers)[:, None] * np.ones(hits.shape[1], dtype=int)[None, :]

    valid = hits[:, :, 3] > 0
    in_box = (
        valid
        & (hits[:, :, 0] > -BOX_HALF_SIZE_MM) & (hits[:, :, 0] < BOX_HALF_SIZE_MM)
        & (hits[:, :, 1] > -BOX_HALF_SIZE_MM) & (hits[:, :, 1] < BOX_HALF_SIZE_MM)
    )
    n_before, n_after = valid.sum(), in_box.sum()
    print(f"Box cut (+-{BOX_HALF_SIZE_MM:.0f}mm): {n_before} -> {n_after} hits ({n_before - n_after} removed)")
    hits = np.where(in_box[:, :, None], hits, 0.0)

    layer_id = np.clip(hits[:, :, 2].astype(int), 0, n_layers - 1)
    z_shift_per_hit = z_shift[shower_idx, layer_id]
    x_shift_per_hit = x_shift[shower_idx, layer_id]
    raw_global_z = np.where(in_box, hits[:, :, 0] + z_shift_per_hit, 0.0)
    raw_global_x = np.where(in_box, hits[:, :, 1] + x_shift_per_hit, 0.0)
    energy = hits[:, :, 3]

    grid_hits_in = np.stack(
        [raw_global_x, raw_global_z, layer_id.astype(np.float32), energy], axis=-1
    ).astype(np.float32)

    print("Projecting: gridcenter (gun at cell center, default convention) ...")
    grid_center, n_points_center = merge_all(grid_hits_in, CELL_SIZE_MM, HALF_MIP_MEV)
    print(f"  -> {int(n_points_center.sum()):,} cells (capacity {grid_center.shape[1]})")

    print("Projecting: gridedge (grid shifted half a cell -> gun at cell boundary) ...")
    half_cell = CELL_SIZE_MM / 2.0
    shifted_hits_in = grid_hits_in.copy()
    # Only real hits (energy > 0) matter here - merge_shower's own valid
    # mask filters by energy before ever looking at x/z, so shifting
    # zero-energy padding rows too is harmless.
    shifted_hits_in[:, :, 0] += half_cell
    shifted_hits_in[:, :, 1] += half_cell
    grid_edge_shifted, n_points_edge = merge_all(shifted_hits_in, CELL_SIZE_MM, HALF_MIP_MEV)
    grid_edge = grid_edge_shifted.copy()
    grid_edge[:, :, 0] -= half_cell
    grid_edge[:, :, 1] -= half_cell
    print(f"  -> {int(n_points_edge.sum()):,} cells (capacity {grid_edge.shape[1]})")

    return {
        "n_layers": n_layers,
        "axis_x_global": x_shift.astype(np.float32),
        "axis_z_global": z_shift.astype(np.float32),
        "gridcenter": (grid_center, n_points_center),
        "gridedge": (grid_edge, n_points_edge),
    }


def write_variant(tag, grid_out, n_points, result, output_root, timestamp, sample_tag=""):
    # The plotting scripts' derive_label() takes this directory's name and strips
    # the trailing _<timestamp>, so whatever sits between tag and timestamp becomes
    # the legend entry. Without a sample_tag two samples both label their lines
    # "gridcenter"/"gridedge" and collide; with e.g. sample_tag="geant4" they read
    # "gridcenter_geant4"/"gridedge_geant4", which is what lets one figure show the
    # gun-edge check for several samples at once.
    name = f"{tag}_{sample_tag}_{timestamp}" if sample_tag else f"{tag}_{timestamp}"
    # Name this level for the number of showers actually written. It used to be
    # hardcoded "generated_showers_5000" no matter what --n-showers was, which
    # made a 10000-shower run look like a 5000-shower one on disk. Only the
    # PARENT directory feeds derive_label(), so this is free to be honest.
    variant_dir = os.path.join(output_root, name, f"generated_showers_{grid_out.shape[0]}")
    os.makedirs(variant_dir, exist_ok=True)
    out_path = os.path.join(variant_dir, "generated_showers_poly_postprocessed.h5")
    with h5py.File(out_path, "w") as hf:
        hf.create_dataset("hits", data=grid_out, compression="gzip")
        hf.create_dataset("n_points", data=n_points)
        hf.create_dataset("axis_x_global", data=result["axis_x_global"])
        hf.create_dataset("axis_z_global", data=result["axis_z_global"])
        hf.attrs["cell_size_mm"] = CELL_SIZE_MM
        hf.attrs["energy_threshold_mev"] = HALF_MIP_MEV
        hf.attrs["n_layers"] = result["n_layers"]
    print(f"Saved {grid_out.shape} -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", default=DEFAULT_INPUT,
                         help="DDML-format h5 (events/phi_global/theta_global/...). Default: same file run.sh uses.")
    parser.add_argument("--n-showers", type=int, default=None,
                         help="Only process the first N showers (default: all showers in the file).")
    parser.add_argument("--output-dir", default=os.path.join(SCRIPT_DIR, "gun_edge_test"),
                         help="Where to write the two gridcenter/gridedge _postprocessed.h5 files.")
    parser.add_argument("--geometry", default="gun_edge_test",
                         help="Suffix for the comparison plot filenames (distribution_<geometry>.<fmt> etc).")
    parser.add_argument("--random-sample", type=int, default=None, metavar="N",
                         help="Draw N showers at random (without replacement) instead of taking the "
                              "first N. Use when the input is much larger than the sample wanted - the "
                              "Geant4 reference holds 35000 showers against the generated files' 10000, "
                              "so --n-showers would only ever read the head of the file. Both grid "
                              "variants use the same drawn showers.")
    parser.add_argument("--seed", type=int, default=12345,
                         help="Seed for --random-sample, so the draw is reproducible (default: 12345).")
    parser.add_argument("--sample-tag", default="",
                         help="Name this sample in the output directory (and hence in the plot "
                              "legend): gridcenter_<tag>_<timestamp> instead of gridcenter_<timestamp>. "
                              "Needed when several samples' gun-edge outputs share one figure, "
                              "e.g. --sample-tag geant4 vs --sample-tag hdbscan_ms3.")
    parser.add_argument("--skip-plots", action="store_true",
                         help="Only write the two h5 files, don't call the plotting scripts.")
    args = parser.parse_args()

    result = project_grid(args.input, args.n_showers,
                           random_sample=args.random_sample, seed=args.seed)

    timestamp = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    paths = {}
    for tag in ("gridcenter", "gridedge"):
        grid_out, n_points = result[tag]
        paths[tag] = write_variant(tag, grid_out, n_points, result, args.output_dir, timestamp,
                                    sample_tag=args.sample_tag)

    if args.skip_plots:
        return

    for plot_script in ("plot_distributions_from_grid.py", "plot_shower_profiles_from_grid.py"):
        cmd = [sys.executable, os.path.join(SCRIPT_DIR, plot_script),
               paths["gridcenter"], paths["gridedge"], "--geometry", args.geometry]
        if plot_script == "plot_distributions_from_grid.py":
            # plot_distributions_from_grid.py's x panel histograms the
            # *already snapped-to-grid* hit x (discrete values, one per
            # cell column - not a continuous position), using a fixed
            # 40-edge (39-bin) histogram over a +-100mm-ish range set via
            # --x-zoom-mm. gridcenter's columns sit at integer multiples of
            # CELL_SIZE_MM (a "comb" with 0 phase); gridedge's sit at the
            # same comb shifted by half a cell. Getting a display bin width
            # that's merely *close* to CELL_SIZE_MM (e.g. the default
            # range's 5.128mm, or a rounded 5.0mm) still aliases: it's easy
            # for a bin edge to land exactly on - or just barely miss - one
            # of these comb points, which either merges two columns into
            # one bin (doubling it) or excludes a column from every bin
            # (zeroing it - verified both failure modes empirically while
            # tuning this).
            #
            # The robust fix is to make the bin width EXACTLY CELL_SIZE_MM,
            # with the whole edges array offset by a quarter cell from 0 so
            # no edge ever sits on either comb's points (0 mod cell, or
            # half a cell mod cell) - not even in principle, regardless of
            # floating-point rounding. That guarantees every bin contains
            # exactly one column from whichever file, for the full display
            # range (verified: ratio stays within ~0.93-1.08 everywhere,
            # vs. spikes to 2x/0x with the naive range).
            _n_edges = 40  # must match plot_distributions_from_grid.py's hardcoded x-panel bin count
            _half_span = (_n_edges - 1) * CELL_SIZE_MM / 2.0
            _lo = -_half_span + CELL_SIZE_MM / 4.0
            _hi = _half_span + CELL_SIZE_MM / 4.0
            cmd += ["--x-zoom-mm", f"{_lo:.6f}", f"{_hi:.6f}"]
        print("\n$ " + " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=SCRIPT_DIR)

    print(f"\nDone. Plots written to {os.path.join(SCRIPT_DIR, 'plots')}/"
          f"{{distribution,shower_profiles}}_{args.geometry}.*")


if __name__ == "__main__":
    main()
