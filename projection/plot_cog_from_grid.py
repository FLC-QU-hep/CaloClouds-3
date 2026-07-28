"""
Plot per-shower center-of-gravity (CoG) distributions from one or more raw
DDML/cc3input-format h5 files (events/[layer_counts]/phi_global/theta_global/
...), in the same comparison style as plot_distributions_from_grid.py - three
panels, one per axis (x, y=depth, z), each an energy-weighted mean position
per shower.

This mirrors the CoG X/Y/Z metrics from the CaloClouds3 paper (arXiv:2511.01460
table 2): the transverse components (our x, z) correspond to the paper's CoG
X/Y (perpendicular to the shower axis), and the depth component (our y)
corresponds to the paper's CoG Z (along the layers). Crucially, CoG x/z is
computed in the LOCAL, shift-aligned frame (box cut + half-MIP merge only, no
shift-undo) - the same frame the paper's own CoG is defined in, where each
layer is already rotated to align the incident photon with the shower axis -
NOT postprocessing.py's raw global reconstruction, which deliberately
re-introduces the real (unaligned) detector position that the paper's own
preprocessing removes before ever computing CoG.

Usage:
    python plot_cog_from_grid.py <input1.h5> [<input2.h5> ...] [--output-dir DIR]
    # default output dir: plots/ next to this script
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

from postprocessing import BOX_HALF_SIZE_MM, CELL_SIZE_MM, HALF_MIP_MEV, merge_all


def derive_label(h5_path):
    """Variant name (merge_within_cell, hdbscan_ms3_mcs10, ...) for the
    legend. Generated-showers filenames are generic (e.g.
    "generated_showers_poly") and don't carry the variant name themselves -
    it lives in the parent "<variant>_<timestamp>" directory instead
    (.../generated_showers/<variant>_<ts>/generated_showers_N/...), so fall
    back to that with the trailing timestamp stripped."""
    base = os.path.basename(h5_path)
    for ext in ("_grid.h5", ".h5"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    label = base.split("ddml_", 1)[-1] if "ddml_" in base else base
    label = label.replace("_unshifted", "").replace("unshifted", "")
    if label.startswith("generated") or label.startswith("poly"):
        # .../generated_showers/<variant>_<ts>/generated_showers_N/generated_showers_poly[...].h5
        variant_dir = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(h5_path))))
        label = re.sub(r"_\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}$", "", variant_dir)
    elif label.startswith("input_cc3"):
        # .../outputs/cc3input_<variant>/input_cc3_file_0[_postprocessed].h5
        variant_dir = os.path.basename(os.path.dirname(os.path.abspath(h5_path)))
        variant = re.sub(r"^cc3input_", "", variant_dir)
        if variant == "merge_within_regular_subcell":
            variant = "25x_subcell"  # this reference file was generated with a 5x5 subcell grid
        label = f"REF: {variant}"
    return label


def load_raw_file(h5_path, n_layers, n_showers=None):
    """Read a raw DDML/cc3input-format h5 and compute each shower's
    energy-weighted centre of gravity in the LOCAL, shift-aligned frame (box
    cut + half-MIP merge, no shift-undo)."""
    print(f"Loading {h5_path} ...")
    sl = slice(None, n_showers)
    with h5py.File(h5_path, "r") as f:
        events = f["events"][sl].astype(np.float32)
        has_layer_counts = "layer_counts" in f

    if not has_layer_counts:
        events[:, n_layers:, 3] *= 1000.0  # GeV -> MeV, same convention as postprocessing.py

    hits = events[:, n_layers:, :]
    valid = hits[:, :, 3] > 0
    in_box = (
        valid
        & (hits[:, :, 0] > -BOX_HALF_SIZE_MM) & (hits[:, :, 0] < BOX_HALF_SIZE_MM)
        & (hits[:, :, 1] > -BOX_HALF_SIZE_MM) & (hits[:, :, 1] < BOX_HALF_SIZE_MM)
    )
    hits = np.where(in_box[:, :, None], hits, 0.0)

    n_showers = hits.shape[0]
    merged, n_points = merge_all(hits, CELL_SIZE_MM, HALF_MIP_MEV)

    valid_m = merged[:, :, 3] > 0
    x = np.where(valid_m, merged[:, :, 0], 0.0)
    z = np.where(valid_m, merged[:, :, 1], 0.0)
    layer = np.where(valid_m, merged[:, :, 2], 0.0)
    e = np.where(valid_m, merged[:, :, 3], 0.0)

    esum = e.sum(axis=1)
    esum_safe = np.where(esum > 0, esum, 1.0)
    cog_x = (x * e).sum(axis=1) / esum_safe
    cog_z = (z * e).sum(axis=1) / esum_safe

    print(f"  {n_showers} showers, {int(n_points.sum()):,} merged cells")
    return {
        "label": derive_label(h5_path),
        "n_events": n_showers,
        "cog_x": cog_x,
        "cog_z": cog_z,
        "layer": layer,
        "e": e,
        "valid": valid_m,
        "n_layers": n_layers,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_h5", nargs="+", help="Path(s) to one or more raw DDML/cc3input-format h5 files")
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots"),
                         help="Directory to write cog_<geometry>.png into (default: plots/ next to this script)")
    parser.add_argument("--geometry", default="regular", help="Names the output cog_<geometry>.png")
    parser.add_argument(
        "--n-showers", type=int, default=None,
        help="Only process the first N showers from each input file (e.g. to match a smaller "
        "generated sample for a fair comparison plot, and to keep the local-frame merge fast on "
        "large real-reference files). Default: all showers in each file.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    sys.path.insert(0, "/eos/user/m/mamozzan/step2point")
    from martina_test.metadata import Metadata  # noqa: E402

    metadata = Metadata()
    n_layers = len(metadata.layer_bottom_pos_global)
    layer_y_mm = metadata.layer_bottom_pos_global
    files = [load_raw_file(p, n_layers, args.n_showers) for p in args.input_h5]

    for f in files:
        y_centers = layer_y_mm[:n_layers]
        layer_idx = np.clip(np.round(f["layer"]).astype(np.int64), 0, n_layers - 1)
        y = y_centers[layer_idx]
        esum_safe = np.where(f["e"].sum(axis=1) > 0, f["e"].sum(axis=1), 1.0)
        f["cog_y"] = (np.where(f["valid"], y, 0.0) * f["e"]).sum(axis=1) / esum_safe
        print(f"{f['label']}: {f['n_events']} events")

    SURFACE = "#fcfcfb"
    INK_PRIMARY = "#0b0b0b"
    GRIDLINE = "#e1e0d9"
    BASELINE = "#c3c2b7"
    CATEGORICAL = ["#2a78d6", "#1baf7a", "#cb8900", "#008300", "#4a3aa7", "#e34948", "#a1215a", "#0e7490"]

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_PRIMARY,
            "xtick.color": INK_PRIMARY,
            "ytick.color": INK_PRIMARY,
            "text.color": INK_PRIMARY,
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.grid": False,
        }
    )

    def style_axes(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)

    def plot_cog(ax_main, ax_ratio, key, bins=30, val_range=None):
        vals_list = [f[key] for f in files]
        if val_range is not None:
            lo, hi = val_range
        else:
            # 1st/99th percentile, not raw min/max: a handful of zero-hit
            # showers fall back to CoG=0 (esum_safe in load_grid_file), which
            # would otherwise drag the range out and squish the real
            # distribution into a sliver.
            lo = min(np.percentile(v, 1) for v in vals_list)
            hi = max(np.percentile(v, 99) for v in vals_list)
        edges = np.linspace(lo, hi, bins)

        counts_list = []
        for i, (f, color) in enumerate(zip(files, CATEGORICAL)):
            counts, _ = np.histogram(f[key], bins=edges)
            counts_list.append(counts)
            if i == 0:
                ax_main.stairs(counts, edges, fill=True, color="#eeede9", edgecolor="#c9c8c2",
                                linewidth=0.8, label=f["label"], zorder=1)
            elif f["label"].startswith("REF:"):
                # A second reference variant (not the primary "Geant4" role) -
                # kept visually distinct from the colored comparison lines.
                ax_main.stairs(counts, edges, color=INK_PRIMARY, linewidth=2, linestyle=":", label=f["label"], zorder=2)
            else:
                ax_main.stairs(counts, edges, color=color, linewidth=2, label=f["label"], zorder=2)

        ref_counts = counts_list[0].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            for f, counts, color in zip(files[1:], counts_list[1:], CATEGORICAL[1:]):
                ratio = np.where(ref_counts > 0, counts / ref_counts, np.nan)
                if f["label"].startswith("REF:"):
                    ax_ratio.stairs(ratio, edges, color=INK_PRIMARY, linewidth=1.6, linestyle=":", zorder=3)
                else:
                    ax_ratio.stairs(ratio, edges, color=color, linewidth=1.6, zorder=3)
        ax_ratio.axhline(1.0, color=BASELINE, linewidth=0.8, zorder=1)
        ax_ratio.set_ylim(0.5, 1.5)
        ax_ratio.set_ylabel("Ratio", fontsize=8)
        style_axes(ax_main)
        style_axes(ax_ratio)
        plt.setp(ax_main.get_xticklabels(), visible=False)

    fig = plt.figure(figsize=(15, 5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.08)
    main_axes = [fig.add_subplot(gs[0, c]) for c in range(3)]
    ratio_axes = [fig.add_subplot(gs[1, c], sharex=main_axes[c]) for c in range(3)]

    n_events_per_file = sorted({f["n_events"] for f in files})
    n_events_label = str(n_events_per_file[0]) if len(n_events_per_file) == 1 else "/".join(str(n) for n in n_events_per_file)
    fig.suptitle(f"Center of Gravity  (n={n_events_label} events/file)", fontsize=13, fontweight="bold", color=INK_PRIMARY)

    for (ax, ax_r), (key, lbl, val_range) in zip(
        zip(main_axes, ratio_axes),
        [("cog_x", "CoG x  [mm]", (-5, 5)), ("cog_y", "CoG y (depth)  [mm]", None), ("cog_z", "CoG z  [mm]", (-5, 5))],
    ):
        plot_cog(ax, ax_r, key, val_range=val_range)
        ax_r.set_xlabel(lbl)
        ax.set_ylabel("Showers")

    main_axes[0].legend(frameon=False, fontsize=9, labelcolor=INK_PRIMARY, loc="upper left")

    out_png = os.path.join(args.output_dir, f"cog_{args.geometry}.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_png}")


if __name__ == "__main__":
    main()
