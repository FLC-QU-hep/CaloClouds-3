"""
Plot distributions (cells/shower, energy/shower, cell energy spectrum, x/y/z
cell distributions) from one or more projected-grid h5 files (the output of
project_ecal_showers.py), in the same style as DDML/scripts/plot_distributions.py
(see distribution_regular.png) - except the underlying quantity is a regular
grid *cell* (a bin of project_ecal_showers.py's histogram, possibly summing
several generated hits that landed in the same cell) rather than a raw
edm4hep hit. Each input file is drawn as its own line/step histogram
(labelled by filename), so multiple files can be compared directly.

Usage:
    python plot_distributions_from_grid.py <input1_grid.h5> [<input2_grid.h5> ...] [--output-dir DIR]
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

sys.path.insert(0, "/eos/user/m/mamozzan/step2point")
from martina_test.metadata import Metadata  # noqa: E402


def derive_label(h5_path):
    """Variant name (merge_within_cell, hdbscan_ms3_mcs10, ...) for the
    legend. Generated-showers filenames are generic (e.g.
    "generated_showers_poly_postprocessed") and don't carry the variant name
    themselves - it lives in the parent "<variant>_<timestamp>" directory
    instead (.../generated_showers/<variant>_<ts>/generated_showers_N/...),
    so fall back to that with the trailing timestamp stripped."""
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


def load_grid_file(h5_path, layer_y_mm, xz_range=None):
    """Read a merged-cell point cloud (the output of project_ecal_showers.py:
    hits (n_showers, capacity, 4) = (x_local, z_local, layer_id, energy_MeV),
    zero-padded per shower up to n_points) and flatten the real (non-padding)
    rows into per-cell arrays. If xz_range=(lo, hi) is given, every quantity
    (not just the x/z display range) is restricted to cells with both x and z
    inside that box, so hits_per_event/energy_per_event/the cell energy
    spectrum all describe the same central region rather than the full shower."""
    print(f"Loading {h5_path} ...")
    with h5py.File(h5_path, "r") as f:
        hits = f["hits"][:]
        n_layers = f.attrs["n_layers"]
        y_centers = layer_y_mm[:n_layers]

    n_events = hits.shape[0]
    valid = hits[:, :, 3] > 0
    evt = np.nonzero(valid)[0]
    x = hits[:, :, 0][valid]
    z = hits[:, :, 1][valid]
    layer = hits[:, :, 2][valid].astype(np.int64)
    e = hits[:, :, 3][valid] / 1000.0  # DDML convention: stored energy is MeV, plots want GeV
    y = y_centers[layer]

    if xz_range is not None:
        lo, hi = xz_range
        in_box = (x >= lo) & (x <= hi) & (z >= lo) & (z <= hi)
        evt, x, z, e, y = evt[in_box], x[in_box], z[in_box], e[in_box], y[in_box]

    print(f"  {n_events} showers, {len(e):,} merged cells")
    return {
        "label": derive_label(h5_path),
        "n_events": n_events,
        "hits_per_event": np.bincount(evt, minlength=n_events).astype(float),
        "energy_per_event": np.bincount(evt, weights=e, minlength=n_events),
        "x": x,
        "y": y,
        "z": z,
        "e": e,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_h5", nargs="+", help="Path(s) to one or more *_grid.h5 files from project_ecal_showers.py")
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots"),
                         help="Directory to write distribution_<geometry>.png into (default: plots/ next to this script)")
    parser.add_argument("--geometry", default="regular", help="Names the output distribution_<geometry>.png")
    parser.add_argument("--xz-range-mm", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                         help="Restrict every quantity (hits/energy per event, cell energy spectrum, x/y/z) "
                              "to cells with both x and z inside [LO, HI] (default: no restriction)")
    parser.add_argument("--x-zoom-mm", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                         help="Display-only zoom for the x panel's histogram range (doesn't filter data "
                              "used by other panels, unlike --xz-range-mm). Default: (-200, 200), symmetric "
                              "around x=0 (gun_xyz_pos_global[0]=0).")
    parser.add_argument("--z-zoom-mm", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                         help="Display-only zoom for the z panel's histogram range. Default: (-300, 200), "
                              "symmetric around z=-50 (gun_xyz_pos_global[2]=-50).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    layer_y_mm = Metadata().layer_bottom_pos_global
    xz_range = tuple(args.xz_range_mm) if args.xz_range_mm else None
    files = [load_grid_file(p, layer_y_mm, xz_range=xz_range) for p in args.input_h5]
    for f in files:
        print(f"\n{f['label']}: {f['n_events']} events, {len(f['e']):,} cells")

    # Chart chrome (light surface, ink roles) and a fixed-order categorical
    # palette - same convention as DDML/scripts/plot_distributions.py, so
    # figures from either pipeline stay visually consistent.
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

    bins = 100

    def style_axes(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)

    def plot_lines(ax_main, ax_ratio, key, log_x=False, log_y=False, bins=bins, quantize_threshold=None, weight_key=None, x_min=None, val_range=None):
        vals_list = [f[key] for f in files]
        if val_range is not None:
            lo, hi = val_range
        else:
            lo = min(v.min() for v in vals_list)
            hi = max(v.max() for v in vals_list)
        uniques = np.unique(np.concatenate(vals_list))
        categorical = len(uniques) <= (quantize_threshold if quantize_threshold is not None else bins)
        if categorical:
            # Data is quantized (e.g. discrete calo layers) - plot against rank
            # position (0, 1, 2, ...) instead of the real value, so every bin
            # is the same width and sits flush against its neighbors regardless
            # of how far apart the real-world values are.
            edges = np.arange(len(uniques) + 1) - 0.5
        elif log_x:
            lo = min(v[v > 0].min() for v in vals_list)
            if x_min is not None:
                lo = max(lo, x_min)
            edges = np.logspace(np.log10(lo), np.log10(hi), bins)
        else:
            edges = np.linspace(lo, hi, bins)

        counts_list = []
        for i, (f, color) in enumerate(zip(files, CATEGORICAL)):
            vals = np.searchsorted(uniques, f[key]) if categorical else f[key]
            weights = f[weight_key] if weight_key else None
            counts, _ = np.histogram(vals, bins=edges, weights=weights)
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

        # Ratio panel: every comparison file (all files after the reference)
        # divided by the reference, all drawn together on one shared ratio axes.
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

        if categorical:
            n_labels = min(len(uniques), 12)
            tick_pos = np.linspace(0, len(uniques) - 1, n_labels).round().astype(int)
            ax_ratio.set_xticks(tick_pos, [f"{uniques[p]:.0f}" for p in tick_pos], rotation=45, ha="right", fontsize=7)
        style_axes(ax_main)
        style_axes(ax_ratio)
        plt.setp(ax_main.get_xticklabels(), visible=False)
        if log_x:
            ax_main.set_xscale("log")
            ax_ratio.set_xscale("log")
        if log_y:
            ax_main.set_yscale("log")

    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    gs = fig.add_gridspec(4, 3, height_ratios=[3, 1, 3, 1], hspace=0.08)
    main_axes = [[fig.add_subplot(gs[0, c]) for c in range(3)], [fig.add_subplot(gs[2, c]) for c in range(3)]]
    ratio_axes = [
        [fig.add_subplot(gs[1, c], sharex=main_axes[0][c]) for c in range(3)],
        [fig.add_subplot(gs[3, c], sharex=main_axes[1][c]) for c in range(3)],
    ]
    n_events_per_file = sorted({f["n_events"] for f in files})
    n_events_label = str(n_events_per_file[0]) if len(n_events_per_file) == 1 else "/".join(str(n) for n in n_events_per_file)
    fig.suptitle(f"Calorimeter Shower Distributions  (n={n_events_label} events/file)", fontsize=13, fontweight="bold", color=INK_PRIMARY)

    ax, ax_r = main_axes[0][0], ratio_axes[0][0]
    plot_lines(ax, ax_r, "hits_per_event", bins=30)
    ax_r.set_xlabel("Cells / shower")
    ax.set_ylabel("Showers")

    ax, ax_r = main_axes[0][1], ratio_axes[0][1]
    plot_lines(ax, ax_r, "energy_per_event", bins=30, log_y=True)
    ax_r.set_xlabel("Total energy / shower  [GeV]")
    ax.set_ylabel("Showers")
    # Placed here (top-middle panel), not the cells/shower panel - that one is
    # densely multi-modal across its whole range, so any in-axes legend sits
    # on top of some line; this panel's plateau leaves a clearer corner.
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_PRIMARY, loc="lower center")

    ax, ax_r = main_axes[0][2], ratio_axes[0][2]
    plot_lines(ax, ax_r, "e", log_x=True, log_y=True, x_min=1e-4)
    ax_r.set_xlabel("Cell energy  [GeV]")
    ax.set_ylabel("Cells")
    ax.set_title("Cell energy spectrum", color=INK_PRIMARY, fontsize=10)

    # y is barrel depth (mapped from layer index to its real detector
    # position, so labels/binning match plot_distributions.py's edm4hep-based
    # y panel exactly); x/z are the regular grid's transverse cell centers -
    # a coarse bin count (well below the ~98 distinct cell positions) avoids
    # aliasing against the grid, same reasoning as plot_distributions.py.
    x_range = tuple(args.x_zoom_mm) if args.x_zoom_mm else (-100, 100)
    z_range = tuple(args.z_zoom_mm) if args.z_zoom_mm else (-150, 50)
    for (ax, ax_r), (key, lbl, key_bins, val_range) in zip(
        zip(main_axes[1], ratio_axes[1]),
        [("x", "x  [mm]", 40, x_range), ("y", "y  [mm]", 400, None), ("z", "z  [mm]", 40, z_range)],
    ):
        plot_lines(ax, ax_r, key, log_y=True, bins=key_bins, weight_key="e", val_range=val_range)
        ax_r.set_xlabel(lbl)
        ax.set_ylabel("Energy  [GeV]")

    out_png = os.path.join(args.output_dir, f"distribution_{args.geometry}.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_png}")


if __name__ == "__main__":
    main()
