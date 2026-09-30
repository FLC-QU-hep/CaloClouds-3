"""
Figure-5-style occupancy comparison plot (see arXiv:2511.01460, "CaloClouds3:
Ultra-Fast Geometry-Independent Highly-Granular Calorimeter Simulation") built
from projected-grid h5 files (postprocessing.py's output), instead of the
paper's own DetectorBinnedData pipeline. Three columns, each with a main panel
on top (log-y, matching the paper) and one ratio-to-reference panel per
comparison file stacked below it:

  * Left   - histogram of the occupancy of the calorimeter in each shower
             ("number of active cells" per shower; y = number of showers).
  * Centre - the number of active cells in concentric rings about the shower
             axis, SUMMED over every shower in the file (not a per-shower
             mean) - matches the paper's "sum active cells" y-label. Radial
             distance is measured from each shower's true axis at that cell's
             depth layer (axis_x_global/axis_z_global from postprocessing.py),
             same convention as plot_shower_profiles_from_grid.py.
  * Right  - the number of active cells in each layer, summed over every
             shower.

The first input file is the reference (solid fill, "Geant4" role, matching
plot_shower_profiles_from_grid.py's convention); every other input file is a
comparison (step line) with its own ratio row per column. Errors are Poisson
(sqrt(N)) counts uncertainty (the paper instead uses the std. dev. across 3
seeds, which needs repeated runs we don't have).

Usage:
    python plot_occupancy_from_grid.py <reference_grid.h5> <comparison1_grid.h5> [...] \
        [--output-dir DIR] --geometry {real,regular}
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.metadata import Metadata  # noqa: E402

from postprocessing import CELL_SIZE_MM  # noqa: E402


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
        variant_dir = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(h5_path))))
        label = re.sub(r"_\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}$", "", variant_dir)
    elif label.startswith("input_cc3"):
        variant_dir = os.path.basename(os.path.dirname(os.path.abspath(h5_path)))
        variant = re.sub(r"^cc3input_", "", variant_dir)
        if variant == "merge_within_regular_subcell":
            variant = "25x_subcell"
        label = f"REF: {variant}"
    return label


def algo_key(h5_path):
    """Variant directory name (merge_within_cell, hdbscan_ms3_mcs10, ...) for a
    path under generated_showers/<variant>_<timestamp>/..., or "" for anything
    else (the Geant4 reference, the gun-edge runs). Used to pick a colour per
    ALGORITHM rather than per file position, so one variant keeps its colour no
    matter where it sits in a given figure's file list."""
    grandparent = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(h5_path))))
    m = re.match(r"^(.+)_\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}$", grandparent)
    return m.group(1) if m else ""

def load_grid_file(h5_path, layer_y_mm):
    """Read a merged-cell point cloud and flatten it into per-cell arrays,
    same convention as plot_shower_profiles_from_grid.py. Also returns
    hits_per_event (occupancy per shower) for the left-column histogram."""
    print(f"Loading {h5_path} ...")
    with h5py.File(h5_path, "r") as f:
        hits = f["hits"][:]
        n_layers = f.attrs["n_layers"]
        y_centers = layer_y_mm[:n_layers]
        axis_x_global = f["axis_x_global"][:]
        axis_z_global = f["axis_z_global"][:]

    n_events = hits.shape[0]
    valid = hits[:, :, 3] > 0

    x = hits[:, :, 0][valid]
    z = hits[:, :, 1][valid]
    layer = hits[:, :, 2][valid].astype(np.int64)
    y = y_centers[layer]
    evt = np.nonzero(valid)[0]

    axis_x_per_hit = axis_x_global[evt, layer]
    axis_z_per_hit = axis_z_global[evt, layer]
    radial = np.sqrt((x - axis_x_per_hit) ** 2 + (z - axis_z_per_hit) ** 2)

    hits_per_event = np.bincount(evt, minlength=n_events).astype(float)

    print(f"  {n_events} showers, {len(x):,} active cells")
    return {
        "label": derive_label(h5_path),
        "n_events": n_events,
        "y": y, "radial": radial,
        "hits_per_event": hits_per_event,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_h5", nargs="+", help="Reference *_grid.h5 first, then one or more comparison *_grid.h5 files")
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots"),
                         help="Directory to write occupancy_<geometry>.png into (default: plots/ next to this script)")
    parser.add_argument("--geometry", default="regular")
    parser.add_argument("--occupancy-range", type=float, nargs=2, default=(0, 1500), metavar=("LO", "HI"),
                         help="Fixed range for the occupancy (active cells/shower) histogram, matching the "
                              "paper's Figure 5 left panel. Default: (0, 1500).")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"],
                         help="Output image format (default: pdf - vector, so it scales in talks and "
                              "papers without resampling). dpi still applies to any rasterised element.")
    args = parser.parse_args()

    if len(args.input_h5) < 2:
        raise SystemExit("Need at least a reference file and one comparison file")

    os.makedirs(args.output_dir, exist_ok=True)
    layer_y_mm = Metadata().layer_bottom_pos_global
    files = [load_grid_file(p, layer_y_mm) for p in args.input_h5]
    ref, comps = files[0], files[1:]
    for f in files:
        print(f"{f['label']}: {f['n_events']} events")

    SURFACE, INK_PRIMARY = "#fcfcfb", "#0b0b0b"
    GRIDLINE, BASELINE = "#e1e0d9", "#c3c2b7"
    # Fallback for any variant ALGO_COLORS below doesn't name, indexed by file
    # position and wrapped modulo its length: zip(files, CATEGORICAL) used to
    # truncate silently, so a 9th input file was dropped from the plot with no
    # error. Slot 0 is a placeholder - files[0] is the reference and is always
    # drawn as the filled grey band, so its entry is never used as a line colour
    # (the --colors override follows the same convention). Hues are the
    # Okabe-Ito colourblind-safe set; its yellow #f0e442 is deliberately absent,
    # at L 0.90 it is too pale to read as a line on the #fcfcfb surface.
    CATEGORICAL = ["#c9c8c2",                                                 # grey reference placeholder
                    "#0072b2", "#e69f00", "#009e73", "#cc79a7", "#d55e00",    # Okabe-Ito
                    "#56b4e9", "#7a21dd",                                     # + violet for a 7th series
                    "#7b3f00", "#a1215a", "#455a64", "#00695c"]               # overflow past the validated 7
    # Colour follows the ALGORITHM, not the file's position in the list: the
    # regular4 figure lists cell/hdbscan/subcell while run.sh's full comparison
    # lists cell/subcell/subcell_6kcut/hdbscan..., so a purely positional
    # palette gave one variant different colours in different figures. Keyed on
    # the run directory (algo_key), so --labels stays purely cosmetic. Subcell
    # keeps the orange and hdbscan the green they already carry in the gun-edge
    # figure, and Geant4 stays the grey band, so those three read the same way
    # in every figure this directory produces.
    ALGO_COLORS = {
        "merge_within_cell": "#0072b2",                   # blue
        "merge_within_regular_subcell": "#e69f00",        # amber      (gun-edge orange)
        "merge_within_regular_subcell_6kcut": "#d55e00",  # vermillion
        "hdbscan_ms3_mcs10": "#009e73",                   # green      (gun-edge green)
        "hdbscan_ms7_mcs7": "#56b4e9",                    # sky blue
        "hdbscan_ms8_mcs40": "#cc79a7",                   # reddish purple
        "hdbscan_ms12_mcs12": "#7a21dd",                  # violet
    }
    file_colors = [ALGO_COLORS.get(algo_key(p), CATEGORICAL[i % len(CATEGORICAL)])
                    for i, p in enumerate(args.input_h5)]


    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_PRIMARY, "xtick.color": INK_PRIMARY, "ytick.color": INK_PRIMARY,
        "text.color": INK_PRIMARY, "font.family": "sans-serif", "font.size": 10, "axes.grid": False,
    })

    n_ratio = len(comps)
    fig = plt.figure(figsize=(16, 4 + 1.6 * n_ratio), constrained_layout=True)
    height_ratios = [3] + [1] * n_ratio
    subfigs = fig.subfigures(1, 3)
    n_events_per_file = sorted({f["n_events"] for f in files})
    n_events_label = str(n_events_per_file[0]) if len(n_events_per_file) == 1 else "/".join(str(n) for n in n_events_per_file)
    fig.suptitle(f"Occupancy  (n={n_events_label} events/file)", fontsize=13, fontweight="bold", color=INK_PRIMARY)

    def style_axes(ax, log_y=True):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        if log_y:
            ax.set_yscale("log")

    def draw_column(subfig, title, xlabel, ylabel, main_vals, edges, is_showers_hist, categorical=False, show_legend=False):
        axes = subfig.subplots(1 + n_ratio, 1, height_ratios=height_ratios, sharex=True,
                                gridspec_kw={"hspace": 0.08})
        ax_main = axes[0] if n_ratio else axes

        uniques = None
        if categorical:
            uniques = np.unique(np.concatenate([main_vals(f) for f in files]))
            edges = np.arange(len(uniques) + 1) - 0.5
            raw_main_vals = main_vals
            main_vals = lambda f: np.searchsorted(uniques, raw_main_vals(f))

        def compute(f):
            counts, _ = np.histogram(main_vals(f), bins=edges)
            return counts.astype(float), np.sqrt(counts)

        centers = 0.5 * (edges[:-1] + edges[1:])
        ref_counts, ref_err = compute(ref)
        ax_main.stairs(ref_counts, edges, fill=True, color="#eeede9", edgecolor="#c9c8c2",
                        linewidth=0.8, label=ref["label"], zorder=1)

        comp_results = []
        for f, color in zip(comps, file_colors[1:]):
            counts, err = compute(f)
            is_ref = f["label"].startswith("REF:")
            line_color = INK_PRIMARY if is_ref else color
            linestyle = ":" if is_ref else "-"
            comp_results.append((f, line_color, linestyle, counts, err))
            ax_main.stairs(counts, edges, color=line_color, linewidth=1.8, linestyle=linestyle, label=f["label"], zorder=3)
            ax_main.fill_between(centers, np.maximum(counts - err, 1e-10), counts + err, step="mid",
                                  color=line_color, alpha=0.15, zorder=2, linewidth=0)

        ax_main.set_title(title, color=INK_PRIMARY, fontsize=11)
        ax_main.set_ylabel(ylabel)
        style_axes(ax_main)
        if show_legend:
            ax_main.legend(frameon=False, fontsize=8, labelcolor=INK_PRIMARY, loc="lower center" if is_showers_hist else "best")
        if categorical:
            n_labels = min(len(uniques), 12)
            tick_pos = np.linspace(0, len(uniques) - 1, n_labels).round().astype(int)
            axes[-1].set_xticks(tick_pos, [f"{uniques[p]:.0f}" for p in tick_pos], rotation=45, ha="right", fontsize=7)

        for ax_ratio, (f, line_color, linestyle, counts, err) in zip(axes[1:], comp_results):
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(ref_counts > 0, counts / ref_counts, np.nan)
                rel_err = np.where(
                    ref_counts > 0,
                    ratio * np.sqrt((err / np.where(counts > 0, counts, np.nan)) ** 2 + (ref_err / ref_counts) ** 2),
                    np.nan,
                )
            ax_ratio.stairs(ratio, edges, color=line_color, linewidth=1.4, linestyle=linestyle, zorder=3)
            ax_ratio.fill_between(centers, ratio - rel_err, ratio + rel_err, step="mid",
                                   color=line_color, alpha=0.2, zorder=2, linewidth=0)
            ax_ratio.axhline(1.0, color=BASELINE, linewidth=0.8, zorder=1)
            ax_ratio.set_ylim(0.5, 1.5)
            ax_ratio.set_ylabel(f"Ratio\n{f['label']}", fontsize=8)
            style_axes(ax_ratio, log_y=False)

        axes[-1].set_xlabel(xlabel)

    occ_edges = np.linspace(args.occupancy_range[0], args.occupancy_range[1], 61)
    radial_edges = np.arange(0, 100 + CELL_SIZE_MM, CELL_SIZE_MM)

    draw_column(subfigs[0], "Occupancy", "number of active cells", "number of showers",
                lambda f: f["hits_per_event"], occ_edges, True, show_legend=True)
    draw_column(subfigs[1], "Radial occupancy", "radius  [mm]", "sum active cells",
                lambda f: f["radial"], radial_edges, False)
    draw_column(subfigs[2], "Longitudinal occupancy", "layers", "sum active cells",
                lambda f: f["y"], None, False, categorical=True)

    out_png = os.path.join(args.output_dir, f"occupancy_{args.geometry}.{args.format}")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_png}")


if __name__ == "__main__":
    main()
