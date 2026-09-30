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


def algo_key(h5_path):
    """Variant directory name (merge_within_cell, hdbscan_ms3_mcs10, ...) for a
    path under generated_showers/<variant>_<timestamp>/..., or "" for anything
    else (the Geant4 reference, the gun-edge runs). Used to pick a colour per
    ALGORITHM rather than per file position, so one variant keeps its colour no
    matter where it sits in a given figure's file list."""
    grandparent = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(h5_path))))
    m = re.match(r"^(.+)_\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}$", grandparent)
    return m.group(1) if m else ""

def _cog_from_merged(merged, h5_path, n_layers, axis_x=None, axis_z=None):
    """Energy-weighted CoG per shower from merged/projected cells
    (n_showers, capacity, 4) = (x_local, z_local, layer_id, energy_MeV).
    Shared by the raw path (which merges first) and the already-grid-projected
    path (which reads the same columns straight out of the file).

    axis_x/axis_z (n_showers, n_layers) subtract each shower's own axis position
    at that cell's depth, and MUST be passed for grid-projected input. The raw
    path works in the local shift-aligned frame where the axis already sits at
    x=z=0, but postprocessing.py/test_gun_position_grid_shift.py undo that shift
    when they project, so their stored cells are in the global frame and their
    CoG would otherwise just trace the gun impact point across the detector
    (measured: x spread +-90mm, z -142..+41mm, against +-3mm for the raw path).
    Subtracting the axis puts both paths in the same frame - the one this
    module's docstring describes and the paper defines CoG in - so the two CoG
    figures are directly comparable."""
    valid_m = merged[:, :, 3] > 0
    x = merged[:, :, 0]
    z = merged[:, :, 1]
    if axis_x is not None:
        # Index the per-shower, per-layer axis position, then mask - masking
        # first would leave padding rows at -axis instead of 0.
        layer_idx = np.clip(np.round(merged[:, :, 2]).astype(np.int64), 0, n_layers - 1)
        rows = np.arange(merged.shape[0])[:, None]
        x = x - axis_x[rows, layer_idx]
        z = z - axis_z[rows, layer_idx]
    x = np.where(valid_m, x, 0.0)
    z = np.where(valid_m, z, 0.0)
    layer = np.where(valid_m, merged[:, :, 2], 0.0)
    e = np.where(valid_m, merged[:, :, 3], 0.0)

    esum = e.sum(axis=1)
    esum_safe = np.where(esum > 0, esum, 1.0)
    cog_x = (x * e).sum(axis=1) / esum_safe
    cog_z = (z * e).sum(axis=1) / esum_safe

    print(f"  {merged.shape[0]} showers, {int(valid_m.sum()):,} merged cells")
    return {
        "label": derive_label(h5_path),
        "n_events": merged.shape[0],
        "cog_x": cog_x,
        "cog_z": cog_z,
        "layer": layer,
        "e": e,
        "valid": valid_m,
        "n_layers": n_layers,
    }


def apply_label_overrides(files, labels):
    """Replace the auto-derived legend labels with user-supplied ones, in input
    order. The "REF:" prefix is load-bearing in this script - it is how the
    reference curves are detected when drawing - so it is preserved: renaming a
    reference changes the text after the prefix, never the prefix itself."""
    if not labels:
        return
    if len(labels) != len(files):
        raise SystemExit(
            f"--labels: got {len(labels)} label(s) for {len(files)} input file(s) - "
            "pass one per input, in the same order (reference first)")
    for f, label in zip(files, labels):
        if f["label"].startswith("REF:") and not label.startswith("REF:"):
            f["label"] = f"REF: {label}"
        else:
            f["label"] = label


def load_raw_file(h5_path, n_layers, n_showers=None):
    """Read a raw DDML/cc3input-format h5 and compute each shower's
    energy-weighted centre of gravity in the LOCAL, shift-aligned frame (box
    cut + half-MIP merge, no shift-undo).

    Also accepts an ALREADY grid-projected file (the hits/n_points schema
    postprocessing.py and test_gun_position_grid_shift.py write). Those have no
    "events" dataset and their cells have already been box-cut and merged, so
    the projection steps below would be wrong to repeat - the CoG is computed
    straight off the stored cells instead. This is what lets the gun-edge-test
    (gridedge) output, which only ever exists post-projection, be plotted as
    just another line here."""
    print(f"Loading {h5_path} ...")
    sl = slice(None, n_showers)
    with h5py.File(h5_path, "r") as f:
        is_grid = "events" not in f and "hits" in f
        if is_grid:
            merged = f["hits"][sl].astype(np.float32)
            axis_x = f["axis_x_global"][sl]
            axis_z = f["axis_z_global"][sl]
        else:
            events = f["events"][sl].astype(np.float32)
            has_layer_counts = "layer_counts" in f

    if is_grid:
        return _cog_from_merged(merged, h5_path, n_layers, axis_x=axis_x, axis_z=axis_z)

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

    merged, _ = merge_all(hits, CELL_SIZE_MM, HALF_MIP_MEV)

    return _cog_from_merged(merged, h5_path, n_layers)


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
    parser.add_argument("--labels", nargs="+", default=None, metavar="LABEL",
                         help="Override the legend labels, one per input file in the same order "
                              "(reference first). Default: each label is derived from the file's "
                              "variant directory name.")
    parser.add_argument("--colors", nargs="+", default=None, metavar="HEX",
                         help="Override the categorical palette, one colour per input file in the "
                              "order given (the first file is drawn as the filled grey reference, so "
                              "its entry is unused). Wrapped modulo its length if shorter than the "
                              "file list. Used for the gun-edge figure, where the lines are two "
                              "samples x two grid phases rather than unrelated variants.")
    parser.add_argument("--cog-xz-range", type=float, nargs=2, default=(-5.0, 5.0), metavar=("LO", "HI"),
                         help="Histogram range for the CoG x and z panels [mm] (default: -5 5). The "
                              "depth panel sets its own range from the data. Widen this only if you "
                              "actually want the global-frame spread: with the axis subtracted both "
                              "input paths peak inside a few mm.")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"],
                         help="Output image format (default: pdf - vector, so it scales in talks and "
                              "papers without resampling). dpi still applies to any rasterised element.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from preprocessing.metadata import Metadata  # noqa: E402

    metadata = Metadata()
    n_layers = len(metadata.layer_bottom_pos_global)
    layer_y_mm = metadata.layer_bottom_pos_global
    files = [load_raw_file(p, n_layers, args.n_showers) for p in args.input_h5]
    apply_label_overrides(files, args.labels)

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
        "hdbscan_ms3_mcs3": "#7b3f00",                    # brown       (overflow past Okabe-Ito 7)
        "hdbscan_ms7_mcs7": "#56b4e9",                    # sky blue
        "hdbscan_ms8_mcs40": "#cc79a7",                   # reddish purple
        "hdbscan_ms12_mcs12": "#7a21dd",                  # violet
    }
    if args.colors:
        # An explicit --colors list wins over both maps: it is positional by
        # definition (the gun-edge figure's lines are two samples x two grid
        # phases, not unrelated variants), so the per-algorithm map must not
        # override it.
        CATEGORICAL = list(args.colors)
        ALGO_COLORS = {}
    file_colors = [ALGO_COLORS.get(algo_key(p), CATEGORICAL[i % len(CATEGORICAL)])
                    for i, p in enumerate(args.input_h5)]

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
        for i, f in enumerate(files):
            color = file_colors[i]
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
            for i, (f, counts) in enumerate(zip(files[1:], counts_list[1:]), start=1):
                color = file_colors[i]
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

    for (ax, ax_r), (key, lbl, val_range) in zip(
        zip(main_axes, ratio_axes),
        [("cog_x", "CoG x  [mm]", tuple(args.cog_xz_range)), ("cog_y", "CoG y (depth)  [mm]", None),
          ("cog_z", "CoG z  [mm]", tuple(args.cog_xz_range))],
    ):
        plot_cog(ax, ax_r, key, val_range=val_range)
        ax_r.set_xlabel(lbl)
        ax.set_ylabel("Showers")

    # One legend for all three panels, above them, so it covers no data.
    handles, labels = main_axes[0].get_legend_handles_labels()
    # "REF:" only marks the reference internally; the grey band identifies it.
    labels = [l.removeprefix("REF: ") for l in labels]
    fig.legend(handles, labels, frameon=False, fontsize=10, labelcolor=INK_PRIMARY,
               loc="outside upper center", ncol=len(labels))

    out_png = os.path.join(args.output_dir, f"cog_{args.geometry}.{args.format}")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_png}")


if __name__ == "__main__":
    main()
