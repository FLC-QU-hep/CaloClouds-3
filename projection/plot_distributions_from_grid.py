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


REFERENCE_LABEL = "Geant4"


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
        # .../cc3input_<variant>/input_cc3_file_0[_postprocessed].h5 - the real
        # detector simulation. Which merging variant it was projected from is a
        # property of how the reference was built, not of what it is, so it stays
        # out of the legend.
        label = REFERENCE_LABEL
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

def apply_label_overrides(files, labels):
    """Replace the auto-derived legend labels with user-supplied ones, in input
    order. Only the displayed label changes - each file's "is_ref" flag keeps
    whatever derive_label() determined, so renaming a variant can neither promote
    it to the reference nor demote the real reference."""
    if not labels:
        return
    if len(labels) != len(files):
        raise SystemExit(
            f"--labels: got {len(labels)} label(s) for {len(files)} input file(s) - "
            "pass one per input, in the same order (reference first)")
    for f, label in zip(files, labels):
        f["label"] = label


def load_incident_energy(h5_path, n_showers=None):
    """Incident photon energy [GeV] per shower. The postprocessed file does not
    carry it, so read it from the raw (pre-postprocessing) file next to it -
    same path without "_postprocessed", as plot_cog_from_grid.py does.
    postprocessing.py keeps the first N showers in order, so row i matches."""
    raw_path = h5_path.replace("_postprocessed", "")
    with h5py.File(raw_path, "r") as f:
        return f["energy"][:n_showers, 0].astype(float)


def load_incident_energy_from(spec, n_showers):
    """Incident energy [GeV] from an explicit source file, for grid files with
    no raw file next to them (the gun-edge test output). spec is "PATH" for
    the first n_showers showers, or "PATH@SEED" for test_gun_position_grid_shift.py's
    --random-sample draw: the same sorted without-replacement choice of
    n_showers out of the file, with that seed, so row i matches."""
    path, _, seed = spec.partition("@")
    with h5py.File(path, "r") as f:
        if not seed:
            return f["energy"][:n_showers, 0].astype(float)
        total = f["events"].shape[0]
        sel = np.sort(np.random.default_rng(int(seed)).choice(total, size=n_showers, replace=False))
        return f["energy"][sel, 0].astype(float)


def load_grid_file(h5_path, layer_y_mm, xz_range=None, min_cell_energy_mev=None, n_showers=None):
    """Read a merged-cell point cloud (the output of project_ecal_showers.py:
    hits (n_showers, capacity, 4) = (x_local, z_local, layer_id, energy_MeV),
    zero-padded per shower up to n_points) and flatten the real (non-padding)
    rows into per-cell arrays. If xz_range=(lo, hi) is given, every quantity
    (not just the x/z display range) is restricted to cells with both x and z
    inside that box, so hits_per_event/energy_per_event/the cell energy
    spectrum all describe the same central region rather than the full shower.
    If min_cell_energy_mev is given, cells at or below it are dropped too - a
    post-projection readout-threshold cut, on top of (not instead of)
    whatever threshold, if any, the input file's own merge step already
    applied (postprocessing.py's merge_all already drops cells <=
    HALF_MIP_MEV=0.1 MeV, but not every input file went through that)."""
    print(f"Loading {h5_path} ...")
    with h5py.File(h5_path, "r") as f:
        hits = f["hits"][:n_showers]
        n_layers = f.attrs["n_layers"]
        y_centers = layer_y_mm[:n_layers]

    n_events = hits.shape[0]
    valid = hits[:, :, 3] > 0
    if min_cell_energy_mev is not None:
        valid &= hits[:, :, 3] > min_cell_energy_mev
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
    label = derive_label(h5_path)
    return {
        "label": label,
        "is_ref": label == REFERENCE_LABEL,
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
    parser.add_argument("--labels", nargs="+", default=None, metavar="LABEL",
                         help="Override the legend labels, one per input file in the same order "
                              "(reference first). Default: each label is derived from the file's "
                              "variant directory name.")
    parser.add_argument("--spectrum-max-gev", type=float, default=None, metavar="HI",
                         help="Display-only upper cap for the cell energy spectrum panel's binning [GeV]. "
                              "Cells above it stay in every other quantity, they are just not drawn. "
                              "Default: the top edge follows the hottest cell in any input file.")
    parser.add_argument("--energy-per-shower-max-gev", type=float, default=None, metavar="HI",
                         help="Display-only upper cap for the total-energy-per-shower panel's binning "
                              "[GeV]. Same reasoning as --spectrum-max-gev.")
    parser.add_argument("--min-cell-energy-mev", type=float, default=None,
                         help="Drop merged cells with energy <= this threshold (MeV) before computing "
                              "any quantity - a post-projection readout-threshold cut applied on top of "
                              "each file's own merge step. Default: no additional cut.")
    parser.add_argument("--n-showers", type=int, default=None,
                         help="Cap every input file to its first N showers. These histograms are raw "
                              "counts, so a reference file holding more showers than the generated files "
                              "sits above them by that ratio; pass the generated files' N to compare "
                              "equal amounts of simulated data. Default: use every shower in each file.")
    parser.add_argument("--colors", nargs="+", default=None, metavar="HEX",
                         help="Override the categorical palette, one colour per input file in the "
                              "order given (the first file is drawn as the filled grey reference, so "
                              "its entry is unused). Wrapped modulo its length if shorter than the "
                              "file list. Used for the gun-edge figure, where the lines are two "
                              "samples x two grid phases rather than unrelated variants.")
    parser.add_argument("--compact", action="store_true",
                         help="Three panels only: cells/shower, total energy/shower and the energy "
                              "response E_vis/E_inc (the cell energy spectrum and x/y/z distributions "
                              "are left to the shower-profile and CoG figures). Needs the raw file "
                              "next to each postprocessed one, for the incident energy.")
    parser.add_argument("--incident-energy-h5", nargs="+", default=None, metavar="PATH[@SEED]",
                         help="--compact only: where each input file's incident energy comes from, one "
                              "per input file in the same order, instead of the raw file next to it. "
                              "PATH takes the first N showers; PATH@SEED repeats "
                              "test_gun_position_grid_shift.py's --random-sample draw with that seed.")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"],
                         help="Output image format (default: pdf - vector, so it scales in talks and "
                              "papers without resampling). dpi still applies to any rasterised element.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    layer_y_mm = Metadata().layer_bottom_pos_global
    xz_range = tuple(args.xz_range_mm) if args.xz_range_mm else None
    files = [load_grid_file(p, layer_y_mm, xz_range=xz_range, min_cell_energy_mev=args.min_cell_energy_mev,
                            n_showers=args.n_showers)
             for p in args.input_h5]
    apply_label_overrides(files, args.labels)
    if args.compact:
        if args.incident_energy_h5 and len(args.incident_energy_h5) != len(files):
            raise SystemExit(f"--incident-energy-h5: got {len(args.incident_energy_h5)} for {len(files)} input files")
        for i, (f, p) in enumerate(zip(files, args.input_h5)):
            if args.incident_energy_h5:
                e_inc = load_incident_energy_from(args.incident_energy_h5[i], f["n_events"])
            else:
                e_inc = load_incident_energy(p, args.n_showers)
            if len(e_inc) != f["n_events"]:
                raise SystemExit(f"{p}: {len(e_inc)} incident energies for {f['n_events']} showers")
            f["response"] = f["energy_per_event"] / e_inc
    for f in files:
        print(f"\n{f['label']}: {f['n_events']} events, {len(f['e']):,} cells")

    # Chart chrome (light surface, ink roles) and a fixed-order categorical
    # palette - same convention as DDML/scripts/plot_distributions.py, so
    # figures from either pipeline stay visually consistent.
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

    bins = 100

    def style_axes(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)

    def plot_lines(ax_main, ax_ratio, key, log_x=False, log_y=False, bins=bins, quantize_threshold=None, weight_key=None, x_min=None, x_max=None, val_range=None):
        vals_list = [f[key] for f in files]
        if val_range is not None:
            lo, hi = val_range
        else:
            lo = min(v.min() for v in vals_list)
            hi = max(v.max() for v in vals_list)
        # Display-only upper cap: without it `hi` is the single most extreme value
        # in any file, so one outlier cell/shower stretches the axis and squeezes
        # the populated region into the first few bins. Values above the cap fall
        # outside the binning and are simply not drawn - every other panel, and
        # the counts below the cap, are untouched.
        if x_max is not None:
            hi = min(hi, x_max)
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
        for i, f in enumerate(files):
            color = file_colors[i]
            vals = np.searchsorted(uniques, f[key]) if categorical else f[key]
            weights = f[weight_key] if weight_key else None
            counts, _ = np.histogram(vals, bins=edges, weights=weights)
            counts_list.append(counts)
            if i == 0:
                ax_main.stairs(counts, edges, fill=True, color="#eeede9", edgecolor="#c9c8c2",
                                linewidth=0.8, label=f["label"], zorder=1)
            elif f["is_ref"]:
                # A second reference variant (not the primary "Geant4" role) -
                # kept visually distinct from the colored comparison lines.
                ax_main.stairs(counts, edges, color=INK_PRIMARY, linewidth=2, linestyle=":", label=f["label"], zorder=2)
            else:
                ax_main.stairs(counts, edges, color=color, linewidth=2, label=f["label"], zorder=2)

        # Ratio panel: every comparison file (all files after the reference)
        # divided by the reference, all drawn together on one shared ratio axes.
        ref_counts = counts_list[0].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            for i, (f, counts) in enumerate(zip(files[1:], counts_list[1:]), start=1):
                color = file_colors[i]
                ratio = np.where(ref_counts > 0, counts / ref_counts, np.nan)
                if f["is_ref"]:
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

    if args.compact:
        fig = plt.figure(figsize=(16, 5.5), constrained_layout=True)
        gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.08)
        main_axes = [fig.add_subplot(gs[0, c]) for c in range(3)]
        ratio_axes = [fig.add_subplot(gs[1, c], sharex=main_axes[c]) for c in range(3)]

        plot_lines(main_axes[0], ratio_axes[0], "hits_per_event", bins=30, log_y=True)
        ratio_axes[0].set_xlabel("Cells / shower")
        main_axes[0].set_ylabel("Showers")

        plot_lines(main_axes[1], ratio_axes[1], "energy_per_event", bins=30, log_y=True,
                   x_max=args.energy_per_shower_max_gev)
        ratio_axes[1].set_xlabel("Total energy / shower  [GeV]")
        main_axes[1].set_ylabel("Showers")
        # One legend for all three panels, above them, so it covers no data.
        handles, labels = main_axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, fontsize=10, labelcolor=INK_PRIMARY,
                   loc="outside upper center", ncol=len(labels))

        # Range from the reference's 1-99.5 percentiles: the few lowest-energy
        # showers (broadest response) fill the outer bins with O(1) counts,
        # which only adds noise to the ratio panel.
        resp_range = tuple(np.percentile(files[0]["response"], [1, 99.5]))
        plot_lines(main_axes[2], ratio_axes[2], "response", bins=25, log_y=True, val_range=resp_range)
        # Axis limits looser than the binned range, so the histogram and its
        # ratio don't run into the panel edges.
        pad = 0.25 * (resp_range[1] - resp_range[0])
        ratio_axes[2].set_xlim(resp_range[0] - pad, resp_range[1] + pad)
        ratio_axes[2].set_xlabel(r"$E_\mathrm{vis} / E_\mathrm{inc}$")
        main_axes[2].set_ylabel("Showers")

        out = os.path.join(args.output_dir, f"distribution_{args.geometry}.{args.format}")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {out}")
        return

    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    gs = fig.add_gridspec(4, 3, height_ratios=[3, 1, 3, 1], hspace=0.08)
    main_axes = [[fig.add_subplot(gs[0, c]) for c in range(3)], [fig.add_subplot(gs[2, c]) for c in range(3)]]
    ratio_axes = [
        [fig.add_subplot(gs[1, c], sharex=main_axes[0][c]) for c in range(3)],
        [fig.add_subplot(gs[3, c], sharex=main_axes[1][c]) for c in range(3)],
    ]
    ax, ax_r = main_axes[0][0], ratio_axes[0][0]
    plot_lines(ax, ax_r, "hits_per_event", bins=30, log_y=True)
    ax_r.set_xlabel("Cells / shower")
    ax.set_ylabel("Showers")

    ax, ax_r = main_axes[0][1], ratio_axes[0][1]
    plot_lines(ax, ax_r, "energy_per_event", bins=30, log_y=True, x_max=args.energy_per_shower_max_gev)
    ax_r.set_xlabel("Total energy / shower  [GeV]")
    ax.set_ylabel("Showers")
    # One legend for all panels, above them, so it covers no data (same as the
    # --compact figure; the shower count is in the paper caption, not a title).
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=10, labelcolor=INK_PRIMARY,
               loc="outside upper center", ncol=len(labels))

    ax, ax_r = main_axes[0][2], ratio_axes[0][2]
    plot_lines(ax, ax_r, "e", log_x=True, log_y=True, x_min=1e-4, x_max=args.spectrum_max_gev)
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

    out_png = os.path.join(args.output_dir, f"distribution_{args.geometry}.{args.format}")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_png}")


if __name__ == "__main__":
    main()
