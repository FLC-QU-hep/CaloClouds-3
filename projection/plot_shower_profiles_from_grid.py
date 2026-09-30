"""
Figure-6-style comparison plot (see arXiv:2601.11716) built from projected-grid
h5 files (the output of project_ecal_showers.py) instead of raw edm4hep hits -
same three columns and layout as DDML/scripts/plot_shower_profiles.py (see
shower_profiles_regular.png): cell energy spectrum, longitudinal profile,
radial profile, each with a main panel on top and one ratio panel per
comparison file stacked below it. The underlying quantity is a regular grid
*cell* (a project_ecal_showers.py histogram bin, possibly summing several
generated hits landing in the same cell) rather than a raw hit.

The first input file is the reference (solid line, "Geant4" role); every
other input file is a comparison (dashed line) and gets its own ratio row
per column, showing that file's profile divided by the reference's.

Longitudinal = mean cell energy per shower binned by y (the barrel depth
axis, read from each cell's layer index via the real detector's layer
positions); radial = mean cell energy per shower binned by sqrt(x^2 + z^2)
(transverse distance from the shower axis, x=z=0). Error bars/bands are the
standard error of the per-shower mean (std across showers / sqrt(n_showers));
the cell energy spectrum uses Poisson (sqrt(N)) errors instead, matching the
paper.

Usage:
    python plot_shower_profiles_from_grid.py <reference_grid.h5> <comparison1_grid.h5> [...] \\
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
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.metadata import Metadata  # noqa: E402

from postprocessing import CELL_SIZE_MM  # noqa: E402


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


def load_grid_file(h5_path, layer_y_mm, min_cell_energy_mev=None, n_showers=None):
    """Read a merged-cell point cloud (the output of project_ecal_showers.py:
    hits (n_showers, capacity, 4) = (x_local, z_local, layer_id, energy_MeV),
    zero-padded per shower up to n_points) and flatten the real (non-padding)
    rows into per-cell arrays (x, y, z, e, evt). Radial distance is measured
    from each shower's true axis at that hit's depth layer - axis_x_global/
    axis_z_global (n_showers, n_layers), written by postprocessing.py from
    the gun position (x=0, z=-50mm) and that shower's own entrance angle via
    get_alignment_shifts(). The axis drifts linearly through x/z with depth
    because of the angle, so a single per-shower centroid would average that
    drift away and smear the core into a ring instead of a peak at r=0.
    If min_cell_energy_mev is given, cells at or below it are dropped too - a
    post-projection readout-threshold cut, on top of (not instead of)
    whatever threshold, if any, the input file's own merge step already
    applied (postprocessing.py's merge_all already drops cells <=
    HALF_MIP_MEV=0.1 MeV, but not every input file went through that).
    If n_showers is given, only the first N showers of the file are read. The
    cell energy spectrum is a raw *count* of cells, so a reference file holding
    more showers than the generated files it is compared against sits above them
    by exactly that ratio and every ratio panel is scaled by it - cap every file
    to the same N to compare like with like."""
    print(f"Loading {h5_path} ...")
    with h5py.File(h5_path, "r") as f:
        hits = f["hits"][:n_showers]
        n_layers = f.attrs["n_layers"]
        y_centers = layer_y_mm[:n_layers]
        axis_x_global = f["axis_x_global"][:n_showers]  # (n_showers, n_layers)
        axis_z_global = f["axis_z_global"][:n_showers]  # (n_showers, n_layers)

    n_events = hits.shape[0]
    valid = hits[:, :, 3] > 0
    if min_cell_energy_mev is not None:
        valid &= hits[:, :, 3] > min_cell_energy_mev

    x = hits[:, :, 0][valid]
    z = hits[:, :, 1][valid]
    layer = hits[:, :, 2][valid].astype(np.int64)
    e = hits[:, :, 3][valid] / 1000.0  # DDML convention: stored energy is MeV, plots want GeV
    y = y_centers[layer]
    evt = np.nonzero(valid)[0]

    axis_x_per_hit = axis_x_global[evt, layer]
    axis_z_per_hit = axis_z_global[evt, layer]

    label = derive_label(h5_path)
    print(f"  {n_events} showers, {len(e):,} merged cells")
    return {
        "label": label,
        "is_ref": label == REFERENCE_LABEL,
        "n_events": n_events,
        "x": x, "y": y, "z": z, "e": e, "evt": evt,
        "radial": np.sqrt((x - axis_x_per_hit) ** 2 + (z - axis_z_per_hit) ** 2),
    }


def per_event_profile(value, energy, evt, n_events, edges):
    """Mean +/- SEM (across showers) of the per-shower binned energy sum, for
    each of len(edges)-1 bins."""
    n_bins = len(edges) - 1
    bin_id = np.clip(np.digitize(value, edges) - 1, 0, n_bins - 1)
    in_range = (value >= edges[0]) & (value <= edges[-1])
    flat_id = bin_id[in_range] * n_events + evt[in_range]
    sums = np.bincount(flat_id, weights=energy[in_range], minlength=n_bins * n_events)
    sums = sums.reshape(n_bins, n_events)
    mean = sums.mean(axis=1)
    sem = sums.std(axis=1, ddof=1) / np.sqrt(n_events)
    return mean, sem


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_h5", nargs="+", help="Reference *_grid.h5 first, then one or more comparison *_grid.h5 files")
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots"),
                         help="Directory to write shower_profiles_<geometry>.png into (default: plots/ next to this script)")
    parser.add_argument("--geometry", default="regular")
    parser.add_argument("--min-cell-energy-mev", type=float, default=None,
                         help="Drop merged cells with energy <= this threshold (MeV) before computing "
                              "any quantity - a post-projection readout-threshold cut applied on top of "
                              "each file's own merge step. Default: no additional cut.")
    parser.add_argument("--n-showers", type=int, default=None,
                         help="Cap every input file to its first N showers. The cell energy spectrum is a "
                              "raw cell count, so a reference file with more showers than the generated "
                              "files sits above them by that ratio; pass the generated files' N to compare "
                              "equal amounts of simulated data. Default: use every shower in each file.")
    parser.add_argument("--labels", nargs="+", default=None, metavar="LABEL",
                         help="Override the legend labels, one per input file in the same order "
                              "(reference first). Default: each label is derived from the file's "
                              "variant directory name.")
    parser.add_argument("--spectrum-zoom-gev", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                         help="Draw an inset zoom box on the cell energy spectrum panel covering this "
                              "energy range [GeV], e.g. --spectrum-zoom-gev 1e-4 5e-3.")
    parser.add_argument("--spectrum-max-gev", type=float, default=None, metavar="HI",
                         help="Display-only upper cap for the cell energy spectrum panel's binning [GeV]. "
                              "Cells above it are not drawn; no other panel is affected. Default: the top "
                              "edge follows the hottest cell in any input file.")
    parser.add_argument("--colors", nargs="+", default=None, metavar="HEX",
                         help="Override the categorical palette, one colour per input file in the "
                              "order given (the first file is drawn as the filled grey reference, so "
                              "its entry is unused). Wrapped modulo its length if shorter than the "
                              "file list. Used for the gun-edge figure, where the lines are two "
                              "samples x two grid phases rather than unrelated variants.")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"],
                         help="Output image format (default: pdf - vector, so it scales in talks and "
                              "papers without resampling). dpi still applies to any rasterised element.")
    args = parser.parse_args()

    if len(args.input_h5) < 2:
        raise SystemExit("Need at least a reference file and one comparison file")

    os.makedirs(args.output_dir, exist_ok=True)
    layer_y_mm = Metadata().layer_bottom_pos_global
    files = [load_grid_file(p, layer_y_mm, min_cell_energy_mev=args.min_cell_energy_mev,
                            n_showers=args.n_showers) for p in args.input_h5]
    apply_label_overrides(files, args.labels)
    ref, comps = files[0], files[1:]
    for f in files:
        print(f"{f['label']}: {f['n_events']} events, {len(f['e']):,} cells")

    # Chart chrome + fixed-order categorical palette (same convention as
    # DDML/scripts/plot_shower_profiles.py, see dataviz skill).
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

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_PRIMARY, "xtick.color": INK_PRIMARY, "ytick.color": INK_PRIMARY,
        "text.color": INK_PRIMARY, "font.family": "sans-serif", "font.size": 10, "axes.grid": False,
    })

    n_ratio = len(comps)
    # One ratio row per comparison file, so the figure grows with every variant
    # added - at 7-8 variants the old 1.6in/row made a ~16in-tall strip. These
    # numbers are chosen so the arithmetic is fixed rather than proportional:
    # with height_ratios [MAIN_H/RATIO_H] + [1]*n against a figure height of
    # MAIN_H + RATIO_H*n, the main panel is always exactly MAIN_H inches tall
    # and each ratio row exactly RATIO_H, no matter how many variants there are.
    MAIN_H, RATIO_H = 4.5, 0.75
    fig = plt.figure(figsize=(16, MAIN_H + RATIO_H * n_ratio), constrained_layout=True)
    height_ratios = [MAIN_H / RATIO_H] + [1] * n_ratio
    subfigs = fig.subfigures(1, 3)

    def style_axes(ax, log_y=True):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        if log_y:
            ax.set_yscale("log")

    def draw_column(subfig, title, xlabel, main_vals, ref_edges_fn, is_spectrum, categorical=False,
                     show_legend=False, zoom_xlim=None):
        axes = subfig.subplots(1 + n_ratio, 1, height_ratios=height_ratios, sharex=True,
                                gridspec_kw={"hspace": 0.08})
        ax_main = axes[0] if n_ratio else axes

        uniques = None
        if categorical:
            uniques = np.unique(np.concatenate([main_vals(f) for f in files]))
            edges = np.arange(len(uniques) + 1) - 0.5
            raw_main_vals = main_vals
            main_vals = lambda f: np.searchsorted(uniques, raw_main_vals(f))
        else:
            edges = ref_edges_fn()

        def compute(f):
            vals = main_vals(f)
            if is_spectrum:
                counts, _ = np.histogram(vals, bins=edges)
                err = np.sqrt(counts)
                return counts.astype(float), err
            mean, sem = per_event_profile(vals, f["e"], f["evt"], f["n_events"], edges)
            return mean, sem

        centers = 0.5 * (edges[:-1] + edges[1:])
        ref_mean, ref_err = compute(ref)
        ax_main.stairs(ref_mean, edges, fill=True, color="#eeede9", edgecolor="#c9c8c2",
                        linewidth=0.8, label=ref["label"], zorder=1)

        comp_results = []
        for i, f in enumerate(comps, start=1):
            color = file_colors[i]
            mean, err = compute(f)
            is_ref = f["is_ref"]
            line_color = INK_PRIMARY if is_ref else color
            linestyle = ":" if is_ref else "-"
            comp_results.append((f, line_color, linestyle, mean, err))
            ax_main.stairs(mean, edges, color=line_color, linewidth=1.8, linestyle=linestyle, label=f["label"], zorder=3)
            ax_main.fill_between(centers, mean - err, mean + err, step="mid", color=line_color, alpha=0.15, zorder=2, linewidth=0)

        ax_main.set_title(title, color=INK_PRIMARY, fontsize=11)
        ax_main.set_ylabel("# Cells" if is_spectrum else "Mean Energy  [GeV]")
        style_axes(ax_main)
        if is_spectrum:
            ax_main.set_xscale("log")
        elif not categorical:
            ymax = max(np.nanmax(ref_mean), max((np.nanmax(m) for _, _, _, m, _ in comp_results), default=0))
            ax_main.set_ylim(bottom=ymax * 1e-4)
        # Inset zoom: the low-energy end of the spectrum carries most of the cells
        # but is squashed into the leftmost slice of a 3-decade log axis, so the
        # variants are indistinguishable there without a magnified copy. Drawn in
        # the empty lower-left corner, with indicate_inset_zoom marking the source
        # region on the main panel.
        if is_spectrum and zoom_xlim is not None:
            axins = ax_main.inset_axes([0.07, 0.07, 0.42, 0.42])
            axins.stairs(ref_mean, edges, fill=True, color="#eeede9", edgecolor="#c9c8c2",
                          linewidth=0.8, zorder=1)
            for f_z, line_color, linestyle, mean, _err in comp_results:
                axins.stairs(mean, edges, color=line_color, linewidth=1.2, linestyle=linestyle, zorder=3)
            axins.set_xscale("log")
            axins.set_yscale("log")
            axins.set_xlim(*zoom_xlim)
            in_zoom = (centers >= zoom_xlim[0]) & (centers <= zoom_xlim[1])
            vals_in_zoom = np.concatenate(
                [v[in_zoom] for v in [ref_mean] + [m for _, _, _, m, _ in comp_results]]
            )
            vals_in_zoom = vals_in_zoom[np.isfinite(vals_in_zoom) & (vals_in_zoom > 0)]
            if vals_in_zoom.size:
                axins.set_ylim(vals_in_zoom.min() * 0.75, vals_in_zoom.max() * 1.3)
            axins.tick_params(labelsize=6, colors=INK_PRIMARY, length=0)
            # Over a narrow span (e.g. 1e-4..1e-3, a single decade) matplotlib
            # labels the log MINOR ticks too, so the inset x axis comes out as
            # overlapping "2x10^-4 4x10^-4 10^-4 10^-3" mush. Keep decade labels
            # only. The y labels go on the right because the inset sits in the
            # main panel's lower-left corner, where left-hand labels collide
            # with the main axis's own tick labels.
            axins.xaxis.set_minor_formatter(mticker.NullFormatter())
            axins.yaxis.set_minor_formatter(mticker.NullFormatter())
            axins.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
            axins.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
            axins.yaxis.tick_right()
            for side in ("top", "right"):
                axins.spines[side].set_visible(False)
            axins.grid(axis="y", color=GRIDLINE, linewidth=0.5, zorder=0)
            axins.set_axisbelow(True)
            ax_main.indicate_inset_zoom(axins, edgecolor=BASELINE, linewidth=0.9, alpha=0.9)

        if show_legend:
            # One legend for all three columns, above them, so it covers no data.
            handles, labels = ax_main.get_legend_handles_labels()
            fig.legend(handles, labels, frameon=False, fontsize=10, labelcolor=INK_PRIMARY,
                       loc="outside upper center", ncol=len(labels))
        if categorical:
            n_labels = min(len(uniques), 12)
            tick_pos = np.linspace(0, len(uniques) - 1, n_labels).round().astype(int)
            axes[-1].set_xticks(tick_pos, [f"{uniques[p]:.0f}" for p in tick_pos], rotation=45, ha="right", fontsize=7)

        for ax_ratio, (f, line_color, linestyle, mean, err) in zip(axes[1:], comp_results):
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(ref_mean > 0, mean / ref_mean, np.nan)
                rel_err = np.where(
                    ref_mean > 0,
                    ratio * np.sqrt((err / np.where(mean > 0, mean, np.nan)) ** 2 + (ref_err / ref_mean) ** 2),
                    np.nan,
                )
            ax_ratio.stairs(ratio, edges, color=line_color, linewidth=1.4, linestyle=linestyle, zorder=3)
            ax_ratio.fill_between(centers, ratio - rel_err, ratio + rel_err, step="mid",
                                   color=line_color, alpha=0.2, zorder=2, linewidth=0)
            ax_ratio.axhline(1.0, color=BASELINE, linewidth=0.8, zorder=1)
            ax_ratio.set_ylim(0.5, 1.5)
            # Explicit sparse ticks + smaller labels: matplotlib's default ~5
            # ticks over [0.5, 1.5] collide in a row this short.
            ax_ratio.set_yticks([0.75, 1.0, 1.25])
            ax_ratio.tick_params(labelsize=7)
            # Just "Ratio": the variant name on a second line crowds a row this
            # short. Each row's line colour matches its entry in the main panel's
            # legend, which is what identifies it.
            ax_ratio.set_ylabel("Ratio", fontsize=7)
            style_axes(ax_ratio, log_y=False)
            if is_spectrum:
                ax_ratio.set_xscale("log")

        axes[-1].set_xlabel(xlabel)

    def spectrum_edges():
        lo = max(min(f["e"][f["e"] > 0].min() for f in files), 1e-4)
        hi = max(f["e"].max() for f in files)
        # Display-only cap: `hi` otherwise tracks the single hottest cell across
        # all files, so one outlier sets the log axis for everybody. Cells above
        # the cap are not binned here; no other panel is affected.
        if args.spectrum_max_gev is not None:
            hi = min(hi, args.spectrum_max_gev)
        return np.logspace(np.log10(lo), np.log10(hi), 60)

    def profile_edges(key, hi_cap=None, step=None):
        lo = min(f[key].min() for f in files)
        hi = max(f[key].max() for f in files)
        if hi_cap is not None:
            hi = min(hi, hi_cap)
        if step is not None:
            # Bins one cell pitch thick, anchored at r=0, so each bin is a ring
            # of cells rather than an arbitrary slice: bin 0 is r < one cell,
            # i.e. the central cell itself, bin 1 the first ring around it, etc.
            #
            # This also fixes the radial panel's peak sitting in bin 1 instead of
            # bin 0. The panel plots total energy per annulus, and an annulus at
            # radius r holds ~2*pi*r*dr worth of cells, so with the old fixed 49
            # bins (2.04mm, well under the 5.088mm pitch) bin 0 spanned less than
            # one cell width and was starved: it held 103k cells against 290k in
            # bin 1, which outweighed its higher energy density and moved the
            # maximum outward. Energy per cell and per unit area always peaked in
            # bin 0. At one pitch per bin the first bin wins outright
            # (5451 vs 3511 GeV on the Geant4 reference).
            n = int(np.floor((hi - lo) / step))
            return lo + step * np.arange(n + 1)
        return np.linspace(lo, hi, 50)

    draw_column(subfigs[0], "Cell energy spectrum", "Cell energy  [GeV]", lambda f: f["e"], spectrum_edges, True,
                 zoom_xlim=tuple(args.spectrum_zoom_gev) if args.spectrum_zoom_gev else None)
    draw_column(subfigs[1], "Longitudinal profile", "y (depth)  [mm]", lambda f: f["y"], lambda: profile_edges("y"), False, categorical=True, show_legend=True)
    # Capped at 100mm, matching the paper's own display range (arXiv:2511.01460
    # figure 7/8) - beyond that, cells are sparse tail/backscatter with too few
    # showers contributing per bin to give a meaningful mean.
    draw_column(subfigs[2], "Radial profile", "Radial distance  [mm]", lambda f: f["radial"], lambda: profile_edges("radial", hi_cap=100, step=CELL_SIZE_MM), False)

    out_png = os.path.join(args.output_dir, f"shower_profiles_{args.geometry}.{args.format}")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_png}")


if __name__ == "__main__":
    main()
