"""
Figure-7-style scan comparison plot (arXiv:2601.11716) built from projected-grid
h5 files (the output of postprocessing.py / project_ecal_showers.py): one figure,
three columns (cell energy spectrum, sum energy vs radius, sum energy vs layers),
each with a main panel (log-y) on top and one
Geant4-vs-CC3 ratio panel per scan point stacked below it.

Unlike plot_distributions_from_grid.py / plot_shower_profiles_from_grid.py (one
shared reference file compared against several *variant* files), each scan point
here needs its *own* real (Geant4) reference: a single shared real file spanning
the full incident-energy/angle range would swamp any one fixed-energy/angle CC3
comparison in raw counts (35000 mixed-energy real showers vs. 5000 showers at one
energy), pushing every ratio to ~0 regardless of CC3's actual quality. Build the
matched real side with filter_real_by_conditioning.py + postprocessing.py first
(it filters the raw input_cc3_file_N.h5 down to real events near a target
energy/theta/phi).

Input files are given as one --group per scan point: the group's first path is
that scan point's real reference, every path after it is a GEN variant (e.g.
one per algorithm) compared against that same real reference - REFERENCE +
REFERENCE1..N, same convention as plot_distributions_from_grid.py, just grouped
per scan point instead of one flat list. Within a color, Geant4 is solid and CC3
dashed, matching the paper's style. Legend labels are auto-derived from each GEN
file's name (the scan suffix generate_showers.py's auto-naming appends, e.g.
"E10GeV", "theta20") prefixed by its algorithm-run directory name when it has
one, so multiple algorithms at the same scan point don't collide on one legend
entry.

There is one ratio row per scan point (--group), not per GEN file: every
algorithm's ratio curve for that scan point is overlaid on the same row, since
splitting them into separate rows made the figure unreadably tall once more
than a couple of algorithms were compared at once. Color is 2D: hue is fixed
per algorithm-slot (a GEN file's position within its --group, so the same
algorithm keeps the same hue family across every scan point), and shade
(lighter -> darker) varies by scan point in the order --group was given, so a
single algorithm's color across energies/thetas stays visually related while
still being distinguishable.

Usage:
    python plot_scan_comparison.py \\
        --group "$REAL_10GEV" "$GEN_10GEV_ALGO1" "$GEN_10GEV_ALGO2" \\
        --group "$REAL_50GEV" "$GEN_50GEV_ALGO1" "$GEN_50GEV_ALGO2" \\
        --group "$REAL_100GEV" "$GEN_100GEV_ALGO1" "$GEN_100GEV_ALGO2" \\
        --scan-type energy --geometry all_algorithms
    # default output dir: plots/ next to this script
    # output: plots/scan_comparison_<scan-type>_<geometry>.png

--scan-type {energy,theta} only changes the figure title/output filename to
say what's varying.
"""

from __future__ import annotations

import argparse
import colorsys
import os
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from postprocessing import CELL_SIZE_MM


def shade_hex(hex_color, lightness_factor):
    """Multiply hex_color's HLS lightness by lightness_factor (>1 lightens,
    <1 darkens), clamped to a valid lightness. Works for grays too (hue/
    saturation are preserved, so a zero-saturation color stays gray)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.0, min(1.0, l * lightness_factor))
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return "#{:02x}{:02x}{:02x}".format(round(r2 * 255), round(g2 * 255), round(b2 * 255))


# Short legend names for the algorithm-run directories under generated_showers/
# (see derive_label) - the full directory names are long and, now that color
# (hue=algorithm, shade=scan point) already carries most of the distinguishing
# information, the text label only needs to be enough to read at a glance.
# Falls back to the raw (long) name for any algorithm not listed here.
ALGO_SHORT_NAMES = {
    "merge_within_cell": "MergeCell",
    "merge_within_regular_subcell_6kcut": "SubCell6k",
    "merge_within_regular_subcell": "SubCell",
    "hdbscan_ms3_mcs10": "HDBSCANms3",
    "hdbscan_ms7_mcs7": "HDBSCANms7",
    "hdbscan_ms8_mcs40": "HDBSCANms8",
    "hdbscan_ms12_mcs12": "HDBSCANms12",
}

# Legend names as in the paper's other figures (plot_*_from_grid.py --labels),
# one entry per algorithm: the shade already encodes the scan point, so the
# legend lists each algorithm once instead of once per scan point.
PAPER_LABELS = {
    "merge_within_cell": "cell",
    "merge_within_regular_subcell": "subcell",
    "merge_within_regular_subcell_6kcut": "subcell, 6k cut",
    "hdbscan_ms3_mcs10": "HDBSCAN (3, 10)",
    "hdbscan_ms3_mcs3": "HDBSCAN (3, 3)",
    "hdbscan_ms7_mcs7": "HDBSCAN (7, 7)",
    "hdbscan_ms8_mcs40": "HDBSCAN (8, 40)",
    "hdbscan_ms12_mcs12": "HDBSCAN (12, 12)",
}


def pretty_scan_label(label):
    """'10GeV' -> '10 GeV', 'theta20' -> 'θ = 20°'; anything else unchanged."""
    m = re.fullmatch(r"E?(\d+)GeV", label)
    if m:
        return f"{m.group(1)} GeV"
    m = re.fullmatch(r"theta(\d+)", label)
    if m:
        return rf"$\theta = {m.group(1)}^\circ$"
    return label


def derive_label(h5_path):
    """Scan-point label for the legend, e.g. 'E10GeV', 'theta20' - the suffix
    generate_showers.py's auto-naming appends to --output_file when
    --fixed_energy_gev/--fixed_theta_deg is set (see generate_showers.py's
    suffix_parts). Falls back to 'baseline' for a file with no such suffix
    (e.g. the shared REFERENCE itself, or an unfixed generated_showers_poly.h5).

    If h5_path sits under a generated_showers/<algo>_<timestamp>/... run
    directory, a short algorithm name (ALGO_SHORT_NAMES) is prefixed onto the
    label (e.g. 'MergeCell E10GeV') so multiple algorithms' GEN files at the
    same scan point - all sharing one real reference - get distinguishable
    legend entries instead of colliding on the bare scan-point label."""
    base = os.path.basename(h5_path)
    for ext in ("_postprocessed.h5", "_grid.h5", ".h5"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    base = re.sub(r"^generated_showers(_poly)?_?", "", base)
    base = re.sub(r"^real_filtered_", "", base)
    base = re.sub(r"^input_cc3_file_\d+_?", "", base)
    scan_label = base if base else "baseline"

    grandparent = os.path.basename(os.path.dirname(os.path.dirname(h5_path)))
    m = re.match(r"^(.+)_\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}$", grandparent)
    if not m:
        return scan_label
    algo = ALGO_SHORT_NAMES.get(m.group(1), m.group(1))
    return f"{algo} {scan_label}"


def algo_key(h5_path):
    """Variant directory name (merge_within_cell, hdbscan_ms3_mcs10, ...) for a
    path under generated_showers/<variant>_<timestamp>/..., or "" for anything
    else (the Geant4 reference, the gun-edge runs). Used to pick a colour per
    ALGORITHM rather than per file position, so one variant keeps its colour no
    matter where it sits in a given figure's file list."""
    grandparent = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(h5_path))))
    m = re.match(r"^(.+)_\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}$", grandparent)
    return m.group(1) if m else ""

def load_grid_file(h5_path, min_cell_energy_mev=None):
    """Read a merged-cell point cloud (see postprocessing.py) and flatten the
    real (non-padding) rows into per-cell arrays. Energy is kept in MeV (the
    DDML/grid.h5 storage convention) since that's what the paper's own Figure 7
    plots in. Radial distance is measured from each shower's true axis at that
    hit's depth layer (axis_x_global/axis_z_global, written by postprocessing.py
    from the gun position and that shower's own entrance angle) - a single
    per-shower centroid would average the angle-driven drift away and smear the
    core into a ring instead of a peak at r=0."""
    print(f"Loading {h5_path} ...")
    with h5py.File(h5_path, "r") as f:
        hits = f["hits"][:]
        n_layers = int(f.attrs["n_layers"])
        axis_x_global = f["axis_x_global"][:]  # (n_showers, n_layers)
        axis_z_global = f["axis_z_global"][:]  # (n_showers, n_layers)

    n_events = hits.shape[0]
    valid = hits[:, :, 3] > 0
    if min_cell_energy_mev is not None:
        valid &= hits[:, :, 3] > min_cell_energy_mev

    x = hits[:, :, 0][valid]
    z = hits[:, :, 1][valid]
    layer = hits[:, :, 2][valid].astype(np.int64)
    e_mev = hits[:, :, 3][valid]
    evt = np.nonzero(valid)[0]

    axis_x_per_hit = axis_x_global[evt, layer]
    axis_z_per_hit = axis_z_global[evt, layer]
    radial = np.sqrt((x - axis_x_per_hit) ** 2 + (z - axis_z_per_hit) ** 2)
    hits_per_event = np.bincount(evt, minlength=n_events).astype(float)

    print(f"  {n_events} showers, {len(e_mev):,} merged cells")
    return {
        "n_events": n_events,
        "n_layers": n_layers,
        "layer": layer,
        "e_mev": e_mev,
        "radial": radial,
        "hits_per_event": hits_per_event,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--group", nargs="+", action="append", required=True, metavar="H5",
        help="One scan point: a Geant4 (real) *_grid.h5 file matched to that scan point's "
             "conditioning (see filter_real_by_conditioning.py), followed by one or more "
             "CaloClouds-3 (generated) *_grid.h5 variant files (e.g. one per algorithm) "
             "compared against that same real reference. Repeat --group once per scan point, "
             "e.g. --group REAL_10GeV GEN_10GeV_ALGO1 GEN_10GeV_ALGO2 --group REAL_50GeV ...",
    )
    parser.add_argument(
        "--scan-type",
        choices=["energy", "theta"],
        default="energy",
        help="What varies across the comparison files. Only affects the figure title/output "
        "filename - the underlying computation is identical regardless.",
    )
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots"),
                         help="Directory to write scan_comparison_<scan-type>_<geometry>.png into "
                              "(default: plots/ next to this script)")
    parser.add_argument("--geometry", default="regular", help="Names the output scan_comparison_<scan-type>_<geometry>.png")
    parser.add_argument("--min-cell-energy-mev", type=float, default=None,
                         help="Drop merged cells with energy <= this threshold (MeV) before computing "
                              "any quantity - a post-projection readout-threshold cut applied on top of "
                              "each file's own merge step. Default: no additional cut.")
    parser.add_argument("--radius-max-mm", type=float, default=100.0,
                         help="Display cap for the radial-profile x-axis [mm] (matching the paper's own "
                              "display range) - beyond this, cells are sparse tail/backscatter.")
    parser.add_argument("--spectrum-range-mev", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                         help="Fixed cell-energy-spectrum bin range [MeV]. Default: the full range across "
                              "all files, so a single hot cell sets the axis. Pass the energy scan's printed "
                              "range to the theta scan to give both figures the same axis.")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"],
                         help="Output image format (default: pdf - vector, so it scales in talks and "
                              "papers without resampling). dpi still applies to any rasterised element.")
    args = parser.parse_args()

    for group in args.group:
        if len(group) < 2:
            raise SystemExit(f"--group needs a real reference plus at least one GEN variant, got: {group}")

    os.makedirs(args.output_dir, exist_ok=True)

    # groups[i] = {"label": scan-point label (from the real reference's own
    # filename, e.g. "10GeV"), "real": grid dict, "gens": [(legend label, grid
    # dict, algo_idx), ...]}. algo_idx is the GEN file's position within its
    # --group, used only for color (see algo_color below) - it assumes every
    # --group lists its algorithm variants in the same order, which is how
    # run.sh builds them (from one shared ALGO_DIRS list).
    groups = []
    for group in args.group:
        real_path, gen_paths = group[0], group[1:]
        real = load_grid_file(real_path, min_cell_energy_mev=args.min_cell_energy_mev)
        gens = [
            (derive_label(gen_path), load_grid_file(gen_path, min_cell_energy_mev=args.min_cell_energy_mev),
              algo_idx, algo_key(gen_path))
            for algo_idx, gen_path in enumerate(gen_paths)
        ]
        groups.append({"label": derive_label(real_path), "real": real, "gens": gens})

    n_groups = len(groups)
    n_layers = groups[0]["real"]["n_layers"]
    all_grid_files = [g["real"] for g in groups] + [gen for g in groups for _, gen, _, _ in g["gens"]]

    # Chart chrome (same convention as the other projection/plot_*_from_grid.py
    # scripts, see dataviz skill). Color is 2D here, not a flat categorical list:
    # ALGO_HUES gives each algorithm-slot a fixed hue, shaded lighter->darker
    # across scan points (see shade_hex/algo_color) so one algorithm's color
    # family stays recognizable across every energy/theta panel; Geant4 is
    # always gray, shaded the same way per scan point, to keep "truth" visually
    # distinct from any CC3 variant regardless of algorithm/energy.
    SURFACE, INK_PRIMARY = "#fcfcfb", "#0b0b0b"
    GRIDLINE, BASELINE = "#e1e0d9", "#c3c2b7"
    # Okabe-Ito colourblind-safe hues. ALGO_COLORS pins the variants that also
    # appear in the other figures in this directory to the same hue there, so
    # SubCell is amber and HDBSCANms3 green in the scans exactly as they are in
    # the gun-edge and comparison plots; ALGO_HUES is the positional fallback
    # for anything ALGO_COLORS doesn't name.
    ALGO_COLORS = {
        "merge_within_cell": "#0072b2",                   # blue
        "merge_within_regular_subcell": "#e69f00",        # amber      (gun-edge orange)
        "merge_within_regular_subcell_6kcut": "#d55e00",  # vermillion
        "hdbscan_ms3_mcs10": "#009e73",                   # green      (gun-edge green)
        "hdbscan_ms7_mcs7": "#56b4e9",                    # sky blue
        "hdbscan_ms8_mcs40": "#cc79a7",                   # reddish purple
        "hdbscan_ms12_mcs12": "#7a21dd",                  # violet
    }
    ALGO_HUES = ["#0072b2", "#e69f00", "#009e73", "#cc79a7", "#d55e00", "#56b4e9", "#7a21dd", "#7b3f00"]
    GRAY_BASE = "#8a8880"

    def shade_factor(group_idx):
        return 1.0 if n_groups <= 1 else np.interp(group_idx, [0, n_groups - 1], [1.4, 0.65])

    def algo_color(akey, algo_idx, group_idx):
        base = ALGO_COLORS.get(akey) or ALGO_HUES[algo_idx % len(ALGO_HUES)]
        return shade_hex(base, shade_factor(group_idx))

    def gray_color(group_idx):
        return shade_hex(GRAY_BASE, shade_factor(group_idx))

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_PRIMARY, "xtick.color": INK_PRIMARY, "ytick.color": INK_PRIMARY,
        "text.color": INK_PRIMARY, "font.family": "sans-serif", "font.size": 10, "axes.grid": False,
    })

    def style_axes(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)

    def draw_panel(ax_main, ratio_axes_col, title, xlabel, ylabel, edges, compute, log_x=False, log_y=True,
                    categorical_uniques=None, show_legend=False):
        """ratio_axes_col: one ratio axis per scan point (groups[i] <-> ratio_axes_col[i]) -
        every algorithm's ratio curve for that scan point is overlaid on the same row."""
        for gi, g in enumerate(groups):
            ax_ratio = ratio_axes_col[gi]
            # Per-shower average (raw totals / n_events), not a raw sum - real and
            # gen files can have very different shower counts (a real matched
            # subset from filter_real_by_conditioning.py is typically much
            # smaller than a fixed 5000/50000-shower CC3 generation run), so an
            # unnormalized ratio would just measure that count mismatch instead
            # of shape/physics agreement.
            real_vals = compute(g["real"]) / g["real"]["n_events"]
            gray = gray_color(gi)
            ax_main.stairs(real_vals, edges, color=gray, linewidth=1.8, linestyle="-",
                            label=f"Geant4 {g['label']}", zorder=2)
            for label, gen, algo_idx, akey in g["gens"]:
                color = algo_color(akey, algo_idx, gi)
                gen_vals = compute(gen) / gen["n_events"]
                ax_main.stairs(gen_vals, edges, color=color, linewidth=1.8, linestyle="--", label=label, zorder=3)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(real_vals > 0, gen_vals / real_vals, np.nan)
                ax_ratio.stairs(ratio, edges, color=color, linewidth=1.4, zorder=3)
            ax_ratio.axhline(1.0, color=BASELINE, linewidth=0.8, zorder=1)
            ax_ratio.set_ylim(0.5, 1.5)
            ax_ratio.set_ylabel(f"Ratio\n{pretty_scan_label(g['label'])}", fontsize=8)
            style_axes(ax_ratio)
            # tick_params, not plt.setp on the current tick labels: set_xscale
            # below rebuilds the ticks, which silently dropped setp's visibility
            # and left the bottom row without x tick labels.
            ax_ratio.tick_params(labelbottom=False)
            if log_x:
                ax_ratio.set_xscale("log")

        ax_main.set_title(title, color=INK_PRIMARY, fontsize=10)
        ax_main.set_ylabel(ylabel)
        if log_y:
            ax_main.set_yscale("log")
        style_axes(ax_main)
        ax_main.tick_params(labelbottom=False)
        if log_x:
            ax_main.set_xscale("log")

        last_ratio = ratio_axes_col[-1]
        last_ratio.tick_params(labelbottom=True)
        if categorical_uniques is not None:
            n_labels = min(len(categorical_uniques), 12)
            tick_pos = np.linspace(0, len(categorical_uniques) - 1, n_labels).round().astype(int)
            last_ratio.set_xticks(tick_pos, [f"{categorical_uniques[p]:.0f}" for p in tick_pos],
                                   rotation=45, ha="right", fontsize=7)
        last_ratio.set_xlabel(xlabel)
        if show_legend:
            # One legend row above all panels, as in the other paper figures:
            # Geant4 once, then each algorithm once, all in the unshaded colour;
            # the light-to-dark shade of every curve encodes the scan point.
            handles = [Line2D([], [], color=GRAY_BASE, linewidth=1.8, label="Geant4")]
            for _, _, algo_idx, akey in groups[0]["gens"]:
                base = ALGO_COLORS.get(akey) or ALGO_HUES[algo_idx % len(ALGO_HUES)]
                handles.append(Line2D([], [], color=base, linewidth=1.8, linestyle="--",
                                      label=PAPER_LABELS.get(akey, akey)))
            fig.legend(handles=handles, frameon=False, fontsize=10, labelcolor=INK_PRIMARY,
                       loc="outside upper center", ncol=len(handles))

    fig = plt.figure(figsize=(16, 4 + 1.8 * n_groups), constrained_layout=True)
    height_ratios = [3] + [1] * n_groups
    gs = fig.add_gridspec(1 + n_groups, 3, height_ratios=height_ratios, hspace=0.08)
    main_axes = [fig.add_subplot(gs[0, c]) for c in range(3)]
    ratio_axes = [[fig.add_subplot(gs[1 + i, c], sharex=main_axes[c]) for i in range(n_groups)] for c in range(3)]

    # --- panel 1: cell energy spectrum ---
    all_e = np.concatenate([f["e_mev"] for f in all_grid_files])
    if args.spectrum_range_mev:
        lo, hi = args.spectrum_range_mev
    else:
        lo = max(all_e[all_e > 0].min(), 1e-3)
        hi = all_e.max()
    print(f"Cell energy spectrum range: {lo:.6g} - {hi:.6g} MeV")
    spectrum_edges = np.logspace(np.log10(lo), np.log10(hi), 60)
    draw_panel(
        main_axes[0], ratio_axes[0], "Cell energy spectrum", "Cell energy  [MeV]", "Cells / shower",
        spectrum_edges, lambda f: np.histogram(f["e_mev"], bins=spectrum_edges)[0].astype(float), log_x=True,
    )

    # --- panel 2: sum energy vs radius ---
    # One cell pitch per bin, anchored at r=0, as in the shower-profile figure's
    # radial panel (plot_shower_profiles_from_grid.py): each bin is a ring of
    # cells rather than an arbitrary slice.
    radius_edges = CELL_SIZE_MM * np.arange(int(np.floor(args.radius_max_mm / CELL_SIZE_MM)) + 1)
    draw_panel(
        main_axes[1], ratio_axes[1], "Sum energy vs radius", "Radius  [mm]", "Energy / shower  [MeV]",
        radius_edges, lambda f: np.histogram(f["radial"], bins=radius_edges, weights=f["e_mev"])[0],
        show_legend=True,
    )

    # --- panel 3: sum energy vs layers ---
    layer_uniques = np.arange(n_layers)
    layer_edges = np.arange(n_layers + 1) - 0.5
    draw_panel(
        main_axes[2], ratio_axes[2], "Sum energy vs layers", "Layer", "Energy / shower  [MeV]",
        layer_edges, lambda f: np.bincount(f["layer"], weights=f["e_mev"], minlength=n_layers)[:n_layers],
        categorical_uniques=layer_uniques,
    )

    out_png = os.path.join(args.output_dir, f"scan_comparison_{args.scan_type}_{args.geometry}.{args.format}")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_png}")


if __name__ == "__main__":
    main()
