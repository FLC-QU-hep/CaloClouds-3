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


def load_grid_file(h5_path, layer_y_mm):
    """Read a merged-cell point cloud (the output of project_ecal_showers.py:
    hits (n_showers, capacity, 4) = (x_local, z_local, layer_id, energy_MeV),
    zero-padded per shower up to n_points) and flatten the real (non-padding)
    rows into per-cell arrays (x, y, z, e, evt). Radial distance is measured
    from each shower's true axis at that hit's depth layer - axis_x_global/
    axis_z_global (n_showers, n_layers), written by postprocessing.py from
    the gun position (x=0, z=-50mm) and that shower's own entrance angle via
    get_alignment_shifts(). The axis drifts linearly through x/z with depth
    because of the angle, so a single per-shower centroid would average that
    drift away and smear the core into a ring instead of a peak at r=0."""
    print(f"Loading {h5_path} ...")
    with h5py.File(h5_path, "r") as f:
        hits = f["hits"][:]
        n_layers = f.attrs["n_layers"]
        y_centers = layer_y_mm[:n_layers]
        axis_x_global = f["axis_x_global"][:]  # (n_showers, n_layers)
        axis_z_global = f["axis_z_global"][:]  # (n_showers, n_layers)

    n_events = hits.shape[0]
    valid = hits[:, :, 3] > 0

    x = hits[:, :, 0][valid]
    z = hits[:, :, 1][valid]
    layer = hits[:, :, 2][valid].astype(np.int64)
    e = hits[:, :, 3][valid] / 1000.0  # DDML convention: stored energy is MeV, plots want GeV
    y = y_centers[layer]
    evt = np.nonzero(valid)[0]

    axis_x_per_hit = axis_x_global[evt, layer]
    axis_z_per_hit = axis_z_global[evt, layer]

    print(f"  {n_events} showers, {len(e):,} merged cells")
    return {
        "label": derive_label(h5_path),
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
    args = parser.parse_args()

    if len(args.input_h5) < 2:
        raise SystemExit("Need at least a reference file and one comparison file")

    os.makedirs(args.output_dir, exist_ok=True)
    layer_y_mm = Metadata().layer_bottom_pos_global
    files = [load_grid_file(p, layer_y_mm) for p in args.input_h5]
    ref, comps = files[0], files[1:]
    for f in files:
        print(f"{f['label']}: {f['n_events']} events, {len(f['e']):,} cells")

    # Chart chrome + fixed-order categorical palette (same convention as
    # DDML/scripts/plot_shower_profiles.py, see dataviz skill).
    SURFACE, INK_PRIMARY = "#fcfcfb", "#0b0b0b"
    GRIDLINE, BASELINE = "#e1e0d9", "#c3c2b7"
    CATEGORICAL = ["#2a78d6", "#1baf7a", "#cb8900", "#008300", "#4a3aa7", "#e34948", "#a1215a", "#0e7490"]

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
    fig.suptitle(f"Shower Profiles  (n={n_events_label} events/file)", fontsize=13, fontweight="bold", color=INK_PRIMARY)

    def style_axes(ax, log_y=True):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        if log_y:
            ax.set_yscale("log")

    def draw_column(subfig, title, xlabel, main_vals, ref_edges_fn, is_spectrum, categorical=False, show_legend=False):
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
        for f, color in zip(comps, CATEGORICAL[1:]):
            mean, err = compute(f)
            is_ref = f["label"].startswith("REF:")
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
        if show_legend:
            ax_main.legend(frameon=False, fontsize=8, labelcolor=INK_PRIMARY, loc="lower center")
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
            ax_ratio.set_ylabel(f"Ratio\n{f['label']}", fontsize=8)
            style_axes(ax_ratio, log_y=False)
            if is_spectrum:
                ax_ratio.set_xscale("log")

        axes[-1].set_xlabel(xlabel)

    def spectrum_edges():
        lo = max(min(f["e"][f["e"] > 0].min() for f in files), 1e-4)
        hi = max(f["e"].max() for f in files)
        return np.logspace(np.log10(lo), np.log10(hi), 60)

    def profile_edges(key, hi_cap=None):
        lo = min(f[key].min() for f in files)
        hi = max(f[key].max() for f in files)
        if hi_cap is not None:
            hi = min(hi, hi_cap)
        return np.linspace(lo, hi, 50)

    draw_column(subfigs[0], "Cell energy spectrum", "Cell energy  [GeV]", lambda f: f["e"], spectrum_edges, True)
    draw_column(subfigs[1], "Longitudinal profile", "y (depth)  [mm]", lambda f: f["y"], lambda: profile_edges("y"), False, categorical=True, show_legend=True)
    # Capped at 100mm, matching the paper's own display range (arXiv:2511.01460
    # figure 7/8) - beyond that, cells are sparse tail/backscatter with too few
    # showers contributing per bin to give a meaningful mean.
    draw_column(subfigs[2], "Radial profile", "Radial distance  [mm]", lambda f: f["radial"], lambda: profile_edges("radial", hi_cap=100), False)

    out_png = os.path.join(args.output_dir, f"shower_profiles_{args.geometry}.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_png}")


if __name__ == "__main__":
    main()
