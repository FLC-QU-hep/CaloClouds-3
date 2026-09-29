"""
Run plot_distributions_from_grid.py, plot_shower_profiles_from_grid.py, and
plot_cog_from_grid.py in one command, on the same REFERENCE/REFERENCE1..N file
list (see each script's own docstring for what it plots and how). This is a
thin driver - it imports each script's module and calls its own main() with a
constructed argv, so all three keep behaving exactly as they do standalone
(same output filenames: distribution_<geometry>.png, shower_profiles_<geometry>.png,
cog_<geometry>.png).

The first two take the *_grid.h5 (postprocessed) files directly. CoG needs the
raw (pre-postprocessing) DDML/cc3input-format file instead, so its input path
per file is derived by stripping "_postprocessed" from each given path -
matching what run.sh used to do by hand with bash's ${VAR/_postprocessed/}
substitution.

Usage:
    python plot_all_from_grid.py <input1_postprocessed.h5> [<input2_postprocessed.h5> ...] \\
        --geometry {real,regular} [--n-showers N] [--min-cell-energy-mev MEV] \\
        [--xz-range-mm LO HI] [--x-zoom-mm LO HI] [--z-zoom-mm LO HI] [--output-dir DIR]
    # default output dir: plots/ next to this script
"""

from __future__ import annotations

import argparse
import os
import sys

import plot_cog_from_grid
import plot_distributions_from_grid
import plot_shower_profiles_from_grid


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_h5", nargs="+", help="Postprocessed *_grid.h5 files, reference first")
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots"),
                         help="Directory all three plots are written into (default: plots/ next to this script)")
    parser.add_argument("--geometry", default="regular", help="Names each output PNG's <geometry> suffix")
    parser.add_argument("--n-showers", type=int, default=None,
                         help="Cap every input file to its first N showers, in all three plots. The "
                              "distributions and cell energy spectrum are raw counts, so a reference file "
                              "holding more showers than the generated files sits above them by exactly "
                              "that ratio - pass the generated files' N to compare equal amounts of "
                              "simulated data.")
    parser.add_argument("--spectrum-zoom-gev", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                         help="Shower profiles only: inset zoom box on the cell energy spectrum panel "
                              "covering this energy range [GeV], e.g. --spectrum-zoom-gev 1e-4 5e-3")
    parser.add_argument("--labels", nargs="+", default=None, metavar="LABEL",
                         help="Override the legend labels in all three plots, one per input file "
                              "in the same order (reference first).")
    parser.add_argument("--spectrum-max-gev", type=float, default=None, metavar="HI",
                         help="Distributions + shower profiles: cap the cell energy spectrum's upper bin "
                              "edge at this energy [GeV]. Display-only - cells above it are still counted "
                              "in every other quantity, they are simply not drawn. Without this the "
                              "spectrum's top edge is set by the single hottest cell in any input file, "
                              "so one outlier stretches the whole log axis.")
    parser.add_argument("--energy-per-shower-max-gev", type=float, default=None, metavar="HI",
                         help="Distributions only: cap the total-energy-per-shower panel's upper bin edge "
                              "at this energy [GeV]. Display-only, same reasoning as --spectrum-max-gev.")
    parser.add_argument("--min-cell-energy-mev", type=float, default=None,
                         help="Distributions + shower profiles only: drop merged cells at or below this "
                              "energy [MeV] before computing any quantity")
    parser.add_argument("--xz-range-mm", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                         help="Distributions only: restrict every quantity to cells inside this x/z box")
    parser.add_argument("--x-zoom-mm", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                         help="Distributions only: display-only zoom for the x panel")
    parser.add_argument("--z-zoom-mm", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                         help="Distributions only: display-only zoom for the z panel")
    parser.add_argument("--colors", nargs="+", default=None, metavar="HEX",
                         help="Override the categorical palette in all three plots, one colour per "
                              "input file in the given order (the first file is the filled grey "
                              "reference, so its entry is unused).")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"],
                         help="Output image format for all three plots (default: pdf).")
    args = parser.parse_args()

    common = ["--output-dir", args.output_dir, "--geometry", args.geometry, "--format", args.format]
    if args.colors:
        common += ["--colors", *args.colors]
    # NOTE: `common` reaches the distributions and shower-profiles argv only -
    # cog_argv below is built by hand and does NOT splat it, so anything added
    # here must be repeated there. Each script keeps its own reference detection
    # independent of the displayed text, so renaming is purely cosmetic.
    if args.labels:
        common += ["--labels", *args.labels]
    if args.min_cell_energy_mev is not None:
        common += ["--min-cell-energy-mev", str(args.min_cell_energy_mev)]
    # --n-showers used to reach CoG only, which left the count-based distributions
    # and cell energy spectrum comparing a 20k-shower reference against 10k-shower
    # generated files - a flat factor-2 offset in every ratio panel.
    if args.n_showers is not None:
        common += ["--n-showers", str(args.n_showers)]

    dist_argv = ["plot_distributions_from_grid.py", *args.input_h5, *common]
    if args.xz_range_mm:
        dist_argv += ["--xz-range-mm", str(args.xz_range_mm[0]), str(args.xz_range_mm[1])]
    if args.x_zoom_mm:
        dist_argv += ["--x-zoom-mm", str(args.x_zoom_mm[0]), str(args.x_zoom_mm[1])]
    if args.z_zoom_mm:
        dist_argv += ["--z-zoom-mm", str(args.z_zoom_mm[0]), str(args.z_zoom_mm[1])]
    if args.spectrum_max_gev is not None:
        dist_argv += ["--spectrum-max-gev", str(args.spectrum_max_gev)]
    if args.energy_per_shower_max_gev is not None:
        dist_argv += ["--energy-per-shower-max-gev", str(args.energy_per_shower_max_gev)]

    profiles_argv = ["plot_shower_profiles_from_grid.py", *args.input_h5, *common]
    if args.spectrum_zoom_gev:
        profiles_argv += ["--spectrum-zoom-gev", str(args.spectrum_zoom_gev[0]), str(args.spectrum_zoom_gev[1])]
    # Cap both spectrum panels together so the two figures stay on the same axis.
    if args.spectrum_max_gev is not None:
        profiles_argv += ["--spectrum-max-gev", str(args.spectrum_max_gev)]

    # CoG wants the raw (pre-postprocessing) file, normally the same path with
    # "_postprocessed" stripped. Some inputs have no raw counterpart though: the
    # gun-edge-test (gridcenter/gridedge) output only ever exists post-projection,
    # since the half-cell grid shift it tests IS the projection step. Fall back to
    # the given path for those - plot_cog_from_grid.load_raw_file detects the
    # hits/n_points grid schema and computes CoG straight off the stored cells.
    raw_paths = []
    for p in args.input_h5:
        raw = p.replace("_postprocessed", "")
        raw_paths.append(raw if os.path.exists(raw) else p)
    cog_argv = ["plot_cog_from_grid.py", *raw_paths, "--output-dir", args.output_dir,
                 "--geometry", args.geometry, "--format", args.format]
    if args.colors:
        cog_argv += ["--colors", *args.colors]
    if args.n_showers is not None:
        cog_argv += ["--n-showers", str(args.n_showers)]
    if args.labels:
        cog_argv += ["--labels", *args.labels]

    for label, module, argv in [
        ("distributions", plot_distributions_from_grid, dist_argv),
        ("shower profiles", plot_shower_profiles_from_grid, profiles_argv),
        ("center of gravity", plot_cog_from_grid, cog_argv),
    ]:
        print(f"\n=== {label} ===")
        old_argv = sys.argv
        sys.argv = argv
        try:
            module.main()
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    main()
