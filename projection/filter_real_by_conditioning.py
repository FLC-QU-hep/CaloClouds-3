"""
Filter one or more raw input_cc3-format h5 files (energy/events/n_points/
p_norm_local/p_norm_global/phi_local/phi_global/theta_local/theta_global - the
output of step2point's convert_to_cc3_format.py, *before* postprocessing.py's
grid projection drops the per-event conditioning columns) down to the real
(Geant4) events matching a fixed energy and/or theta and/or phi window.

This exists so a real (Geant4) reference can be matched to a CaloClouds-3
fixed-conditioning scan point (see generate_showers.py's --fixed_energy_gev/
--fixed_theta_deg/--fixed_phi_deg and projection/plot_scan_comparison.py):
comparing a CC3 sample generated at a fixed 10 GeV against the *full*
all-energies-mixed real distribution isn't a fair ratio, so this script
produces a real subset restricted to the same conditioning window instead.
Multiple input files can be given to gather enough matching statistics from a
narrow window; matches are concatenated across all of them, in order, capped
at --n-showers if given.

The output keeps the same schema as the input (all datasets, same dtypes),
just with events selected - so it's a drop-in input to postprocessing.py.

Usage:
    python filter_real_by_conditioning.py <input1.h5> [<input2.h5> ...] \\
        --energy-range-gev 9 11 \\
        --label 10GeV \\
        [--output-dir DIR]
    # default output dir: filtered/ next to this script
    # output: filtered/real_filtered_<label>.h5

    # combine windows (all given ones apply together, AND'ed):
    python filter_real_by_conditioning.py input_cc3_file_0.h5 \\
        --theta-range-deg -2 2 --phi-range-deg -2 2 --label theta0
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_h5", nargs="+", help="Path(s) to one or more raw input_cc3_file_N.h5 files "
                         "(pre-postprocessing.py). Matches are concatenated across all of them, in order.")
    parser.add_argument("--energy-range-gev", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                         help="Keep only events with energy in [LO, HI] GeV. Default: no energy cut.")
    parser.add_argument("--theta-range-deg", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                         help="Keep only events with theta_local in [LO, HI] degrees. Default: no theta cut.")
    parser.add_argument("--phi-range-deg", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                         help="Keep only events with phi_local in [LO, HI] degrees. Default: no phi cut.")
    parser.add_argument("--label", required=True, help="Names the output real_filtered_<label>.h5 "
                         "(e.g. '10GeV', 'theta0') - should match the label used for the corresponding "
                         "CC3 --fixed_energy_gev/--fixed_theta_deg/--fixed_phi_deg generation run.")
    parser.add_argument("--n-showers", type=int, default=None,
                         help="Cap the number of matching events kept (after concatenating across all "
                         "input files, in order). Default: keep every match.")
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "filtered"),
                         help="Directory to write real_filtered_<label>.h5 into (default: filtered/ next to this script)")
    args = parser.parse_args()

    if args.energy_range_gev is None and args.theta_range_deg is None and args.phi_range_deg is None:
        parser.error("Give at least one of --energy-range-gev / --theta-range-deg / --phi-range-deg.")

    os.makedirs(args.output_dir, exist_ok=True)

    per_file_data = []
    n_total_matched = 0
    for h5_path in args.input_h5:
        print(f"Loading {h5_path} ...")
        with h5py.File(h5_path, "r") as f:
            keys = list(f.keys())
            energy = f["energy"][:, 0] if "energy" in keys else None
            theta_local = f["theta_local"][:] if "theta_local" in keys else None
            phi_local = f["phi_local"][:] if "phi_local" in keys else None
            n_events = f[keys[0]].shape[0]

            mask = np.ones(n_events, dtype=bool)
            if args.energy_range_gev is not None:
                if energy is None:
                    parser.error(f"{h5_path} has no 'energy' dataset, can't apply --energy-range-gev.")
                lo, hi = args.energy_range_gev
                mask &= (energy >= lo) & (energy <= hi)
            if args.theta_range_deg is not None:
                if theta_local is None:
                    parser.error(f"{h5_path} has no 'theta_local' dataset, can't apply --theta-range-deg.")
                lo, hi = args.theta_range_deg
                mask &= (theta_local >= lo) & (theta_local <= hi)
            if args.phi_range_deg is not None:
                if phi_local is None:
                    parser.error(f"{h5_path} has no 'phi_local' dataset, can't apply --phi-range-deg.")
                lo, hi = args.phi_range_deg
                mask &= (phi_local >= lo) & (phi_local <= hi)

            idx = np.nonzero(mask)[0]
            if args.n_showers is not None:
                remaining = args.n_showers - n_total_matched
                idx = idx[:max(remaining, 0)]
            print(f"  {n_events} events, {mask.sum()} match ({len(idx)} kept)")
            n_total_matched += len(idx)

            per_file_data.append({k: f[k][:][idx] for k in keys})

    if n_total_matched == 0:
        raise SystemExit("No events matched the given range(s) across all input files.")

    keys = per_file_data[0].keys()
    merged = {k: np.concatenate([d[k] for d in per_file_data], axis=0) for k in keys}

    out_path = os.path.join(args.output_dir, f"real_filtered_{args.label}.h5")
    with h5py.File(out_path, "w") as hf:
        for k, v in merged.items():
            hf.create_dataset(k, data=v)

    print(f"\nTotal matched: {n_total_matched} events")
    print(f"Saved: {out_path}")
    print(f"\nNext step: python postprocessing.py {out_path}")


if __name__ == "__main__":
    main()
