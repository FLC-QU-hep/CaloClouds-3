#!/usr/bin/env python3
"""
Compare CaloClouds-3 generated showers against real showers drawn from the
input_cc3 files for the same dataset variant.

Accepts generated showers in either format:
  - .npz, as written by inferepy.ipynb's cell-save: arrays "showers" (col 2 =
    global z in mm) and "cond". Hit energy already in MeV.
  - .h5, as written by scripts/generate_showers.py: dataset "events" in the
    same layout as input_cc3_file_*.h5 (col 2 = layer_idx + fuzz, col 3 = hit
    energy in GeV).

Both are normalized here to the same internal convention (layer index,
energy in MeV) before comparison.

Usage:
    python compare_generated_vs_input_cc3.py /path/to/generated_showers.{npz,h5} \
        --data-used withincell [--n-showers 5000] [--out-dir ./cc3_vs_real_plots]
    example: python scripts/plotting/compare_generated_vs_input_cc3.py generated_showers/CD_S2P_HDBScan_2026_06_17__13_46_20/generated_showers_10000/generated_showers.h5 --data-used hdbscan_ms8_mcs40
"""

from __future__ import annotations

import argparse
import importlib
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt

from pointcloud.utils.metadata import Metadata
from pointcloud.utils.detector_map import floors_ceilings
from pointcloud.data.conditioning import read_raw_regaxes_withcond

CONFIG_MODULES = {
    "withincell": "pointcloud.config_varients.caloclouds_3_S2P_withincell",
    "hdbscan_ms8_mcs40": "pointcloud.config_varients.caloclouds_3_S2P_hdbscan_ms8_mcs40",
    "hdbscan_ms3_mcs10": "pointcloud.config_varients.caloclouds_3_S2P_hdbscan_ms3_mcs10",
    "hdbscan_ms12_mcs12": "pointcloud.config_varients.caloclouds_3_S2P_hdbscan_ms12_mcs12",
    "hdbscan_ms40_mcs40": "pointcloud.config_varients.caloclouds_3_S2P_hdbscan_ms40_mcs40",
    "steps": "pointcloud.config_varients.caloclouds_3_S2P_steps",
    "subcell": "pointcloud.config_varients.caloclouds_3_S2P_subcell",
    "subcell_6kcut": "pointcloud.config_varients.caloclouds_3_S2P_subcell_6kcut",
}


def load_config(data_used: str):
    module = importlib.import_module(CONFIG_MODULES[data_used])
    return module.Configs()


def load_generated(path: str, meta):
    """
    Returns events (N, max_hits, 4) normalized to (x, y, layer_idx, energy_MeV),
    and incident energy (N,) in GeV.
    """
    if path.endswith(".npz"):
        data = np.load(path)
        events = data["showers"].copy()
        inc_energy = np.asarray(data["cond"])[:, 0].astype(np.float32)
        layer_mid_global = meta.layer_bottom_pos_global + meta.cell_thickness_global / 2
        events[:, :, 2] = np.abs(events[:, :, 2][:, :, None] - layer_mid_global).argmin(axis=2)
        # energy already in MeV
    else:
        with h5py.File(path, "r") as f:
            events = np.asarray(f["events"], dtype=np.float32)
            inc_energy = np.asarray(f["energy"], dtype=np.float32).reshape(-1)
            # generate_showers.py writes hit energy as GeV by default, or MeV when the
            # output is meant for DDML's LoadHdf5 loader (see --energy_units there).
            # Older files predate this attr and are always GeV.
            hit_energy_units = f.attrs.get("hit_energy_units", "GeV")
            if "layer_counts" in f:
                # new layer-counts-prefix format (generate_showers.py):
                # events[:, :n_layers, :] are per-layer hit counts, not real hits.
                n_layers = f["layer_counts"].shape[1]
                events = events[:, n_layers:, :]
        events[:, :, 2] = np.floor(events[:, :, 2])
        if hit_energy_units == "GeV":
            events[:, :, 3] *= 1000  # GeV -> MeV, same as real_events
        # if already MeV, no conversion needed
    return events, inc_energy


def load_real(config, n_showers: int):
    cond, events = read_raw_regaxes_withcond(config, total_size=n_showers, for_model="diffusion")
    events = np.array(events)
    cond = np.array(cond)
    return events, cond


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generated", help="Path to generated_showers.npz")
    parser.add_argument("--data-used", choices=list(CONFIG_MODULES), default="withincell")
    parser.add_argument(
        "--n-showers", type=int, default=None,
        help="Number of real showers to load (default: match the generated count)",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Where to save plots (default: same folder as the generated showers file)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.generated))
    args.out_dir = out_dir
    os.makedirs(args.out_dir, exist_ok=True)

    config = load_config(args.data_used)
    meta = Metadata(config)

    print(f"Loading generated showers from {args.generated} ...")
    fake_showers, gen_inc_energy = load_generated(args.generated, meta)
    print(f"  shape: {fake_showers.shape}")

    n_showers = args.n_showers or fake_showers.shape[0]
    print(f"Loading {n_showers} real showers from input_cc3 files ({args.data_used}) ...")
    real_events, real_cond = load_real(config, n_showers)
    real_events = real_events.copy()
    real_events[:, :, 3] *= 1000  # GeV -> MeV
    real_inc_energy = real_cond[:, 0].astype(np.float32)
    print(f"  shape: {real_events.shape}")

    gen_mask = fake_showers[:, :, 3] > 0
    real_mask = real_events[:, :, 3] > 0

    # both datasets now have a layer index (not fuzzed) in column 2
    n_layers = len(meta.layer_bottom_pos_global)
    gen_z_layer = fake_showers[:, :, 2].astype(int)
    real_z_layer = np.floor(real_events[:, :, 2]).astype(int).clip(0, n_layers - 1)

    # ── Figure 1: mean hit energy vs x / y / z, plus the energy spectrum ────
    def mean_energy_profile(coord, energy, edges):
        centers = (edges[:-1] + edges[1:]) / 2
        idx = np.clip(np.searchsorted(edges, coord, side="right") - 1, 0, len(centers) - 1)
        sums = np.bincount(idx, weights=energy, minlength=len(centers))[: len(centers)]
        counts = np.bincount(idx, minlength=len(centers))[: len(centers)]
        with np.errstate(invalid="ignore"):
            mean_e = sums / counts
        return centers, mean_e

    real_xs, real_ys = real_events[:, :, 0][real_mask], real_events[:, :, 1][real_mask]
    gen_xs, gen_ys = fake_showers[:, :, 0][gen_mask], fake_showers[:, :, 1][gen_mask]
    real_zs, gen_zs = real_z_layer[real_mask].astype(float), gen_z_layer[gen_mask].astype(float)
    real_es, gen_es = real_events[:, :, 3][real_mask], fake_showers[:, :, 3][gen_mask]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for ax, label, gcoord, rcoord, is_layer in zip(
        axes[:3], ["x [mm]", "y [mm]", "z (layer index)"],
        [gen_xs, gen_ys, gen_zs], [real_xs, real_ys, real_zs], [False, False, True],
    ):
        if is_layer:
            # z is a discrete layer index (0..n_layers-1): align bins to integers so
            # every layer gets its own bin, instead of percentile-based bins that can
            # straddle/skip integers and leave empty (NaN) gaps in the line.
            edges = np.arange(-0.5, n_layers + 0.5, 1)
        else:
            lo = min(np.percentile(gcoord, 1), np.percentile(rcoord, 1))
            hi = max(np.percentile(gcoord, 99), np.percentile(rcoord, 99))
            edges = np.linspace(lo, hi, 29)
        rc, rmean = mean_energy_profile(rcoord, real_es, edges)
        gc, gmean = mean_energy_profile(gcoord, gen_es, edges)
        ax.step(rc, rmean, where="mid", color="grey", linewidth=2, label="Real")
        ax.step(gc, gmean, where="mid", color="steelblue", linewidth=1.5, label="Generated")
        ax.set_xlabel(label)
        ax.set_ylabel("Mean hit energy [MeV]")
        ax.legend()

    ax = axes[3]
    lo = min(np.percentile(np.log10(gen_es.clip(1e-6)), 1), np.percentile(np.log10(real_es.clip(1e-6)), 1))
    hi = max(np.percentile(np.log10(gen_es.clip(1e-6)), 99), np.percentile(np.log10(real_es.clip(1e-6)), 99))
    bins = np.linspace(lo, hi, 28)
    ax.hist(np.log10(real_es.clip(1e-6)), bins=bins, density=True, histtype="stepfilled", color="grey", alpha=0.6, label="Real")
    ax.hist(np.log10(gen_es.clip(1e-6)), bins=bins, density=True, histtype="step", color="steelblue", linewidth=1.5, label="Generated")
    ax.set_xlabel("log$_{10}$(hit energy [MeV])")
    ax.set_ylabel("Density")
    ax.set_yscale("log")
    ax.legend()

    plt.suptitle(f"Hit-level coordinates — {args.data_used}")
    plt.tight_layout()
    out = os.path.join(args.out_dir, "hit_level_coords.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # ── Figure 2: shower-level observables ──────────────────────────────────
    # n_hits and total deposited energy both scale strongly with incident energy,
    # so an unconditioned histogram over the whole dataset is dominated by that
    # spread and looks bumpy/multimodal. Bin by incident energy instead, and show
    # a gen/real ratio panel under each profile so deviations are easy to read.
    gen_n_hits, real_n_hits = gen_mask.sum(axis=1), real_mask.sum(axis=1)
    gen_e_sum = (fake_showers[:, :, 3] * gen_mask).sum(axis=1)
    real_e_sum = (real_events[:, :, 3] * real_mask).sum(axis=1)
    gen_response = gen_e_sum / (gen_inc_energy * 1000)
    real_response = real_e_sum / (real_inc_energy * 1000)

    def binned_profile(x_real, y_real, x_gen, y_gen, n_bins=12):
        lo = min(x_real.min(), x_gen.min())
        hi = max(x_real.max(), x_gen.max())
        edges = np.linspace(lo, hi, n_bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2

        def means(x, y):
            idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_bins - 1)
            sums = np.bincount(idx, weights=y, minlength=n_bins)[:n_bins]
            counts = np.bincount(idx, minlength=n_bins)[:n_bins]
            with np.errstate(invalid="ignore"):
                return sums / counts

        return centers, means(x_real, y_real), means(x_gen, y_gen)

    fig = plt.figure(figsize=(16, 8))
    gs = plt.GridSpec(2, 3, height_ratios=[3, 1], hspace=0.08, wspace=0.32)

    panels = [
        ("Mean N hits", "N hits per shower", gen_n_hits, real_n_hits, "steelblue"),
        ("Mean dep. E [MeV]", "Total deposited energy", gen_e_sum, real_e_sum, "coral"),
    ]
    for col, (ylabel, title, gen_y, real_y, color) in enumerate(panels):
        centers, real_mean, gen_mean = binned_profile(real_inc_energy, real_y, gen_inc_energy, gen_y)
        ax = fig.add_subplot(gs[0, col])
        ax_r = fig.add_subplot(gs[1, col], sharex=ax)
        ax.step(centers, real_mean, where="mid", color="grey", linewidth=2, label="Real")
        ax.step(centers, gen_mean, where="mid", color=color, linewidth=1.5, label="Generated")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.tick_params(labelbottom=False)
        ax.legend()
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(real_mean > 0, gen_mean / real_mean, np.nan)
        ax_r.step(centers, ratio, where="mid", color=color, linewidth=1.5)
        ax_r.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        ax_r.set_ylim(0.5, 1.5)
        ax_r.set_xlabel("Incident energy [GeV]")
        ax_r.set_ylabel("Gen/Real")

    ax = fig.add_subplot(gs[:, 2])
    lo = min(np.percentile(gen_response, 1), np.percentile(real_response, 1))
    hi = max(np.percentile(gen_response, 99), np.percentile(real_response, 99))
    bins = np.linspace(lo, hi, 40)
    ax.hist(real_response, bins=bins, density=True, histtype="stepfilled", color="grey", alpha=0.6, label="Real")
    ax.hist(gen_response, bins=bins, density=True, histtype="step", color="forestgreen", linewidth=1.5, label="Generated")
    ax.set_xlabel("Deposited / incident energy")
    ax.set_ylabel("Density")
    ax.set_title("Energy response")
    ax.legend()

    plt.suptitle(f"Shower-level observables — {args.data_used}")
    out = os.path.join(args.out_dir, "shower_level_observables.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # ── Figure 3: center of gravity and radial profile ──────────────────────
    def e_cog(arr, energy, mask):
        e = energy * mask
        a = arr * mask
        return (a * e).sum(axis=1) / e.sum(axis=1)

    def mean_center(arr, mask):
        return (arr * mask).sum(axis=1) / mask.sum(axis=1)

    gen_cog_x = e_cog(fake_showers[:, :, 0], fake_showers[:, :, 3], gen_mask)
    gen_cog_y = e_cog(fake_showers[:, :, 1], fake_showers[:, :, 3], gen_mask)
    gen_cog_z = e_cog(gen_z_layer.astype(float), fake_showers[:, :, 3], gen_mask)
    real_cog_x = e_cog(real_events[:, :, 0], real_events[:, :, 3], real_mask)
    real_cog_y = e_cog(real_events[:, :, 1], real_events[:, :, 3], real_mask)
    real_cog_z = e_cog(real_z_layer.astype(float), real_events[:, :, 3], real_mask)

    gen_mean_x = mean_center(fake_showers[:, :, 0], gen_mask)
    gen_mean_y = mean_center(fake_showers[:, :, 1], gen_mask)
    real_mean_x = mean_center(real_events[:, :, 0], real_mask)
    real_mean_y = mean_center(real_events[:, :, 1], real_mask)

    gen_r = np.sqrt((fake_showers[:, :, 0] - gen_mean_x[:, None]) ** 2 + (fake_showers[:, :, 1] - gen_mean_y[:, None]) ** 2)
    real_r = np.sqrt((real_events[:, :, 0] - real_mean_x[:, None]) ** 2 + (real_events[:, :, 1] - real_mean_y[:, None]) ** 2)

    gen_r_energy = fake_showers[:, :, 3][gen_mask]
    real_r_energy = real_events[:, :, 3][real_mask]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    pairs = [
        ("CoG x [mm]", gen_cog_x, real_cog_x, None, None),
        ("CoG y [mm]", gen_cog_y, real_cog_y, None, None),
        ("CoG z (layer index)", gen_cog_z, real_cog_z, None, None),
        ("Radial r [mm]", gen_r[gen_mask], real_r[real_mask], gen_r_energy, real_r_energy),
    ]
    for j, (ax, (label, gv, rv, gw, rw)) in enumerate(zip(axes, pairs)):
        # CoG values are NaN for zero-hit showers (0/0 in e_cog/mean_center);
        # use nan-safe percentiles and drop them before histogramming.
        if gw is None and rw is None:
            gv, rv = gv[np.isfinite(gv)], rv[np.isfinite(rv)]
        if label in ("CoG x [mm]", "CoG y [mm]"):
            bound = max(np.nanpercentile(np.abs(gv), 99), np.nanpercentile(np.abs(rv), 99))
            lo, hi = -bound, bound
        else:
            lo = min(np.nanpercentile(gv, 1), np.nanpercentile(rv, 1))
            hi = max(np.nanpercentile(gv, 99), np.nanpercentile(rv, 99))
        bins = np.linspace(lo, hi, 40)
        ax.hist(rv, bins=bins, weights=rw, density=True, histtype="stepfilled", color="grey", alpha=0.6, label="Real")
        ax.hist(gv, bins=bins, weights=gw, density=True, histtype="step", color="steelblue", linewidth=1.5, label="Generated")
        ax.set_xlabel(label)
        ax.set_xlim([lo, hi])
        ax.set_ylabel("Energy density" if label == "Radial r [mm]" else "Density")
        ax.legend()
        ax.set_title(label)
    plt.suptitle(f"Center of gravity / radial profile — {args.data_used}")
    plt.tight_layout()
    out = os.path.join(args.out_dir, "cog_radial_profile.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # ── Figure 4: shower distributions (hits/shower, energy/shower, hit-energy
    # spectrum, x/y/z hit distributions) ────────────────────────────────────
    def dist_panel(ax, real_v, gen_v, log_x=False, log_y=False, n_bins=100, edges=None):
        if edges is not None:
            pass
        elif log_x:
            lo = min(real_v[real_v > 0].min(), gen_v[gen_v > 0].min())
            hi = max(real_v.max(), gen_v.max())
            edges = np.logspace(np.log10(lo), np.log10(hi), n_bins)
        else:
            lo = min(real_v.min(), gen_v.min())
            hi = max(real_v.max(), gen_v.max())
            edges = np.linspace(lo, hi, n_bins)
        ax.hist(real_v, bins=edges, histtype="stepfilled", color="grey", alpha=0.6, label="Real")
        ax.hist(gen_v, bins=edges, histtype="step", color="steelblue", linewidth=1.5, label="Generated")
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.legend()

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"Shower distributions — {args.data_used}  "
        f"(real n={real_events.shape[0]}, gen n={fake_showers.shape[0]})",
        fontsize=13, fontweight="bold",
    )

    dist_panel(axes[0, 0], real_n_hits, gen_n_hits)
    axes[0, 0].set_xlabel("Hits / shower")
    axes[0, 0].set_ylabel("Showers")

    resp_lo = min(np.percentile(real_response, 1), np.percentile(gen_response, 1))
    resp_hi = max(np.percentile(real_response, 99), np.percentile(gen_response, 99))
    dist_panel(axes[0, 1], real_response, gen_response, edges=np.linspace(resp_lo, resp_hi, 100))
    axes[0, 1].set_xlim(resp_lo, resp_hi)
    axes[0, 1].set_xlabel("Deposited / incident energy")
    axes[0, 1].set_ylabel("Showers")

    dist_panel(axes[0, 2], real_es / 1000, gen_es / 1000, log_x=True, log_y=True)
    axes[0, 2].set_xlabel("Hit energy [GeV]")
    axes[0, 2].set_ylabel("Hits")
    axes[0, 2].set_title("Hit energy spectrum")

    dist_panel(axes[1, 0], real_xs, gen_xs, log_y=True)
    axes[1, 0].set_xlabel("x [mm]")
    axes[1, 0].set_ylabel("Hits")

    dist_panel(axes[1, 1], real_ys, gen_ys, log_y=True)
    axes[1, 1].set_xlabel("y [mm]")
    axes[1, 1].set_ylabel("Hits")

    dist_panel(axes[1, 2], real_zs, gen_zs, log_y=True, edges=np.arange(-0.5, n_layers + 0.5, 1))
    axes[1, 2].set_xlabel("z (layer index)")
    axes[1, 2].set_ylabel("Hits")

    plt.tight_layout()
    out = os.path.join(args.out_dir, "shower_distributions.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────")
    print(f"{'':<20}{'N showers':>12}{'mean n_hits':>14}{'mean dep E [MeV]':>18}")
    print(f"{'Real':<20}{real_events.shape[0]:>12d}{real_n_hits.mean():>14.1f}{real_e_sum.mean():>18.2f}")
    print(f"{'Generated':<20}{fake_showers.shape[0]:>12d}{gen_n_hits.mean():>14.1f}{gen_e_sum.mean():>18.2f}")


if __name__ == "__main__":
    main()
