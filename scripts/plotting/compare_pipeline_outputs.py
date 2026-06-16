#!/usr/bin/env python3
"""
Compare CC3-generated showers (after the full step2point pipeline) across
different log_dirs against real GEANT4 data processed with merge_within_cell.

Both reference and generated data are in step2point HDF5 format:
  steps/{event_id, energy, position, cell_id}
  primary/{event_id, momentum, pdg, vertex}

Usage:
    python compare_pipeline_outputs.py \
        --generated-dir /eos/user/m/mamozzan/CaloClouds-3/generated_showers \
        --reference-dir /eos/user/m/mamozzan/step2point/outputs/pipeline2_merge_within_cell \
        --output-dir ./comparison_plots \
        [--n-showers 10000] [--max-ref 50000] [--bins 50]

Reference is loaded from all file_*/compressed_merge_within_cell.h5 under --reference-dir.
For each log_dir under --generated-dir the script tries (in order):
  1. generated_showers_{n}/compressed_merge_within_cell.h5   (step2point format)
  2. generated_showers_{n}/generated_showers.h5              (CC3 format, fallback)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import matplotlib.gridspec as mgs
import matplotlib.pyplot as plt
import numpy as np

_PALETTE = ["tab:orange", "tab:blue", "tab:green", "tab:red", "tab:purple"]
_REF_COLOR = "black"
_REF_LABEL = "Reference (GEANT4 + MWC)"


# ── Loading ───────────────────────────────────────────────────────────────────


def _load_step2point(paths: list[str], max_showers: int | None = None) -> dict:
    """Load one or more step2point HDF5 files and concatenate."""
    n_hits_list, total_E_list, incident_E_list = [], [], []
    hit_E_all, x_all, y_all, z_all = [], [], [], []
    loaded = 0

    for path in paths:
        with h5py.File(path, "r", locking=False) as f:
            event_id = np.asarray(f["steps/event_id"], dtype=np.int32)
            hit_E = np.asarray(f["steps/energy"], dtype=np.float32)
            pos = np.asarray(f["steps/position"], dtype=np.float32)
            p_mom = np.asarray(f["primary/momentum"], dtype=np.float32)

        incident_E = np.linalg.norm(p_mom, axis=1).astype(np.float32)

        unique_ids, counts = np.unique(event_id, return_counts=True)
        boundaries = np.append(np.searchsorted(event_id, unique_ids), len(event_id))

        if max_showers is not None:
            keep = min(len(unique_ids), max_showers - loaded)
            unique_ids = unique_ids[:keep]
            counts = counts[:keep]
            boundaries = boundaries[: keep + 1]
            incident_E = incident_E[:keep]
            # trim hits to only those belonging to kept events
            end_hit = int(boundaries[-1])
            event_id = event_id[:end_hit]
            hit_E = hit_E[:end_hit]
            pos = pos[:end_hit]

        total_E = np.array(
            [
                hit_E[boundaries[i] : boundaries[i + 1]].sum()
                for i in range(len(unique_ids))
            ],
            dtype=np.float32,
        )

        n_hits_list.append(counts.astype(np.int64))
        total_E_list.append(total_E)
        incident_E_list.append(incident_E)
        hit_E_all.append(hit_E)
        x_all.append(pos[:, 0])
        y_all.append(pos[:, 1])
        z_all.append(pos[:, 2])

        loaded += len(unique_ids)
        if max_showers is not None and loaded >= max_showers:
            break

    return {
        "n_hits": np.concatenate(n_hits_list),
        "total_E": np.concatenate(total_E_list),
        "incident_E": np.concatenate(incident_E_list),
        "hit_E": np.concatenate(hit_E_all),
        "x": np.concatenate(x_all),
        "y": np.concatenate(y_all),
        "z": np.concatenate(z_all),
    }


def _load_cc3_fallback(path: str) -> dict:
    """Fallback: load CC3-format generated_showers.h5 (coordinate-invariant fields only)."""
    with h5py.File(path, "r", locking=False) as f:
        events = np.asarray(f["events"], dtype=np.float32)  # (N, max_hits, 4)
        n_pts = np.asarray(f["n_points"], dtype=np.int64)  # (N,)
        inc_E = np.asarray(f["energy"], dtype=np.float32).reshape(-1)

    hit_e = events[:, :, 3]
    mask = hit_e > 0
    return {
        "n_hits": n_pts,
        "total_E": hit_e.sum(axis=1),
        "incident_E": inc_E,
        "hit_E": hit_e[mask],
        # no global x/y/z available in CC3 local format
    }


def _short_label(log_dir_name: str) -> str:
    if "WithinCell" in log_dir_name:
        return "CC3 WithinCell"
    if "HDBScan" in log_dir_name:
        return "CC3 HDBScan"
    parts = log_dir_name.split("_")
    return "_".join(parts[:4]) if len(parts) > 4 else log_dir_name


# ── Discovery ─────────────────────────────────────────────────────────────────


def discover_generated(
    generated_dir: str, n_showers: int
) -> list[tuple[str, str, dict]]:
    results = []
    for log_dir_path in sorted(Path(generated_dir).iterdir()):
        if not log_dir_path.is_dir():
            continue
        subdir = log_dir_path / f"generated_showers_{n_showers}"
        compressed = subdir / "compressed_merge_within_cell.h5"
        cc3_raw = subdir / "generated_showers.h5"

        if compressed.exists():
            print(f"  [{log_dir_path.name}] step2point format: {compressed.name}")
            data = _load_step2point([str(compressed)])
        elif cc3_raw.exists():
            print(
                f"  [{log_dir_path.name}] CC3 fallback (pipeline not run yet): {cc3_raw.name}"
            )
            data = _load_cc3_fallback(str(cc3_raw))
        else:
            print(f"  [{log_dir_path.name}] nothing found in {subdir}, skipping.")
            continue

        results.append((log_dir_path.name, _short_label(log_dir_path.name), data))
    return results


# ── Plot helpers ──────────────────────────────────────────────────────────────


def _hist(ax, datasets, palette, key, bins, xlabel, title, *,
          logy=False, logx=False, xlim=None, ax_ratio=None):
    vals = [d[key] for _, _, d in datasets if key in d]
    if not vals:
        ax.set_visible(False)
        if ax_ratio is not None:
            ax_ratio.set_visible(False)
        return
    all_v = np.concatenate(vals)
    if xlim is not None:
        lo, hi = xlim
    else:
        lo, hi = np.nanpercentile(all_v, 0.5), np.nanpercentile(all_v, 99.5)
    if logx:
        lo = max(lo, all_v[all_v > 0].min() if (all_v > 0).any() else 1e-6)
        hi = max(hi, lo * 10)
        edges = np.logspace(np.log10(lo), np.log10(hi), bins + 1)
    else:
        edges = np.linspace(lo, hi, bins + 1)

    counts_per_ds: dict[str, np.ndarray] = {}
    for (name, label, d), color in zip(datasets, palette):
        if key not in d:
            continue
        counts, _ = np.histogram(d[key], bins=edges)
        width = np.diff(edges)
        counts_per_ds[name] = counts / max(counts.sum(), 1) / width
        ax.hist(d[key], bins=edges, density=True, histtype="step",
                color=color, linewidth=1.5, label=label)

    ax.set_ylabel("Density")
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")

    if ax_ratio is not None:
        ax.tick_params(labelbottom=False)
        ref_name = datasets[0][0]
        ref_counts = counts_per_ds.get(ref_name)
        if ref_counts is not None:
            for (name, label, d), color in zip(datasets[1:], palette[1:]):
                gen_counts = counts_per_ds.get(name)
                if gen_counts is None:
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(ref_counts > 0, gen_counts / ref_counts, np.nan)
                ax_ratio.step(edges, np.append(ratio, ratio[-1]),
                              where="post", color=color, linewidth=1.2)
            ax_ratio.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
            ax_ratio.set_ylim(0.0, 2.0)
            ax_ratio.set_yticks([0.5, 1.0, 1.5])
            if logx:
                ax_ratio.set_xscale("log")
            ax_ratio.set_xlabel(xlabel)
            ax_ratio.set_ylabel("Gen/Ref", fontsize=7)
            ax_ratio.tick_params(labelsize=7)
        else:
            ax_ratio.set_visible(False)
            ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(xlabel)


def _profile(ax, datasets, palette, x_key, e_key, bins, xlabel, title, *,
             xlim=None, ax_ratio=None):
    """Profile histogram: mean hit energy per spatial bin."""
    vals_x = [d[x_key] for _, _, d in datasets if x_key in d and e_key in d]
    if not vals_x:
        ax.set_visible(False)
        if ax_ratio is not None:
            ax_ratio.set_visible(False)
        return
    all_x = np.concatenate(vals_x)
    if xlim is not None:
        lo, hi = xlim
    else:
        lo, hi = np.nanpercentile(all_x, 0.1), np.nanpercentile(all_x, 99.9)
    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    profiles: dict[str, np.ndarray] = {}
    for (name, label, d), color in zip(datasets, palette):
        if x_key not in d or e_key not in d:
            continue
        x_data = d[x_key]
        e_data = d[e_key]
        mean_e = np.array([
            e_data[(x_data >= edges[i]) & (x_data < edges[i + 1])].mean()
            if ((x_data >= edges[i]) & (x_data < edges[i + 1])).any() else np.nan
            for i in range(len(centers))
        ])
        profiles[name] = mean_e
        ax.step(edges, np.append(mean_e, mean_e[-1]),
                where="post", color=color, linewidth=1.5, label=label)

    ax.set_ylabel("Mean hit E [GeV]")
    ax.set_title(title)
    yvals_all = np.concatenate([v for v in profiles.values() if v is not None])
    ylo = np.nanmin(yvals_all)
    yhi = np.nanmax(yvals_all)
    pad = 0.05 * (yhi - ylo) if yhi > ylo else 1e-6
    ax.set_ylim(max(0, ylo - pad), yhi + pad)

    if ax_ratio is not None:
        ax.tick_params(labelbottom=False)
        ref_name = datasets[0][0]
        ref_prof = profiles.get(ref_name)
        if ref_prof is not None:
            for (name, label, d), color in zip(datasets[1:], palette[1:]):
                gen_prof = profiles.get(name)
                if gen_prof is None:
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(
                        np.isfinite(ref_prof) & (ref_prof > 0),
                        gen_prof / ref_prof, np.nan,
                    )
                ax_ratio.step(edges, np.append(ratio, ratio[-1]),
                              where="post", color=color, linewidth=1.2)
            ax_ratio.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
            ax_ratio.set_ylim(0.0, 2.0)
            ax_ratio.set_yticks([0.5, 1.0, 1.5])
            ax_ratio.set_xlabel(xlabel)
            ax_ratio.set_ylabel("Gen/Ref", fontsize=7)
            ax_ratio.tick_params(labelsize=7)
        else:
            ax_ratio.set_visible(False)
            ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(xlabel)


def _longitudinal_profile(ax, datasets, palette, title, n_layers=30, *, ax_ratio=None):
    """Mean deposited energy per calorimeter layer (longitudinal shower profile).

    Layer assignment is done by radial coordinate sqrt(x^2+y^2).  Uses
    n_layers equal-width bins over the reference radial range (transverse cell
    offsets smear r by ~5 mm, so per-mm unique values do not resolve layers).
    """
    ref_d = datasets[0][2]
    if "x" not in ref_d or "y" not in ref_d or "hit_E" not in ref_d:
        ax.set_visible(False)
        if ax_ratio is not None:
            ax_ratio.set_visible(False)
        return

    # Equal-width bins over reference radial range
    r_ref = np.sqrt(ref_d["x"] ** 2 + ref_d["y"] ** 2)
    r_min, r_max = float(r_ref.min()), float(r_ref.max())
    boundaries = np.linspace(r_min - 0.5, r_max + 0.5, n_layers + 1)
    layer_x = np.arange(n_layers)

    profiles: dict[str, np.ndarray] = {}
    for (name, label, d), color in zip(datasets, palette):
        if "x" not in d or "y" not in d or "hit_E" not in d:
            continue
        n_showers = len(d["n_hits"])
        r_data = np.sqrt(d["x"] ** 2 + d["y"] ** 2)
        e_data = d["hit_E"]

        hit_layer = np.clip(np.searchsorted(boundaries, r_data) - 1, 0, n_layers - 1)
        mean_e = np.array(
            [e_data[hit_layer == li].sum() / n_showers for li in range(n_layers)]
        )
        profiles[name] = mean_e
        ax.step(layer_x, mean_e, where="mid", color=color, linewidth=1.5, label=label)

    ax.set_ylabel("Mean dep. E / shower [GeV]")
    ax.set_title(title)
    ax.set_xlim(-0.5, n_layers - 0.5)
    yhi = max((np.nanmax(v) for v in profiles.values()), default=1.0)
    ax.set_ylim(0, yhi * 1.12)

    if ax_ratio is not None:
        ax.tick_params(labelbottom=False)
        ref_name = datasets[0][0]
        ref_prof = profiles.get(ref_name)
        if ref_prof is not None:
            for (name, label, d), color in zip(datasets[1:], palette[1:]):
                gen_prof = profiles.get(name)
                if gen_prof is None:
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(ref_prof > 0, gen_prof / ref_prof, np.nan)
                ax_ratio.step(layer_x, ratio, where="mid", color=color, linewidth=1.2)
            ax_ratio.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
            ax_ratio.set_ylim(0.0, 2.0)
            ax_ratio.set_yticks([0.5, 1.0, 1.5])
            ax_ratio.set_xlim(-0.5, n_layers - 0.5)
            ax_ratio.set_xlabel("Layer")
            ax_ratio.set_ylabel("Gen/Ref", fontsize=7)
            ax_ratio.tick_params(labelsize=7)
        else:
            ax_ratio.set_visible(False)
            ax.set_xlabel("Layer")
    else:
        ax.set_xlabel("Layer")



def _legend(fig, datasets, palette):
    handles = [
        plt.Line2D([0], [0], color=c, linewidth=2, label=label)
        for (_, label, _), c in zip(datasets, palette)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, -0.01),
        fontsize=9,
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generated-dir", default="/eos/user/m/mamozzan/CaloClouds-3/generated_showers"
    )
    parser.add_argument(
        "--reference-dir",
        default="/eos/user/m/mamozzan/step2point/outputs/pipeline2_merge_within_cell",
    )
    parser.add_argument("--output-dir", default="./comparison_plots")
    parser.add_argument("--n-showers", type=int, default=10000)
    parser.add_argument(
        "--max-ref", type=int, default=50000, help="Max reference showers to load."
    )
    parser.add_argument("--bins", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(42)

    # ── Reference ──────────────────────────────────────────────────────────
    ref_files = sorted(
        Path(args.reference_dir).glob("file_*/compressed_merge_within_cell.h5")
    )
    if not ref_files:
        raise FileNotFoundError(
            f"No file_*/compressed_merge_within_cell.h5 found in {args.reference_dir}"
        )
    print(
        f"Loading reference ({len(ref_files)} file(s), up to {args.max_ref} showers)…"
    )
    ref_data = _load_step2point([str(p) for p in ref_files], max_showers=args.max_ref)
    print(f"  → {len(ref_data['n_hits'])} reference showers\n")

    # ── Generated ──────────────────────────────────────────────────────────
    print("Discovering generated datasets…")
    generated = discover_generated(args.generated_dir, args.n_showers)
    if not generated:
        raise RuntimeError("No generated datasets found.")
    print(f"  → {len(generated)} dataset(s)\n")

    # Build ordered list: reference first
    palette = [_REF_COLOR] + _PALETTE[: len(generated)]
    all_ds = [("reference", _REF_LABEL, ref_data)] + generated

    # Add derived per-hit quantities to each dataset
    for _, _, d in all_ds:
        inc = d["incident_E"]
        dep = d["total_E"][: len(inc)]
        d["response"] = np.where(inc > 0, dep / inc, np.nan)
        if "x" in d and "y" in d:
            d["r"]   = np.sqrt(d["x"] ** 2 + d["y"] ** 2)
            d["phi"] = np.degrees(np.arctan2(d["y"], d["x"]))

    # ── Figure 1: shower-level ──────────────────────────────────────────────
    fig1 = plt.figure(figsize=(15, 12))
    fig1.suptitle("Shower-level comparison: generated vs reference", fontsize=13)
    outer1 = mgs.GridSpec(2, 3, figure=fig1, hspace=0.55, wspace=0.35)

    def _pair(gs, row, col):
        """Return (ax_main, ax_ratio) as a stacked 3:1 pair inside one GridSpec cell."""
        inner = mgs.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs[row, col], height_ratios=[3, 1], hspace=0.08
        )
        ax_m = fig1.add_subplot(inner[0])
        ax_r = fig1.add_subplot(inner[1], sharex=ax_m)
        return ax_m, ax_r

    m00, r00 = _pair(outer1, 0, 0)
    m01, r01 = _pair(outer1, 0, 1)
    m02, r02 = _pair(outer1, 0, 2)
    m10, r10 = _pair(outer1, 1, 0)
    m11, r11 = _pair(outer1, 1, 1)
    m12, r12 = _pair(outer1, 1, 2)

    _hist(m00, all_ds, palette, "n_hits",     args.bins, "N hits per shower",       "Hit multiplicity",   ax_ratio=r00)
    _hist(m01, all_ds, palette, "total_E",    args.bins, "Total deposited E [GeV]", "Deposited energy",   ax_ratio=r01)
    _hist(m02, all_ds, palette, "incident_E", args.bins, "Incident E [GeV]",        "Incident energy",    ax_ratio=r02)
    _hist(m10, all_ds, palette, "response",   args.bins, "Deposited / Incident E",  "Energy response",    ax_ratio=r10)
    _hist(m11, all_ds, palette, "hit_E",      args.bins, "Hit energy [GeV]",        "Hit energy spectrum", logy=True, logx=True, ax_ratio=r11)
    _longitudinal_profile(m12, all_ds, palette, "Longitudinal profile", ax_ratio=r12)

    _legend(fig1, all_ds, palette)
    fig1.tight_layout(rect=[0, 0.04, 1, 0.97])
    out1 = Path(args.output_dir) / "shower_level_comparison.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Saved {out1}")

    # ── Figure 2: spatial mean hit energy profiles ──────────────────────────
    ref_r = ref_data.get("r", np.array([1900.0]))
    xlim_r = (ref_r.min() * 0.999, ref_r.max() * 1.001)
    xlim_xz = (-400.0, 400.0)

    fig2 = plt.figure(figsize=(15, 6))
    fig2.suptitle("Mean hit energy vs shower coordinates", fontsize=13)
    outer2 = mgs.GridSpec(1, 3, figure=fig2, hspace=0.45, wspace=0.35)

    def _pair2(gs, col):
        inner = mgs.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs[0, col], height_ratios=[3, 1], hspace=0.08
        )
        ax_m = fig2.add_subplot(inner[0])
        ax_r = fig2.add_subplot(inner[1], sharex=ax_m)
        return ax_m, ax_r

    mr, rr = _pair2(outer2, 0)
    mx, rx = _pair2(outer2, 1)
    mz, rz = _pair2(outer2, 2)

    _profile(mr, all_ds, palette, "r",   "hit_E", args.bins,
             "Radial depth r [mm]", "Radial depth (longitudinal)", xlim=xlim_r,  ax_ratio=rr)
    _profile(mx, all_ds, palette, "x",   "hit_E", args.bins,
             "Global x [mm]",       "Global x",                    xlim=xlim_xz, ax_ratio=rx)
    _profile(mz, all_ds, palette, "z",   "hit_E", args.bins,
             "Global z [mm]",       "Global z",                    xlim=xlim_xz, ax_ratio=rz)

    _legend(fig2, all_ds, palette)
    fig2.tight_layout(rect=[0, 0.06, 1, 0.97])
    out2 = Path(args.output_dir) / "spatial_comparison.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved {out2}")

    # ── Summary table ───────────────────────────────────────────────────────
    print("\n── Summary ─────────────────────────────────────────────────────────")
    hdr = f"{'Dataset':<35}  {'N showers':>10}  {'mean n_hits':>11}  {'mean dep E':>11}  {'mean inc E':>11}  {'mean resp':>10}"
    print(hdr)
    print("─" * len(hdr))
    for _, label, d in all_ds:
        print(
            f"{label:<35}  {len(d['n_hits']):>10d}  "
            f"{d['n_hits'].mean():>11.1f}  "
            f"{d['total_E'].mean():>11.4f}  "
            f"{d['incident_E'].mean():>11.4f}  "
            f"{np.nanmean(d['response']):>10.4f}"
        )


if __name__ == "__main__":
    main()
