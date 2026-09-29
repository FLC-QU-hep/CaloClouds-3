"""
Aggregate every trial's metrics.json into a ranked table + scatter plot.

Usage:
    python scripts/training/showerflow_search/summarize_results.py \
        --search_root /path/to/search_root
"""

import argparse
import csv
import glob
import json
import os

import numpy as np

# Above this combined Wasserstein distance a trial hasn't fit badly, it has run
# away entirely (observed: 4e3 up to 2e9, against ~40-290 for trials that
# trained). Used only to keep blow-ups from dominating the plots' scales - the
# ranking table below still lists them.
DIVERGED_THRESHOLD = 1000.0


def load_history(row):
    """Per-epoch [epoch, val_loss, train_loss] for one trial, or None.

    Derived from best_model_path rather than globbed, because a trial dir can
    also hold a stale history from an earlier aborted run of a different
    version (e.g. trial_0002 has both a log1 and a log1_stable history).
    """
    best = row.get("best_model_path")
    if not best or not best.endswith("_best.pth"):
        return None
    path = best[: -len("_best.pth")] + "_history.npy"
    if not os.path.exists(path):
        return None
    return np.asarray(np.load(path, allow_pickle=True), dtype=float)


def plot_loss_curves(rows, search_root, plt):
    """Validation-loss curves per version, plus best-NLL vs sample quality.

    Kept separate from the hyperparameter scatter because the NLL scale is not
    comparable between versions - a log-space flow's Jacobian shifts the
    density, so its "better" NLL says nothing about how good its samples are.
    """
    histories = {r["trial_id"]: load_history(r) for r in rows}
    have = [r for r in rows if histories.get(r["trial_id"]) is not None]
    if not have:
        return None

    versions = sorted({r.get("shower_flow_version", "alt1") for r in have})
    fig, axes = plt.subplots(1, len(versions) + 1, figsize=(5.9 * (len(versions) + 1), 5.4))
    axes = np.atleast_1d(axes)

    for ax, version in zip(axes, versions):
        group = sorted(
            [r for r in have if r.get("shower_flow_version", "alt1") == version],
            key=lambda r: r["combined_wasserstein"],
        )
        n_nan = 0
        for i, r in enumerate(group):
            hist = histories[r["trial_id"]]
            epochs, val, train = hist[0], hist[1], hist[2]
            color = plt.cm.viridis(i / max(1, len(group) - 1))
            ax.plot(
                epochs, val, lw=1.5, color=color,
                label=f"#{r['trial_id']} af{r['af_dim']}/nb{r['num_blocks']}  "
                      f"W={r['combined_wasserstein']:.3g}",
            )
            # NaN training batches are the symptom to watch for: mark the
            # epochs where the training loss went non-finite.
            nan_epochs = epochs[~np.isfinite(train)]
            n_nan += len(nan_epochs)
            if len(nan_epochs):
                ax.plot(
                    nan_epochs, np.full_like(nan_epochs, ax.get_ylim()[0]), "|",
                    color=color, ms=7, mew=1.4, clip_on=False,
                )
        if any(r["combined_wasserstein"] >= DIVERGED_THRESHOLD for r in group):
            ax.set_yscale("symlog", linthresh=200)
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation NLL loss")
        ax.set_title(
            f"{version}: {len(group)} trial(s)\n"
            f"{n_nan} NaN training epoch(s); ticks on the axis mark them",
            fontsize=11,
        )
        ax.grid(alpha=0.25, linestyle=":")
        ax.legend(fontsize=7.4, loc="upper right", framealpha=0.95)

    # Final panel: best NLL against sample quality, usable trials only.
    ax = axes[-1]
    ok = [r for r in have if r["combined_wasserstein"] < DIVERGED_THRESHOLD]
    n_bad = len(have) - len(ok)
    colors = {"alt1": "steelblue", "log1_stable": "darkorange"}
    styles = {"alt1": "o", "log1_stable": "^"}
    for version in versions:
        subset = [r for r in ok if r.get("shower_flow_version", "alt1") == version]
        if not subset:
            continue
        ax.scatter(
            [r["best_val_nll_loss"] for r in subset],
            [r["combined_wasserstein"] for r in subset],
            marker=styles.get(version, "s"), s=190, edgecolor="k", linewidth=0.9,
            c=colors.get(version, "grey"), label=version, zorder=3,
        )
        for r in subset:
            ax.annotate(
                f"#{r['trial_id']}",
                (r["best_val_nll_loss"], r["combined_wasserstein"]),
                textcoords="offset points", xytext=(9, -3.5), fontsize=9,
            )
    ax.set_xlabel("best validation NLL  (lower = 'better' loss)")
    ax.set_ylabel("combined Wasserstein distance (lower = better samples)")
    ax.set_title(
        "Best NLL vs. sample quality"
        + (f"\n({n_bad} diverged trial(s) off-scale, not shown)" if n_bad else ""),
        fontsize=11,
    )
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(title="version", fontsize=9, loc="best", framealpha=0.95)

    fig.tight_layout()
    path = os.path.join(search_root, "loss_curves.png")
    fig.savefig(path, dpi=150)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search_root", required=True)
    args = parser.parse_args()

    metrics_paths = sorted(
        glob.glob(os.path.join(args.search_root, "trials", "trial_*", "metrics.json"))
    )
    if not metrics_paths:
        print(f"No metrics.json files found under {args.search_root}/trials/")
        return

    rows = []
    failed = []
    for p in metrics_paths:
        with open(p) as f:
            m = json.load(f)
        if m.get("failed"):
            failed.append(m)
            continue
        m["combined_wasserstein"] = m["w_num_points"] + m["w_visible_energy"]
        rows.append(m)

    rows.sort(key=lambda m: m["combined_wasserstein"])

    if failed:
        print(f"{len(failed)} trial(s) failed and are excluded from ranking:")
        for m in failed:
            print(
                f"  trial {m['trial_id']}: af_dim={m['af_dim']} num_blocks={m['num_blocks']} "
                f"version={m.get('shower_flow_version', 'alt1')} - {m.get('error', 'unknown error')}"
            )
        print()

    if not rows:
        print("No successful trials to rank.")
        return

    fieldnames = [
        "trial_id",
        "af_dim",
        "num_blocks",
        "shower_flow_version",
        "epochs",
        "best_val_nll_loss",
        "best_epoch",
        "w_num_points",
        "w_visible_energy",
        "combined_wasserstein",
        "best_model_path",
    ]
    csv_path = os.path.join(args.search_root, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    print(f"{'trial':>5} {'af_dim':>7} {'nb':>3} {'version':>12} {'val_nll':>12} "
          f"{'w_points':>14} {'w_energy':>12} {'combined':>14}")
    for row in rows:
        flag = " DIVERGED" if row["combined_wasserstein"] >= DIVERGED_THRESHOLD else ""
        print(
            f"{row['trial_id']:>5} {row['af_dim']:>7} {row['num_blocks']:>3} "
            f"{row.get('shower_flow_version', 'alt1'):>12} "
            f"{row['best_val_nll_loss']:>12.2f} {row['w_num_points']:>14.2f} "
            f"{row['w_visible_energy']:>12.4f} {row['combined_wasserstein']:>14.2f}"
            f"{flag}"
        )
    print(f"\nWrote {csv_path}")
    best = rows[0]
    print(
        f"\nBest by combined Wasserstein distance: trial {best['trial_id']} "
        f"(af_dim={best['af_dim']}, num_blocks={best['num_blocks']}, "
        f"shower_flow_version={best.get('shower_flow_version', 'alt1')})"
    )
    print(f"Checkpoint: {best['best_model_path']}")

    # Break out log-vs-linear specifically, since that's a categorical choice
    # rather than a point on the af_dim/num_blocks grid.
    by_version = {}
    for r in rows:
        by_version.setdefault(r.get("shower_flow_version", "alt1"), []).append(
            r["combined_wasserstein"]
        )
    if len(by_version) > 1:
        # Report the best and median over trials that actually trained, plus
        # how many diverged. A plain mean is meaningless here: a single blow-up
        # at 2e9 sets the average for its whole version.
        print("\nBy shower_flow_version (over trials that did not diverge, lower = better):")
        stats = []
        for version, values in by_version.items():
            usable = sorted(v for v in values if v < DIVERGED_THRESHOLD)
            n_bad = len(values) - len(usable)
            stats.append((usable[0] if usable else float("inf"), version, usable, n_bad))
        for best_v, version, usable, n_bad in sorted(stats):
            if usable:
                median = usable[len(usable) // 2]
                print(
                    f"  {version:>12}: best={usable[0]:9.2f}  median={median:9.2f}  "
                    f"(n={len(usable)} usable, {n_bad} diverged)"
                )
            else:
                print(f"  {version:>12}: no usable trials ({n_bad} diverged)")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # A trial whose training ran away produces a Wasserstein distance orders of
    # magnitude above every real one (up to ~2e9). Left in, it flattens the
    # colour scale so all the usable trials come out the same shade, so plot
    # them separately rather than letting one blow-up hide the whole result.
    ok = [r for r in rows if r["combined_wasserstein"] < DIVERGED_THRESHOLD]
    diverged = [r for r in rows if r["combined_wasserstein"] >= DIVERGED_THRESHOLD]
    if not ok:
        print("\nEvery trial diverged - skipping plots.")
        return

    # Two trials with the same (af_dim, num_blocks) but different versions would
    # sit on top of each other, so nudge them apart along x.
    dx = {"alt1": -0.20, "log1_stable": 0.38}
    styles = {"alt1": "o", "log1_stable": "^"}

    def offset(r):
        return r["af_dim"] + dx.get(r.get("shower_flow_version", "alt1"), 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

    ax = axes[0]
    vmin = min(r["combined_wasserstein"] for r in ok)
    vmax = max(r["combined_wasserstein"] for r in ok)
    for version, marker in styles.items():
        subset = [r for r in ok if r.get("shower_flow_version", "alt1") == version]
        if not subset:
            continue
        sc = ax.scatter(
            [offset(r) for r in subset],
            [r["num_blocks"] for r in subset],
            c=[r["combined_wasserstein"] for r in subset],
            cmap="viridis_r",
            vmin=vmin,
            vmax=vmax,
            marker=marker,
            s=300,
            edgecolor="k",
            linewidth=1.0,
            label=version,
            zorder=3,
        )
    for r in ok:
        ax.annotate(
            f"#{r['trial_id']}", (offset(r), r["num_blocks"]),
            textcoords="offset points", xytext=(0, 17), ha="center", fontsize=9,
        )
        ax.annotate(
            f"{r['combined_wasserstein']:.0f}", (offset(r), r["num_blocks"]),
            textcoords="offset points", xytext=(0, -25), ha="center",
            fontsize=8.5, color="0.3",
        )
    best = ok[0]
    ax.scatter(
        offset(best), best["num_blocks"], s=900, facecolor="none",
        edgecolor="crimson", linewidth=2.4, zorder=2,
    )
    ax.annotate(
        "best", (offset(best), best["num_blocks"]), textcoords="offset points",
        xytext=(0, 33), ha="center", fontsize=10, color="crimson", weight="bold",
    )
    plt.colorbar(sc, ax=ax, label="combined Wasserstein distance (lower = better)")
    # Explicit padding: the per-point annotations sit above and below each
    # marker, so matplotlib's autoscale clips the top and bottom rows.
    af = [r["af_dim"] for r in ok]
    nb = [r["num_blocks"] for r in ok]
    ax.set_xlim(min(af) - 1.5, max(af) + 1.8)
    ax.set_ylim(min(nb) - 0.6, max(nb) + 0.9)
    ax.set_xticks(sorted(set(af)))
    ax.set_yticks(sorted(set(nb)))
    ax.set_xlabel("af_dim")
    ax.set_ylabel("shower_flow_num_blocks")
    ax.set_title(
        "Shower-flow search over the (af_dim, num_blocks) grid"
        + (f"\n{len(diverged)} diverged trial(s) excluded" if diverged else ""),
        fontsize=11,
    )
    ax.legend(title="version", loc="lower right", framealpha=0.95, fontsize=9)
    ax.grid(alpha=0.25, linestyle=":")

    ax = axes[1]
    for version, marker in styles.items():
        subset = [r for r in ok if r.get("shower_flow_version", "alt1") == version]
        if not subset:
            continue
        sc2 = ax.scatter(
            [r["num_blocks"] + dx.get(version, 0.0) * 0.4 for r in subset],
            [r["w_num_points"] for r in subset],
            marker=marker, s=220, edgecolor="k", linewidth=0.9,
            c=[r["af_dim"] for r in subset], cmap="plasma",
            vmin=min(r["af_dim"] for r in ok), vmax=max(r["af_dim"] for r in ok),
            label=version, zorder=3,
        )
        for r in subset:
            ax.annotate(
                f"#{r['trial_id']}",
                (r["num_blocks"] + dx.get(version, 0.0) * 0.4, r["w_num_points"]),
                textcoords="offset points", xytext=(11, -3.5), fontsize=9,
            )
    plt.colorbar(sc2, ax=ax, label="af_dim")
    ax.set_xlabel("shower_flow_num_blocks")
    ax.set_ylabel("Wasserstein distance, num_points")
    ax.set_title("Quality vs. flow depth", fontsize=11)
    ax.legend(title="version", loc="upper right", framealpha=0.95, fontsize=9)
    ax.grid(alpha=0.25, linestyle=":")

    if diverged:
        fig.text(
            0.012, 0.015,
            "excluded (diverged):  " + "   ".join(
                f"#{r['trial_id']} af{r['af_dim']}/nb{r['num_blocks']} "
                f"W={r['combined_wasserstein']:.2g}" for r in diverged
            ),
            fontsize=8.2, color="crimson",
        )
    fig.tight_layout(rect=[0, 0.045 if diverged else 0, 1, 1])
    plot_path = os.path.join(args.search_root, "summary_scatter.png")
    fig.savefig(plot_path, dpi=150)
    print(f"Wrote {plot_path}")

    loss_path = plot_loss_curves(rows, args.search_root, plt)
    if loss_path:
        print(f"Wrote {loss_path}")


if __name__ == "__main__":
    main()
