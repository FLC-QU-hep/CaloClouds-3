import numpy as np
from matplotlib import pyplot as plt

dataset_name = "sim-E1261AT600AP180-180"
loaded = np.load(
    f"/gpfs/dust/user/dayhallh/point-cloud-diffusion-logs/{dataset_name}/time_shower_flow_cuda_disk.npz"
)

conds = loaded["cond"]
names = loaded["names"]
versions = loaded["versions"]
# versions = [n.split('_')[0] for n in names]
set_version = set(versions)
print(f"Versions are {set_version}")
num_blocks = loaded["num_blocks"]
cut_inputs = loaded["cut_inputs"]
set_cut_inputs = set(cut_inputs)
print(f"Input cuts are {set_cut_inputs}")
best_loss = loaded["best_loss"]
times = loaded["times"]
for key in loaded:
    items = set([str(l) for l in loaded[key]])
    if len(items) == len(num_blocks):
        continue

    print(key)
    print(f'\t{", ".join(items)}')

version_cut_mask = {
    f"{v}_{c}": np.array(
        [(ver == v) & (cu == c) for ver, cu in zip(versions, cut_inputs)], dtype=bool
    )
    for v in set_version
    for c in set_cut_inputs
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
from pointcloud.utils.plotting import nice_hex

version_styles = {
    "original": {"marker": "x", "color": nice_hex[2][0], "label": "original"},
    "alt1": {"marker": "^", "color": nice_hex[2][2], "label": "alt1"},
    "alt2": {"marker": "o", "color": nice_hex[2][4], "label": "alt2"},
}
for cut, ax in zip(set_cut_inputs, axes):
    if cut:
        ax.set_title(f"Cutting {cut}")
    else:
        ax.set_title("All inputs")
    for version in set_version:
        mask = version_cut_mask[f"{version}_{cut}"]
        if not np.any(mask):
            continue
        style = version_styles[version]
        ax.scatter(num_blocks[mask], best_loss[mask], **style)
    ax.set_xlabel("num blocks")
    ax.set_ylabel("best loss")
    ax.legend()

plt.tight_layout()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
from pointcloud.utils.plotting import nice_hex

version_styles = {
    "original": {"marker": "x", "color": nice_hex[2][0], "label": "original"},
    "alt1": {"marker": "^", "color": nice_hex[2][2], "label": "alt1"},
    "alt2": {"marker": "o", "color": nice_hex[2][4], "label": "alt2"},
}
for cut, ax in zip(set_cut_inputs, axes):
    if cut:
        ax.set_title(f"Cutting {cut}")
    else:
        ax.set_title("All inputs")
    for version in set_version:
        mask = version_cut_mask[f"{version}_{cut}"]
        if not np.any(mask):
            continue
        style = version_styles[version]
        ax.scatter(num_blocks[mask], np.mean(times[mask], axis=1), **style)
    ax.set_xlabel("num blocks")
    ax.set_ylabel("mean time")
    ax.legend()

plt.tight_layout()

fig, axes = plt.subplots(1, 6, figsize=(15, 5))
reached = 0
imshow_settings = {"vmin": 0, "vmax": 0.1, "origin": "lower"}
for c, cut in enumerate(set_cut_inputs):
    for v, version in enumerate(set_version):
        mask = version_cut_mask[f"{version}_{cut}"]
        if not np.any(mask):
            continue
        ax = axes[reached]
        reached += 1
        if cut:
            ax.set_title(f"Cutting {cut} - {version}")
        else:
            ax.set_title(f"All inputs - {version}")
        times_here = times[mask]
        ax.imshow(times_here, **imshow_settings)
        ax.set_ylabel("num blocks")
        ax.set_yticks(range(times_here.shape[0]))
        ax.set_yticklabels(range(1, times_here.shape[0] + 1))
        ax.set_xlabel("incident energy")
        ax.set_xticks(range(times_here.shape[1]))
        ax.set_xticklabels(conds.flatten())


plt.tight_layout()

colours = nice_hex[0]
i = 0
import os

fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
line_styles = ["-", ":", "-.", "--"]
for ax, version in zip(axes, ["original", "alt1"]):
    for num_blocks in range(11):
        history_path = f"../../../point-cloud-diffusion-data/showerFlow/{dataset_name}/ShowerFlow_{version}_nb{num_blocks}_inputs36893488147419103231_history.npy"
        if not os.path.exists(history_path):
            continue
        colour = colours[i % len(colours)]
        ls = line_styles[i % len(line_styles)]
        i += 1
        history = np.load(history_path)
        xs = np.linspace(np.min(history[0]), np.max(history[0]), 100)
        half_window = 0.2 * (xs[1] - xs[0])
        window_filter = [
            (history[0] > (x - half_window)) & (history[0] < (x + half_window))
            for x in xs
        ]
        ys = np.fromiter((np.min(history[1, f]) for f in window_filter), dtype=float)
        ax.scatter(history[0], history[1], c=colour, alpha=0.1)
        ax.plot(xs, ys, c=colour, alpha=0.3)
        ys_fit = np.polyfit(xs, ys, 1)
        ys_lin = np.polyval(ys_fit, xs)
        name = f"{version}, {num_blocks} blocks, grad {ys_fit[0]:.4e}"
        ax.plot(xs, ys_lin, c=colour, ls=ls, label=name)
    ax.legend()
    ax.set_ylim(-200, 100)
colours = nice_hex[0]
i = 0
import os

fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
line_styles = ["-", ":", "-.", "--"]
for ax, version in zip(axes, ["original", "alt1"]):
    for num_blocks in range(11):
        history_path = f"../../../point-cloud-diffusion-data/showerFlow/{dataset_name}/ShowerFlow_{version}_nb{num_blocks}_inputs8070450532247928831_fnorms_history.npy"
        if not os.path.exists(history_path):
            continue
        colour = colours[i % len(colours)]
        ls = line_styles[i % len(line_styles)]
        i += 1
        history = np.load(history_path)
        xs = np.linspace(np.min(history[0]), np.max(history[0]), 100)
        half_window = 0.2 * (xs[1] - xs[0])
        window_filter = [
            (history[0] > (x - half_window)) & (history[0] < (x + half_window))
            for x in xs
        ]
        ys = np.fromiter((np.min(history[1, f]) for f in window_filter), dtype=float)
        ax.scatter(history[0], history[1], c=colour, alpha=0.1)
        ax.plot(xs, ys, c=colour, alpha=0.3)
        ys_fit = np.polyfit(xs, ys, 1)
        ys_lin = np.polyval(ys_fit, xs)
        name = f"{version}, {num_blocks} blocks, grad {ys_fit[0]:.4e}"
        ax.plot(xs, ys_lin, c=colour, ls=ls, label=name)
    ax.legend()
    ax.set_ylim(-200, 100)
