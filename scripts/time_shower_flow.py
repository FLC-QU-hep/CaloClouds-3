"""
Check how long it takes to draw from variations of shower flow
"""

import matplotlib.pyplot as plt
from pointcloud.models.shower_flow import HybridTanH_factory, versions_dict
from pointcloud.configs import Configs
from pointcloud.utils import showerflow_utils

# from pointcloud.config_varients.wish import Configs
# from pointcloud.config_varients.caloclouds_3 import Configs
import time
import numpy as np
import torch
import os

DEFAULT_NUM_INPUTS = 65
DEFAULT_NUM_COND = 1
DEFAULT_COUNT_BINS = 8


def time_pattern(factory, pattern, count_bins, cond, pattern_repeats):
    """
    Generate a flow dist, and time how long it takes to draw from it.
    """
    sample_size = torch.Size([10000])
    times = np.zeros((len(pattern_repeats), len(cond)))
    first = True
    for q in range(10):
        print(f"{q/10:.0%}", end="\r")
        for r, repeats in reversed(list(enumerate(pattern_repeats))):
            # for r, repeats in enumerate(pattern_repeats):
            _, flow_dist = factory.create(repeats, pattern, count_bins=count_bins)
            for i, this_cond in enumerate(cond):
                dist = flow_dist.condition(this_cond)
                t0 = time.time()
                dist.sample(sample_size)
                t1 = time.time()
                if first:  # Ignore first run
                    first = False
                else:
                    times[r, i] += t1 - t0
    times /= 9
    print()
    return times


def time_saved(models, cond):
    """
    Given a flow dist, time how long it takes to draw from it.
    """
    sample_size = torch.Size([10000])
    times = np.zeros((len(models["names"]), len(cond)))
    repeats = 10
    ticks = repeats * len(models)
    for r, name in enumerate(models["names"]):
        config = showerflow_utils.construct_config(Configs(), models, r)
        flow_dist = model_loader(config, models["paths"][r])
        big_tick = r * repeats
        first = True
        for q in range(repeats):
            print(f"{(big_tick + q)/ticks:.0%}", end="\r")
            for i, this_cond in enumerate(cond):
                dist = flow_dist.condition(this_cond)
                t0 = time.time()
                dist.sample(sample_size)
                t1 = time.time()
                if not first:  # Ignore first run
                    times[r, i] += t1 - t0
            first = False
    times /= repeats - 1
    print()
    return times


patterns = {
    "permute_only": ["permutation"],
    "affine_only": ["affine_coupling"],
    "spline_only": ["spline_coupling"],
    "permute_affine": ["permutation", "affine_coupling"],
    "permute_spline": ["permutation", "spline_coupling"],
    "affine_spline": ["affine_coupling", "spline_coupling"],
    "permute_affine_spline": ["permutation", "affine_coupling", "spline_coupling"],
}
pattern_repeats = np.arange(1, 11) * 10
config = Configs()
n_cond_inputs = 4
config.logdir = "/data/dust/user/dayhallh/point-cloud-diffusion-logs/"
config.dataset_path = (
    "/beegfs/desy/user/akorol/data/AngularShowers_RegularDetector/"
    "hdf5_for_CC/sim-E1261AT600AP180-180_file_{}slcio.hdf5"
)

try:
    config.device = "cuda"
    cond = torch.linspace(0.1, 1.0, 5)[:, None].to(config.device)

except Exception as e:
    config.device = "cpu"
    cond = torch.linspace(0.1, 1.0, 5)[:, None].to(config.device)

if n_cond_inputs > 1:
    cond = torch.tile(cond, (1, n_cond_inputs))

# dataset_basename = "p22_th90_ph90_en10-100"
dataset_basename = "sim-E1261AT600AP180-180"
save_path = os.path.join(
    config.logdir, dataset_basename, f"time_shower_flow_{config.device}_disk.npz"
)


def model_loader(config, model_path):
    shower_flow_compiler = versions_dict[config.shower_flow_version]

    cond_used = showerflow_utils.get_cond_mask(config)
    cond_dim = np.sum(cond_used)
    inputs_used = showerflow_utils.get_input_mask(config)
    input_dim = np.sum(inputs_used)

    # use cuda if avaliable
    if torch.cuda.is_available():
        config.device = "cuda"
    else:
        config.device = "cpu"
    device = torch.device(config.device)

    model, distribution = shower_flow_compiler(
        num_blocks=config.shower_flow_num_blocks,
        num_inputs=input_dim,
        num_cond_inputs=cond_dim,
        device=device,
    )  # num_cond_inputs
    model.load_state_dict(torch.load(model_path, weights_only=False)["model"])
    return distribution


def get_saved_models(config):
    config.shower_flow_fixed_input_norms = True
    saved_models = showerflow_utils.existing_models(config)
    config.shower_flow_fixed_input_norms = False
    other_models = showerflow_utils.existing_models(config)
    for k in other_models:
        saved_models[k].extend(other_models[k])
    return saved_models


def time_all():
    to_save = {}

    to_save["pattern_repeats"] = pattern_repeats
    to_save["cond"] = cond.cpu().numpy()
    factory = HybridTanH_factory(DEFAULT_NUM_INPUTS, DEFAULT_NUM_COND, config.device)

    for pattern in patterns:
        print(f"Pattern: {pattern}")
        times = time_pattern(
            factory,
            patterns[pattern],
            DEFAULT_COUNT_BINS,
            cond,
            pattern_repeats,
        )
        to_save[pattern] = times
        np.savez(save_path, **to_save)


def time_all_saved():
    saved_models = get_saved_models(config)
    times = time_saved(saved_models, cond)
    to_save = {"cond": cond.cpu().numpy()}
    to_save.update(saved_models)
    to_save["times"] = times
    np.savez(save_path, **to_save)


def update_losses_only():
    saved_models = showerflow_utils.existing_models(config)
    from_disk = np.load(save_path)
    new_data = {k: from_disk[k] for k in from_disk.files}
    for i, name in enumerate(saved_models["names"]):
        if name in saved_models["names"]:
            idx = np.where(new_data["names"] == name)[0]
            new_data["best_loss"][idx] = saved_models["best_loss"][i]
        else:
            for k in new_data:
                new_data[k].append(saved_models[k][i])
    np.savez(save_path, **new_data)


def plot_pattern(name, axes=None, cmap="viridis", cmin=None, cmax=None):
    with np.load(save_path) as data:
        times = data[name]
        pattern_repeats = data["pattern_repeats"]
        cond = data["cond"]

    if axes is None:
        fig, axes = plt.subplots(1, 2)

    ax = axes[0]
    if cmin is None:
        cmin = 0.0
    if cmax is None:
        cmax = times.max()
    heatmap = ax.imshow(times, origin="lower", cmap=cmap, vmin=cmin, vmax=cmax)
    plt.colorbar(heatmap, ax=ax)
    ax.set_xlabel("Incident Energy")
    ax.set_ylabel("Pattern Repeats")
    ax.set_xticks(np.arange(len(cond)))
    ax.set_xticklabels(cond.squeeze())
    ax.set_yticks(np.arange(len(pattern_repeats)))
    ax.set_yticklabels(pattern_repeats)

    # ax[1] is the marginal in pattern repeats
    ax = axes[1]
    ax.plot(pattern_repeats, times.mean(axis=1))
    ax.set_xlabel("Pattern Repeats")
    ax.set_ylabel("Mean Time (s)")


if __name__ == "__main__":
    # update_losses_only()
    time_all_saved()
