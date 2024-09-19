"""
Check how long it takes to draw from variations of shower flow
"""

import matplotlib.pyplot as plt
from pointcloud.models.shower_flow import HybridTanH_factory
from pointcloud.configs import Configs
#from pointcloud.config_varients.wish import Configs
import time
import numpy as np
import torch
import os

DEFAULT_NUM_INPUTS = 65
DEFAULT_NUM_COND = 1
DEFAULT_COUNT_BINS = 8


def time_pattern(factory, pattern, count_bins, incident_energies, pattern_repeats):
    """
    Generate a flow dist, and time how long it takes to draw from it.
    """
    sample_size = torch.Size([10000])
    times = np.zeros((len(pattern_repeats), len(incident_energies)))
    first = True
    for q in range(10):
        print(f"{q/10:.0%}", end="\r")
        for r, repeats in reversed(list(enumerate(pattern_repeats))):
        #for r, repeats in enumerate(pattern_repeats):
            _, flow_dist = factory.create(repeats, pattern, count_bins=count_bins)
            for i, incident_energy in enumerate(incident_energies):
                dist = flow_dist.condition(incident_energy)
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


def time_saved(models, incident_energies):
    """
    Given a flow dist, time how long it takes to draw from it.
    """
    sample_size = torch.Size([10000])
    times = np.zeros((len(models), len(incident_energies)))
    repeats = 10
    ticks = repeats * len(models)
    for r, model_params in enumerate(models):
        flow_dist = model_loader(*model_params)
        big_tick = r*repeats
        first = True
        for q in range(repeats):
            print(f"{(big_tick + q)/ticks:.0%}", end="\r")
            for i, incident_energy in enumerate(incident_energies):
                dist = flow_dist.condition(incident_energy)
                t0 = time.time()
                dist.sample(sample_size)
                t1 = time.time()
                if not first:  # Ignore first run
                    times[r, i] += t1 - t0
            first = False
    times /= (repeats - 1)
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
configs = Configs()

try:
    configs.device = "cuda"
    incident_energies = torch.linspace(0.1, 1.0, 5)[:, None].to(configs.device)
except Exception as e:
    configs.device = "cpu"
    incident_energies = torch.linspace(0.1, 1.0, 5)[:, None].to(configs.device)

#dataset_basename = "p22_th90_ph90_en10-100"
dataset_basename = "sim-E1261AT600AP180-180"
save_path = os.path.join(configs.logdir, dataset_basename, f"time_shower_flow_{configs.device}_disk.npz")


def saved_name(version, num_blocks, cut_inputs):
    max_input_dims = 65
    inputs_used = np.ones(max_input_dims, dtype=bool)
    for i in range(5):
        if str(i) in cut_inputs:
            inputs_used[i] = False
    inputs_used_as_binary = ''.join(['1' if i else '0' for i in inputs_used])
    inputs_used_as_base10 = int(inputs_used_as_binary, 2)
    name_base = f"ShowerFlow_{version}_nb{num_blocks}_inputs{inputs_used_as_base10}"
    data_dir = os.path.join(configs.storage_base, "dayhallh/point-cloud-diffusion-data")
    # data_dir = "/home/dayhallh/Data/"
    showerflow_dir = os.path.join(data_dir, f"showerFlow/{dataset_basename}")
    best_model_path = os.path.join(showerflow_dir, f"{name_base}_best.pth")
    best_data_path = os.path.join(showerflow_dir, f"{name_base}_best_data.txt")

    nice_name = f"{version}_nb{num_blocks}"
    if cut_inputs:
        nice_name += f"_wo{cut_inputs}"

    return nice_name, best_model_path, best_data_path

from pointcloud.models.shower_flow import (
    compile_HybridTanH_model,
    compile_HybridTanH_alt1,
    compile_HybridTanH_alt2,
)


versions_dict = {
    "original": compile_HybridTanH_model,
    "alt1": compile_HybridTanH_alt1,
    "alt2": compile_HybridTanH_alt2,
}


def model_loader(version_name, num_blocks, cut_inputs):
    model, distribution = versions_dict[version_name](
        num_blocks=num_blocks,
        num_inputs=65 - len(cut_inputs),
        num_cond_inputs=1,
        device=configs.device,
    )
    model_path = saved_name(version_name, num_blocks, cut_inputs)[1]
    model.load_state_dict(torch.load(model_path)["model"])
    return distribution


def get_saved_models(configs):
    versions = []
    names = []
    num_blocks = []
    cut_inputs = []
    best_loss = []
    for version in versions_dict:
        for nb in range(1, 100):
            for ci in ["", "01", "01234"]:
                name, model_path, data_path = saved_name(version, nb, ci)
                if not os.path.exists(model_path):
                    #print(f"Skipping {name}")
                    continue
                names.append(name)
                versions.append(version)
                num_blocks.append(nb)
                cut_inputs.append(ci)
                with open(data_path, "r") as f:
                    text = f.read().split()
                    best_loss.append(float(text[0]))
                if nb == 4:
                    print(f"{name} has best loss {best_loss[-1]}")
    print(f"Found {len(names)} saved models")

    saved_models = {"names": names, "version": versions, "num_blocks": num_blocks, "cut_inputs": cut_inputs, "best_loss": best_loss}
    return saved_models


def time_all():
    to_save = {}

    to_save["pattern_repeats"] = pattern_repeats
    to_save["incident_energies"] = incident_energies.cpu().numpy()
    factory = HybridTanH_factory(DEFAULT_NUM_INPUTS, DEFAULT_NUM_COND, configs.device)

    for pattern in patterns:
        print(f"Pattern: {pattern}")
        times = time_pattern(
            factory,
            patterns[pattern],
            DEFAULT_COUNT_BINS,
            incident_energies,
            pattern_repeats,
        )
        to_save[pattern] = times
        np.savez(save_path, **to_save)


def time_all_saved():
    saved_models = get_saved_models(configs)
    model_params = list(zip(saved_models["version"], saved_models["num_blocks"], saved_models["cut_inputs"]))
    times = time_saved(model_params, incident_energies)
    to_save = {"incident_energies": incident_energies.cpu().numpy()}
    to_save.update(saved_models)
    to_save["times"] = times
    np.savez(save_path, **to_save)


def update_losses_only():
    saved_models = get_saved_models(configs)
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
        incident_energies = data["incident_energies"]

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
    ax.set_xticks(np.arange(len(incident_energies)))
    ax.set_xticklabels(incident_energies.squeeze())
    ax.set_yticks(np.arange(len(pattern_repeats)))
    ax.set_yticklabels(pattern_repeats)

    # ax[1] is the marginal in pattern repeats
    ax = axes[1]
    ax.plot(pattern_repeats, times.mean(axis=1))
    ax.set_xlabel("Pattern Repeats")
    ax.set_ylabel("Mean Time (s)")



if __name__ == "__main__":
    #update_losses_only()
    time_all_saved()
