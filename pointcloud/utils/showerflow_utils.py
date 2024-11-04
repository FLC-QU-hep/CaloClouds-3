import os
import numpy as np
import copy

from pointcloud.models.shower_flow import versions_dict
from .metadata import Metadata


def get_data_dir(configs, last_resort="/home/{}/Data/"):
    user_name = os.getenv("USER", os.getenv("USERNAME", ""))
    second_choice = os.path.join(configs.storage_base, "point-cloud-diffusion-data")
    data_dir = getattr(configs, "shower_flow_data_dir", second_choice)
    if not os.path.exists(data_dir):
        data_dir = os.path.join(
            configs.storage_base, user_name, "point-cloud-diffusion-data"
        )
    if not os.path.exists(data_dir):
        data_dir = last_resort.format(user_name)
    return data_dir


def get_showerflow_dir(configs):
    dataset_path = configs.dataset_path
    base_path = get_data_dir(configs)
    dataset_name_key = ".".join(os.path.basename(dataset_path).split(".")[:-1])
    if "{" in dataset_name_key:
        dataset_name_key = dataset_name_key.split("{")[0]
    if "seed" in dataset_name_key:
        dataset_name_key = dataset_name_key.split("seed")[0]
    if "file" in dataset_name_key:
        dataset_name_key = dataset_name_key.split("file")[0]
    dataset_name_key = dataset_name_key.strip("_")

    showerflow_dir = os.path.join(base_path, "showerFlow", dataset_name_key)
    return showerflow_dir


def model_save_paths(configs, version, num_blocks, cut_inputs):
    showerflow_dir = get_showerflow_dir(configs)
    max_input_dims = 65
    inputs_used = np.ones(max_input_dims, dtype=bool)
    cut_inputs = [int(i) for i in cut_inputs]
    for i in range(5):
        if i in cut_inputs:
            inputs_used[i] = False
    inputs_used_as_binary = "".join(["1" if i else "0" for i in inputs_used])
    inputs_used_as_base10 = int(inputs_used_as_binary, 2)
    name_base = f"ShowerFlow_{version}_nb{num_blocks}_inputs{inputs_used_as_base10}"
    nice_name = f"{version}_nb{num_blocks}"
    if getattr(configs, "shower_flow_fixed_input_norms", False):
        name_base += "_fnorms"
        nice_name += "_fnorms"
    best_model_path = os.path.join(showerflow_dir, f"{name_base}_best.pth")
    best_data_path = os.path.join(showerflow_dir, f"{name_base}_best_data.txt")

    if cut_inputs:
        nice_name += f"_wo{cut_inputs}"

    return nice_name, best_model_path, best_data_path


def get_cond_mask(configs):
    max_cond_inputs = 4
    inputs_used = np.zeros(max_cond_inputs, dtype=bool)
    if "energy" in configs.shower_flow_cond_features:
        inputs_used[0] = True
    if "p_norm_local" in configs.shower_flow_cond_features:
        inputs_used[1:] = True
    return inputs_used


def get_input_mask(configs):
    max_input_dims = 65
    inputs_used = np.zeros(max_input_dims, dtype=bool)
    if "total_clusters" in configs.shower_flow_inputs:
        inputs_used[0] = True
    if "total_energy" in configs.shower_flow_inputs:
        inputs_used[1] = True
    for c in "xyz":
        if f"cog_{c}" in configs.shower_flow_inputs:
            inputs_used[2 + "xyz".index(c)] = True
    if "clusters_per_layer" in configs.shower_flow_inputs:
        inputs_used[5:35] = True
    if "energy_per_layer" in configs.shower_flow_inputs:
        inputs_used[35:] = True
    return inputs_used


def existing_models(configs):
    saved_models = {}
    saved_models["versions"] = []
    saved_models["names"] = []
    saved_models["num_blocks"] = []
    saved_models["cut_inputs"] = []
    saved_models["best_loss"] = []
    saved_models["paths"] = []
    saved_models["cond_features"] = []
    saved_models["fixed_input_norms"] = []

    for version in versions_dict:
        for nb in range(1, 100):
            for ci in ["", "01", "014", "01234"]:
                name, model_path, data_path = model_save_paths(configs, version, nb, ci)
                fixed_input_norms = getattr(
                    configs, "shower_flow_fixed_input_norms", False
                )
                if not os.path.exists(model_path):
                    # print(f"Skipping {name}")
                    continue
                saved_models["paths"].append(model_path)
                saved_models["names"].append(name)
                saved_models["versions"].append(version)
                saved_models["num_blocks"].append(nb)
                saved_models["cut_inputs"].append(ci)
                saved_models["cond_features"].append(configs.shower_flow_cond_features)
                saved_models["fixed_input_norms"].append(fixed_input_norms)
                with open(data_path, "r") as f:
                    text = f.read().split()
                    saved_models["best_loss"].append(float(text[0]))
                if nb == 4:
                    print(f"{name} has best loss {saved_models['best_loss'][-1]}")
    names = saved_models["names"]
    print(f"Found {len(names)} saved models")

    return saved_models


def construct_configs(config_base, saved_models, idx):
    configs = copy.deepcopy(config_base)
    configs.model_name = "shower_flow"
    configs.shower_flow_version = saved_models["versions"][idx]
    configs.shower_flow_cond_features = saved_models["cond_features"][idx]
    shower_flow_inputs = [
        "total_clusters",
        "total_energy",
        "cog_x",
        "cog_y",
        "cog_z",
        "clusters_per_layer",
        "energy_per_layer",
    ]
    cut_ints = [int(c) for c in saved_models["cut_inputs"][idx]]
    cut_ints = sorted(cut_ints)
    for c in cut_ints[::-1]:
        del shower_flow_inputs[c]
    configs.shower_flow_inputs = shower_flow_inputs
    configs.shower_flow_num_blocks = saved_models["num_blocks"][idx]
    configs.shower_flow_fixed_input_norms = saved_models["fixed_input_norms"][idx]
    return configs


def truescale_showerflow_output(samples, config):
    # check what inputs are expected
    inputs_mask = get_input_mask(config)
    bs = samples.shape[0]
    metadata = Metadata(config)
    # name samples
    reached = 0
    if inputs_mask[0]:
        num_clusters = np.clip(
            (samples[:, 0] * metadata.n_pts_rescale).reshape(bs, 1),
            1,
            config.max_points,
        )
        reached += 1
    else:
        num_clusters = None

    gev_to_mev = 1000
    if inputs_mask[1]:
        energies = (
            samples[:, reached] * metadata.vis_eng_rescale * gev_to_mev
        ).reshape(bs, 1)
        # in MeV  (clip to a minimum energy of 40 MeV)
        energies = np.clip(energies, 40, None)
        reached += 1
    else:
        energies = None

    cogs = []
    for i in range(3):
        if inputs_mask[2 + i]:
            cog = (samples[:, reached] * metadata.std_cog[i]) + metadata.mean_cog[i]
            cogs.append(cog)
            reached += 1
        else:
            cogs.append(None)
    cog_x, cog_y, cog_z = cogs

    n_layers = len(metadata.layer_bottom_pos_hdf5)
    cluster_start = np.sum(inputs_mask[:5])
    cluster_end = np.sum(inputs_mask[: 5 + n_layers])
    if np.any(inputs_mask[5 : 5 + n_layers]):
        clusters_per_layer_gen = samples[:, cluster_start:cluster_end]
        if getattr(config, "shower_flow_fixed_input_norms", False):
            clusters_per_layer_gen *= metadata.n_pts_rescale / n_layers
            clusters_per_layer_gen = np.clip(clusters_per_layer_gen, 0, None)
        else:
            clusters_per_layer_gen = np.clip(clusters_per_layer_gen, 0, 1)
    else:
        clusters_per_layer_gen = None
    if np.any(inputs_mask[35:]):
        e_per_layer_gen = samples[:, cluster_end:]
        if getattr(config, "shower_flow_fixed_input_norms", False):
            e_per_layer_gen *= gev_to_mev * metadata.vis_eng_rescale / n_layers
            e_per_layer_gen = np.clip(e_per_layer_gen, 0, None)
        else:
            e_per_layer_gen = np.clip(e_per_layer_gen, 0, 1)
    else:
        e_per_layer_gen = None
    return (
        num_clusters,
        energies,
        cog_x,
        cog_y,
        cog_z,
        clusters_per_layer_gen,
        e_per_layer_gen,
    )
