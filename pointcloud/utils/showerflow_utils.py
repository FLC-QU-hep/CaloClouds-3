import os
import numpy as np
import copy
import warnings
import itertools

from pointcloud.data.naming import dataset_name_from_path
from pointcloud.models.shower_flow import versions_dict
from .metadata import Metadata
from ..data.conditioning import get_cond_features_names


def get_data_dir(config, last_resort="/home/{}/Data/"):
    user_name = os.getenv("USER", os.getenv("USERNAME", ""))
    second_choice = os.path.join(config.storage_base, "point-cloud-diffusion-data")
    data_dir = getattr(config, "shower_flow_data_dir", second_choice)
    if not os.path.exists(data_dir):
        data_dir = os.path.join(
            config.storage_base, user_name, "point-cloud-diffusion-data"
        )
    if not os.path.exists(data_dir):
        data_dir = last_resort.format(user_name)
    return data_dir


def get_showerflow_dir(config):
    dataset_path = config.dataset_path
    base_path = get_data_dir(config)
    dataset_name_key = dataset_name_from_path(dataset_path)

    showerflow_dir = os.path.join(base_path, "showerFlow", dataset_name_key)
    return showerflow_dir


def model_save_paths(config, version, num_blocks, cut_inputs):
    showerflow_dir = get_showerflow_dir(config)
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
    if getattr(config, "shower_flow_fixed_input_norms", False):
        name_base += "_fnorms"
        nice_name += "_fnorms"

    if getattr(config, "shower_flow_train_base", False):
        name_base += "_tbase"
        nice_name += "_tbase"

    if getattr(config, "shower_flow_detailed_history", False):
        name_base += "_dhist"
        nice_name += "_dhist"

    if getattr(config, "shower_flow_weight_decay", 0):
        wd_string = f"{config.shower_flow_weight_decay:.1e}".replace(".", "p")
        name_base += f"_wd{wd_string}"
        nice_name += f"_wd{wd_string}"

    best_model_path = os.path.join(showerflow_dir, f"{name_base}_best.pth")
    best_data_path = os.path.join(showerflow_dir, f"{name_base}_best_data.txt")

    if cut_inputs:
        nice_name += f"_wo{cut_inputs}"

    return nice_name, best_model_path, best_data_path


def get_cond_mask(config):
    max_cond_inputs = 4
    inputs_used = np.zeros(max_cond_inputs, dtype=bool)
    names = get_cond_features_names(config, "showerflow")
    inputs_used[0] = "energy" in names
    inputs_used[1:] = "p_norm_local" in names
    return inputs_used


def get_input_mask(config):
    max_input_dims = 65
    inputs_used = np.zeros(max_input_dims, dtype=bool)
    if "total_clusters" in config.shower_flow_inputs:
        inputs_used[0] = True
    if "total_energy" in config.shower_flow_inputs:
        inputs_used[1] = True
    for c in "xyz":
        if f"cog_{c}" in config.shower_flow_inputs:
            inputs_used[2 + "xyz".index(c)] = True
    if "clusters_per_layer" in config.shower_flow_inputs:
        inputs_used[5:35] = True
    if "energy_per_layer" in config.shower_flow_inputs:
        inputs_used[35:] = True
    return inputs_used


def input_mask_to_list(input_mask):
    max_input_dims = 65
    assert len(input_mask) == max_input_dims
    inputs = []
    if input_mask[0]:
        inputs.append("total_clusters")
    if input_mask[1]:
        inputs.append("total_energy")
    for i, c in enumerate("xyz"):
        if input_mask[2 + i]:
            inputs.append(f"cog_{c}")
    if np.all(input_mask[5:35]):
        inputs.append("clusters_per_layer")
    elif np.any(input_mask[5:35]):
        raise NotImplementedError("Cannot handle partial layer inputs")
    if np.all(input_mask[35:]):
        inputs.append("energy_per_layer")
    elif np.any(input_mask[35:]):
        raise NotImplementedError("Cannot handle partial layer inputs")
    return inputs


def existing_models(config):
    saved_models = {}
    saved_models["versions"] = []
    saved_models["names"] = []
    saved_models["num_blocks"] = []
    saved_models["cut_inputs"] = []
    saved_models["best_loss"] = []
    saved_models["paths"] = []
    saved_models["cond_features"] = []
    saved_models["weight_decay"] = []
    saved_models["fixed_input_norms"] = []

    combinations = itertools.product(
        versions_dict.keys(), range(1, 100), ["", "01", "014", "01234"]
    )

    for version, nb, ci in combinations:
        name, model_path, data_path = model_save_paths(config, version, nb, ci)
        fixed_input_norms = getattr(config, "shower_flow_fixed_input_norms", False)
        weight_decay = getattr(config, "shower_flow_weight_decay", 0)
        if not os.path.exists(model_path):
            continue
        cond_feature_names = get_cond_features_names(config, "showerflow")
        saved_models["paths"].append(model_path)
        saved_models["names"].append(name)
        saved_models["versions"].append(version)
        saved_models["num_blocks"].append(nb)
        saved_models["cut_inputs"].append(ci)
        saved_models["cond_features"].append(cond_feature_names)
        saved_models["fixed_input_norms"].append(fixed_input_norms)
        saved_models["weight_decay"].append(weight_decay)
        with open(data_path, "r") as f:
            text = f.read().split()
            saved_models["best_loss"].append(float(text[0]))
        if nb == 4:
            print(f"{name} has best loss {saved_models['best_loss'][-1]}")
    names = saved_models["names"]
    print(f"Found {len(names)} saved models")
    return saved_models


def models_at_paths(cond_features, check_paths):
    saved_models = {}
    saved_models["versions"] = []
    saved_models["names"] = []
    saved_models["num_blocks"] = []
    saved_models["cut_inputs"] = []
    saved_models["best_loss"] = []
    saved_models["paths"] = []
    saved_models["cond_features"] = []
    saved_models["fixed_input_norms"] = []

    for model_path in check_paths:
        if isinstance(model_path, str):
            name = os.path.basename(model_path).split("ShowerFlow_")[1].split(".pth")[0]
            data_path = model_path.replace(".pth", "_data.txt")
        else:
            assert len(model_path) == 3
            name, model_path, data_path = model_path

        if not os.path.exists(model_path):
            warnings.warn(
                f"Path {model_path} given to showerflow_utils.models_at_paths but does not exist"
            )
        version, nb, ci, _, fixed_input_norms = config_params_from_showerflow_path(
            model_path
        )
        saved_models["paths"].append(model_path)
        saved_models["names"].append(name)
        saved_models["versions"].append(version)
        saved_models["num_blocks"].append(nb)
        saved_models["cut_inputs"].append(ci)
        saved_models["cond_features"].append(cond_features)
        saved_models["fixed_input_norms"].append(fixed_input_norms)
        if not os.path.exists(data_path):
            warnings.warn(f"Data path {data_path} does not exist")
            saved_models["best_loss"].append(np.nan)
        else:
            with open(data_path, "r") as f:
                text = f.read().split()
                saved_models["best_loss"].append(float(text[0]))

    return saved_models


def config_params_from_showerflow_path(showerflow_path):
    # otherwise, we can guess from the path
    base_name = os.path.basename(showerflow_path)
    assert base_name.startswith(
        "ShowerFlow_"
    ), f"Base name of path {showerflow_path} does not start with 'ShowerFlow_'"
    # we can reverse engineer the path
    parts = base_name.split(".pth")[0].split("_")
    version = parts[1]
    num_blocks = int(next(part[2:] for part in parts if part.startswith("nb")))
    inputs_used_as_base10 = int(
        next(part[6:] for part in parts if part.startswith("inputs"))
    )
    inputs_used_mask = [x == "1" for x in f"{inputs_used_as_base10:b}"]
    max_input_dims = 65
    if len(inputs_used_mask) < max_input_dims:
        # pad at the start
        inputs_used_mask = np.pad(inputs_used_mask, (max_input_dims - len(inputs_used_mask), 0))
    inputs = input_mask_to_list(inputs_used_mask)
    fixed_input_norms = "fnorms" in parts
    cut_inputs = "".join([str(i) for i in range(5) if not inputs_used_mask[i]])
    return version, num_blocks, cut_inputs, inputs, fixed_input_norms


def config_from_showerflow_path(config, showerflow_path):
    # get existing models
    saved_models = existing_models(config)
    # check if the path is in the list
    if showerflow_path in saved_models["paths"]:
        idx = saved_models["paths"].index(showerflow_path)
        config = construct_config(config, saved_models, idx)
        return config
    # try just the base name
    base_name = os.path.basename(showerflow_path)
    found_base_names = [os.path.basename(p) for p in saved_models["paths"]]
    if found_base_names.count(base_name) == 1:  # unique match
        idx = found_base_names.index(base_name)
        config = construct_config(config, saved_models, idx)
        return config

    warnings.warn(
        f"Did not generate the path {showerflow_path} when looking for existing models."
        " This may indicate there is something unusual about the path."
        " Deducing the config by processing the file name."
    )
    version, num_blocks, cut_inputs, inputs, fixed_input_norms = (
        config_params_from_showerflow_path(showerflow_path)
    )
    # otherwise, we can guess from the path
    config = copy.deepcopy(config)
    config.shower_flow_version = version
    config.shower_flow_num_blocks = num_blocks
    config.shower_flow_inputs = inputs
    config.shower_flow_fixed_input_norms = fixed_input_norms
    return config


def construct_config(config_base, saved_models, idx):
    config = copy.deepcopy(config_base)
    config.model_name = "shower_flow"
    config.shower_flow_version = saved_models["versions"][idx]
    config.shower_flow_cond_features = saved_models["cond_features"][idx]
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
    config.shower_flow_inputs = shower_flow_inputs
    config.shower_flow_num_blocks = saved_models["num_blocks"][idx]
    config.shower_flow_fixed_input_norms = saved_models["fixed_input_norms"][idx]
    return config


def truescale_showerflow_output(samples, config):
    # check what inputs are expected
    inputs_mask = get_input_mask(config)
    bs = samples.shape[0]
    # if config has a metadata attr, that would be pulled here
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
