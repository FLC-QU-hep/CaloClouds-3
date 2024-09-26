# # Showerflow
#
# utilty functions for scripts/Showerflow.py

import pandas as pd
import torch
import numpy as np
import os

from pointcloud.utils.metadata import Metadata
from pointcloud.data.read_write import get_n_events, read_raw_regaxes
from pointcloud.utils.detector_map import floors_ceilings
from pointcloud.utils.gen_utils import get_cog as cog_from_kinematics
from pointcloud.models.shower_flow import versions_dict


def get_data_dir(configs):
    data_dir = os.path.join(configs.storage_base, "dayhallh/point-cloud-diffusion-data")
    if not os.path.exists(data_dir):
        data_dir = os.path.join(configs.storage_base, "point-cloud-diffusion-data")
    if not os.path.exists(data_dir):
        data_dir = "/home/dayhallh/Data/"
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
    for i in range(5):
        if str(i) in cut_inputs:
            inputs_used[i] = False
    inputs_used_as_binary = "".join(["1" if i else "0" for i in inputs_used])
    inputs_used_as_base10 = int(inputs_used_as_binary, 2)
    name_base = f"ShowerFlow_{version}_nb{num_blocks}_inputs{inputs_used_as_base10}"
    best_model_path = os.path.join(showerflow_dir, f"{name_base}_best.pth")
    best_data_path = os.path.join(showerflow_dir, f"{name_base}_best_data.txt")

    nice_name = f"{version}_nb{num_blocks}"
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
    versions = []
    names = []
    num_blocks = []
    cut_inputs = []
    best_loss = []
    paths = []
    for version in versions_dict:
        for nb in range(1, 100):
            for ci in ["", "01", "01234"]:
                name, model_path, data_path = model_save_paths(configs, version, nb, ci)
                if not os.path.exists(model_path):
                    # print(f"Skipping {name}")
                    continue
                paths.append(model_path)
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

    saved_models = {
        "names": names,
        "version": versions,
        "num_blocks": num_blocks,
        "cut_inputs": cut_inputs,
        "best_loss": best_loss,
        "path": paths,
    }
    return saved_models


def get_incident_npts_visible(
    configs, showerflow_dir, redo=False, local_batch_size=10_000
):
    """
    Save and return the incident energy and the number of
    visible points in the shower for all showers in the dataset.
    If found on disk, will just return.

    Parameters
    ----------
    configs : pointcloud.configs.Configs
        Description of setup, including location of dataset.
    showerflow_dir : str
        Location to store the data, should exist.
    redo : bool, optional
        If `True`, will recalculate the data even if it is already
        found in `showerflow_dir`. The default is False.
    local_batch_size : int, optional
        Number of events to process at once. Should fit in memory.
        The default is 10_000.

    Returns
    -------
    pointsE_path : str
        Path to the file containing the incident energy and number of
        visible points in the shower for all showers in the dataset.
    """
    assert os.path.exists(showerflow_dir)
    meta = Metadata(configs)
    pointsE_path = os.path.join(showerflow_dir, "pointsE.npz")
    if os.path.exists(pointsE_path) and not redo:
        print("Using precaluclated energies and counts", flush=True)
    else:
        print("Recalculating energies and counts", flush=True)
        n_events = np.sum(get_n_events(configs.dataset_path, configs.n_dataset_files))
        floors, ceilings = floors_ceilings(
            meta.layer_bottom_pos_hdf5, meta.cell_thickness_hdf5
        )
        energy = np.zeros(n_events)
        num_points = np.zeros(n_events)
        visible_energy = np.zeros(n_events)
        for start_idx in range(0, n_events, local_batch_size):
            print(f"{start_idx/n_events:.0%}", end="\r")
            my_slice = slice(start_idx, start_idx + local_batch_size)
            energies_batch, events_batch = read_raw_regaxes(
                configs, pick_events=my_slice
            )
            energy[my_slice] = energies_batch
            num_points[my_slice] = (events_batch[:, :, 3] > 0).sum(axis=1)
            visible_energy[my_slice] = (events_batch[:, :, 3]).sum(axis=1)
        np.savez(
            pointsE_path,
            energy=energy,
            num_points=num_points,
            visible_energy=visible_energy,
        )
        print(f"Energy goes from {visible_energy.min()}, to {visible_energy.max()}")
        print(f"Num points goes from {num_points.min()}, to {num_points.max()}")
    return pointsE_path


def get_gun_direction(configs, showerflow_dir, redo=False, local_batch_size=10_000):
    """
    Save and return the unti vector for the gun direction for all showers in the dataset.
    If found on disk, will just return.

    Parameters
    ----------
    configs : pointcloud.configs.Configs
        Description of setup, including location of dataset.
    showerflow_dir : str
        Location to store the data, should exist.
    redo : bool, optional
        If `True`, will recalculate the data even if it is already
        found in `showerflow_dir`. The default is False.
    local_batch_size : int, optional
        Number of events to process at once. Should fit in memory.
        The default is 10_000.

    Returns
    -------
    direction_path : str
        Path to the file containing the unit vector for the gun direction
        for all showers in the dataset.
    """
    assert os.path.exists(showerflow_dir)
    direction_path = os.path.join(showerflow_dir, "direction.npy")
    if os.path.exists(direction_path) and not redo:
        print("Using precaluclated gun direction", flush=True)
    else:
        print("Recalculating gun direction", flush=True)
        n_events = np.sum(get_n_events(configs.dataset_path, configs.n_dataset_files))
        gun_direction = np.zeros((n_events, 3))
        for start_idx in range(0, n_events, local_batch_size):
            print(f"{start_idx/n_events:.0%}", end="\r")
            my_slice = slice(start_idx, start_idx + local_batch_size)
            per_event_batch, _ = read_raw_regaxes(
                configs, pick_events=my_slice, per_event_cols=["p_norm_local"]
            )
            gun_direction[my_slice] = per_event_batch
        np.save(
            direction_path,
            gun_direction,
        )
        print(f"Direction includes {gun_direction[0]}")
    return direction_path


def get_clusters_per_layer(
    configs, showerflow_dir, redo=False, local_batch_size=10_000
):
    """
    Save and return the number of clusters in each layer for all showers
    in the dataset. If found on disk, will just return.

    Parameters
    ----------
    configs : pointcloud.configs.Configs
        Description of setup, including location of dataset.
    showerflow_dir : str
        Location to store the data.
    redo : bool, optional
        If `True`, will recalculate the data even if it is already
        found in `showerflow_dir`. The default is False.
    local_batch_size : int, optional
        Number of events to process at once. Should fit in memory.
        The default is 10_000.

    Returns
    -------
    clusters_per_layer_path : str
        Path to the file containing the number of clusters in each layer
        for all showers in the dataset.
    """
    assert os.path.exists(showerflow_dir)
    meta = Metadata(configs)
    n_events = np.sum(get_n_events(configs.dataset_path, configs.n_dataset_files))
    clusters_per_layer_path = os.path.join(showerflow_dir, "clusters_per_layer.npz")
    if os.path.exists(clusters_per_layer_path) and not redo:
        print("Using precaluclated clusters per layer", flush=True)
    else:
        print("Recalculating clusters per layer", flush=True)
        floors, ceilings = floors_ceilings(
            meta.layer_bottom_pos_hdf5, meta.cell_thickness_hdf5
        )
        clusters_per_layer = np.zeros((n_events, len(floors)))
        for start_idx in range(0, n_events, local_batch_size):
            print(f"{start_idx/n_events:.0%}", end="\r")
            my_slice = slice(start_idx, start_idx + local_batch_size)
            _, events_batch = read_raw_regaxes(configs, pick_events=my_slice)
            mask = events_batch[:, :, 3] > 0
            clusters_here = [
                ((events_batch[:, :, 2] < c) & (events_batch[:, :, 2] > f) & mask).sum(
                    axis=1
                )
                for f, c in zip(floors, ceilings)
            ]
            clusters_per_layer[my_slice] = np.vstack(clusters_here).T
        rescales = clusters_per_layer / clusters_per_layer.max(axis=1)[:, np.newaxis]
        np.savez(
            clusters_per_layer_path,
            clusters_per_layer=clusters_per_layer,
            rescaled_clusters_per_layer=rescales,
        )
        assert np.all(~np.isnan(clusters_per_layer))
    return clusters_per_layer_path


def get_energy_per_layer(configs, showerflow_dir, redo=False, local_batch_size=10_000):
    """
    Save and return the observed total energy in each layer
    for all showers in the dataset.

    Parameters
    ----------
    configs : pointcloud.configs.Configs
        Description of setup, including location of dataset.
    showerflow_dir : str
        Location to store the data.
    redo : bool, optional
        If `True`, will recalculate the data even if it is already
        found in `showerflow_dir`. The default is False.
    local_batch_size : int, optional
        Number of events to process at once. Should fit in memory.
        The default is 10_000.

    Returns
    -------
    energy_per_layer_path : str
        Path to the file containing the observed total energy in each layer
        for all showers in the dataset.

    """
    assert os.path.exists(showerflow_dir)
    meta = Metadata(configs)
    n_events = np.sum(get_n_events(configs.dataset_path, configs.n_dataset_files))
    energy_per_layer_path = os.path.join(showerflow_dir, "energy_per_layer.npz")
    if os.path.exists(energy_per_layer_path) and not redo:
        print("Using precaluclated energy per layer", flush=True)
    else:
        print("Recalculating energy per layer", flush=True)
        floors, ceilings = floors_ceilings(
            meta.layer_bottom_pos_hdf5, meta.cell_thickness_hdf5
        )
        energy_per_layer = np.zeros((n_events, len(floors)))
        for start_idx in range(0, n_events, local_batch_size):
            print(f"{start_idx/n_events:.0%}", end="\r")
            my_slice = slice(start_idx, start_idx + local_batch_size)
            _, events_batch = read_raw_regaxes(configs, pick_events=my_slice)
            energy_here = [
                (
                    events_batch[..., 3]
                    * (events_batch[:, :, 2] < c)
                    * (events_batch[:, :, 2] > f)
                ).sum(axis=1)
                for f, c in zip(floors, ceilings)
            ]
            energy_per_layer[my_slice] = np.vstack(energy_here).T
        rescaled = energy_per_layer / energy_per_layer.max(axis=1)[:, np.newaxis]
        np.savez(
            energy_per_layer_path,
            energy_per_layer=energy_per_layer,
            rescaled_energy_per_layer=rescaled,
        )
        assert np.all(~np.isnan(energy_per_layer))
    return energy_per_layer_path


def get_cog(configs, showerflow_dir, redo=False, local_batch_size=10_000):
    """
    Save and return the number of center of gravity of each shower
    in the dataset. If found on disk, will just return.
    Also returns a sample equal to the local_batch_size.

    Parameters
    ----------
    configs : pointcloud.configs.Configs
        Description of setup, including location of dataset.
    showerflow_dir : str
        Location to store the data.
    redo : bool, optional
        If `True`, will recalculate the data even if it is already
        found in `showerflow_dir`. The default is False.
    local_batch_size : int, optional
        Number of events to process at once. Should fit in memory.
        The default is 10_000.

    Returns
    -------
    cog_path : str
        Path to the file containing the center of gravity of each shower
        for all showers in the dataset.
    cog_batch : np.ndarray
        A sample of the center of gravity of each shower for the first
        `local_batch_size` showers in the dataset.
    """
    assert os.path.exists(showerflow_dir)
    n_events = np.sum(get_n_events(configs.dataset_path, configs.n_dataset_files))
    cog_path = os.path.join(showerflow_dir, "cog.npy")
    if os.path.exists(cog_path) and not redo:
        print("Using precaluclated cog", flush=True)
        my_slice = slice(0, local_batch_size)
        _, events_batch = read_raw_regaxes(configs, pick_events=my_slice)
        cog_batch = cog_from_kinematics(
            events_batch[..., 0],
            events_batch[..., 1],
            events_batch[..., 2],
            events_batch[..., 3],
        )
    else:
        print("Recalculating cog", flush=True)
        cog = np.zeros((n_events, 3))
        for start_idx in range(0, n_events, local_batch_size):
            print(f"{start_idx/n_events:.0%}", end="\r")
            my_slice = slice(start_idx, start_idx + local_batch_size)
            _, events_batch = read_raw_regaxes(configs, pick_events=my_slice)
            cog_batch = cog_from_kinematics(
                events_batch[..., 0],
                events_batch[..., 1],
                events_batch[..., 2],
                events_batch[..., 3],
            )
            cog[my_slice] = np.vstack(cog_batch).T
        np.save(cog_path, cog)
        assert np.all(~np.isnan(cog))
    return cog_path, cog_batch


def train_ds_function_factory(
    pointsE_path,
    cog_path,
    clusters_per_layer_path,
    energy_per_layer_path,
    configs,
    direction_path=None,
):
    """
    Function factory that returns a function to create training datasets
    for selected index ranges of the dataset.

    Parameters
    ----------
    pointsE_path : str
        Path to the file containing the incident energy and number of
        visible points in the shower for all showers in the dataset.
    cog_path : str
        Path to the file containing the center of gravity of each shower
        for all showers in the dataset.
    clusters_per_layer_path : str
        Path to the file containing the number of clusters in each layer
        for all showers in the dataset.
    energy_per_layer_path : str
        Path to the file containing the observed total energy in each layer
        for all showers in the dataset.
    configs : pointcloud.configs.Configs
        Description of setup, including location of dataset.
    direction_path : str, optional
        If the gun direction is used for conditioning, the path to the
        file containing the unit vector for the gun direction for all showers
        in the dataset. The default is None.

    Returns
    -------
    make_train_ds : function
        Function that returns a training dataset for a given index range.
        Arguments are `start_idx` and `end_idx`.
    """
    meta = Metadata(configs)
    device = configs.device

    def make_train_ds(start_idx, end_idx):
        my_slice = slice(start_idx, end_idx)
        # mem-map the files to avoid loading all data
        pointE = np.load(pointsE_path, mmap_mode="r")

        # for conditioning
        output = []
        if "energy" in configs.shower_flow_cond_features:
            energy = torch.tensor(pointE["energy"][my_slice]) / meta.incident_rescale
            energy = energy.view(-1, 1).to(device).float()
            output.append(energy)
        if "p_norm_local" in configs.shower_flow_cond_features:
            assert direction_path is not None
            direction = torch.tensor(np.load(direction_path, mmap_mode="r")[my_slice])
            direction = direction.to(device).float()
            output.append(direction)

        # for predicted values
        if "total_clusters" in configs.shower_flow_inputs:
            num_points = (
                torch.tensor(pointE["num_points"][my_slice]) / meta.n_pts_rescale
            )
            num_points = num_points.view(-1, 1).to(device).float()
            output.append(num_points)
        if "total_energy" in configs.shower_flow_inputs:
            visible_energy = (
                torch.tensor(pointE["visible_energy"][my_slice]) / meta.vis_eng_rescale
            )
            visible_energy = visible_energy.view(-1, 1).to(device).float()
            output.append(visible_energy)

        normed_cog = np.load(cog_path, mmap_mode="r")[my_slice].copy()
        normed_cog = (normed_cog - meta.mean_cog) / meta.std_cog

        if "cog_x" in configs.shower_flow_inputs:
            cog_x = torch.tensor(normed_cog[:, 0])
            cog_x = cog_x.view(-1, 1).to(device).float()
            output.append(cog_x)
        if "cog_y" in configs.shower_flow_inputs:
            cog_y = torch.tensor(normed_cog[:, 1])
            cog_y = cog_y.view(-1, 1).to(device).float()
            output.append(cog_y)
        if "cog_z" in configs.shower_flow_inputs:
            cog_z = torch.tensor(normed_cog[:, 2])
            cog_z = cog_z.view(-1, 1).to(device).float()
            output.append(cog_z)

        if "clusters_per_layer" in configs.shower_flow_inputs:
            clusters = np.load(clusters_per_layer_path, mmap_mode="r")
            clusters_per_layer = torch.tensor(
                clusters["rescaled_clusters_per_layer"][my_slice]
            ).to(device)
            output.append(clusters_per_layer)
        if "energy_per_layer" in configs.shower_flow_inputs:
            energies = np.load(energy_per_layer_path, mmap_mode="r")
            e_per_layer = torch.tensor(
                energies["rescaled_energy_per_layer"][my_slice]
            ).to(device)
            output.append(e_per_layer)

        for values in output:
            assert values.shape[0] <= end_idx - start_idx
            assert not torch.any(torch.isnan(values))

        output = torch.cat(output, 1)
        return output

    return make_train_ds
