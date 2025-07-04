# # Showerflow
#
# utilty functions for scripts/Showerflow.py

import torch
import numpy as np
import os

from pointcloud.utils.metadata import Metadata
from pointcloud.data.read_write import get_n_events, read_raw_regaxes
from pointcloud.utils.detector_map import floors_ceilings
from pointcloud.utils.gen_utils import get_cog as cog_from_kinematics

from ..data.conditioning import get_cond_features_names


def get_incident_npts_visible(
    config, showerflow_dir, redo=False, local_batch_size=10_000
):
    """
    Save and return the incident energy and the number of
    visible points in the shower for all showers in the dataset.
    If found on disk, will just return.

    Parameters
    ----------
    config : pointcloud.configs.Configs
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
    meta = Metadata(config)
    pointsE_path = os.path.join(showerflow_dir, "pointsE.npz")
    if os.path.exists(pointsE_path) and not redo:
        print("Using precaluclated energies and counts", flush=True)
    else:
        print("Recalculating energies and counts", flush=True)
        n_events = np.sum(get_n_events(config.dataset_path, config.n_dataset_files))
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
                config, pick_events=my_slice
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


def get_gun_direction(config, showerflow_dir, redo=False, local_batch_size=10_000):
    """
    Save and return the unti vector for the gun direction
    for all showers in the dataset. If found on disk, will just return.

    Parameters
    ----------
    config : pointcloud.configs.Configs
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
        n_events = np.sum(get_n_events(config.dataset_path, config.n_dataset_files))
        gun_direction = np.zeros((n_events, 3))
        try:
            for start_idx in range(0, n_events, local_batch_size):
                print(f"{start_idx/n_events:.0%}", end="\r")
                my_slice = slice(start_idx, start_idx + local_batch_size)
                per_event_batch, _ = read_raw_regaxes(
                    config, pick_events=my_slice, per_event_cols=["p_norm_local"]
                )
                gun_direction[my_slice] = per_event_batch
        except KeyError:
            print(
                "This file doesn't record p_norm_local - likely a fixed gun direction"
            )
            print("The gun will be assumed to always point in the z direction")
            gun_direction[:, 2] = 1
        np.save(
            direction_path,
            gun_direction,
        )
        print(f"Direction includes {gun_direction[0]}")
    return direction_path


def get_clusters_per_layer(config, showerflow_dir, redo=False, local_batch_size=10_000):
    """
    Save and return the number of clusters in each layer for all showers
    in the dataset. If found on disk, will just return.

    Parameters
    ----------
    config : pointcloud.configs.Configs
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
    meta = Metadata(config)
    n_events = np.sum(get_n_events(config.dataset_path, config.n_dataset_files))
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
            _, events_batch = read_raw_regaxes(config, pick_events=my_slice)
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


def get_energy_per_layer(config, showerflow_dir, redo=False, local_batch_size=10_000):
    """
    Save and return the observed total energy in each layer
    for all showers in the dataset.

    Parameters
    ----------
    config : pointcloud.configs.Configs
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
    meta = Metadata(config)
    n_events = np.sum(get_n_events(config.dataset_path, config.n_dataset_files))
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
            _, events_batch = read_raw_regaxes(config, pick_events=my_slice)
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


def get_cog(config, showerflow_dir, redo=False, local_batch_size=10_000):
    """
    Save and return the number of center of gravity of each shower
    in the dataset. If found on disk, will just return.
    Also returns a sample equal to the local_batch_size.

    Parameters
    ----------
    config : pointcloud.configs.Configs
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
    n_events = np.sum(get_n_events(config.dataset_path, config.n_dataset_files))
    cog_path = os.path.join(showerflow_dir, "cog.npy")
    if os.path.exists(cog_path) and not redo:
        print("Using precaluclated cog", flush=True)
        my_slice = slice(0, local_batch_size)
        _, events_batch = read_raw_regaxes(config, pick_events=my_slice)
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
            _, events_batch = read_raw_regaxes(config, pick_events=my_slice)
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


def _train_ds_function_factory_args(*args, **kwargs):
    input_args = {
        "pointsE_path": None,  # optional
        "cog_path": None,  # optional
        "clusters_per_layer_path": None,  # optional
        "energy_per_layer_path": None,  # optional
        "config": None,  # required
        "direction_path": None,  # optional
        "showerflow_dir": None,  # required if we miss the paths
    }
    status = f"args given: {args}, kwargs given: {kwargs}"
    for key in kwargs:
        if key in input_args:
            input_args[key] = kwargs[key]
        else:
            raise TypeError(f"{status}; Got unexpected keyword argument {key}")
    for i, key in enumerate(input_args):
        if not key.endswith("_path"):
            break
        if len(args) <= i or not isinstance(args[i], str):
            break  # no longer a path
        assert input_args[key] is None, status
        input_args[key] = args[i]
    if input_args["config"] is None:
        if len(args) < i:
            raise ValueError(
                f"{status}; Must provide a config, either as a kwarg or as a positional arg"
            )
        assert not isinstance(args[i], str), f"{status}; {i} Expected a config object"
        input_args["config"] = args[i]
        i += 1
    if input_args["direction_path"] is None:
        if len(args) > i:
            assert isinstance(args[i], str), f"{status}; {i} Expected a path"
            input_args["direction_path"] = args[i]
            i += 1
    if input_args["showerflow_dir"] is None:
        if len(args) > i:
            assert isinstance(args[i], str), f"{status}; {i} Expected a path"
            input_args["showerflow_dir"] = args[i]
        elif None in [
            input_args["pointsE_path"],
            input_args["cog_path"],
            input_args["clusters_per_layer_path"],
            input_args["energy_per_layer_path"],
        ]:
            raise ValueError(
                f"{status}; Must provide a showerflow_dir, either as a kwarg or as a positional"
                " arg, or provide paths to all of pointsE_path, cog_path, "
                "clusters_per_layer_path, and energy_per_layer_path"
            )
    return input_args


def train_ds_function_factory(*args, **kwargs):
    input_args = _train_ds_function_factory_args(*args, **kwargs)
    config = input_args["config"]
    showerflow_dir = input_args["showerflow_dir"]
    if input_args["pointsE_path"] is None:
        input_args["pointsE_path"] = get_incident_npts_visible(config, showerflow_dir)
    if input_args["cog_path"] is None:
        input_args["cog_path"], _ = get_cog(config, showerflow_dir)
    if input_args["clusters_per_layer_path"] is None:
        input_args["clusters_per_layer_path"] = get_clusters_per_layer(
            config, showerflow_dir
        )
    if input_args["energy_per_layer_path"] is None:
        input_args["energy_per_layer_path"] = get_energy_per_layer(
            config, showerflow_dir
        )
    if input_args["direction_path"] is None and showerflow_dir is not None:
        input_args["direction_path"] = get_gun_direction(config, showerflow_dir)
    del input_args["showerflow_dir"]

    return _train_ds_function_factory(**input_args)


def _train_ds_function_factory(
    pointsE_path,
    cog_path,
    clusters_per_layer_path,
    energy_per_layer_path,
    config,
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
    config : pointcloud.configs.Configs
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
    meta = Metadata(config)
    device = config.device

    if getattr(config, "shower_flow_fixed_input_norms", False):
        clusters_per_layer_key = "clusters_per_layer"
        energy_per_layer_key = "energy_per_layer"
        n_layers = len(meta.layer_bottom_pos_hdf5)
        clusters_per_layer_norm = n_layers / meta.n_pts_rescale
        energy_per_layer_norm = n_layers / meta.vis_eng_rescale
    else:
        clusters_per_layer_key = "rescaled_clusters_per_layer"
        energy_per_layer_key = "rescaled_energy_per_layer"
        clusters_per_layer_norm = 1.0
        energy_per_layer_norm = 1.0

    condition_energy = "energy" in get_cond_features_names(config, "showerflow")
    condition_direction = "p_norm_local" in get_cond_features_names(
        config, "showerflow"
    )

    min_cond_energy = config.shower_flow_min_cond_energy
    min_event_points = config.shower_flow_min_train_points

    def make_train_ds(start_idx, end_idx):
        my_slice = slice(start_idx, end_idx)
        # mem-map the files to avoid loading all data
        pointE = np.load(pointsE_path, mmap_mode="r")

        # for conditioning
        output = []
        energy = torch.tensor(pointE["energy"][my_slice])
        num_points = torch.tensor(pointE["num_points"][my_slice])
        if min_cond_energy or min_event_points:
            mask = energy > min_cond_energy and num_points >= min_event_points
            energy = energy[mask]
            my_slice = [i for i, m in enumerate(mask, start_idx) if m]
        if condition_energy:
            energy = energy / meta.incident_rescale
            energy = energy.view(-1, 1).to(device).float()
            output.append(energy)
        if condition_direction:
            assert direction_path is not None
            direction = torch.tensor(np.load(direction_path, mmap_mode="r")[my_slice])
            direction = direction.to(device).float()
            output.append(direction)

        # for predicted values
        if "total_clusters" in config.shower_flow_inputs:
            num_points = (
                num_points / meta.n_pts_rescale
            )
            num_points = num_points.view(-1, 1).to(device).float()
            output.append(num_points)
        if "total_energy" in config.shower_flow_inputs:
            visible_energy = (
                torch.tensor(pointE["visible_energy"][my_slice]) / meta.vis_eng_rescale
            )
            visible_energy = visible_energy.view(-1, 1).to(device).float()
            output.append(visible_energy)

        normed_cog = np.load(cog_path, mmap_mode="r")[my_slice].copy()
        normed_cog = (normed_cog - meta.mean_cog) / meta.std_cog

        if "cog_x" in config.shower_flow_inputs:
            cog_x = torch.tensor(normed_cog[:, 0])
            cog_x = cog_x.view(-1, 1).to(device).float()
            output.append(cog_x)
        if "cog_y" in config.shower_flow_inputs:
            cog_y = torch.tensor(normed_cog[:, 1])
            cog_y = cog_y.view(-1, 1).to(device).float()
            output.append(cog_y)
        if "cog_z" in config.shower_flow_inputs:
            cog_z = torch.tensor(normed_cog[:, 2])
            cog_z = cog_z.view(-1, 1).to(device).float()
            output.append(cog_z)

        if "clusters_per_layer" in config.shower_flow_inputs:
            clusters = np.load(clusters_per_layer_path, mmap_mode="r")
            clusters_per_layer = (
                torch.tensor(clusters[clusters_per_layer_key][my_slice]).to(device)
                * clusters_per_layer_norm
            )
            output.append(clusters_per_layer)
        if "energy_per_layer" in config.shower_flow_inputs:
            energies = np.load(energy_per_layer_path, mmap_mode="r")
            e_per_layer = (
                torch.tensor(energies[energy_per_layer_key][my_slice]).to(device)
                * energy_per_layer_norm
            )
            output.append(e_per_layer)

        for values in output:
            assert values.shape[0] <= end_idx - start_idx
            assert not torch.any(torch.isnan(values))

        output = torch.cat(output, 1)
        return output

    return make_train_ds
