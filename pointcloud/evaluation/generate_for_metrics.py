"""
Create a merge_dict object for other plotting scripts to take as input.
"""
import argparse
import os
import gc

import torch
import pickle
import h5py

from ..configs import Configs
from ..utils.detector_map import get_projections, create_map
from ..utils import metrics, plotting
from ..utils.dataset import dataset_class_from_config
from ..utils.metadata import Metadata
from ..utils.misc import regularise_shower_axes

from .generate import load_flow_model, load_diffusion_model, generate_showers


def get_cli_args() -> argparse.Namespace:
    """
    Get the model name as a command line argument.

    Returns
    -------
    args : argparse.Namespace
        Namespace with the attribute `caloclouds` containing the model name.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--caloclouds",
        "-cc",
        type=str,
        default="cm",
        help="caloclouds model to use, choose from: ddpm, edm, cm, or g4",
    )
    args = parser.parse_args()
    return args


def make_params_dict(model_name: str = "cm") -> dict:
    """
    Generate the default param dict for the generation of shower metrics
    on a single model.
    It is expected that the user will modify this function to suit their needs.

    Returns
    -------
    params_dict : dict
        Dictionary containing the parameters for the generation of showers.
    """
    params_dict = {}
    params_dict["total_events"] = 500_000  # total events to process
    params_dict["n_events"] = 50_000  # in chunks of n_events
    params_dict["min_energy"] = 10
    params_dict["max_energy"] = 90
    params_dict[
        "pickle_path"
    ] = "/beegfs/desy/user/buhmae/6_PointCloudDiffusion/output/metrics/"

    # COMMON PARAMETERS
    params_dict["n_scaling"] = True  # default True
    params_dict["prefix"] = ""  # default ''
    # list of all high level features (docs of plotting.get_features)
    # - e_radial : array
    #       radial profile of the energy deposition
    # - e_sum : array
    #       total energy deposited in the detector
    # - hits : array
    #       energy deposited in each cell
    # - occ : array
    #       number of cells hit
    # - e_layers_distibution : array
    #       energy deposited in each layer
    # - e_layers : array
    #       average energy deposited in each layer
    # - e_layers_std : array
    #       standard deviation of the energy deposited in each layer
    # - occ_layers : array
    #       number of cells hit in each layer
    # - e_radial_lists : list
    #       list of radial profiles of the energy deposition for each layer
    # - hits_noThreshold : array
    #       energy deposited in each cell without threshold
    # - binned_layer_e : array
    #       binned energy deposited in each layer
    # - binned_radial_e : array
    #       binned radial profile of the energy deposition
    params_dict["key_exceptions"] = [
        "e_radial",
        "e_layers",
        "e_layers_distibution",
        "occ_layers",
        "e_radial_lists",
    ]  # not saved in dict

    params_dict["caloclouds"] = model_name

    if model_name == "ddpm":
        params_dict["seed"] = 12345
        params_dict["batch_size"] = 64
    elif model_name == "edm":
        params_dict["seed"] = 123456
        params_dict["batch_size"] = 64
    elif model_name == "cm":
        params_dict["seed"] = 1234567
        params_dict["batch_size"] = 16
    elif model_name == "g4":
        pass
    else:
        raise ValueError(f"Model name {model_name} not recognized.")
    return params_dict


def get_g4_data(path_or_config: str | Configs) -> tuple:
    """
    Get the validation G4 data for comparison to the simulated data.

    Parameters
    ----------
    path_or_config : str or Configs
        If str, the path to the G4 data. If Configs, the Configs object
        from which the storage base can be extracted.

    Returns
    -------
    all_events : lazy_ops.lazy_loading.DatasetViewh5py (n_events, max_num_hits, 4)
        The G4 events.
    all_energy : h5py._hl.dataset.Dataset (n_events, 1)
        The incident particle energy for the G4 events.
    """
    if isinstance(path_or_config, str):
        path = path_or_config
    else:
        storage_base = path_or_config.storage_base
        path = os.path.join(
            storage_base,
            "akorol/data/calo-clouds/hdf5/all_steps/validation",
            "10-90GeV_x36_grid_regular_712k.hdf5",
        )
    try:
        all_events = h5py.File(path, "r")["events"][:]
    except KeyError:
        all_events = h5py.File(path, "r")["event"][:]
    all_events = regularise_shower_axes(all_events, known_transposed=True)
    all_energy = h5py.File(path, "r")["energy"]
    return all_events, all_energy


# GENERATE EVENTS


def yield_g4_showers(
    config: Configs, param_dict: dict, g4_data: tuple[h5py.Group, h5py.Group]
) -> tuple:
    """
    Reformat the G4 data into the shower format that is produced by
    the generative models, and yield the showers in chunks of n_events,
    where n_events is taken from the param_dict.

    Parameters
    ----------
    config : Configs
        The Configs object.
    param_dict : dict
        Dictionary containing the parameters for the generation of showers,
        as generated by make_params_dict.
    g4_data : tuple[h5py.Group, h5py.Group]
        The G4 data, as returned by get_g4_data.

    Yields
    ------
    showers : np.ndarray (n_events, max_num_hits, 4
        The showers in the format produced by the generative models.
    cond_E : np.ndarray (n_events, 1)
        The incident particle energy for the showers.
    """
    dataset_class = dataset_class_from_config(config)
    all_events, all_energy = g4_data
    n_events = param_dict["n_events"]
    for i in range(0, all_events.shape[0], n_events):
        showers, cond_E = all_events[i : i + n_events], all_energy[i : i + n_events]
        showers[:, -1] = showers[:, -1] * dataset_class.energy_scale
        yield showers, cond_E


def shower_generator_factory(config, param_dict):
    """
    Create a shower generator function that yields showers and incident
    particle energies from the dataset specified by the param_dict,
    in chunks of n_events, where n_events is taken from the param_dict.

    Parameters
    ----------
    config : Configs
        The Configs object.
    param_dict : dict
        Dictionary containing the parameters for the generation of showers,
        as generated by make_params_dict.
    """
    if param_dict["caloclouds"] == "g4":
        if "g4_data_path" in param_dict:
            args = (param_dict["g4_data_path"],)
        else:
            args = (config,)
        g4_data = get_g4_data(*args)

        def shower_generator(param_dict=param_dict, g4_data=g4_data):
            for showers, cond_E in yield_g4_showers(config, param_dict, g4_data):
                yield showers, cond_E

    else:
        torch.manual_seed(param_dict["seed"])
        print(" one random torch number: ", torch.rand(1))
        min_energy = param_dict["min_energy"]
        max_energy = param_dict["max_energy"]

        kwargs = {}
        if "flow_model_path" in param_dict:
            kwargs = {"model_path": param_dict["flow_model_path"]}
        flow_model = load_flow_model(config, **kwargs)

        kwargs = {}
        if "diffusion_model_path" in param_dict:
            kwargs = {"model_path": param_dict["diffusion_model_path"]}
        diffusion_model = load_diffusion_model(
            config, param_dict["caloclouds"], **kwargs
        )

        def shower_generator(
            min_energy=min_energy,
            max_energy=max_energy,
            param_dict=param_dict,
            flow_model=flow_model,
            diffusion_model=diffusion_model,
        ):
            while True:
                yield generate_showers(
                    config,
                    min_energy,
                    max_energy,
                    param_dict,
                    flow_model,
                    diffusion_model,
                )

    return shower_generator


def add_chunk(config, params_dict, shower_generator, merge_dict):
    """
    Add a chunk of params_dict["n_events"] events to the merge_dict,
    using the shower_generator to generate the showers.
    Stores the events in the merge_dict, no return value.

    Parameters
    ----------
    config : Configs
        The Configs object.
    params_dict : dict
        Dictionary containing the parameters for the generation of showers,
        as generated by make_params_dict.
    shower_generator : generator
        The generator that yields showers and incident particle energies.
    merge_dict : dict
        The dictionary where the events are being gathered.

    """
    showers, cond_E = next(shower_generator)
    print("projecting showers")
    metadata = Metadata(config)
    X, Y, Z, _ = metadata.load_muon_map()
    map_layers, _ = create_map(
        X,
        Y,
        Z,
        metadata.layer_bottom_pos,
        metadata.half_cell_size,
        metadata.cell_thickness,
    )
    events, clouds = get_projections(
        showers,
        map_layers,
        metadata.layer_bottom_pos,
        metadata.half_cell_size,
        metadata.cell_thickness,
        max_num_hits=6000,
        return_cell_point_cloud=True,
    )

    print("get features and center of gravities")
    plt_config = plotting.PltConfigs()
    features_dict = plotting.get_features(
        plt_config, map_layers, metadata.half_cell_size, events
    )

    features_dict["incident_energy"] = cond_E.reshape(-1)  # GeV  shape: (n_events,)
    cog_list = plotting.get_cog(clouds)
    features_dict["cog_x"] = cog_list[0]
    features_dict["cog_y"] = cog_list[1]
    features_dict["cog_z"] = cog_list[2]

    print("merging dicts")
    combined = metrics.merge_dicts(
        [merge_dict, features_dict], key_exceptions=params_dict["key_exceptions"]
    )
    merge_dict.update(combined)

    print("current shape of occupancy in merge_dict: ", merge_dict["occ"].shape)


def write_merge_dict(params_dict, merge_dict):
    """
    Makes a checkpoint of the current merge_dict in a pickle file.
    Will overwrite the last checkpoint if it exists.

    Parameters
    ----------
    params_dict : dict
        Dictionary containing the parameters for the generation of showers,
        as generated by make_params_dict.
        Determines the name of the output file.
    merge_dict : dict
        The dictionary to be saved in the pickle file.

    Returns
    -------
    file_path : str
        The path to the saved pickle file.
    """
    file_name = (
        f"merge_dict_{params_dict['min_energy']}-{params_dict['max_energy']}GeV"
        + f"_{params_dict['total_events']}_{params_dict['caloclouds']}.pickle"
    )
    file_path = os.path.join(params_dict["pickle_path"], file_name)
    with open(file_path, "wb") as pickle_file:
        pickle.dump(merge_dict, pickle_file)
    print("merge_dict saved in pickle file")
    return file_path


def write_all_events(config, params_dict):
    """
    For the number of total events specified in params_dict, generate
    showers and write them to a pickle file.

    Parameters
    ----------
    config : Configs
        The Configs object.
    params_dict : dict
        Dictionary containing the parameters for the generation of showers,
        as generated by make_params_dict.
    """
    shower_generator = shower_generator_factory(config, params_dict)
    merge_dict = {}
    n_chunks = int(params_dict["total_events"] / params_dict["n_events"])
    for chunk_n in range(n_chunks):
        print(f"processing chunk {chunk_n} of {n_chunks}")
        add_chunk(config, params_dict, shower_generator, merge_dict)
        write_merge_dict(params_dict, merge_dict)
        gc.collect()


if __name__ == "__main__":
    config = Configs()
    params_dict = make_params_dict()
    # write_merge_dict(config, params_dict)
