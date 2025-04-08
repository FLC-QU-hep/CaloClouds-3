import warnings
import torch
import h5py
from functools import lru_cache
from ..utils.metadata import Metadata
from ..utils import precision
from .read_write import read_raw_regaxes, get_files


def get_cond_features_names(config, for_model):
    """
    Get the names of the conditioning features for the model.
    Designed to insulate the script from changes to our config files.

    Parameters
    ----------
    config : pointcloud.configs.Config
        The configuration object for the model.
        Can have attributes `cond_features` and/or `shower_flow_cond_features`.
        The `cond_features` attribute may be an int or a list of strings.
    for_model : str
        Either "diffusion" or "showerflow".

    Returns
    -------
    list of str
        The names of the conditioning features for the model.
    """
    value = None
    if for_model == "diffusion" and hasattr(config, "cond_features_names"):
        value = config.cond_features_names
    elif for_model == "diffusion" and hasattr(config, "cond_features"):
        value = config.cond_features
    elif hasattr(config, "shower_flow_cond_features"):
        value = config.shower_flow_cond_features
        if for_model == "diffusion":
            warnings.warn(
                "Did not find cond_features in config, "
                "using shower_flow_cond_features instead. "
                "If there are dimension issues in the diffusion model "
                "this might be the cause"
            )
    if value is None:
        warnings.warn(
            "Did not find cond_features or shower_flow_cond_features in config, "
            "using default ['energy', 'n_points']. "
            f"If there are dimension issues in the {for_model} model "
            "this might be the cause"
        )
        return ["energy", "n_points"]
    if isinstance(value, int):
        if value == 1:
            return ["energy"]
        elif value == 2:
            return ["energy", "n_points"]
        elif value == 4:
            return ["energy", "p_norm_local"]
        elif value == 5:
            return ["energy", "n_points", "p_norm_local"]
        else:
            raise ValueError(
                f"Unsupported number of conditioning features: {value}"
                "Can process a list of strings of length 1, 2, 4 or 5"
            )
    return value


def get_cond_feats(config, batch, for_model):
    """
    Get the values of the conditioning features for the model.

    Parameters
    ----------
    config : pointcloud.configs.Config
        The configuration object for the model.
        Can have attributes `cond_features` and/or `shower_flow_cond_features`.
        The `cond_features` attribute may be an int or a list of strings.
    batch : dict
        The batch of data, with component names as keys.
        Exceptionally, n_points may have the key points.
    for_model : str
        Either "diffusion" or "showerflow".

    Returns
    -------
    cond_feats : torch.Tensor
        The values of the conditioning features for the model.
    """
    names = get_cond_features_names(config, for_model)
    if "points" in batch and "n_points" in names:
        names[names.index("n_points")] = "points"
    elif "n_points" in batch and "points" in names:
        names[names.index("points")] = "n_points"
    dtype = precision.get("diffusion", config)
    cond_feats = [batch[k][0].to(config.device, dtype=dtype) for k in names]
    cond_feats = torch.cat(cond_feats, -1).to(config.device, dtype=dtype)  # B, C
    return cond_feats


feature_lengths = {"energy": 1, "points": 1, "n_points": 1, "p_norm_local": 3}


def normalise_cond_feats(config, cond_feats, for_model):
    """
    Normalise the conditioning features for the model.
    Doesn't modify the input tensor, returns a new

    Parameters
    ----------
    config : pointcloud.configs.Config
        The configuration object for the model.
        Has attribute `norm_cond`, if this is false, no normalisation is done.
        Has attribute `max_points`, the maximum number of points in a cloud.
        Can have attributes `cond_features` and/or `shower_flow_cond_features`.
        The `cond_features` attribute may be an int or a list of strings.
    cond_feats : torch.Tensor
        The values of the conditioning features for the model, as one
        continuos tensor.
    for_model : str
        Either "diffusion" or "showerflow".

    Returns
    -------
    cond_feats : torch.Tensor
        Conditioned features, if changes where required, the tensor has been
        coppied.
    """
    if config.norm_cond:
        cond_feats = cond_feats.clone()
        meta = Metadata(config)
        names = get_cond_features_names(config, for_model)
        if "points" in names:
            names[names.index("points")] = "n_points"
        lengths = [feature_lengths[k] for k in names]
        start_positions = [sum(lengths[:i]) for i in range(len(lengths))]
        if "energy" in names:
            idx = start_positions[names.index("energy")]
            if for_model == "showerflow":
                cond_feats[:, idx] /= torch.tensor(meta.incident_rescale)
            else:
                cond_feats[:, idx] = (cond_feats[:, idx] / 100) * 2 - 1
        if "n_points" in names:
            idx = start_positions[names.index("n_points")]
            n = cond_feats[:, idx]
            if config.dataset == "calo_challenge":
                n = torch.log((n + 1) / config.min_points) / torch.log(
                    config.max_points / config.min_points
                )
            else:
                n = (n / config.max_points) * 2 - 1
            cond_feats[:, idx] = n
    return cond_feats


def get_cond_dim(config, for_model):
    """
    Get the dimension of the conditioning features for the model.

    Parameters
    ----------
    config : pointcloud.configs.Config
        The configuration object for the model.
        Can have attributes `cond_features` and/or `shower_flow_cond_features`.
        The `cond_features` attribute may be an int or a list of strings.
    for_model : str
        Either "diffusion" or "showerflow".

    Returns
    -------
    int
        The dimension of the conditioning features for the model.
    """
    names = get_cond_features_names(config, for_model)
    lengths = [feature_lengths[k] for k in names]
    return sum(lengths)


@lru_cache(maxsize=1)
def has_n_points(file_name):
    """
    Check if this dataset file has an n_points/points data field.
    """
    loaded = h5py.File(file_name)
    name = next((key for key in loaded.keys() if key in ["n_points", "points"]), False)
    return name


def padding_position(events, roll_axis=False):
    try:
        if roll_axis:
            first_energies = events[:, 3, 0]
            last_energies = events[:, 3, -1]
        else:
            first_energies = events[:, 0, 3]
            last_energies = events[:, -1, 3]
    except IndexError:
        padding = "unknown"
    else:
        start_zeros = (first_energies <= 0).sum()
        end_zeros = (last_energies <= 0).sum()
        if start_zeros > end_zeros:
            padding = "front"
        elif start_zeros < end_zeros:
            padding = "back"
        else:
            padding = "unknown"
    return padding


def calculate_n_points_from_events(events, padding="unknown"):
    """
    Calculate the number of points in the event.
    Assumes we don't have a rolled axis, i.e. the last axis is xyze
    and the second to last is the points.

    Parameters
    ----------
    events : numpy.ndarray
        The events data, with the last axis being xyze.
    padding : str
        The padding position, either "front", "back" or "unknown".

    Returns
    -------
    n_points : numpy.ndarray
        The number of points
    """
    if len(events) == 0:
        return events[..., :1]
    if padding == "unknown":
        padding = padding_position(events)
        if padding == "unknown":
            raise ValueError("Cannot determine padding position")
    padded_length = events.shape[-2]
    if padding == "front":
        # index of the first non-zero energy from the front
        n_points = (events[..., 3] > 0).argmax(-1)[..., None] + 1
        # anywhere the first energy is non-zero, the number of points is the padded length
        n_points[events[..., 0, 3] > 0] = padded_length
    else:
        # index of the first non-zero energy from the back
        padding_len = (events[..., ::-1, 3] > 0).argmax(-1)[..., None]
        n_points = padded_length - padding_len
    return n_points


def read_raw_regaxes_withcond(
    config, pick_events=None, total_size=None, for_model=("showerflow", "diffusion")
):
    """
    Like read_raw_regaxes, but automatically determines the conditioning features,
    and reads them out. Exceptionally, will determine points/n_points if it is a
    conditioning feature and not present in the dataset.
    """
    if isinstance(for_model, str):
        for_model = [for_model]
        cond_names = get_cond_features_names(config, for_model)
    else:
        cond_names_per_model = [get_cond_features_names(config, m) for m in for_model]
        cond_names = list(set(sum(cond_names_per_model, [])))
    # don't have numpy imported here, so manual cumsum
    starts = [0]
    for name in cond_names:
        starts.append(starts[-1] + feature_lengths[name])
    # decide if number of points per event is wanted
    requires_points = next(
        (i for i, name in enumerate(cond_names) if name in ["n_points", "points"]), -1
    )
    readable_conds = cond_names[:]
    if requires_points >= 0:
        n_points_name = has_n_points(get_files(config.dataset_path, 1)[0])
        if n_points_name:
            readable_conds[requires_points] = n_points_name
        else:  # we can't read it, so remove it for now
            readable_conds.pop(requires_points)
    cond, events = read_raw_regaxes(config, pick_events, total_size, readable_conds)
    cond = torch.tensor(cond)
    if requires_points >= 0 and not n_points_name:
        if len(cond.shape) == 1:
            cond = cond[:, None]
        # If there is padding in the config, use that.
        known_padding = getattr(config, "event_padding_position", "unknown")
        n_points = torch.tensor(calculate_n_points_from_events(events, known_padding))
        # we need to insert the n_points into the cond tensor
        cond = torch.cat(
            [cond[:, :requires_points], n_points, cond[:, requires_points:]], -1
        )
    if starts[-1] > 1:
        # if there are 0 events, but multiple conditions,
        # this ensures the indexing works
        cond = cond.reshape(-1, starts[-1])
    if len(for_model) == 1:
        return cond, events
    if len(cond_names) == 1:
        # pottentially dimension length 1, so avoid slicing
        conds = {model_name: cond for model_name in for_model}
        return conds, events
    conds = {}
    for model_name, names in zip(for_model, cond_names_per_model):
        indices = []
        for cond_name in names:
            start = starts[cond_names.index(cond_name)]
            indices += [start + i for i in range(feature_lengths[cond_name])]
        if len(indices) > 1:
            conds[model_name] = cond[:, indices]
        else:
            conds[model_name] = cond[:, indices[0]]
    return conds, events


def cond_dim_at_path(model_path, for_model):
    """
    Check if the model is compatible with the conditioning features.
    """
    loaded = torch.load(model_path, map_location="cpu", weights_only=False)
    if "args" in loaded:
        args = loaded["args"]
        found = get_cond_dim(args, for_model)
    elif for_model == "diffusion":
        state_dict = loaded["state_dict"]
        hyper_gate_layer0 = state_dict[
            "diffusion.inner_model.layers.0._hyper_gate.weight"
        ]
        found = hyper_gate_layer0.shape[1] - 64
    elif for_model == "showerflow":
        state_dict = loaded["model"]
        found = (
            state_dict["showerflow.inner_model.layers.0._hyper_gate.weight"].shape[1]
            - 64
        )
    else:
        raise ValueError(f"Unknown model {for_model}")
    return found
