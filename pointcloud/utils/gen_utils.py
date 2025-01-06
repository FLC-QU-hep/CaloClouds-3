import warnings
import numpy as np
import torch

from .metadata import Metadata

from ..configs import Configs

from .showerflow_utils import truescale_showerflow_output
from ..data.conditioning import get_cond_features_names, normalise_cond_feats


def get_cog(x, y, z, e):
    """
    Calculate the center of gravity of a shower.

    Parameters
    ----------
    x : np.array
        The x positions of the shower particles.
    y : np.array
        The y positions of the shower particles.
    z : np.array
        The z positions of the shower particles.
    e : np.array
        The energies of the shower particles.

    Returns
    -------
    cog_x : np.array or float
        The x position of the center of gravity.
        If the input has multiple dimensions, this has one less,
        otherwise it is a float.
    cog_y : np.array or float
        The y position of the center of gravity.
        If the input has multiple dimensions, this has one less,
        otherwise it is a float.
    cog_z : np.array or float
        The z position of the center of gravity.
        If the input has multiple dimensions, this has one less,
        otherwise it is a float.
    """
    summed_e = e.sum(axis=-1)
    if len(e.shape) > 1:
        summed_e[summed_e == 0] = 1  # avoid division by zero
    elif not summed_e:
        return 0, 0, 0
    return (
        np.sum((x * e), axis=-1) / summed_e,
        np.sum((y * e), axis=-1) / summed_e,
        np.sum((z * e), axis=-1) / summed_e,
    )


def get_scale_factor(
    num_clusters, coef_real, coef_fake, n_splines
):  # num_clusters: (bs, 1)
    if n_splines is not None:
        spline_real = n_splines["spline_real"]
        spline_fake = n_splines["spline_fake"]
        scale_factor = (
            spline_fake.predict(
                spline_real.predict(num_clusters).reshape(-1, 1)
            ).reshape(-1, 1)
            / num_clusters
        )
    else:
        poly_fn_real = np.poly1d(coef_real)
        poly_fn_fake = np.poly1d(coef_fake)
        scale_factor = poly_fn_fake(poly_fn_real(num_clusters)) / num_clusters

    scale_factor = np.clip(scale_factor, 0.0, None)  # cannot be negative
    return scale_factor  # (bs, 1)


# TODO
# Consider changin how the cond_feats is passed in for
# backward compatibility with older versions?
# Maybe achived
def get_shower(model, num_points, cond_feats, cond_N=None, bs=1, config=Configs()):
    """
    Get a shower from the diffusion model.
    """
    try:
        n_conds = len(cond_feats)
    except TypeError:
        cond_feats = torch.tensor([cond_feats] * bs).float().reshape(bs, 1)
    else:
        cond_feats = torch.atleast_2d(torch.tensor(cond_feats).float()).reshape(
            n_conds, -1
        )
    if cond_N is not None:
        try:
            n_conds = len(cond_N)
        except TypeError:
            cond_N = torch.tensor([cond_N] * bs).float().reshape(bs, 1)
        else:
            cond_N = torch.atleast_2d(torch.tensor(cond_N).float()).reshape(n_conds, -1)
        if len(cond_N.shape) == 1:
            cond_N = cond_N[:, None]
        cond_feats = torch.cat([cond_feats, cond_N], dim=1)
    fake_shower = model.sample(cond_feats, num_points, config)
    return fake_shower


def is_config(config):
    return hasattr(config, "model_name")


# for historic reasons, I don't want to mess with the
# argument parsing of this function, but it's complex
# so lets just make a separate function for it.
# This is a fairly defensive function, that will complain
# about any missing/unexpected arguments.
def _shower_batch_arg_parser(*args, **kwargs):
    # read the args differently for different models
    n_required_args = 4
    arg_values = {
        "model": None,
        "shower_flow": None,
        "e_min": None,
        "e_max": None,
        "num": 2000,
        "bs": 32,
        "config": Configs(),
        "coef_real": None,
        "coef_fake": None,
        "n_scaling": True,
        "n_splines": None,
    }
    arg_positions = [
        "model",
        "shower_flow",
        "e_min",
        "e_max",
        "num",
        "bs",
        "config",
        "coef_real",
        "coef_fake",
        "n_scaling",
        "n_splines",
    ]
    config_pos = arg_positions.index("config")
    if "config" in kwargs:
        assert (
            len(args) <= config_pos
        ), "config given twice, once as positional, once as kwarg"
        config = kwargs["config"]
    elif config_pos < len(args) + 1:
        if config_pos < len(args) and is_config(args[config_pos]):
            config = args[config_pos]
        elif is_config(args[config_pos - 1]):
            config = args[config_pos - 1]
        else:
            raise ValueError(
                f"Expected a Configs object in position {config_pos} or {config_pos-1}"
            )
    else:
        config = Configs()

    if config.model_name in ["wish", "fish"]:
        n_required_args -= 1
        del arg_positions[1]
        assert (
            "shower_flow" not in kwargs
        ), "For models wish and fish, the shower_flow argument is not expected"

    assert len(args) + len(kwargs) <= (
        len(arg_positions)
    ), f"Too many arguments given, expected {len(arg_positions)} at most"
    for i, name in enumerate(arg_positions):
        in_kwargs = name in kwargs
        if len(args) > i:
            assert (
                not in_kwargs
            ), f"Argument {name} given twice, once as positional, once as kwarg"
            arg_values[name] = args[i]
        elif in_kwargs:
            arg_values[name] = kwargs.pop(name)
        else:
            assert i >= n_required_args, f"Argument {name} (position {i}) is required"
    assert not kwargs, f"Unknown arguments given: {kwargs.keys()}"
    # do some sanity checks
    assert arg_values.get("e_min", 1) >= 0, "e_min must be non-negative"
    assert arg_values.get("e_min", 1) <= arg_values.get(
        "e_max", 2
    ), "e_min must be less than e_max"
    assert arg_values["num"] >= 0, "num must be positive"
    assert arg_values["bs"] >= 0, "bs must be positive"
    return arg_values.values()


# batch inference
def gen_showers_batch(
    *args,
    **kwargs,
):
    """
    Generate a batch of showers for evaluation purposes.
    Not sutable for training.

    Parameters
    ----------
    models : torch.nn.Module
        The model to generate showers from.
        If the configs.model_name specifies a model
        that uses a shower flow, like caloclouds, then this should
        be both the model and the flow distribution;
    shower_flow : torch.distributions.Distribution, (optional)
        The flow to sample from. Only needed for models that use one,
        for models that don't, like Wish, this argument
        isn't needed (either positionally, or as a kwarg).
    e_min : float
        The minimum incident energy of the showers.
    e_max : float
        The maximum incident energy of the showers.
        It is required for all models.
    num : int
        Total number of showers to generate.
        (default is 2000)
    bs : int
        The inner batch size to generate the batch in ....
        (default is 32)
    config : configs.Configs
        The configuration object. Used to determine the device
        and the model.
        (default is Configs())
    coef_real : list
        The coefficients of the real energy calibration polynomial,
        only used for v1 style models.
        (default is None)
    coef_fake : list
        The coefficients of the fake energy calibration polynomial,
        only used for v1 style models.
        (default is None)
    n_scaling : bool
        Whether to scale the number of clusters or not.
        Only used for v1 style models.
        (default is True)
    n_splines : dict
        The spline models for the number of clusters.
        Alterative to coef_real and coef_fake.
        Only used for v1 style models.
        (default is None)

    Returns
    -------
    showers : np.array (num, configs.max_points, 4)
        The generated showers. The third dimension is (x, y, z, e)
    cond : np.array (num, C)
        The conditioning variables for the showers,
        in the case that the showers are conditioned on just energy, this
        is just a (num, 1) array of the incident energies.

    """
    (
        model,
        shower_flow,
        e_min,
        e_max,
        num,
        bs,
        config,
        coef_real,
        coef_fake,
        n_scaling,
        n_splines,
    ) = _shower_batch_arg_parser(*args, **kwargs)

    # TODO generalise
    if get_cond_features_names(config, "diffusion") != ["energy"]:
        raise NotImplementedError(
            "Currently only energy conditioning is supported for batch generation"
        )
    if get_cond_features_names(config, "showerflow") != ["energy"]:
        raise NotImplementedError(
            "Currently only energy conditioning is supported for batch generation"
        )
    cond = torch.FloatTensor(num, 1).uniform_(e_min, e_max)  # B,1
    cond = cond.to(config.device)
    fake_showers = gen_cond_showers_batch(
        model,
        shower_flow,
        cond,
        bs,
        config,
        coef_real,
        coef_fake,
        n_scaling,
        n_splines,
    )
    cond = cond.detach().cpu().numpy()
    return fake_showers, cond


def gen_cond_showers_batch(
    model,
    shower_flow,
    cond=None,
    bs=32,
    config=Configs(),
    coef_real=None,
    coef_fake=None,
    n_scaling=True,
    n_splines=None,
):
    """
    Generate a batch of showers for evaluation purposes.
    Not sutable for training.

    Parameters
    ----------
    model : torch.nn.Module
        The model to generate showers from.
    shower_flow : torch.distributions.Distribution (optional)
        The flow to sample from. Only needed for models that use one,
        for models that don't like Wish, the e_min can be the second
        positional argument, or shower_flow can be None.
    cond : torch.Tensor (num, C)
        The conditioning variable for the showers,
        unnormalised.
        Optionally, may be given seperately for showerflow then the diffusion model.
        in the case that the showers are conditioned on just energy, this
        is just a (num, 1) array of the incident energies.
    bs : int
        The inner batch size to generate the batch in ....
    config : configs.Configs
        The configuration object. Used to determine the device
        and the model.
    coef_real : list
        The coefficients of the real energy calibration polynomial,
        only used for v1 style models.
    coef_fake : list
        The coefficients of the fake energy calibration polynomial,
        only used for v1 style models.
    n_scaling : bool
        Whether to scale the number of clusters or not.
        Only used for v1 style models.
    n_splines : dict
        The spline models for the number of clusters.
        Alterative to coef_real and coef_fake.
        Only used for v1 style models.

    Returns
    -------
    showers : np.array (condE.shape[0], configs.max_points, 4)
        The generated showers. The third dimension is (x, y, z, e)

    """
    if config.model_name in ["wish", "fish"]:
        # for this model, we don't have to give a shower_flow
        if cond is None:
            cond = torch.tensor(shower_flow).to(config.device).float()
        kw_args = {
            "model": model,
        }
        inner_batch_func = gen_wish_inner_batch
        showerflow_cond = None  # not needed for these models
    else:
        assert cond is not None, (
            f"For model named {config.model}, "
            + "3 positional arguments are expected; "
            + "model, shower_flow, cond"
        )
        inner_batch_func = gen_v1_inner_batch
        kw_args = {
            "model": model,
            "shower_flow": shower_flow,
            "config": config,
            "coef_real": coef_real,
            "coef_fake": coef_fake,
            "n_scaling": n_scaling,
            "n_splines": n_splines,
        }
    seperate_showerflow_cond = isinstance(cond, dict)
    if seperate_showerflow_cond:
        # it's a seperate showerflow cond
        showerflow_cond = torch.tensor(cond["showerflow"]).to(config.device).float()
        cond = torch.tensor(cond["diffusion"]).to(config.device).float()
    else:
        # give showerflow the same cond as the diffusion model
        cond = torch.tensor(cond).to(config.device).float()
        showerflow_cond = cond
    num = cond.shape[0]
    fake_showers = np.empty((num, config.max_points, 4))
    for i, cond_batch in enumerate(cond_batcher(bs, cond, showerflow_cond)):
        inner_batch_func(cond_batch, fake_showers, i * bs, **kw_args)

    return fake_showers


def cond_batcher(batch_size, cond, showerflow_cond=None):
    """
    Yield batches of incident energies to condition the showers on.

    Parameters
    ----------
    batch_size : int
        Max batch size of incident energies to generate.
    cond : torch.Tensor (num, C)
        The conditioning variable for the showers,
        in the case that the showers are conditioned on just energy, this
        is just a (num, 1) array of the incident energies.
        Energies is generaly expected to be the final value at the lowest dimension.

    Yields
    ------
    cond_batch : torch.Tensor (batch_size or smaller, C)
        The batch of incident energies to condition the showers on.

    """
    # sort by energies for better batching and faster inference
    if len(cond.shape) > 1:
        mask = torch.argsort(cond[..., -1].squeeze())
    else:
        mask = torch.argsort(cond)
    # change torch tensor to numpyarray
    mask = mask.cpu().numpy()
    mask = np.atleast_1d(mask)
    cond = cond[mask]
    num = cond.size(0)
    for evt_id in range(0, num, batch_size):
        cond_batch = {"diffusion": cond[evt_id : evt_id + batch_size]}
        if showerflow_cond is not None:
            cond_batch["showerflow"] = showerflow_cond[evt_id : evt_id + batch_size]
        yield cond_batch


def gen_wish_inner_batch(cond_batch, destination_array, first_index, model):
    """
    Generate a batch of showers using the Wish model
    according to the given incident energies.
    Also works on fish.

    Parameters
    ----------
    cond_batch : dict with str: torch.Tensor (batch_size, C)
        In the dict is a key "diffusion" with the incident particle
        properties to condition the showers on.
    destination_array : np.array (large, max_points, 4)
        The array to store the generated showers in.
    first_index : int
        The index to start storing the generated showers at.
    model : torch.nn.Module
        The model to generate showers from.
    config : configs.Configs
        The configuration object, used to determine the max points.
    """
    cond_batch = cond_batch["diffusion"]
    max_points = destination_array.shape[1]
    last_index = first_index + cond_batch.size(0)
    destination_array[first_index:last_index] = model.sample(cond_batch, max_points)


def gen_v1_inner_batch(
    cond_batch,
    destination_array,
    first_index,
    model,
    shower_flow,
    config=Configs(),
    coef_real=None,
    coef_fake=None,
    n_scaling=True,
    n_splines=None,
):
    """
    Generate a batch of showers using any of the v1 style models
    according to the given incident energies.

    Parameters
    ----------
    cond_batch : torch.Tensor (batch_size, cond_features)
        The conditioning
    model : torch.nn.Module
        The model to generate showers from.
    destination_array : np.array (large, max_points, 4)
        The array to store the generated showers in.
    first_index : int
        The index to start storing the generated showers at.
    shower_flow : torch.distributions.Distribution (optional)
        The flow to sample from. Only needed for models that use one,
        for models that don't like Wish, the e_min can be the second
        positional argument, or shower_flow can be None.
    config : configs.Configs
        The configuration object. Used to determine the device
        and the model.
    coef_real : list
        The coefficients of the real energy calibration polynomial,
        only used for v1 style models.
    coef_fake : list
        The coefficients of the fake energy calibration polynomial,
        only used for v1 style models.
    n_scaling : bool
        Whether to scale the number of clusters or not.
        Only used for v1 style models.
    n_splines : dict
        The spline models for the number of clusters.
        Alterative to coef_real and coef_fake.
        Only used for v1 style models.

    Returns
    -------
    output : np.array (cond_batch.size(0), configs.max_points, 4)
        The generated showers. The third dimension is (x, y, z, e)
    """
    if isinstance(cond_batch, dict):
        showerflow_cond = cond_batch.get("showerflow", cond_batch["diffusion"][:])
        if len(showerflow_cond.shape) == 1:
            showerflow_cond = showerflow_cond[:, None]
        cond_batch = cond_batch["diffusion"]
        if len(cond_batch.shape) == 1:
            cond_batch = cond_batch[:, None]
    else:
        if len(cond_batch.shape) == 1:
            cond_batch = cond_batch[:, None]
        showerflow_cond = cond_batch
    cond_batch = normalise_cond_feats(config, cond_batch, "diffusion")
    showerflow_cond = normalise_cond_feats(config, showerflow_cond, "showerflow")

    metadata = Metadata(config)
    bs = cond_batch.size(0)

    # sample from shower flow
    samples = (
        shower_flow.condition(showerflow_cond)
        .sample(
            torch.Size(
                [
                    bs,
                ]
            )
        )
        .cpu()
        .numpy()
    )

    (
        num_clusters,
        energies,
        cog_x,
        cog_y,
        _,
        clusters_per_layer_gen,
        e_per_layer_gen,
    ) = truescale_showerflow_output(samples, config)

    scale_factor = 1.0
    if n_scaling:
        if all(v is None for v in [n_splines, coef_real, coef_fake]):
            warnings.warn(
                "Warning; not scaling number of clusters as no scaling parameters given"
            )
        else:
            scale_factor = get_scale_factor(
                num_clusters, coef_real, coef_fake, n_splines
            )  # B,1
    num_clusters = (num_clusters * scale_factor).astype(int)  # B,1

    # scale relative clusters per layer to actual number of clusters per layer
    # and same for energy
    clusters_per_layer_gen = (
        clusters_per_layer_gen
        / clusters_per_layer_gen.sum(axis=1, keepdims=True)
        * num_clusters
    ).astype(
        int
    )  # B,30
    e_per_layer_gen = (
        e_per_layer_gen / e_per_layer_gen.sum(axis=1, keepdims=True) * energies
    )  # B,30

    # convert clusters_per_layer_gen to a fractions of points in the layer
    # out of sum(points in the layer) of event
    # multuply clusters_per_layer_gen by corrected tottal num of points
    hits_per_layer_all = (
        clusters_per_layer_gen  # [evt_id : evt_id+bs] # shape (bs, num_layers)
    )
    e_per_layer_all = e_per_layer_gen  # [evt_id : evt_id+bs] # shape (bs, num_layers)
    max_num_clusters = hits_per_layer_all.sum(axis=1).max()

    fake_showers = get_shower(
        model,
        max_num_clusters,
        cond_batch,
        bs=bs,
        config=config,
    )
    fake_showers = fake_showers.detach().cpu().numpy()

    # if np.isnan(fake_showers).sum() != 0:
    #       return fake_showers
    #     print('nans in showers!')
    #     fake_showers[np.isnan(fake_showers)] = 0
    #       break

    # loop over events
    z_positions = metadata.layer_bottom_pos_global + metadata.cell_thickness_global / 2
    for i, hits_per_layer in enumerate(hits_per_layer_all):
        # for i, (hits_per_layer, e_per_layer) in enumerate(
        #     zip(hits_per_layer_all, e_per_layer_all)
        # ):

        n_hits_to_concat = max_num_clusters - hits_per_layer.sum()

        z_flow = np.repeat(z_positions, hits_per_layer)
        z_flow = np.concatenate([z_flow, np.zeros(n_hits_to_concat)])

        mask = np.concatenate(
            [np.ones(hits_per_layer.sum()), np.zeros(n_hits_to_concat)]
        )

        fake_showers[i, :, 2][mask == 0] = 10  # move them out of the sort
        idx_dm = np.argsort(fake_showers[i, :, 2])
        fake_showers[i, :, 2][idx_dm] = z_flow

        fake_showers[i, :, :][mask == 0] = 0

        # fake_showers[:, :, -1] = abs(fake_showers[:, :, -1])
        fake_showers[fake_showers[:, :, -1] <= 0] = (
            0  # setting events with negative energy to zero
        )
        # fake_showers[:, :, -1] = (
        #     fake_showers[:, :, -1]
        #     / fake_showers[:, :, -1].sum(axis=1).reshape(bs, 1)
        #     * energies[evt_id : evt_id + bs]
        # )  # energy rescaling to predicted e_sum

        # energy per layer calibration
        for j in range(len(z_positions)):
            mask = fake_showers[i, :, 2] == z_positions[j]
            fake_showers[i, :, -1][mask] = (
                fake_showers[i, :, -1][mask]
                / fake_showers[i, :, -1][mask].sum()
                * e_per_layer_all[i, j]
            )
    length = config.max_points - fake_showers.shape[1]
    fake_showers = np.concatenate(
        (fake_showers, np.zeros((bs, length, 4))), axis=1
    )  # B, max_points, 4

    # move to be roughly bettween 0 and 1
    fake_showers[:, :, 0] = (fake_showers[:, :, 0] + 1) / 2
    fake_showers[:, :, 1] = (fake_showers[:, :, 1] + 1) / 2

    assert metadata.orientation[:16] == "hdf5:xyz==local:"
    assert metadata.orientation_global[:17] == "hdf5:xyz==global:"
    local_ori = metadata.orientation[16:]
    global_ori = metadata.orientation_global[17:]
    if global_ori[local_ori.index("x")] == "z":
        assert global_ori[local_ori.index("y")] == "x"
        # rotated coordinates mean local x==global z and local y==global x and local z==global y
        fake_showers[:, :, 0] = (
            fake_showers[:, :, 0] * (metadata.Zmin_global - metadata.Zmax_global)
            + metadata.Zmax_global
        )
        fake_showers[:, :, 1] = (
            fake_showers[:, :, 1] * (metadata.Xmin_global - metadata.Xmax_global)
            + metadata.Xmax_global
        )
    elif global_ori[local_ori.index("x")] == "x":
        assert global_ori[local_ori.index("y")] == "z"
        fake_showers[:, :, 0] = (
            fake_showers[:, :, 0] * (metadata.Xmin_global - metadata.Xmax_global)
            + metadata.Xmax_global
        )
        fake_showers[:, :, 1] = (
            fake_showers[:, :, 1] * (metadata.Zmin_global - metadata.Zmax_global)
            + metadata.Zmax_global
        )
    else:
        raise NotImplementedError(
            f"Global orientation {global_ori} not supported for local orientation {local_ori}"
        )

    # CoG calibration
    cog = get_cog(
        fake_showers[:, :, 0],
        fake_showers[:, :, 1],
        fake_showers[:, :, 2],
        fake_showers[:, :, 3],
    )
    fake_showers[:, :, 0] -= (cog[0] - cog_x)[:, None]
    fake_showers[:, :, 1] -= (cog[1] - cog_y)[:, None]
    destination_array[first_index : first_index + bs] = fake_showers
