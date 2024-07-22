import warnings
import numpy as np
import torch
from tqdm import tqdm

from .metadata import Metadata

from ..configs import Configs


def get_cog(x, y, z, e):
    return (
        np.sum((x * e), axis=1) / e.sum(axis=1),
        np.sum((y * e), axis=1) / e.sum(axis=1),
        np.sum((z * e), axis=1) / e.sum(axis=1),
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


def get_shower(model, num_points, energy, cond_N, bs=1, config=Configs()):
    e = torch.ones((bs, 1), device=config.device) * energy
    n = torch.ones((bs, 1), device=config.device) * cond_N

    if config.norm_cond:
        e = e / 100 * 2 - 1  # max incident energy: 100 GeV
        n = n / config.max_points * 2 - 1
    cond_feats = torch.cat([e, n], -1)

    fake_shower = model.sample(cond_feats, num_points, config)

    return fake_shower


# batch inference
def gen_showers_batch(
    model,
    shower_flow,
    e_min,
    e_max=None,
    num=2000,
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
    e_min : float
        The minimum incident energy of the showers.
    e_max : float
        The maximum incident energy of the showers.
    num : int
        Total number of showers to generate.
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
    showers : np.array (num, configs.max_points, 4)
        The generated showers. The third dimension is (x, y, z, e)
    cond_E : np.array (num, 1)
        The energy of the incident particle that conditions the showers.

    """
    if config.model_name == "wish":
        # for this model, we don't have to give a shower_flow
        if e_max is None:
            e_min = shower_flow
            e_max = e_min
            shower_flow = None
    else:
        assert e_max is not None, (
            f"For model named {config.model}, "
            + "4 positional arguments are expected; "
            + "model, shower_flow, e_min, e_max"
        )
    cond_E = torch.FloatTensor(num, 1).uniform_(e_min, e_max)  # B,1
    fake_showers = gen_condE_showers_batch(
        model,
        shower_flow,
        cond_E,
        bs,
        config,
        coef_real,
        coef_fake,
        n_scaling,
        n_splines,
    )
    cond_E = cond_E.detach().cpu().numpy()
    return fake_showers, cond_E


def gen_condE_showers_batch(
    model,
    shower_flow,
    cond_E=None,
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
    cond_E : int
        The incident energy of the showers.
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
    if config.model_name == "wish":
        # for this model, we don't have to give a shower_flow
        if cond_E is None:
            cond_E = shower_flow
        kw_args = {
            "model": model,
        }
        inner_batch_func = gen_wish_inner_batch
    else:
        assert cond_E is not None, (
            f"For model named {config.model}, "
            + "3 positional arguments are expected; "
            + "model, shower_flow, condE"
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
    num = cond_E.shape[0]
    fake_showers = np.empty((num, config.max_points, 4))
    for i, cond_E_batch in enumerate(cond_E_batcher(cond_E, bs)):
        inner_batch_func(cond_E_batch, fake_showers, i * bs, **kw_args)

    return fake_showers


def cond_E_batcher(cond_E, batch_size):
    """
    Yield batches of incident energies to condition the showers on.

    Parameters
    ----------
    cond_E : torch.Tensor (num, 1)
        The incident energies to condition the showers on.
    batch_size : int
        Max batch size of incident energies to generate.

    Yields
    ------
    cond_E_batch : torch.Tensor (batch_size or smaller, 1)
        The batch of incident energies to condition the showers on.

    """
    # sort by energyies for better batching and faster inference
    mask = torch.argsort(cond_E.squeeze())
    mask = np.atleast_1d(mask)
    cond_E = cond_E[mask]
    num = cond_E.size(0)
    for evt_id in range(0, num, batch_size):
        cond_E_batch = cond_E[evt_id : evt_id + batch_size]
        yield cond_E_batch


def gen_wish_inner_batch(cond_E_batch, destination_array, first_index, model):
    """
    Generate a batch of showers using the Wish model
    according to the given incident energies.

    Parameters
    ----------
    cond_E_batch : torch.Tensor (batch_size, 1)
        The incident energies to condition the showers on.
    destination_array : np.array (large, max_points, 4)
        The array to store the generated showers in.
    first_index : int
        The index to start storing the generated showers at.
    model : torch.nn.Module
        The model to generate showers from.
    config : configs.Configs
        The configuration object, used to determine the max points.

    Returns
    -------
    output : np.array (cond_E_batch.size(0), configs.max_points, 4)
        The generated showers. The third dimension is (x, y, z, e)
    """
    max_points = destination_array.shape[1]
    last_index = first_index + cond_E_batch.size(0)
    destination_array[first_index:last_index] = model.sample(cond_E_batch, max_points)


def truescale_showerflow_output(samples, bs, config):
    metadata = Metadata(config)
    # name samples
    num_clusters = np.clip(
        (samples[:, 0] * metadata.n_pts_rescale).reshape(bs, 1), 1, config.max_points
    )
    gev_to_mev = 1000
    energies = (samples[:, 1] * metadata.vis_eng_rescale * gev_to_mev).reshape(bs, 1)
    # in MeV  (clip to a minimum energy of 40 MeV)
    energies = np.clip(energies, 40, None)
    cog_x = (samples[:, 2] * metadata.std_cog[0]) + metadata.mean_cog[0]
    # cog_y = (samples[:, 3] * metadata.std_cog[1]) + metadata.mean_cog[1]
    cog_z = (samples[:, 4] * metadata.std_cog[2]) + metadata.mean_cog[2]

    clusters_per_layer_gen = np.clip(samples[:, 5:35], 0, 1)  # B,30
    e_per_layer_gen = np.clip(samples[:, 35:], 0, 1)  # B,30
    return num_clusters, energies, cog_x, cog_z, clusters_per_layer_gen, e_per_layer_gen


def gen_v1_inner_batch(
    cond_E_batch,
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
    cond_E_batch : torch.Tensor (batch_size, 1)
        The incident energies to condition the showers on.
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
    output : np.array (cond_E_batch.size(0), configs.max_points, 4)
        The generated showers. The third dimension is (x, y, z, e)
    """
    metadata = Metadata(config)
    bs = cond_E_batch.size(0)

    # sample from shower flow
    samples = (
        shower_flow.condition(cond_E_batch / metadata.incident_rescale)
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
        cog_z,
        clusters_per_layer_gen,
        e_per_layer_gen,
    ) = truescale_showerflow_output(samples, bs, config)

    scale_factor = 1.
    if n_scaling:
        if all(v is None for v in [n_splines, coef_real, coef_fake]):
            warnings.warn("Warning; not scaling number of clusters as no scaling parameters given")
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
    cond_N = (
        torch.Tensor(hits_per_layer_all.sum(axis=1)).to(config.device).unsqueeze(-1)
    )

    fake_showers = get_shower(
        model,
        max_num_clusters,
        cond_E_batch,
        cond_N,
        bs=bs,
        config=config,
    )
    fake_showers = fake_showers.detach().numpy()

    # if np.isnan(fake_showers).sum() != 0:
    #       return fake_showers
    #     print('nans in showers!')
    #     fake_showers[np.isnan(fake_showers)] = 0
    #       break

    # loop over events
    y_positions = metadata.layer_bottom_pos + metadata.cell_thickness / 2
    for i, hits_per_layer in enumerate(hits_per_layer_all):
        # for i, (hits_per_layer, e_per_layer) in enumerate(
        #     zip(hits_per_layer_all, e_per_layer_all)
        # ):

        n_hits_to_concat = max_num_clusters - hits_per_layer.sum()

        y_flow = np.repeat(y_positions, hits_per_layer)
        y_flow = np.concatenate([y_flow, np.zeros(n_hits_to_concat)])

        mask = np.concatenate(
            [np.ones(hits_per_layer.sum()), np.zeros(n_hits_to_concat)]
        )

        fake_showers[i, :, 1][mask == 0] = 10
        idx_dm = np.argsort(fake_showers[i, :, 1])
        fake_showers[i, :, 1][idx_dm] = y_flow

        fake_showers[i, :, :][y_flow == 0] = 0

        # fake_showers[:, :, -1] = abs(fake_showers[:, :, -1])
        fake_showers[
            fake_showers[:, :, -1] <= 0
        ] = 0  # setting events with negative energy to zero
        # fake_showers[:, :, -1] = (
        #     fake_showers[:, :, -1]
        #     / fake_showers[:, :, -1].sum(axis=1).reshape(bs, 1)
        #     * energies[evt_id : evt_id + bs]
        # )  # energy rescaling to predicted e_sum

        # energy per layer calibration
        for j in range(len(y_positions)):
            mask = fake_showers[i, :, 1] == y_positions[j]
            fake_showers[i, :, -1][mask] = (
                fake_showers[i, :, -1][mask]
                / fake_showers[i, :, -1][mask].sum()
                * e_per_layer_all[i, j]
            )

    length = config.max_points - fake_showers.shape[1]
    fake_showers = np.concatenate(
        (fake_showers, np.zeros((bs, length, 4))), axis=1
    )  # B, max_points, 4

    fake_showers[:, :, 0] = (fake_showers[:, :, 0] + 1) / 2
    fake_showers[:, :, 2] = (fake_showers[:, :, 2] + 1) / 2

    fake_showers[:, :, 0] = (
        fake_showers[:, :, 0] * (metadata.Xmin - metadata.Xmax) + metadata.Xmax
    )
    fake_showers[:, :, 2] = (
        fake_showers[:, :, 2] * (metadata.Zmin - metadata.Zmax) + metadata.Zmax
    )

    # CoG calibration
    cog = get_cog(
        fake_showers[:, :, 0],
        fake_showers[:, :, 1],
        fake_showers[:, :, 2],
        fake_showers[:, :, 3],
    )
    fake_showers[:, :, 0] -= (cog[0] - cog_x)[:, None]
    fake_showers[:, :, 2] -= (cog[2] - cog_z)[:, None]
    destination_array[first_index : first_index + bs] = fake_showers
