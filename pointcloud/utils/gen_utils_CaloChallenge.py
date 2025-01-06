import numpy as np
import torch
from tqdm import tqdm

from .utils.conditioning import get_cond_feature_names


def get_cog(x, y, z, e):
    return (
        np.sum((x * e), axis=1) / e.sum(axis=1),
        np.sum((y * e), axis=1) / e.sum(axis=1),
        np.sum((z * e), axis=1) / e.sum(axis=1),
    )


def cylindrical_histogram(
    point_cloud,
    num_layers=45,
    num_radial_bins=18,
    num_angular_bins=50,
    range_limit=(-1, 1),
):
    # Convert Cartesian coordinates (x, y, z) to cylindrical coordinates (z, phi, r)
    x, y, z, e = point_cloud
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    # Normalize phi to [0, 2 * pi]
    phi = (phi + 2 * np.pi) % (2 * np.pi)

    # Create histogram edges
    r_edges = np.linspace(0, 1, num_radial_bins + 1)
    phi_edges = np.linspace(0, 2 * np.pi, num_angular_bins + 1)
    z_edges = np.linspace(-1, 1, num_layers + 1)

    # Create the cylindrical histogram
    histogram, _ = np.histogramdd(
        np.stack((z, phi, r), axis=-1), bins=(z_edges, phi_edges, r_edges), weights=e
    )

    return histogram


def get_scale_factor(num_clusters, coef_real, coef_fake):
    poly_fn_real = np.poly1d(coef_real)
    poly_fn_fake = np.poly1d(coef_fake)

    scale_factor = poly_fn_real(num_clusters) / poly_fn_fake(num_clusters)

    return scale_factor


def get_shower(model, num_points, energy, cond_N, config, bs=1):
    e = torch.ones((bs, 1), device=config.device) * energy
    n = torch.ones((bs, 1), device=config.device) * cond_N

    if config.norm_cond:
        #         e = torch.log((e + 1e-5)/config.min_energy) / np.log(config.max_energy/config.min_energy)
        n = torch.log((n + 1) / config.min_points) / np.log(
            config.max_points / config.min_points
        )

    cond_feats = torch.cat([e, n], -1)

    with torch.no_grad():
        if config.kdiffusion:
            fake_shower = model.sample(cond_feats, num_points, config)
        else:
            fake_shower = model.sample(cond_feats, num_points, config.flexibility)

    return fake_shower


def invers_transform_energy(energy):
    # energy_min, energy_max = 944.6402, 811494.44
    energy_min, energy_max = 10.004840714091406, 89.99100699024089

    return energy_min * (np.e ** (energy * np.log(energy_max / energy_min)))


def invers_transform_points(n_points):
    # points_min, points_max = 201, 19206
    points_min, points_max = 330, 4961

    return points_min * (np.e ** (n_points * np.log(points_max / points_min)))


# batch inference
def gen_showers_batch(model, shower_flow, e_min, e_max, config, num=2000, bs=32):
    if (get_cond_feature_names(config)[0] != ["energy"]) or (
            get_cond_feature_names(config)[1] not in ["n_points", "points"]):
        raise NotImplementedError(
            "Currently only energy and points conditioning in diffusion "
            "is supported for batch generation"
        )
    if get_cond_feature_names(config, model="showerflow") != ["energy"]:
        raise NotImplementedError(
            "Currently only energy conditioning in showerflow "
            "is supported for batch generation"
        )
    output = {}

    leyer_pos = np.arange(-0.98, 1, 0.0444)

    low_log = np.log10(e_min)  # convert to log space
    high_log = np.log10(e_max)  # convert to log space
    uniform_samples = np.random.uniform(low_log, high_log, num)

    # uniform_samples = showers[0]
    # showers = [energy, energysum, num_points, e_per_layer, clusters_per_layer]

    # apply exponential function (base 10)
    log_uniform_samples = np.power(10, uniform_samples)
    log_uniform_samples.sort()
    output["energy"] = log_uniform_samples
    log_uniform_samples = (
        np.log(log_uniform_samples / log_uniform_samples.min())
        / np.log(log_uniform_samples.max() / log_uniform_samples.min())
    ).reshape(-1)
    cond_E = torch.tensor(log_uniform_samples).view(num, 1).to(config.device).float()

    samples = (
        shower_flow.condition(cond_E)
        .sample(
            torch.Size(
                [
                    num,
                ]
            )
        )
        .cpu()
        .numpy()
    )
    # samples = showers

    # visible_energy, num_points, e_per_layer, clusters_per_layer
    energy_sum = samples[:, 0]
    energy_sum = invers_transform_energy(energy_sum)
    energy_sum = energy_sum.reshape(num, 1)

    num_clasters = samples[:, 1]
    num_clasters = invers_transform_points(num_clasters)

    e_per_layer = samples[:, 2:47]
    e_per_layer[e_per_layer < 0] = 0
    e_per_layer = e_per_layer / e_per_layer.sum(axis=1).reshape(num, 1)
    e_per_layer[e_per_layer < 0] = e_per_layer[e_per_layer < 0] * (-1)

    clusters_per_layer = samples[:, 47:]
    clusters_per_layer = clusters_per_layer / clusters_per_layer.sum(axis=1).reshape(
        num, 1
    )
    clusters_per_layer = clusters_per_layer * num_clasters.reshape(num, 1)

    energy_sum[energy_sum < 0] = energy_sum[energy_sum < 0] * (-1)
    clusters_per_layer[clusters_per_layer < 0] = clusters_per_layer[
        clusters_per_layer < 0
    ] * (-1)
    clusters_per_layer = clusters_per_layer.astype(int)

    output["num_points"] = clusters_per_layer.sum(axis=1)

    fake_showers_list = []

    for evt_id in tqdm(range(0, num, bs)):
        if (num - evt_id) < bs:
            bs = num - evt_id
        #     num = num - 75 # substruct correction factor
        # convert clusters_per_layer_gen to a fractions of points in the layer out of sum(points in the layer) of event
        # multuply clusters_per_layer_gen by corrected tottal num of points
        hits_per_layer_all = clusters_per_layer[
            evt_id : evt_id + bs
        ]  # shape (bs, num_layers)
        max_num_clusters = hits_per_layer_all.sum(axis=1).max()
        cond_N = (
            torch.Tensor(hits_per_layer_all.sum(axis=1)).to(config.device).unsqueeze(-1)
        )
        fs = get_shower(
            model,
            max_num_clusters,
            cond_E[evt_id : evt_id + bs],
            cond_N,
            bs=bs,
            config=config,
        )

        fs = fs.cpu().numpy()

        if np.isnan(fs).sum() != 0:
            print("nans in showers!")
            fs[np.isnan(fs)] = 0

        for i, hits_per_layer in enumerate(hits_per_layer_all):
            n_hits_to_concat = max_num_clusters - hits_per_layer.sum()

            z_flow = np.repeat(leyer_pos, hits_per_layer)
            z_flow = np.concatenate([z_flow, np.zeros(n_hits_to_concat)])

            mask = np.concatenate(
                [np.ones(hits_per_layer.sum()), np.zeros(n_hits_to_concat)]
            )

            fs[i, :, 2][mask == 0] = 10
            idx_dm = np.argsort(fs[i, :, 2])
            fs[i, :, 2][idx_dm] = z_flow

            fs[i, :, :][z_flow == 0] = 0

        fs[:, :, -1][fs[:, :, -1] < 0] = 0
        length = 20000 - fs.shape[1]
        fs = np.concatenate((fs, np.zeros((bs, length, 4))), axis=1)
        fs = [[cylindrical_histogram(fs)] for fs in np.moveaxis(fs, -1, 1)]
        fs = np.vstack(fs)
        fake_showers_list.append(fs)

    fake_showers = np.vstack(fake_showers_list)

    # energy per layer calibration
    fake_showers = fake_showers / fake_showers.sum(axis=(2, 3)).reshape(num, 45, 1, 1)
    fake_showers[np.isnan(fake_showers)] = 0
    fake_showers = fake_showers * (e_per_layer * energy_sum).reshape(num, 45, 1, 1)

    output["showers"] = fake_showers

    return output


# batch inference
def gen_showers_batch_r(model, showers, e_min, e_max, config, num=2000, bs=32):
    """
    version with deterministic calibration of energy per layer w/o using the shower flow
    """

    output = {}

    leyer_pos = np.arange(-0.98, 1, 0.0444)

    # low_log = np.log10(e_min)  # convert to log space
    # high_log = np.log10(e_max)  # convert to log space
    # uniform_samples = np.random.uniform(low_log, high_log, num)

    # showers = [ energysum, num_points, e_per_layer, clusters_per_layer]

    log_uniform_samples = showers[:, -1]  # incident energies

    # apply exponential function (base 10)
    # log_uniform_samples = np.power(10, uniform_samples)
    log_uniform_samples.sort()
    output["energy"] = log_uniform_samples
    log_uniform_samples = (
        np.log(log_uniform_samples / log_uniform_samples.min())
        / np.log(log_uniform_samples.max() / log_uniform_samples.min())
    ).reshape(-1)
    cond_E = torch.tensor(log_uniform_samples).view(num, 1).to(config.device).float()

    # samples = shower_flow.condition(cond_E).sample(torch.Size([num, ])).cpu().numpy()
    samples = showers

    # visible_energy, num_points, e_per_layer, clusters_per_layer
    energy_sum = samples[:, 0]
    # energy_sum = invers_transform_energy(energy_sum)
    energy_sum = energy_sum.reshape(num, 1)

    num_clasters = samples[:, 1]
    # num_clasters = invers_transform_points(num_clasters)

    e_per_layer = samples[:, 2:47]
    e_per_layer[e_per_layer < 0] = 0
    e_per_layer = e_per_layer / e_per_layer.sum(axis=1).reshape(num, 1)
    e_per_layer[e_per_layer < 0] = e_per_layer[e_per_layer < 0] * (-1)

    clusters_per_layer = samples[:, 47:92]
    clusters_per_layer = clusters_per_layer / clusters_per_layer.sum(axis=1).reshape(
        num, 1
    )
    clusters_per_layer = clusters_per_layer * num_clasters.reshape(num, 1)

    energy_sum[energy_sum < 0] = energy_sum[energy_sum < 0] * (-1)
    clusters_per_layer[clusters_per_layer < 0] = clusters_per_layer[
        clusters_per_layer < 0
    ] * (-1)
    clusters_per_layer = clusters_per_layer.astype(int)

    output["num_points"] = clusters_per_layer.sum(axis=1)

    fake_showers_list = []

    for evt_id in tqdm(range(0, num, bs)):
        if (num - evt_id) < bs:
            bs = num - evt_id
        #         num = num - 75 # substruct correction factor
        # convert clusters_per_layer_gen to a fractions of points in the layer out of sum(points in the layer) of event
        # multuply clusters_per_layer_gen by corrected tottal num of points
        hits_per_layer_all = clusters_per_layer[
            evt_id : evt_id + bs
        ]  # shape (bs, num_layers)
        max_num_clusters = hits_per_layer_all.sum(axis=1).max().astype(int)
        cond_N = (
            torch.Tensor(hits_per_layer_all.sum(axis=1)).to(config.device).unsqueeze(-1)
        )
        fs = get_shower(
            model,
            max_num_clusters,
            cond_E[evt_id : evt_id + bs],
            cond_N,
            bs=bs,
            config=config,
        )

        fs = fs.cpu().numpy()

        if np.isnan(fs).sum() != 0:
            print("nans in showers!")
            fs[np.isnan(fs)] = 0

        for i, hits_per_layer in enumerate(hits_per_layer_all):
            n_hits_to_concat = max_num_clusters - hits_per_layer.sum()

            z_flow = np.repeat(leyer_pos, hits_per_layer)
            z_flow = np.concatenate([z_flow, np.zeros(n_hits_to_concat)])

            mask = np.concatenate(
                [np.ones(hits_per_layer.sum()), np.zeros(n_hits_to_concat)]
            )

            fs[i, :, 2][mask == 0] = 10
            idx_dm = np.argsort(fs[i, :, 2])
            fs[i, :, 2][idx_dm] = z_flow

            fs[i, :, :][z_flow == 0] = 0

        fs[:, :, -1][fs[:, :, -1] < 0] = 0  # TODO try to put threshold 0.01515
        length = 20000 - fs.shape[1]
        fs = np.concatenate((fs, np.zeros((bs, length, 4))), axis=1)
        fs = [[cylindrical_histogram(fs)] for fs in np.moveaxis(fs, -1, 1)]
        fs = np.vstack(fs)
        fake_showers_list.append(fs)

    fake_showers = np.vstack(fake_showers_list)

    # energy per layer calibration
    fake_showers = fake_showers / fake_showers.sum(axis=(2, 3)).reshape(num, 45, 1, 1)
    fake_showers[np.isnan(fake_showers)] = 0
    fake_showers = fake_showers * (e_per_layer * energy_sum).reshape(num, 45, 1, 1)

    # point energy theshold AGAIN

    output["showers"] = fake_showers

    return output
