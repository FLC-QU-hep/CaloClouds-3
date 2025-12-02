"""
Script to create the metrics used in scripts/plotting/standard_metrics.ipynb
Please do add any new models!

TODO; should the binned data be stored inside the repo?
it's very lightweight after binning...
"""

import numpy as np
import torch
import os
import time

from pointcloud.config_varients import default, caloclouds_2, caloclouds_3

from pointcloud.utils.metadata import Metadata
from pointcloud.utils import detector_map
from pointcloud.data.read_write import get_n_events
from pointcloud.data.conditioning import read_raw_regaxes_withcond, get_cond_dim
from pointcloud.utils.gen_utils import gen_cond_showers_batch
from pointcloud.utils.misc import seed_all


# imports specific to this evaluation
from pointcloud.evaluation.bin_standard_metrics import get_caloclouds_models
from pointcloud.evaluation.calculate_scale_factors import get_path as factor_get_path


def get_path(config, name, min_energy, max_energy, n_events, seed):
    out_path = config.logdir
    timestamp = time.strftime("%Y-%m-%d_%H-%M")
    scaled = config.shower_flow_n_scaling
    if scaled:
        name += "_scaled"
    else:
        name += "_unscaled"
    path = os.path.join(
        out_path,
        "raw_events",
        f"{name}_{min_energy}-{max_energy}GeV_{n_events}_seed{seed}_{timestamp}.npz",
    )
    out_dir = os.path.dirname(path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return path


def sample_g4(config, n_events):
    """
    Draw samples from the g4 data and add them to the binned data.

    Parameters
    ----------
    config : Configs
        The configuration used to find the dataset, etc.
    binned : BinnedData
        The binned data to add the samples to, modified inplace.
    n_events : int
        The number of events to sample.
    """
    batch_len = 1000
    batch_starts = np.arange(0, n_events, batch_len)
    n_batches = np.ceil(n_events / batch_len)
    all_cond = []
    all_events = []

    for b, start in enumerate(batch_starts):
        print(f"g4 {b/n_batches:.1%}", end="\r", flush=True)
        cond, events = read_raw_regaxes_withcond(
            config, pick_events=slice(start, start + batch_len)
        )
        all_cond.append(cond["diffusion"])
        all_events.append(events)
    print()
    print("Done")
    return np.concatenate(all_cond, axis=0), np.concatenate(all_events, axis=0)


def conditioned_sample_model(model_config, cond, model, shower_flow=None):
    """
    Use a model to produce events and add them to the binned data.

    Parameters
    ----------
    model_config : Configs
        The configuration used for model metadata
    model_config : torch.Tensor (n_events, C)
        The conditioning for the sample
    model : torch.nn.Module
        The model to sample from, will be sampled by
        pointcloud.utils.gen_utils.gen_cond_showers_batch.
    shower_flow : distribution, optional
        The flow to sample from, if the model needs it, by default None
        also used by gen_cond_showers_batch.

    Returns
    -------
    events : np.array (n_events, n_points, 4)
        The events produced by the model
        with the last axis being [x, y, z, energy].

    """
    if isinstance(cond, dict):
        n_events = len(next(iter(cond.values())))
    else:
        n_events = len(cond)
    if n_events == 0:
        return np.zeros(0), np.zeros((0, 0, 4))
    if model_config.model_name in ["fish", "wish"]:
        batch_len = min(1000, n_events)
    else:
        batch_len = min(100, n_events)

    batch_starts = np.arange(0, n_events, batch_len)
    n_batches = np.ceil(n_events / batch_len)
    all_events = []

    for b, start in enumerate(batch_starts):
        print(f" model {b/n_batches:.1%}", end="\r", flush=True)
        if isinstance(cond, dict):
            cond_here = {key: cond[key][start : start + batch_len] for key in cond}
        else:
            cond_here = cond[start : start + batch_len]
        events = gen_cond_showers_batch(
            model, shower_flow, cond_here, bs=batch_len, config=model_config
        )
        all_events.append(events)
    print()
    print("Done")
    return np.concatenate(all_events, axis=0)


def make_cond(model_config, min_energy, max_energy, n_events):
    cond_E = torch.FloatTensor(n_events, 1).uniform_(min_energy, max_energy)
    direction = torch.zeros(n_events, 3)
    direction[:, 2] = 1
    sf_dim = get_cond_dim(model_config, "showerflow")
    cond = {}
    if sf_dim > 1:
        cond["showerflow"] = torch.hstack([cond_E, direction])
    else:
        cond["showerflow"] = cond_E
    dif_dim = get_cond_dim(model_config, "diffusion")
    if dif_dim > 2:
        cond["diffusion"] = torch.hstack([cond_E, direction])
    elif dif_dim > 1:
        cond["diffusion"] = torch.hstack([cond_E, torch.zeros(n_events, 1)])
    else:
        cond["diffusion"] = cond_E
    return cond


def main(
    config,
    models,
    redo_g4_data=False,
    redo_model_data=True,
    min_energy=10,
    max_energy=100,
    seed=42,
    max_g4_events=0,
    max_model_events=40_000,
):
    """
    Run me like a script to create the metrics used in
    scripts/plotting/standard_metrics.ipynb
    """
    # The input config that will be used for the g4 data
    # plus to get detector ceilings and floors.
    # also, it's dataset path must be correct, or the g4 data will not be found.
    # so also hold onto it for the config of the models, which may have
    # incorrect dataset paths.
    seed_all(seed)
    meta = Metadata(config)
    floors, ceilings = detector_map.floors_ceilings(
        meta.layer_bottom_pos_hdf5, meta.cell_thickness_hdf5, 0
    )

    g4_name = "Geant 4"
    n_g4_events = np.sum(get_n_events(config.dataset_path, config.n_dataset_files))
    g4_save_path = get_path(config, g4_name, min_energy, max_energy, n_g4_events, seed)
    if max_g4_events:
        n_g4_events = min(n_g4_events, max_g4_events)

    # Get the g4 data
    if redo_g4_data or not os.path.exists(g4_save_path):
        print(f"Need to process {g4_name}")

        g4_cond, g4_events = sample_g4(config, n_g4_events)
        np.savez(
            g4_save_path,
            cond=g4_cond,
            events=g4_events,
        )

    for model_name in models:
        model, shower_flow, model_config = models[model_name]

        model_config.dataset_path_in_storage = False
        model_config.dataset_path = config.dataset_path
        model_config.n_dataset_files = config.n_dataset_files

        save_path = get_path(
            config, model_name, min_energy, max_energy, max_model_events, seed
        )
        # scale_factor_save_path = factor_get_path(config, model_name)

        if redo_model_data or not os.path.exists(save_path):
            print(f"Need to process {model_name}")

            if max_model_events:
                n_events = max_model_events
            else:
                n_events = n_g4_events
            cond = make_cond(model_config, min_energy, max_energy, n_events)

            model_events = conditioned_sample_model(
                model_config, cond, model, shower_flow
            )

            model_save_dict = {
                "cond_showerflow": cond["showerflow"],
                "cond_diffusion": cond["diffusion"],
                "events": model_events,
            }

            print(f"Saving {model_name} to {save_path}")
            np.savez(save_path, **model_save_dict)


scale_n = False
scale_e = False

# Gather the models to evaluate
# the dict has the format {model_name: (model, shower_flow, config)}
# the config should hold correct hyperparameters for the model,
# but the dataset_path may be incorrect.
models = {}
log_base = "../point-cloud-diffusion-logs/"
# log_base = "/beegfs/desy/user/dayhallh/point-cloud-diffusion-logs/"
log_base = "/data/dust/user/dayhallh/point-cloud-diffusion-logs"
# data_base = "../point-cloud-diffusion-data/"
# data_base = "/beegfs/desy/user/dayhallh/point-cloud-diffusion-data/"
data_base = "/data/dust/user/dayhallh/point-cloud-diffusion-data/"
torch.set_default_dtype(torch.float32)

# static_dataset = "/data/dust/user/dayhallh/data/ILCsoftEvents/p22_th90_ph90_en10-100_joined/p22_th90_ph90_en10-100_seed{}_all_steps.hdf5"
# static_n_files = 10
static_dataset = "/data/dust/user/dayhallh/data/ILCsoftEvents/highGran_g40_p22_th90_ph90_en10-100.hdf5"
static_n_files = 1

static_stats = np.load(
    "/data/dust/user/dayhallh/data/ILCsoftEvents/p22_th90_ph90_en10-100_joined/stats.npz"
)

angular_dataset = caloclouds_3.Configs().dataset_path
angular_n_files = caloclouds_3.Configs().n_dataset_files
try:
    pass
    if False:  # new a1 model
        model_name = "CaloClouds3-ShowerFlow_a1_fnorms_7"
        config = caloclouds_3.Configs()
        config.dataset_tag = "high_gran_g40_p22_th90_ph90_en10-100"
        config.device = "cpu"
        config.cond_features = 4
        config.diffusion_pointwise_hidden_l1 = 32
        config.distillation = True
        config.cond_features_names = ["energy", "p_norm_local"]

        if scale_n:
            config.shower_flow_n_scaling = True
            loaded = np.load(
                "/data/dust/user/dayhallh/CC2-out/nhits_rescale_cc3_unscaled_2.npz"
            )
            config.shower_flow_coef_real = np.zeros(2)
            # config.shower_flow_coef_real[0] = np.mean(loaded["single_factor"])
            config.shower_flow_coef_real[0] = 0.65
        else:
            config.shower_flow_n_scaling = False

        if scale_e:
            raise NotImplementedError

        caloclouds_paths = [
            "/data/dust/group/ilc/sft-ml/model_weights/CaloClouds/CC3/ckpt_0.000000_6135000.pt"
        ]
        # showerflow_paths = ["/data/dust/group/ilc/sft-ml/model_weights/CaloClouds/CC3/ShowerFlow_alt1_nb2_inputs8070450532247928831_fnorms_dhist_best.pth"]
        showerflow_paths = [
            "/data/dust/group/ilc/sft-ml/model_weights/CaloClouds/CC3/ShowerFlow_alt1_nb2_inputs8070450532247928831_fnorms_best.pth"
        ]
        # showerflow_paths = ["/data/dust/user/dayhallh/point-cloud-diffusion-data/showerFlow/sim-E1261AT600AP180-180/ShowerFlow_log1_nb2_inputs8070450532247928831_fnorms_dhist_try6_0000200.pth"]
        # showerflow_paths = [
        #    "/data/dust/user/dayhallh/point-cloud-diffusion-data/showerFlow/sim-E1261AT600AP180-180/ShowerFlow_alt1_nb2_inputs8070450532247928831_fnorms_dhist_try7_best.pth"
        # ]
        config.shower_flow_version = "alt1"

        caloclouds = get_caloclouds_models(
            caloclouds_paths=caloclouds_paths,
            showerflow_paths=showerflow_paths,
            caloclouds_names=["CaloClouds3"],
            showerflow_names=["ShowerFlow_a1_fnorms_7"],
            config=config,
        )

        # cc3_stats = np.load(showerflow_paths[0].replace(".pth", "_stats_cond_p22_th90_ph90_en10-100.npz"))

        # generate some custom metadata that will allow comparison between this model and the old model
        train_dataset_meta = Metadata(caloclouds_3.Configs())
        meta_here = Metadata(caloclouds_2.Configs())

        meta_here.incident_rescale = 127
        meta_here.n_pts_rescale = train_dataset_meta.n_pts_rescale
        meta_here.vis_eng_rescale = 3.4

        # try as in interance
        # meta_here.mean_cog[:] = [-4.06743696e-03, -2.27790998e-01,  1.10137465e+01]
        meta_here.mean_cog[:] = [-4.06743696e-03, 0.321829, 1.10137465e01]
        meta_here.std_cog[:] = [1.24559791, 0.95357278, 2.59475371]

        meta_here.log_incident_mean = train_dataset_meta.log_incident_mean
        meta_here.log_incident_std = train_dataset_meta.log_incident_std
        meta_here.found_attrs += ["log_incident_mean", "log_incident_std"]

        # internally, showers are assumed to be scaled between 0 and 1
        # but in cc3, they are actually normalised to std=0.5 mean=0
        # so we can alter Zmax_global, Zmin_global, Xmax_global and Xmin_global
        # to get the scaling needed
        Xmean, Ymean, Zmean = -0.0074305227, -0.21205868, 12.359252
        Xstd, Ystd, Zstd = 22.4728036, 23.65837968, 5.305082

        # meta_here.Xmax_global = Ymean
        # meta_here.Xmin_global = 2*Ystd - Ymean
        # meta_here.Zmax_global = Xmean
        # meta_here.Zmin_global = 2*Xstd - Xmean
        meta_here.Xmax_global = 2 * Ymean
        meta_here.Xmin_global = 2 * (2 * Ystd - Ymean)
        meta_here.Zmax_global = 2 * Xmean
        meta_here.Zmin_global = 2 * (2 * Xstd - Xmean)

        print("\n~~~~~~~~\n")
        print(repr(meta_here))
        print("\n~~~~~~~~\n")

        caloclouds["CaloClouds3-ShowerFlow_a1_fnorms_7"][2].metadata = meta_here

        models.update(caloclouds)

    if True:
        model_name = "CaloClouds2-ShowerFlow_CC2_8"
        config = caloclouds_2.Configs()
        config.dataset_tag = "highGran_g40_p22_th90_ph90_en10-100"
        config.device = "cpu"
        config.cond_features = (
            2  # number of conditioning features (i.e. energy+points=2)
        )
        config.cond_features_names = ["energy", "points"]
        config.shower_flow_cond_features = ["energy"]
        config.n_dataset_files = static_n_files
        config.dataset_path_in_storage = False
        config.dataset_path = static_dataset
        config.shower_flow_roll_xyz = True
        config.distillation = True

        # config.max_points = 6_000
        # config.max_points = 30_000

        if scale_n:
            config.shower_flow_n_scaling = True
            loaded = np.load(factor_get_path(config, model_name), allow_pickle=True)
            config.shower_flow_coef_real = loaded["real_coeff"]
            config.shower_flow_coef_fake = loaded["fake_coeff"]
        else:
            config.shower_flow_n_scaling = False

        if scale_e:
            raise NotImplementedError

        # showerflow_paths = ["/data/dust/group/ilc/sft-ml/model_weights/CaloClouds/CC2/220714_cog_e_layer_ShowerFlow_best.pth"]
        # showerflow_paths = ["/data/dust/user/dayhallh/point-cloud-diffusion-data/showerFlow/p22_th90_ph90_en10-100/ShowerFlow_original_nb10_inputs36893488147419103231_dhist_best.pth"]
        # showerflow_paths = [
        #    "/data/dust/user/dayhallh/point-cloud-diffusion-data/showerFlow/highGran_g40_p22_th90_ph90_en10-100/ShowerFlow_original_nb10_inputs36893488147419103231_dhist_best.pth"
        # ]
        showerflow_paths = [
            "/data/dust/user/dayhallh/point-cloud-diffusion-data/showerFlow/highGran_g40_p22_th90_ph90_en10-100/ShowerFlow_original_nb10_inputs36893488147419103231_dhist_try8_best.pth"
        ]
        caloclouds_paths = [
            "/data/dust/group/ilc/sft-ml/model_weights/CaloClouds/CC2/ckpt_0.000000_1000000.pt"
        ]

        caloclouds = get_caloclouds_models(
            caloclouds_paths=caloclouds_paths,
            showerflow_paths=showerflow_paths,
            caloclouds_names=["CaloClouds2"],
            showerflow_names=["ShowerFlow_CC2_8"],
            config=config,
        )

        # cc2_stats = np.load(showerflow_paths[0].replace(".pth", "_stats_cond_p22_th90_ph90_en10-100.npz"))

        train_dataset_meta = Metadata(config)
        # meta_here = Metadata(caloclouds_2.Configs())
        meta_here = Metadata(config)
        # meta_here.n_pts_rescale = 5000
        # meta_here.vis_eng_rescale = 2.5
        # meta_here.incident_rescale = 100
        # meta_here.std_cog = 1/cc2_stats["cog_x_std"], 1/cc2_stats["cog_y_std"], 1/cc2_stats["cog_z_std"]
        # meta_here.mean_cog = -cc2_stats["cog_x_mean"], -cc2_stats["cog_y_mean"], -cc2_stats["cog_z_mean"]
        # meta_here.mean_cog[:] = [0.3842599999999834, 0, 0.12772120012000343]
        # meta_here.std_cog[:] = [4.864242154098466**2, 1, 3.2035606359259203**2]

        print("\n~~~~~~~~\n")
        print("CC2")
        print(repr(meta_here))
        print(caloclouds["CaloClouds2-ShowerFlow_CC2_8"][2].max_points)
        print("\n~~~~~~~~\n")

        caloclouds["CaloClouds2-ShowerFlow_CC2_8"][2].metadata = meta_here

        models.update(caloclouds)
except FileNotFoundError as e:
    print("CaloClouds models not found")
    print(e)


if __name__ == "__main__":
    config = caloclouds_3.Configs()
    config.shower_flow_n_scaling = scale_n
    config.device = "cpu"
    config.dataset_path_in_storage = False
    config._dataset_path = static_dataset
    config.n_dataset_files = static_n_files
    config.logdir = "/data/dust/user/dayhallh/point-cloud-diffusion-logs/"
    # config._dataset_path = angular_dataset
    # config.n_dataset_files = angular_n_files
    # config.dataset_tag = "sim-E1261AT600AP180-180"
    # config.dataset_tag = "p22_th90_ph90_en10-100"
    config.dataset_tag = "highGran_p22_th90_ph90_en10-100"
    if not scale_n:
        config.dataset_tag += "_noNFactor"
    if not scale_e:
        config.dataset_tag += "_noEFactor"
    main(config, models)
