# in the public repo this is calc_timing.py
# using a single thread
import os

import time
import numpy as np
import torch
import os

from pointcloud.config_varients import (
    caloclouds_3_simple_shower,
    caloclouds_3,
    default,
    caloclouds_2,
)


from pointcloud.utils.metadata import Metadata
from pointcloud.utils import detector_map
from pointcloud.data.read_write import read_raw_regaxes, get_n_events
from pointcloud.data.conditioning import read_raw_regaxes_withcond

from pointcloud.utils.gen_utils import gen_cond_showers_batch

# imports specific to this evaluation
from pointcloud.evaluation.bin_standard_metrics import (
    get_wish_models,
    get_fish_models,
    get_caloclouds_models,
)
from pointcloud.evaluation.bin_standard_metrics import get_path as base_get_path
from pointcloud.evaluation.calculate_scale_factors import get_path as factor_get_path


os.environ["OPENBLAS_NUM_THREADS"] = "1"  # to run numpy single threaded
detector_projection = True
scale_e_n = True

smallest_batch = 1
largest_batch = 100


def time_model_fixed(
    model_config,
    model,
    shower_flow,
    MAP=None,
    layer_bottom_pos=None,
    half_cell_size_global=None,
    cell_thickness=None,
):
    model_config.device = "cpu"
    batch_size = 100
    results = {"batch_size": [], "energy": [], "time": []}
    cond, _ = read_raw_regaxes_withcond(
        model_config,
        total_size=batch_size,
        for_model=["showerflow", "diffusion"],
    )

    print()
    for energy_here in range(9, 101):
        print(f"{energy_here-100} runs remaining", end="\n")
        if len(cond["showerflow"].shape) > 1:
            cond["showerflow"][:, 0] = energy_here
        else:
            cond["showerflow"][:] = energy_here
        cond["diffusion"][:, 0] = energy_here

        if detector_projection:
            start_time = time.time()
            events = gen_cond_showers_batch(
                model, shower_flow, cond, bs=batch_size, config=model_config
            )
            detector_map.get_projections(
                events,
                MAP,
                layer_bottom_pos=layer_bottom_pos,
                half_cell_size_global=half_cell_size_global,
                cell_thickness_global=cell_thickness,
                return_cell_point_cloud=False,
                include_artifacts=False,
            )
            end_time = time.time()
            del events
            del events_as_cells
        else:
            start_time = time.time()
            events = gen_cond_showers_batch(
                model, shower_flow, cond, bs=batch_size, config=model_config
            )
            end_time = time.time()
            del events
        if not energy_here < 10:
            results["batch_size"].append(batch_size)
            results["energy"].append(energy_here)
            results["time"].append(end_time - start_time)
        now = end_time
    print()
    return results


def save_results_fixed(config, model_name, results):
    path = base_get_path(config, model_name, True)
    if detector_projection:
        path = path.replace("_detectorProj", "_timingb100_detectorProj")
    else:
        path = path.replace("_detectorProj", "_timingb100")
    print(f"Saving to {path}")
    np.savez(path, **results)


def time_model(
    model_config,
    model,
    shower_flow,
    MAP=None,
    layer_bottom_pos=None,
    half_cell_size_global=None,
    cell_thickness=None,
    total_test_time=60 * 60,
):
    model_config.device = "cpu"
    test_end = time.time() + total_test_time
    n_cond = 10_000
    cond, _ = read_raw_regaxes_withcond(
        model_config,
        total_size=n_cond,
        for_model=["showerflow", "diffusion"],
    )
    n_cond = len(cond["showerflow"])
    results = {"batch_size": [], "energy": [], "time": []}
    first_pull = True

    now = time.time()
    print()
    i = 0
    while now < test_end:
        i += 1
        if i > 10:
            i = 0
            print(f"{test_end-now} seconds remaining", end="\r")

        if first_pull:
            batch_size = largest_batch
            first_pull = False
        else:
            batch_size = np.random.randint(smallest_batch, largest_batch)
        cond_idx = np.random.randint(0, n_cond - 1)
        if len(cond["showerflow"].shape) > 1:
            energy_here = cond["showerflow"][cond_idx, 0]
        else:
            energy_here = cond["showerflow"][cond_idx]
        cond_here = {k: cond[k][[cond_idx] * batch_size] for k in cond.keys()}
        if detector_projection:
            start_time = time.time()
            events = gen_cond_showers_batch(
                model, shower_flow, cond_here, bs=batch_size, config=model_config
            )
            detector_map.get_projections(
                events,
                MAP,
                layer_bottom_pos=layer_bottom_pos,
                half_cell_size_global=half_cell_size_global,
                cell_thickness_global=cell_thickness,
                return_cell_point_cloud=False,
                include_artifacts=False,
            )
            end_time = time.time()
            del events
            del events_as_cells
        else:
            start_time = time.time()
            events = gen_cond_showers_batch(
                model, shower_flow, cond_here, bs=batch_size, config=model_config
            )
            end_time = time.time()
            del events
        if not first_pull:
            results["batch_size"].append(batch_size)
            results["energy"].append(energy_here)
            results["time"].append(end_time - start_time)
        now = end_time
    print()
    return results


def save_results(config, model_name, results):
    path = base_get_path(config, model_name, True)
    if detector_projection:
        path = path.replace("_detectorProj", "_timing_detectorProj")
    else:
        path = path.replace("_detectorProj", "_timing")
    print(f"Saving to {path}")
    np.savez(path, **results)


# time_model = time_model_fixed
# save_results = save_results_fixed

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

static_dataset = "/data/dust/user/dayhallh/data/ILCsoftEvents/p22_th90_ph90_en10-100_joined/p22_th90_ph90_en10-100_seed{}_all_steps.hdf5"
static_n_files = 10

static_stats = np.load(
    "/data/dust/user/dayhallh/data/ILCsoftEvents/p22_th90_ph90_en10-100_joined/stats.npz"
)

angular_dataset = caloclouds_3_simple_shower.Configs().dataset_path
angular_n_files = caloclouds_3_simple_shower.Configs().n_dataset_files


try:
    pass
#    wish_path = os.path.join(
#        log_base, "wish/dataset_accumulators/p22_th90_ph90_en10-1/try2_wish_poly{}.pt"
#    )
#    models.update(
#        get_wish_models(
#            wish_path=wish_path,
#            n_poly_degrees=4,
#        )
#    )
except FileNotFoundError as e:
    print("Wish models not found")
    print(e)

try:
    pass
#    fish_path = os.path.join(
#        log_base, "wish/fish/fish.npz"
#    )
#    models.update(
#        get_fish_models(fish_path=fish_path)
#    )
except FileNotFoundError as e:
    print("Wish models not found")
    print(e)
try:
    pass
    if True:  # new a1 model
        model_name = "CaloClouds3-ShowerFlow_a1_fnorms_2"
        config = caloclouds_3_simple_shower.Configs()
        config.dataset_tag = "p22_th90_ph90_en10-100"
        config.device = "cpu"
        config.cond_features = 4
        config.diffusion_pointwise_hidden_l1 = 32
        config.distillation = True
        config.cond_features_names = ["energy", "p_norm_local"]

        if scale_e_n:
            config.shower_flow_n_scaling = True
            loaded = np.load(factor_get_path(config, model_name), allow_pickle=True)
            config.shower_flow_coef_real = loaded["final_n_coeff"]
            # config.shower_flow_coef_real = np.zeros(2)
            # config.shower_flow_coef_real[0] =  0.7
        else:
            config.shower_flow_n_scaling = False

        caloclouds_paths = [
            "/data/dust/group/ilc/sft-ml/model_weights/CaloClouds/CC3/ckpt_0.000000_6135000.pt"
        ]
        # showerflow_paths = ["/data/dust/group/ilc/sft-ml/model_weights/CaloClouds/CC3/ShowerFlow_alt1_nb2_inputs8070450532247928831_fnorms_dhist_best.pth"]
        showerflow_paths = [
            "/data/dust/group/ilc/sft-ml/model_weights/CaloClouds/CC3/ShowerFlow_alt1_nb2_inputs8070450532247928831_fnorms_best.pth"
        ]

        caloclouds = get_caloclouds_models(
            caloclouds_paths=caloclouds_paths,
            showerflow_paths=showerflow_paths,
            caloclouds_names=["CaloClouds3"],
            showerflow_names=["ShowerFlow_a1_fnorms_2"],
            config=config,
        )

        dataset_stats = np.load(
            "/data/dust/user/dayhallh/data/ILCsoftEvents/p22_th90_ph90_en10-100_joined/stats.npz"
        )
        # cc3_stats = np.load(showerflow_paths[0].replace(".pth", "_stats_cond_p22_th90_ph90_en10-100.npz"))

        # generate some custom metadata that will allow comparison between this model and the old model
        train_dataset_meta = Metadata(caloclouds_3_simple_shower.Configs())
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

        caloclouds["CaloClouds3-ShowerFlow_a1_fnorms_2"][2].metadata = meta_here

        models.update(caloclouds)

    if False:  #  old model, angular dataset
        config = caloclouds_3.Configs()
        config.device = "cpu"
        config.cond_features = 4
        config.cond_features_names = ["energy", "p_norm_local"]
        parts = ["original_nb10_inputs36893488147419103231"]
        showerflow_paths = [
            os.path.join(
                data_base,
                "showerFlow/sim-E1261AT600AP180-180",
                f"ShowerFlow_{part}_best.pth",
            )
            for part in parts
        ]

        caloclouds = get_caloclouds_models(
            caloclouds_paths=[caloclouds_path],
            showerflow_paths=showerflow_paths,
            caloclouds_names=["CaloClouds3"],
            showerflow_names=[f"ShowerFlow_original_{i}" for i in [10]],
            config=config,
        )
        models.update(caloclouds)

    if True:
        model_name = "CaloClouds2-ShowerFlow_CC2"
        config = caloclouds_2.Configs()
        config.dataset_tag = "p22_th90_ph90_en10-100"
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

        if scale_e_n:
            config.shower_flow_n_scaling = True
            loaded = np.load(factor_get_path(config, model_name), allow_pickle=True)
            config.shower_flow_coef_real = loaded["real_coeff"]
            config.shower_flow_coef_fake = loaded["fake_coeff"]
        else:
            config.shower_flow_n_scaling = False

        # showerflow_paths = ["/data/dust/group/ilc/sft-ml/model_weights/CaloClouds/CC2/220714_cog_e_layer_ShowerFlow_best.pth"]
        showerflow_paths = [
            "/data/dust/user/dayhallh/point-cloud-diffusion-data/showerFlow/p22_th90_ph90_en10-100/ShowerFlow_original_nb10_inputs36893488147419103231_dhist_best.pth"
        ]
        caloclouds_paths = [
            "/data/dust/group/ilc/sft-ml/model_weights/CaloClouds/CC2/ckpt_0.000000_1000000.pt"
        ]

        caloclouds = get_caloclouds_models(
            caloclouds_paths=caloclouds_paths,
            showerflow_paths=showerflow_paths,
            caloclouds_names=["CaloClouds2"],
            showerflow_names=["ShowerFlow_CC2"],
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
        print(caloclouds["CaloClouds2-ShowerFlow_CC2"][2].max_points)
        print("\n~~~~~~~~\n")

        caloclouds["CaloClouds2-ShowerFlow_CC2"][2].metadata = meta_here

        models.update(caloclouds)
except FileNotFoundError as e:
    print("CaloClouds models not found")
    print(e)


def main(
    config,
    models=models,
):
    meta = Metadata(config)
    MAP, _ = detector_map.create_map(config=config)

    for model_name in models:
        model, shower_flow, model_config = models[model_name]

        model_config.dataset_path_in_storage = False
        model_config.dataset_path = config.dataset_path
        model_config.n_dataset_files = config.n_dataset_files

        print(f"Need to process {model_name}")

        # Standard normalised model output
        layer_bottom_pos = np.linspace(-0.1, 28.9, 30)
        cell_thickness_global = 0.5
        rescale_energy = 1e3
        print(f"{model_name}")

        if "caloclouds" in model_name.lower():  # this model unnorms itself.
            layer_bottom_pos = meta.layer_bottom_pos_global
        elif "fish" in model_name.lower():
            layer_bottom_pos = np.linspace(-0.75, 0.75, 30)
            cell_thickness = layer_bottom_pos[1] - layer_bottom_pos[0]

        meta = Metadata(config)
        results = time_model(
            model_config,
            model,
            shower_flow,
            MAP,
            layer_bottom_pos,
            meta.half_cell_size_global,
            meta.cell_thickness_global,
        )
        save_results(config, model_name, results)


if __name__ == "__main__":
    config = caloclouds_3_simple_shower.Configs()
    config.device = "cpu"
    config.dataset_path_in_storage = False
    config._dataset_path = static_dataset
    config.n_dataset_files = static_n_files
    # config._dataset_path = angular_dataset
    # config.n_dataset_files = angular_n_files
    config.dataset_tag = "p22_th90_ph90_en10-100"
    # config.dataset_tag = "sim-E1261AT600AP180-180"
    main(config)
