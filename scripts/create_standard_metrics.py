"""
Script to create the metrics used in scripts/plotting/standard_metrics.ipynb
Please do add any new models!

TODO; should the binned data be stored inside the repo?
it's very lightweight after binning...
"""

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
from pointcloud.utils import detector_map, gen_utils
from pointcloud.data.read_write import read_raw_regaxes, get_n_events
from pointcloud.data.conditioning import read_raw_regaxes_withcond

from pointcloud.utils.stats_accumulator import StatsAccumulator

# imports specific to this evaluation
from pointcloud.evaluation.bin_standard_metrics import (
    DetectorBinnedData,
    BinnedData,
    sample_g4,
    conditioned_sample_model,
    sample_accumulator,
    get_wish_models,
    get_fish_models,
    get_caloclouds_models,
    get_path,
)
from pointcloud.evaluation.calculate_scale_factors import get_path as factor_get_path


detector_projection = True
scale_e_n = False

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

angular_dataset = caloclouds_3_simple_shower.Configs().dataset_path
angular_n_files = caloclouds_3_simple_shower.Configs().n_dataset_files

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
            "/data/dust/group/ilc/sft-ml/model_weights/CaloClouds/CC3/ShowerFlow_alt1_nb2_inputs8070450532247928831_fnorms_dhist_best.pth"
        ]
        # showerflow_paths = ["/data/dust/user/dayhallh/point-cloud-diffusion-data/showerlow/sim-E1261AT600AP180-180/ShowerFlow_log1_nb2_inputs8070450532247928831_fnorms_dhist_try6_0000200.pth"]
        config.shower_flow_version = "alt1"

        caloclouds = get_caloclouds_models(
            caloclouds_paths=caloclouds_paths,
            showerflow_paths=showerflow_paths,
            caloclouds_names=["CaloClouds3"],
            showerflow_names=["ShowerFlow_a1_fnorms_2"],
            config=config,
        )

        # generate some custom metadata that will allow comparison between this model and the old model
        train_dataset_meta = Metadata(caloclouds_3_simple_shower.Configs())
        meta_here = Metadata(caloclouds_2.Configs())

        meta_here.incident_rescale = 127
        meta_here.n_pts_rescale = train_dataset_meta.n_pts_rescale
        meta_here.vis_eng_rescale = 3.4

        # try as in interance
        meta_here.mean_cog[:] = [-4.06743696e-03, 0.321829, 1.10137465e01]
        meta_here.mean_cog[0] -= 40
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

        meta_here.Xmax_global = 2 * Ymean
        meta_here.Xmin_global = 2 * (2 * Ystd - Ymean)
        meta_here.Zmax_global = 2 * Xmean
        meta_here.Zmin_global = 2 * (2 * Xstd - Xmean)

        print("\n~~~~~~~~\n")
        print(repr(meta_here))
        print("\n~~~~~~~~\n")

        caloclouds["CaloClouds3-ShowerFlow_a1_fnorms_2"][2].metadata = meta_here

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
        config.cheat=False

        # config.max_points = 6_000
        # config.max_points = 30_000

        if scale_e_n:
            config.shower_flow_n_scaling = True
            loaded = np.load(factor_get_path(config, model_name), allow_pickle=True)
            config.shower_flow_coef_real = loaded["real_coeff"]
            #config.shower_flow_coef_fake = loaded["fake_coeff"]
            coeff_path = "/data/dust/user/dayhallh/CC2-out/nhits_rescale_cc2_try8_unscaled_3.npz"
            #config.shower_flow_coef_real = np.load(coeff_path)['coef_real']
            config.shower_flow_coef_fake = np.load(coeff_path)['coef_fake']
        else:
            config.shower_flow_n_scaling = False

        # showerflow_paths = ["/data/dust/group/ilc/sft-ml/model_weights/CaloClouds/CC2/220714_cog_e_layer_ShowerFlow_best.pth"]
        # showerflow_paths = ["/data/dust/user/dayhallh/point-cloud-diffusion-data/showerFlow/p22_th90_ph90_en10-100/ShowerFlow_original_nb10_inputs36893488147419103231_dhist_best.pth"]
        showerflow_paths = [
            "/data/dust/user/dayhallh/point-cloud-diffusion-data/showerFlow/highGran_g40_p22_th90_ph90_en10-100/ShowerFlow_original_nb10_inputs36893488147419103231_dhist_best.pth"
        ]
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
        meta_here.mean_cog[:] = [-40, 40, 0]
        meta_here.std_cog[:] = 1

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


# Toggle the redo flags to ignore files already on disk and redo datasets
# for example if you have retrained a model.
accum_path = os.path.join(
    log_base,
    "wish/dataset_accumulators/p22_th90_ph90_en10-1/p22_th90_ph90_en10-100_seedAll_all_steps.h5",
)
accum_path = None


def main(
    config,
    redo_g4_data=False,
    redo_g4_acc_data=False,
    redo_model_data=True,
    max_g4_events=10_000,
    max_model_events=10_000,
    models=models,
    accumulator_path=None,  # accum_path,
):
    """
    Run me like a script to create the metrics used in
    scripts/plotting/standard_metrics.ipynb

    Parameters
    ----------
    config : Configs
        The config used to get g4 data, must have the correct dataset path.
    redo_g4_data : bool
        If True, ignore the g4 data on disk and recreate it.
    redo_g4_acc_data : bool
        If True, ignore the g4 accumulator data on disk and recreate it.
    redo_model_data : bool
        If True, ignore any model data on disk and recreate data for all models given.
    max_g4_events : int
        Max number of G4 events to include in the bins. If 0, all events are included.
    max_model_events : int
        Max number of model events to include in the bins. If 0, as many events
        as there are G4 events are included.
    models : dict
        The models to evaluate, in the format
        {model_name: (model, shower_flow, config)}
        where model is a torch model, shower_flow is a shower flow model, and config
        is a Configs object with the correct dataset path.
        The shower_flow model is only used for caloclouds models, so it can be None
    accumulator_path : str
        The path to the accumulator to use for the g4 accumulator data.
    """
    # The input config that will be used for the g4 data
    # plus to get detector ceilings and floors.
    # also, it's dataset path must be correct, or the g4 data will not be found.
    # so also hold onto it for the config of the models, which may have
    # incorrect dataset paths.
    meta = Metadata(config)
    if detector_projection:
        MAP, _ = detector_map.create_map(config=config)
        shifted_MAP = MAP[:]
        for layer in shifted_MAP:
            layer["xedges"] -= 50
            layer["zedges"] -= 50
    floors, ceilings = detector_map.floors_ceilings(
        meta.layer_bottom_pos_hdf5, meta.cell_thickness_hdf5, 0
    )

    g4_name = "Geant 4"
    g4_save_path = get_path(config, g4_name, detector_projection)
    n_g4_events = np.sum(get_n_events(config.dataset_path, config.n_dataset_files))
    if max_g4_events:
        n_g4_events = min(n_g4_events, max_g4_events)

    # Get the g4 data
    if redo_g4_data or not os.path.exists(g4_save_path):
        print(f"Need to process {g4_name}")

        raw_floors, raw_ceilings = detector_map.floors_ceilings(
            meta.layer_bottom_pos_hdf5, meta.cell_thickness_hdf5, 0
        )
        xyz_limits = [
            [meta.Zmin_global, meta.Zmax_global],
            [meta.Xmin_global, meta.Xmax_global],
            [raw_floors[0], raw_ceilings[-1]],
        ]
        if detector_projection:
            binned_g4 = DetectorBinnedData(
                "Geant 4",
                xyz_limits,
                1.0,
                shifted_MAP,
                meta.layer_bottom_pos_hdf5,
                meta.half_cell_size_global,
                meta.cell_thickness_hdf5,
                # np.array([10, 40, 0]),
                np.array([-50, 0, 0]),
                no_box_cut=True,
            )
        else:
            binned_g4 = BinnedData(
                "Geant 4",
                xyz_limits,
                1.0,
                meta.layer_bottom_pos_hdf5,
                meta.cell_thickness_hdf5,
                # meta.gun_xz_pos_raw)
                np.array([10, 40, 0]),
                no_box_cut=True,
            )
        sample_g4(config, binned_g4, n_g4_events)
        binned_g4.save(g4_save_path)
    else:
        print(f"Found binned data for {g4_name} in {g4_save_path}")
        if detector_projection:
            binned_g4 = DetectorBinnedData.load(g4_save_path)
        else:
            binned_g4 = BinnedData.load(g4_save_path)

    for model_name in models:
        model, shower_flow, model_config = models[model_name]

        model_config.dataset_path_in_storage = False
        model_config.dataset_path = config.dataset_path
        model_config.n_dataset_files = config.n_dataset_files

        save_path = get_path(config, model_name, detector_projection)
        scale_factor_save_path = factor_get_path(config, model_name)

        if redo_model_data or not os.path.exists(save_path):
            print(f"Need to process {model_name}")

            if max_model_events:
                n_events = min(n_g4_events, max_model_events)
            else:
                n_events = n_g4_events
            sample_cond, _ = read_raw_regaxes_withcond(
                model_config,
                total_size=n_events,
                for_model=["showerflow", "diffusion"],
            )

            # Standard normalised model output
            xyz_limits = [[-1, 1], [-1, 1], [0, 29]]
            layer_bottom_pos = np.linspace(-0.1, 28.9, 30)
            cell_thickness_global = 0.5
            rescale_energy = 1e3
            if detector_projection:
                if scale_e_n:
                    energy_correction = np.load(
                        scale_factor_save_path, allow_pickle=True
                    )["final_e_coeff"]
                    rescale_energy /= energy_correction[-2]
                    # rescale_energy /= 0.9880121837394529
                    pass
                if "3" in model_name:
                    gun_pos = np.array([0, 0, 0])
                else:
                    gun_pos = np.array([0, 0, 0])
            else:
                if "3" in model_name:
                    gun_pos = np.array([60, 40, 0])
                else:
                    gun_pos = np.array([60, 40, 0])
            print(f"{model_name} gun pos: {gun_pos}")

            if "caloclouds" in model_name.lower():  # this model unnorms itself.

                cc_floors, cc_ceilings = detector_map.floors_ceilings(
                    meta.layer_bottom_pos_global, meta.cell_thickness_global, 0
                )
                xyz_limits = [
                    [meta.Zmin_global, meta.Zmax_global],
                    [meta.Xmin_global, meta.Xmax_global],
                    [cc_floors[0], cc_ceilings[-1]],
                ]
                layer_bottom_pos = meta.layer_bottom_pos_global
                rescale_energy = 1e3
            elif "fish" in model_name.lower():
                xyz_limits = [
                    [-1, 1],
                    [-1, 1],
                    [-1, 1],
                ]
                layer_bottom_pos = np.linspace(-0.75, 0.75, 30)
                cell_thickness = layer_bottom_pos[1] - layer_bottom_pos[0]
                gun_pos = np.array([0, -70, 0])

            if detector_projection:
                binned = DetectorBinnedData(
                    model_name,
                    xyz_limits,
                    rescale_energy,
                    shifted_MAP,
                    layer_bottom_pos,
                    meta.half_cell_size_global,
                    cell_thickness_global,
                    gun_pos,
                )
            else:
                binned = BinnedData(
                    model_name,
                    xyz_limits,
                    rescale_energy,
                    layer_bottom_pos,
                    cell_thickness_global,
                    gun_pos,
                )
            conditioned_sample_model(
                model_config, binned, sample_cond, model, shower_flow
            )

            print(f"Saving {model_name} to {save_path}")
            binned.save(save_path)
        else:
            print(f"Found binned data for {model_name} in {save_path}")
            if detector_projection:
                binned = DetectorBinnedData.load(save_path)
            else:
                binned = BinnedData.load(save_path)


if __name__ == "__main__":
    config = caloclouds_3_simple_shower.Configs()
    config.device = "cpu"
    config.dataset_path_in_storage = False
    config._dataset_path = static_dataset
    config.n_dataset_files = static_n_files
    # config._dataset_path = angular_dataset
    # config.n_dataset_files = angular_n_files
    # config.dataset_tag = "sim-E1261AT600AP180-180"
    # config.dataset_tag = "p22_th90_ph90_en10-100"
    config.dataset_tag = "p22_th90_ph90_en10-100"
    if not scale_e_n:
        config.dataset_tag += "_noFactor"
    main(config)
