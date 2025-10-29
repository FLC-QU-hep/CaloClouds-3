"""
Script to create the metrics used in scripts/plotting/standard_metrics.ipynb
Please do add any new models!

TODO; should the binned data be stored inside the repo?
it's very lightweight after binning...
"""

import numpy as np
import torch
import os
import sys

from pointcloud.config_varients import (
    caloclouds_2,
    caloclouds_3,
)


from pointcloud.utils.metadata import Metadata
from pointcloud.utils import detector_map
from pointcloud.utils.misc import seed_all
from pointcloud.data.read_write import get_n_events
from pointcloud.data.conditioning import read_raw_regaxes_withcond

# imports specific to this evaluation
from pointcloud.evaluation.bin_standard_metrics import (
    DetectorBinnedData,
    BinnedData,
    sample_g4,
    conditioned_sample_model,
    get_caloclouds_models,
    get_path,
)
from pointcloud.evaluation.calculate_scale_factors import get_path as factor_get_path


def get_args():
    if len(sys.argv) > 1:
        detector_projection = sys.argv[1].strip().lower() == "true"
    else:
        detector_projection = True
    if len(sys.argv) > 2:
        scale_e_n = sys.argv[2].strip().lower() == "true"
    else:
        scale_e_n = False

    if len(sys.argv) > 3:
        cog_calibration = sys.argv[3].strip().lower() == "true"
    else:
        cog_calibration = True

    if len(sys.argv) > 4:
        seed = int(sys.argv[4])
        seed_all(seed)
    else:
        seed = None
    return detector_projection, scale_e_n, cog_calibration, seed


def get_dataset_path(static=True):
    static_dataset = "/data/dust/user/dayhallh/data/ILCsoftEvents/highGran_g40_p22_th90_ph90_en10-100.hdf5"
    static_n_files = 10

    angular_dataset = caloclouds_3.Configs().dataset_path
    angular_n_files = caloclouds_3.Configs().n_dataset_files

    if static:
        return static_dataset, static_n_files
    else:
        return angular_dataset, angular_n_files


def get_models(scale_e_n, cog_calibration):
    # Gather the models to evaluate
    # the dict has the format {model_name: (model, shower_flow, config)}
    # the config should hold correct hyperparameters for the model,
    # but the dataset_path may be incorrect.
    models = {}
    torch.set_default_dtype(torch.float32)

    try:
        pass
        if True:  # new a1 model
            config = caloclouds_3.Configs()
            config.dataset_tag = "p22_th90_ph90_en10-100"
            config.device = "cpu"
            config.cond_features = 4
            config.diffusion_pointwise_hidden_l1 = 32
            config.distillation = True
            config.cond_features_names = ["energy", "p_norm_local"]
            caloclouds_paths = [
                "/data/dust/group/ilc/sft-ml/model_weights/CaloClouds/CC3/ckpt_0.000000_6135000.pt"
            ]
            # showerflow_paths = ["/data/dust/group/ilc/sft-ml/model_weights/CaloClouds/CC3/ShowerFlow_alt1_nb2_inputs8070450532247928831_fnorms_dhist_best.pth"]
            showerflow_paths = [
                "/data/dust/group/ilc/sft-ml/model_weights/CaloClouds/CC3/ShowerFlow_alt1_nb2_inputs8070450532247928831_fnorms_best.pth"
            ]

            model_name = "CaloClouds3-ShowerFlow_a1_fnorms_2"
            if scale_e_n:
                config.shower_flow_n_scaling = True
                factor_path = factor_get_path(config, model_name)
                print(f"CC3 factor path: {factor_path}")
                loaded = np.load(factor_path, allow_pickle=True)
                config.shower_flow_coef_real = loaded["final_n_coeff"]
                print(f"CC3 real coeff: {config.shower_flow_coef_real}")
                config.shower_flow_scale_e = loaded["final_e_coeff"][-2]
                print(f"CC3 e coeff: {config.shower_flow_scale_e}")
            else:
                config.shower_flow_n_scaling = False
                config.shower_flow_coef_fake = None
                config.shower_flow_coef_real = None
                config.shower_flow_scale_e = None
            config.cog_calibration = cog_calibration

            caloclouds = get_caloclouds_models(
                caloclouds_paths=caloclouds_paths,
                showerflow_paths=showerflow_paths,
                caloclouds_names=["CaloClouds3"],
                showerflow_names=["ShowerFlow_a1_fnorms_2"],
                config=config,
            )

            # generate some custom metadata that will allow comparison between this model and the old model
            train_dataset_meta = Metadata(caloclouds_3.Configs())
            meta_here = Metadata(caloclouds_2.Configs())

            meta_here.incident_rescale = 127
            meta_here.n_pts_rescale = train_dataset_meta.n_pts_rescale
            meta_here.vis_eng_rescale = 3.5

            # meta_here.mean_cog[:] = [-4.06743696e-03, -2.27790998e-01,  1.10137465e+01]
            if config.cog_calibration:
                meta_here.mean_cog[:] = [40, 0, 0]
                meta_here.mean_cog[1] += -0.33
                meta_here.std_cog[:] = [0.53 / 0.39, 0.52 / 0.54, 1]
            else:
                meta_here.mean_cog[:] = [85, 45, 0]
                # x= 0.4, lp, y = 0.5, lp
                # meta_here.mean_cog[0] += -0.04  # from -40.04 to -40
                # meta_here.mean_cog[1] += 1.88  # from 2.24 to 0.36
                # 1 changes x coord, 1.88 is too left, 1.5 is too right
                meta_here.mean_cog[1] += 1.85  # c1
                meta_here.mean_cog[0] += -0.33
                meta_here.std_cog[:] = [0.53 / 0.39, 0.52 / 0.54, 1]

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

            caloclouds[model_name][2].metadata = meta_here

            models.update(caloclouds)

        if True:
            config = caloclouds_2.Configs()
            config.dataset_tag = "p22_th90_ph90_en10-100"
            config.device = "cpu"
            config.cond_features = (
                2  # number of conditioning features (i.e. energy+points=2)
            )
            config.cond_features_names = ["energy", "points"]
            config.shower_flow_cond_features = ["energy"]
            static_dataset, static_n_files = get_dataset_path(static=True)
            config.n_dataset_files = static_n_files
            config.dataset_path_in_storage = False
            config.dataset_path = static_dataset
            config.shower_flow_roll_xyz = True
            config.distillation = True
            showerflow_paths = [
                "/data/dust/user/dayhallh/point-cloud-diffusion-data/showerFlow/highGran_g40_p22_th90_ph90_en10-100/ShowerFlow_original_nb10_inputs36893488147419103231_dhist_try8_best.pth"
            ]

            caloclouds_paths = [
                "/data/dust/group/ilc/sft-ml/model_weights/CaloClouds/CC2/ckpt_0.000000_1000000.pt"
            ]
            model_name = "CaloClouds2-ShowerFlow_CC2"

            if scale_e_n:
                config.shower_flow_n_scaling = True
                factor_path = factor_get_path(config, model_name)
                print(f"CC2 factor path: {factor_path}")
                loaded = np.load(factor_path, allow_pickle=True)
                config.shower_flow_coef_fake = loaded["fake_coeff"]
                print(f"CC2 fake coeff: {config.shower_flow_coef_fake}")
                config.shower_flow_coef_real = loaded["real_coeff"]
                print(f"CC2 real coeff: {config.shower_flow_coef_real}")
                config.shower_flow_scale_e = loaded["final_e_coeff"][-2]
                print(f"CC2 e coeff: {config.shower_flow_scale_e}")
            else:
                config.shower_flow_n_scaling = False
                config.shower_flow_coef_fake = None
                config.shower_flow_coef_real = None
                config.shower_flow_scale_e = None
            config.cog_calibration = cog_calibration

            caloclouds = get_caloclouds_models(
                caloclouds_paths=caloclouds_paths,
                showerflow_paths=showerflow_paths,
                caloclouds_names=["CaloClouds2"],
                showerflow_names=["ShowerFlow_CC2"],
                config=config,
            )

            train_dataset_meta = Metadata(config)
            meta_here = Metadata(config)
            meta_here.mean_cog[:] = [0, 0, 0]
            meta_here.mean_cog[2] += -0.3
            meta_here.mean_cog[0] += -0.3
            # meta_here.mean_cog[2] += -0.38  # from -40.38 to -40
            # meta_here.mean_cog[0] += -0.48  # from -0.12 to 0.36
            meta_here.std_cog[:] = [0.52 / 0.37, 0, 0.53 / 0.86]
            meta_here.Zmin_global = -160
            meta_here.Zmax_global = 240
            meta_here.Xmin_global = -160
            meta_here.Xmax_global = 240

            print("\n~~~~~~~~\n")
            print("CC2")
            print(repr(meta_here))
            print(caloclouds[model_name][2].max_points)
            print("\n~~~~~~~~\n")

            caloclouds[model_name][2].metadata = meta_here

            models.update(caloclouds)
    except FileNotFoundError as e:
        print("CaloClouds models not found")
        print(e)
    return models


def get_configs(static=True):
    config = caloclouds_3.Configs()
    config.device = "cpu"
    config.dataset_path_in_storage = False
    dataset, n_files = get_dataset_path(static=static)
    config._dataset_path = dataset
    config.n_dataset_files = n_files
    if static:
        config.dataset_tag = "p22_th90_ph90_en10-100"
    else:
        config.dataset_tag = "sim-E1261AT600AP180-180"
    g4_gun = np.array([40, 50, 0])
    cc3_model_gun = cc2_model_gun = g4_gun
    cc2_model_gun[2] -= 0.1
    cc2_model_gun[0] += 0.2

    return config, g4_gun, cc2_model_gun, cc3_model_gun


def main(
    config,
    guns,
    detector_projection,
    scale_e_n,
    cog_calibration,
    redo_g4_data=True,
    redo_model_data=False,
    max_g4_events=10_000,
    max_model_events=10_000,
    models=None,
    **config_values,
):
    """
    Run me like a script to create the metrics used in
    scripts/plotting/standard_metrics.ipynb

    Parameters
    ----------
    detector_projection : bool
        If True, use the detector projection.
    scale_e_n : bool
        If True, use the ShowerFlow energy scaling.
    cog_calibration : bool
        If True, use the cog calibration.
    redo_g4_data : bool
        If True, ignore the g4 data on disk and recreate it.
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
    """
    g4_gun, cc2_model_gun, cc3_model_gun = guns
    # The input config that will be used for the g4 data
    # plus to get detector ceilings and floors.
    # also, it's dataset path must be correct, or the g4 data will not be found.
    # so also hold onto it for the config of the models, which may have
    # incorrect dataset paths.
    if models is None:
        models = get_models(scale_e_n, cog_calibration)
    meta = Metadata(config)
    if detector_projection:
        MAP, _ = detector_map.create_map(config=config)
        shifted_MAP = MAP[:]
        for layer in shifted_MAP:
            layer["xedges"] -= 30
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
                1e-3,
                shifted_MAP,
                meta.layer_bottom_pos_hdf5,
                meta.half_cell_size_global,
                meta.cell_thickness_hdf5,
                g4_gun,
                no_box_cut=True,
            )
        else:
            binned_g4 = BinnedData(
                "Geant 4",
                xyz_limits,
                1e-3,
                meta.layer_bottom_pos_hdf5,
                meta.cell_thickness_hdf5,
                g4_gun,
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
            rescale_energy = 1.0
            if detector_projection:
                if scale_e_n:
                    energy_correction = model_config.shower_flow_scale_e
                    print(f"Energy correction: {energy_correction}")
                    rescale_energy /= energy_correction
                    # rescale_energy /= 0.9880121837394529

            if "caloclouds" in model_name.lower():  # this model unnorms itself.
                cc_floors, cc_ceilings = detector_map.floors_ceilings(
                    meta.layer_bottom_pos_global, meta.cell_thickness_global, 0
                )
                raw_floors, raw_ceilings = detector_map.floors_ceilings(
                    meta.layer_bottom_pos_hdf5, meta.cell_thickness_hdf5, 0
                )
                xyz_limits = [
                    [meta.Zmin_global, meta.Zmax_global],
                    [meta.Xmin_global, meta.Xmax_global],
                    [cc_floors[0], cc_ceilings[-1]],
                    # [raw_floors[0], raw_ceilings[-1]],
                ]
                layer_bottom_pos = meta.layer_bottom_pos_global
                if "3" in model_name:
                    gun_pos = cc3_model_gun
                else:
                    gun_pos = cc2_model_gun
            print(f"{model_name} gun pos: {gun_pos}")

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
    detector_projection, scale_e_n, cog_calibration, seed = get_args()
    config, g4_gun, cc2_model_gun, cc3_model_gun = get_configs(
        cog_calibration, static=True
    )
    guns = [g4_gun, cc2_model_gun, cc3_model_gun]
    # for seed in range(100):
    if True:
        config.dataset_tag = "p22_th90_ph90_en10-100"
        if not scale_e_n:
            config.dataset_tag += "_noFactor"
        if not cog_calibration:
            config.dataset_tag += "_noCoGCalebration"
            config.cog_calibration = False
        if seed is not None:
            config.dataset_tag += f"_seed{seed}"
        # Only do g4 once
        redo_g4 = seed == 0 or seed is None
        main(config, guns, detector_projection, scale_e_n, cog_calibration, seed, redo_g4)
