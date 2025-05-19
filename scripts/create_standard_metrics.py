"""
Script to create the metrics used in scripts/plotting/standard_metrics.ipynb
Please do add any new models!

TODO; should the binned data be stored inside the repo?
it's very lightweight after binning...
"""

import numpy as np
import torch
import os

from pointcloud.config_varients import caloclouds_3_simple_shower, caloclouds_3, default, caloclouds_2


from pointcloud.utils.metadata import Metadata
from pointcloud.utils.detector_map import floors_ceilings
from pointcloud.data.read_write import read_raw_regaxes, get_n_events
from pointcloud.data.conditioning import read_raw_regaxes_withcond

from pointcloud.utils.stats_accumulator import StatsAccumulator

# imports specific to this evaluation
from pointcloud.evaluation.bin_standard_metrics import (
    BinnedData,
    sample_g4,
    conditioned_sample_model,
    sample_accumulator,
    get_wish_models,
    get_fish_models,
    get_caloclouds_models,
)
from pointcloud.evaluation.bin_standard_metrics import get_path as base_get_path

def get_path(configs, dataset_name):
    if hasattr(configs, "dataset_tag"):
        dataset_name += "_" + configs.dataset_tag
    return base_get_path(configs, dataset_name)

# Gather the models to evaluate
# the dict has the format {model_name: (model, shower_flow, configs)}
# the configs should hold correct hyperparameters for the model,
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

static_stats = np.load("/data/dust/user/dayhallh/data/ILCsoftEvents/p22_th90_ph90_en10-100_joined/stats.npz")

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
        configs = caloclouds_3_simple_shower.Configs()
        configs.device = 'cpu'
        configs.cond_features = 4
        configs.cond_features_names = ["energy", "p_norm_local"]
        caloclouds_path = os.path.join(
           log_base,
           "sim-E1261AT600AP180-180/Anatoliis_cc_2.pt"
           #"p22_th90_ph90_en10-100/CD_2024_08_23__16_13_16/ckpt_0.439563_30000.pt"
           #"p22_th90_ph90_en10-100/CD_2024_08_23__16_13_16/ckpt_0.447468_870000.pt",
        )
        parts = [
           f"alt1_nb{repeats}_inputs8070450532247928831_fnorms"
           for repeats in [2, 10]
        ]
        showerflow_paths = [
           os.path.join(data_base, "showerFlow/sim-E1261AT600AP180-180",
           f"ShowerFlow_{part}_dhist_best.pth")
           for part in parts
        ]


        caloclouds = get_caloclouds_models(
            caloclouds_paths=[caloclouds_path], showerflow_paths=showerflow_paths, caloclouds_names=["CaloClouds3"], showerflow_names=[f"ShowerFlow_a1_fnorms_{i}" for i in [2, ]],
            configs=configs
        )

        # generate some custom metadata that will allow comparison between this model and the old model
        train_dataset_meta = Metadata(caloclouds_3_simple_shower.Configs())
        meta_here = Metadata(caloclouds_2.Configs())
        meta_here.n_pts_rescale = train_dataset_meta.n_pts_rescale

        #meta_here.n_pts_rescale = 500
        #meta_here.vis_eng_rescale = train_dataset_meta.vis_eng_rescale/meta_here.vis_eng_rescale
        #meta_here.std_cog = train_dataset_meta.std_cog/meta_here.std_cog
        meta_here.std_cog = train_dataset_meta.std_cog
        #meta_here.mean_cog = train_dataset_meta.mean_cog/meta_here.mean_cog
        meta_here.mean_cog = train_dataset_meta.mean_cog
        meta_here.incident_rescale = 127
    
        print('\n~~~~~~~~\n')
        print(repr(meta_here))
        print('\n~~~~~~~~\n')

        caloclouds["CaloClouds3-ShowerFlow_a1_fnorms_2"][2].metadata = meta_here

        models.update(caloclouds)

    if False:  #  old model, angular dataset
        configs = caloclouds_3.Configs()
        configs.device = 'cpu'
        configs.cond_features = 4
        configs.cond_features_names = ["energy", "p_norm_local"]
        parts = [
           "original_nb10_inputs36893488147419103231"
        ]
        showerflow_paths = [
           os.path.join(data_base, "showerFlow/sim-E1261AT600AP180-180",
           f"ShowerFlow_{part}_best.pth")
           for part in parts
        ]

        caloclouds = get_caloclouds_models(
            caloclouds_paths=[caloclouds_path], showerflow_paths=showerflow_paths, caloclouds_names=["CaloClouds3"], showerflow_names=[f"ShowerFlow_original_{i}" for i in [10]],
            configs=configs
        )
        models.update(caloclouds)

    if True:
        configs = caloclouds_2.Configs()
        configs.device = 'cpu'
        configs.cond_features = 2  # number of conditioning features (i.e. energy+points=2)
        configs.cond_features_names = ["energy", "points"]
        configs.shower_flow_cond_features = ["energy"]
        configs.n_dataset_files = static_n_files
        configs.dataset_path_in_storage = False
        configs.dataset_path = static_dataset
        showerflow_paths = ["/data/dust/user/dayhallh/point-cloud-diffusion-data/showerFlow/p22_th90_ph90_en10-100/ShowerFlow_original_nb10_inputs36893488147419103231_best.pth"]
        caloclouds_paths = ["/data/dust/user/dayhallh/point-cloud-diffusion-logs/from_anatoli/CC2/CD_2023_07_07__16_32_09/ckpt_0.000000_1120000.pt"]
        caloclouds = get_caloclouds_models(
            caloclouds_paths=caloclouds_paths, showerflow_paths=showerflow_paths, caloclouds_names=["CaloClouds2"], showerflow_names=["ShowerFlow_CC2"],
            configs=configs
        )

        train_dataset_meta = Metadata(configs)
        meta_here = Metadata(caloclouds_2.Configs())
        #meta_here.n_pts_rescale = train_dataset_meta.n_pts_rescale
        meta_here.n_pts_rescale = 7864
        meta_here.vis_eng_rescale = train_dataset_meta.vis_eng_rescale
        #meta_here.vis_eng_rescale = 3.4
        meta_here.std_cog = static_stats["std_cog"][[2, 0, 1]]
        meta_here.mean_cog = static_stats["mean_cog"][[2, 0, 1]]
        meta_here.mean_cog[:] = 0
        meta_here.incident_rescale = 100
        #meta_here.incident_rescale = 127

        #caloclouds["CaloClouds2-ShowerFlow_CC2"][2].max_points = 6_000
        caloclouds["CaloClouds2-ShowerFlow_CC2"][2].max_points = 5_000

        print('\n~~~~~~~~\n')
        print("CC2")
        print(repr(meta_here))
        print(caloclouds["CaloClouds2-ShowerFlow_CC2"][2].max_points)
        print('\n~~~~~~~~\n')

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
    configs,
    redo_g4_data=False,
    redo_g4_acc_data=False,
    redo_model_data=False,
    max_g4_events=10_000,
    max_model_events=10_000,
    models=models,
    accumulator_path=None, #accum_path,
):
    """
    Run me like a script to create the metrics used in
    scripts/plotting/standard_metrics.ipynb

    Parameters
    ----------
    configs : Configs
        The configs used to get g4 data, must have the correct dataset path.
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
        {model_name: (model, shower_flow, configs)}
        where model is a torch model, shower_flow is a shower flow model, and configs
        is a Configs object with the correct dataset path.
        The shower_flow model is only used for caloclouds models, so it can be None
    accumulator_path : str
        The path to the accumulator to use for the g4 accumulator data.
    """
    # The input configs that will be used for the g4 data
    # plus to get detector ceilings and floors.
    # also, it's dataset path must be correct, or the g4 data will not be found.
    # so also hold onto it for the configs of the models, which may have
    # incorrect dataset paths.
    meta = Metadata(configs)
    floors, ceilings = floors_ceilings(
        meta.layer_bottom_pos_hdf5, meta.cell_thickness_hdf5, 0
    )

    g4_name = "Geant 4"
    g4_save_path = get_path(configs, g4_name)
    n_g4_events = np.sum(get_n_events(configs.dataset_path, configs.n_dataset_files))
    if max_g4_events:
        n_g4_events = min(n_g4_events, max_g4_events)

    # Get the g4 data
    if redo_g4_data or not os.path.exists(g4_save_path):
        print(f"Need to process {g4_name}")

        raw_floors, raw_ceilings = floors_ceilings(
            meta.layer_bottom_pos_hdf5, meta.cell_thickness_hdf5, 0
        )
        xyz_limits = [
            [meta.Zmin_global, meta.Zmax_global],
            [meta.Xmin_global, meta.Xmax_global],
            [raw_floors[0], raw_ceilings[-1]],
        ]
        binned_g4 = BinnedData(
            "Geant 4",
            xyz_limits,
            1.0,
            meta.layer_bottom_pos_hdf5,
            meta.cell_thickness_hdf5,
            # meta.gun_xz_pos_raw)
            np.array([0, -50, 0]),
        )
        sample_g4(configs, binned_g4, n_g4_events)
        binned_g4.save(g4_save_path)
    else:
        binned_g4 = BinnedData.load(g4_save_path)

    # Get the model data
    model_data = []

    for model_name in models:
        model, shower_flow, model_configs = models[model_name]

        model_configs.dataset_path_in_storage = False
        model_configs.dataset_path = configs.dataset_path
        model_configs.n_dataset_files = configs.n_dataset_files


        save_path = get_path(configs, model_name)

        if redo_model_data or not os.path.exists(save_path):
            print(f"Need to process {model_name}")

            if max_model_events:
                n_events = min(n_g4_events, max_model_events)
            else:
                n_events = n_g4_events
            sample_cond, _= read_raw_regaxes_withcond(
                model_configs,
                total_size=n_events,
                for_model=["showerflow", "diffusion"],
            )

            # Standard normalised model output
            xyz_limits = [[-1, 1], [-1, 1], [0, 29]]
            layer_bottom_pos = np.linspace(-0.1, 28.9, 30)
            cell_thickness_global = 0.5
            rescale_energy = 1e3
            gun_pos = np.array([50, -50, 0])

            if "caloclouds" in model_name.lower():  # this model unnorms itself.

                cc_floors, cc_ceilings = floors_ceilings(
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

            binned = BinnedData(
                model_name,
                xyz_limits,
                rescale_energy,
                layer_bottom_pos,
                cell_thickness_global,
                gun_pos,
            )
            conditioned_sample_model(
                model_configs, binned, sample_cond, model, shower_flow
            )

            print(f"Saving {model_name} to {save_path}")
            binned.save(save_path)
        else:
            binned = BinnedData.load(save_path)
        model_data.append(binned)

    # Get some data from an accumulator too for good measure
    acc_name = "Geant 4 Accumulator"
    acc_save_path = get_path(configs, acc_name)

    if (
        (redo_g4_acc_data
        or not os.path.exists(acc_save_path))
        and accumulator_path is not None
    ):
        print(f"Need to process {acc_name}")

        xyz_limits = [[-1, 1], [-1, 1], [0, 29]]
        layer_bottom_pos = np.linspace(-0.1, 28.9, 30)
        cell_thickness_global = 1
        gun_pos = np.array([0, -70, 0])
        rescale_energy = 1e4
        binned_acc = BinnedData(
            acc_name,
            xyz_limits,
            rescale_energy,
            layer_bottom_pos,
            cell_thickness_global,
            gun_pos,
        )

        acc = StatsAccumulator.load(accumulator_path)
        sample_accumulator(configs, binned_acc, acc, n_g4_events)
        binned_acc.save(acc_save_path)
    else:
        binned_acc = BinnedData.load(acc_save_path)


if __name__ == "__main__":
    configs = caloclouds_3_simple_shower.Configs()
    configs.device = 'cpu'
    configs.dataset_path_in_storage = False
    configs._dataset_path = static_dataset
    configs.n_dataset_files = static_n_files
    configs.dataset_tag = "p22_th90_ph90_en10-100"
    main(configs)
