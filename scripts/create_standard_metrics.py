"""
Script to create the metrics used in scripts/plotting/standard_metrics.ipynb
Please do add any new models!

TODO; should the binned data be stored inside the repo?
it's very lightweight after binning...
"""

import numpy as np
import torch
import os

from pointcloud.config_varients.caloclouds_3_simple_shower import Configs

from pointcloud.utils.metadata import Metadata
from pointcloud.utils.detector_map import floors_ceilings
from pointcloud.data.read_write import read_raw_regaxes, get_n_events
from pointcloud.utils.stats_accumulator import StatsAccumulator

# imports specific to this evaluation
from pointcloud.evaluation.bin_standard_metrics import (
    BinnedData,
    sample_g4,
    sample_model,
    sample_accumulator,
    get_path,
    get_wish_models,
    get_fish_models,
    get_caloclouds_models,
)

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
precision = "float32"
sf_precision = "float32"
torch.set_default_dtype(torch.float32)

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
    configs = Configs()
    configs.device = 'cpu'
    configs.diffusion_precision = precision
    configs.showerflow_precision = sf_precision
    configs.cond_features = 4
    configs.cond_features_names = ["energy", "p_norm_local"]
    caloclouds_path = os.path.join(
       log_base,
       "sim-E1261AT600AP180-180/Anatoliis_cc_2.pt"
       #"p22_th90_ph90_en10-100/CD_2024_08_23__16_13_16/ckpt_0.439563_30000.pt"
       #"p22_th90_ph90_en10-100/CD_2024_08_23__16_13_16/ckpt_0.447468_870000.pt",
    )
    caloclouds_16_path = "/data/dust/user/dayhallh/point-cloud-diffusion-logs/CD_sim-E1261AT600AP180-1802025_02_04__15_00_20_float16/ckpt_unscored_2300000.pt"
    caloclouds_64_path = "/data/dust/user/dayhallh/point-cloud-diffusion-logs/CD_sim-E1261AT600AP180-1802025_02_04__15_07_10_float64/ckpt_0.428667_1530000.pt"
    parts = [
       f"alt1_nb{repeats}_inputs8070450532247928831_fnorms"
       for repeats in [2, 10]
    ]
    showerflow_paths = [
       os.path.join(data_base, "showerFlow/sim-E1261AT600AP180-180",
       f"ShowerFlow_{part}_dhist_best.pth")
       for part in parts
    ]
    showerflow_16_path = "/data/dust/user/dayhallh/point-cloud-diffusion-data/showerFlow/sim-E1261AT600AP180-180/ShowerFlow_alt1_nb2_inputs8070450532247928831_fnorms_precfloat16_best.pth"
    showerflow_64_path = "/data/dust/user/dayhallh/point-cloud-diffusion-data/showerFlow/sim-E1261AT600AP180-180/ShowerFlow_alt1_nb2_inputs8070450532247928831_fnorms_precfloat64_best.pth"

    caloclouds = get_caloclouds_models(
        caloclouds_paths=[caloclouds_path], showerflow_paths=showerflow_paths, caloclouds_names=["CaloClouds3"], showerflow_names=[f"ShowerFlow_a1_fnorms_{i}" for i in [2, ]],
        configs=configs
    )
    models.update(caloclouds)
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
    configs=Configs(),
    redo_g4_data=False,
    redo_g4_acc_data=False,
    redo_model_data=False,
    max_g4_events=10_000,
    max_model_events=10_000,
    models=models,
    accumulator_path=accum_path,
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
    configs.showerflow_precision = sf_precision
    configs.diffusion_precision = precision
    # The input configs that will be used for the g4 data
    # plus to get detector ceilings and floors.
    # also, it's dataset path must be correct, or the g4 data will not be found.
    # so also hold onto it for the configs of the models, which may have
    # incorrect dataset paths.
    dataset_path = configs.dataset_path
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
        model_configs.showerflow_precision = sf_precision
        model_configs.diffusion_precision = precision
        # configs.logdir = "/home/dayhallh/training/point-cloud-diffusion-logs"
        model_configs.dataset_path = dataset_path
        model_configs.n_dataset_files = 10

        save_path = get_path(configs, model_name)

        if redo_model_data or not os.path.exists(save_path):
            print(f"Need to process {model_name}")

            if max_model_events:
                n_events = min(n_g4_events, max_model_events)
            else:
                n_events = n_g4_events

            # Standard normalised model output
            xyz_limits = [[-1, 1], [-1, 1], [0, 29]]
            layer_bottom_pos = np.linspace(-0.1, 28.9, 30)
            cell_thickness_global = 0.5
            rescale_energy = 1e3
            gun_pos = np.array([0, -70, 0])

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
                gun_pos = np.array([0, -50, 0])
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
            some_energies, some_events = sample_model(
                model_configs, binned, n_events, model, shower_flow
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

    try:
        # Print some stats for a sanity check
        g4_energy, g4_events = read_raw_regaxes(configs, pick_events=slice(0, 100))
        some_energies = some_energies.detach().cpu().numpy()
        for name, (
            type_inci,
            type_events,
        ) in {
            "g4": (g4_energy, g4_events),
            "model": (some_energies, some_events),
        }.items():
            print(name)
            es = type_events[:, :, 3]
            mask = es > 0
            es = es[mask]
            xs = type_events[:, :, 0][mask]
            ys = type_events[:, :, 1][mask]
            zs = type_events[:, :, 2][mask]

            for func in [np.min, np.max, np.mean, np.std]:
                print(
                    f"{func.__name__} \t [{func(type_inci):.2f}] -> "
                    f"({func(xs):.2f}, {func(ys):.2f}, "
                    f"{func(zs):.2f}, {func(es*100):.2f})"
                )
            print()
    except Exception:
        print("Could not print stats")


if __name__ == "__main__":
    main()
