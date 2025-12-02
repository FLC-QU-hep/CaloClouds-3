# # Showerflow
#
# Building and training showerflow from data.
#
# Start by setting up notebook enviroment.

from tqdm import tqdm

import torch
import glob

import ot
import numpy as np
import os
import time

from pointcloud.config_varients import (
    wish,
    wish_maxwell,
    caloclouds_2,
    caloclouds_2_v2,
    caloclouds_2,
    caloclouds_2_v4,
    caloclouds_3,
    caloclouds_3,
)
from pointcloud.data.read_write import get_n_events
from pointcloud.data.conditioning import get_cond_dim
from pointcloud.utils import showerflow_training, showerflow_utils, misc
from pointcloud.models import shower_flow


def get_data(config, model_path, batch_size=2048, n_showerflow_events=100_000):
    """
    Really this is a script, but for ease of testing, it's the main function.
    """

    shower_flow_compiler = shower_flow.versions_dict[config.shower_flow_version]

    cond_dim = get_cond_dim(config, "showerflow")
    inputs_used = showerflow_utils.get_input_mask(config)
    input_dim = np.sum(inputs_used)

    # use cuda if avaliable
    if torch.cuda.is_available():
        config.device = "cuda"
    else:
        config.device = "cpu"
    device = torch.device(config.device)

    model, distribution, transforms = shower_flow_compiler(
        num_blocks=config.shower_flow_num_blocks,
        num_inputs=input_dim,
        num_cond_inputs=cond_dim,
        device=device,
    )  # num_cond_inputs
    model.load_state_dict(
        torch.load(model_path, map_location=config.device, weights_only=False)["model"]
    )
    # ## Setup
    #
    # Load the data and check it's properties.
    print(f"Using dataset from {config.dataset_path}")
    # ## Data prep
    #
    # Now some values are needed to aid the showerflow training.

    model.to(device)
    model.eval()

    input_data, context = get_input_data(
        config, model_path, n_input_events=n_showerflow_events
    )
    context = torch.tensor(context).float().to(device)
    n_showerflow_events = len(context)

    model_data = []
    for start_point in tqdm(range(0, n_showerflow_events, batch_size)):
        context_here = context[start_point : start_point + batch_size]
        model_out = distribution.condition(context_here).sample([len(context_here)])
        model_data.append(model_out.detach().cpu().numpy())
    model_data = np.concatenate(model_data, axis=0)

    return model_data, input_data, context


def get_input_data(config, model_path, n_input_events=100_000):
    """
    Really this is a script, but for ease of testing, it's the main function.
    """
    misc.seed_all(50)
    _, input_save_path, _ = get_save_paths(config, model_path)

    if os.path.exists(input_save_path):
        loaded = np.load(input_save_path)
        return loaded["input_data"], loaded["context"]

    # ## Data prep
    #
    # Now some values are needed to aid the showerflow training.

    cond_dim = get_cond_dim(config, "showerflow")
    showerflow_dir = showerflow_utils.get_showerflow_dir(config)
    print(f"Showerflow data will be saved to {showerflow_dir}")
    os.makedirs(showerflow_dir, exist_ok=True)
    if config.device == "cuda":
        # on the gpu we can use all the data
        # local_batch_size = n_events
        local_batch_size = 100_000
    else:
        local_batch_size = 10_000

    # ## Setup
    #
    # Load the data and check it's properties.
    print(f"Using dataset from {config.dataset_path}")

    pointsE_path = showerflow_training.get_incident_npts_visible(
        config, showerflow_dir, redo=False, local_batch_size=local_batch_size
    )

    if "p_norm_local" in config.shower_flow_cond_features:
        direction_path = showerflow_training.get_gun_direction(
            config, showerflow_dir, redo=False, local_batch_size=local_batch_size
        )
    else:
        direction_path = None
    # Calculating clusters per layer takes ~ 5 mins, so it's saved between runs.

    clusters_per_layer_path = showerflow_training.get_clusters_per_layer(
        config, showerflow_dir, redo=False, local_batch_size=local_batch_size
    )

    energy_per_layer_path = showerflow_training.get_energy_per_layer(
        config, showerflow_dir, redo=False, local_batch_size=local_batch_size
    )

    # center of gravity

    cog_path, cog = showerflow_training.get_cog(
        config, showerflow_dir, redo=False, local_batch_size=local_batch_size
    )

    # ## Training data
    #
    # We now have everything needed to trian showerflow,
    # we can add it to a pandas dataframe and run the training loop
    # As the dataset may be larger than can be carried in the ram,
    # this will be repeated for each epoch.

    make_train_ds = showerflow_training.train_ds_function_factory(
        pointsE_path,
        cog_path,
        clusters_per_layer_path,
        energy_per_layer_path,
        config,
        direction_path=direction_path,
    )

    print(f"Getting input data sample")
    dataset = make_train_ds(0, n_input_events)

    input_data = dataset[:, cond_dim:]
    context = dataset[:, :cond_dim]
    save_dict = {
        "input_data": input_data,
        "context": context,
    }
    np.savez(input_save_path, **save_dict)
    return input_data, context


def get_save_paths(config, model_path):
    out_path = "/data/dust/user/dayhallh/point-cloud-diffusion-data/showerFlow/sampling"
    input_save_path = config.dataset_path.format("stack_of_training_data")
    if "stack_of_training_data" not in input_save_path:
        input_save_path += "stack_of_training_data.npz"
    else:
        input_save_path += ".npz"
    train_ds_name = model_path.split("/")[-2]
    showerflow_config = model_path.split("/")[-1][11:-4]
    f_start = os.path.join(
        out_path, f"pureShowerflow_{train_ds_name}_{showerflow_config}"
    )
    done = glob.glob(f_start + "*.npz")
    timestamp = time.strftime("%Y-%m-%d_%H-%M")
    output_save_path = f"{f_start}_{timestamp}.npz"
    return done, input_save_path, output_save_path


def calculate_wass(model_data_path, g4_data_path):
    model_data = np.load(model_data_path)["showerflow_out"]
    g4_data = np.load(g4_data_path)
    wass = ot.sliced.sliced_wasserstein_distance(
        model_data, g4_data, n_projections=1000
    )
    return wass


def main(
    config,
    model_path,
    n_showerflow_events=100_000,
):
    done, input_save_path, output_save_path = get_save_paths(config, model_path)
    if done:
        print("Already done")
        # return

    # get the data
    model_data, input_data, context = get_data(
        config,
        model_path,
        n_showerflow_events=n_showerflow_events,
    )

    if input_data is not None:
        np.save(input_save_path, input_data)
    else:
        input_data = np.load(input_save_path)

    wass = ot.sliced.sliced_wasserstein_distance(
        np.array(model_data), np.array(input_data), n_projections=1000
    )

    if done:
        existing_wass = np.load(done[0])["wass"]
        if existing_wass.shape == 0:
            existing_wass = np.array([existing_wass])
        wass = np.append(existing_wass, wass)

    np.savez(output_save_path, showerflow_out=model_data, context=context, wass=wass)


if __name__ == "__main__":
    # check user input
    # and if not given get it from the user
    cc_version = 3
    if cc_version == 3:
        config = caloclouds_3.Configs()
        config.dataset_path = "/data/dust/group/ilc/sft-ml/datasets/sim-E1261AT600AP180-180/sim-E1261AT600AP180-180_file_{}.slcio.hdf5"
        config.n_dataset_files = 88
        config.shower_flow_version = "alt1"
        # config.shower_flow_version = "log1"
        config.shower_flow_num_blocks = 2
        config.shower_flow_detailed_history = True
        config.shower_flow_weight_decay = 0.0
        config.shower_flow_tag = "try6"
        config.max_energy = 127
    elif cc_version == 2:
        config = caloclouds_2.Configs()
        config.dataset_path = "/data/dust/user/dayhallh/data/ILCsoftEvents/highGran_g40_p22_th90_ph90_en10-100.hdf5"
        config.n_dataset_files = 1
        config.shower_flow_version = "original"
        config.shower_flow_num_blocks = 10
        config.shower_flow_detailed_history = True
        config.shower_flow_weight_decay = 0.0
        config.shower_flow_tag = ""
        config.max_energy = 100

    print(config.max_energy)

    # model_path_base1 = "/data/dust/user/dayhallh/point-cloud-diffusion-data/showerFlow/sim-E1261AT600AP180-180/ShowerFlow_alt1_nb2_inputs8070450532247928831_fnorms_dhist_try3_*.pth"
    # model_path_base2 = "/data/dust/user/dayhallh/point-cloud-diffusion-data/showerFlow/sim-E1261AT600AP180-180/ShowerFlow_alt1_nb2_inputs8070450532247928831_fnorms_dhist_try5*.pth"
    model_path_base = "/data/dust/user/dayhallh/point-cloud-diffusion-data/showerFlow/sim-E1261AT600AP180-180/ShowerFlow_alt1_nb2_inputs8070450532247928831_fnorms_dhist_try9*.pth"
    # model_path_base = "/data/dust/user/dayhallh/point-cloud-diffusion-data/showerFlow/highGran_g40_p22_th90_ph90_en10-100/ShowerFlow_original_nb10_inputs36893488147419103231_dhist_try8_*.pth"
    # model_path_base = "/data/dust/user/dayhallh/point-cloud-diffusion-data/showerFlow/sim-E1261AT600AP180-180/ShowerFlow_log1_nb2_inputs8070450532247928831_fnorms_dhist_try6_*.pth"

    # for model_path_base in [model_path_base1, model_path_base2]:
    if True:
        for model_path in glob.glob(model_path_base):
            if "alt1" in model_path:
                assert config.shower_flow_version == "alt1"
            if "log1" in model_path:
                assert config.shower_flow_version == "log1"
            print(model_path)
            try:
                main(config, model_path)
            except Exception as e:
                print(e)
