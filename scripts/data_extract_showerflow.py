# # Showerflow
#
# Building and training showerflow from data.
#
# Start by setting up notebook enviroment.

from tqdm import tqdm

import torch
import glob

import numpy as np
import os
import time

from pointcloud.config_varients import (
    wish,
    wish_maxwell,
    caloclouds_2,
    caloclouds_2_v2,
    caloclouds_2_v3,
    caloclouds_2_v4,
    caloclouds_3,
    caloclouds_3_simple_shower,
)
from pointcloud.data.read_write import get_n_events
from pointcloud.data.conditioning import get_cond_dim
from pointcloud.utils import showerflow_training, showerflow_utils
from pointcloud.models import shower_flow


def get_data(
    config, model_path, batch_size=2048, n_input_events=0, n_showerflow_events=100_000
):
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
    n_events = np.sum(get_n_events(config.dataset_path, config.n_dataset_files))
    if config.device == "cuda":
        # on the gpu we can use all the data
        # local_batch_size = n_events
        local_batch_size = 100_000
    else:
        local_batch_size = 10_000

    showerflow_dir = showerflow_utils.get_showerflow_dir(config)
    print(f"Showerflow data will be saved to {showerflow_dir}")
    os.makedirs(showerflow_dir, exist_ok=True)

    print(f"Of the {n_events} avaliable, we are batching into {local_batch_size}")
    # ## Data prep
    #
    # Now some values are needed to aid the showerflow training.

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

    model.to(device)
    model.eval()

    if n_input_events:
        print(f"Getting input data sample")
        dataset = make_train_ds(0, n_input_events)
        input_data = dataset[:, cond_dim:].to(device).float()
        context = dataset[:, :cond_dim].to(device).float()
        # np.save(input_save_path, input_data)
        del dataset
    else:
        input_data = None

    min_energy = 10
    max_energy = 90
    wrong_context = (
        torch.FloatTensor(n_showerflow_events, 1)
        .uniform_(min_energy, max_energy)
        .to(config.device)
    ) / config.max_energy  #  B,1
    if cond_dim > 1:
        direction = (torch.zeros(n_showerflow_events, 3)).to(config.device)
        direction[:, 2] = 1
        wrong_context = torch.cat((wrong_context, direction), 1)

    model_data = []
    for start_point in tqdm(range(0, n_showerflow_events, batch_size)):
        #context_here = wrong_context[start_point : start_point + batch_size]
        context_here = context[start_point : start_point + batch_size]
        model_out = distribution.condition(context_here).sample([len(context_here)])
        model_data.append(model_out.detach().cpu().numpy())
    model_data = np.concatenate(model_data, axis=0)

    return model_data, input_data, wrong_context, context


def main(
    config,
    model_path,
    n_input_events=1_000_000,
    n_showerflow_events=100_000,
):
    # decide on the save paths
    out_path = "/data/dust/user/dayhallh/point-cloud-diffusion-data/showerFlow/sampling"
    input_save_path = config.dataset_path.format("stack_of_training_data")
    if "stack_of_training_data" not in input_save_path:
        input_save_path += "stack_of_training_data.npy"
    else:
        input_save_path += ".npy"

    if os.path.exists(input_save_path):
        n_input_events = 0

    train_ds_name = model_path.split("/")[-2]
    showerflow_config = model_path.split("/")[-1][11:-4]
    f_start = os.path.join(
        out_path, f"pureShowerflow_{train_ds_name}_{showerflow_config}"
    )
    if glob.glob(f_start + "*"):
        print(f"Done: {f_start}")
        return
    timestamp = time.strftime("%Y-%m-%d_%H-%M")
    output_save_path = f"{f_start}_{timestamp}.npy"

    # get the data
    model_data, input_data = get_data(
        config,
        model_path,
        n_input_events=n_input_events,
        n_showerflow_events=n_showerflow_events,
    )

    if input_data is not None:
        np.save(input_save_path, input_data)

    np.save(output_save_path, model_data)


if __name__ == "__main__":

    # check user input
    # and if not given get it from the user
    cc_version = 2
    if cc_version == 3:
        config = caloclouds_3_simple_shower.Configs()
        config.dataset_path = "/data/dust/group/ilc/sft-ml/datasets/sim-E1261AT600AP180-180/sim-E1261AT600AP180-180_file_{}.slcio.hdf5"
        config.n_dataset_files = 88
        config.shower_flow_version = "alt1"
        #config.shower_flow_version = "log1"
        config.shower_flow_num_blocks = 2
        config.shower_flow_detailed_history = True
        config.shower_flow_weight_decay = 0.
        config.shower_flow_tag = "try6"
        config.max_energy = 127
    elif cc_version == 2:
        config = caloclouds_2_v3.Configs()
        config.dataset_path = "/data/dust/user/dayhallh/data/ILCsoftEvents/highGran_g40_p22_th90_ph90_en10-100.hdf5"
        config.n_dataset_files = 1
        config.shower_flow_version = "original"
        config.shower_flow_num_blocks = 10
        config.shower_flow_detailed_history = True
        config.shower_flow_weight_decay = 0.
        config.shower_flow_tag = ""
        config.max_energy = 100 

    print(config.max_energy)

    model_path_base = "/data/dust/user/dayhallh/point-cloud-diffusion-data/showerFlow/highGran_g40_p22_th90_ph90_en10-100/ShowerFlow_original_nb10_inputs36893488147419103231_dhist_try2_*.pth"
    #model_path_base = "/data/dust/user/dayhallh/point-cloud-diffusion-data/showerFlow/sim-E1261AT600AP180-180/ShowerFlow_alt1_nb2_inputs8070450532247928831_fnorms_dhist_try3_*.pth"
    #model_path_base = "/data/dust/user/dayhallh/point-cloud-diffusion-data/showerFlow/sim-E1261AT600AP180-180/ShowerFlow_alt1_nb2_inputs8070450532247928831_fnorms_dhist_try5_*.pth"

    for model_path in glob.glob(model_path_base):
        print(model_path)
        main(config, model_path)
