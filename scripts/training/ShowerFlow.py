# # Showerflow
#
# Building and training showerflow from data.
#
# Start by setting up notebook enviroment.

from tqdm import tqdm

import torch
from torch import optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
import os
import sys
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
    caloclouds_3_v_simple_shower,
)
from pointcloud.data.read_write import get_n_events
from pointcloud.data.conditioning import get_cond_dim
from pointcloud.utils import showerflow_training, showerflow_utils
from pointcloud.utils.metadata import Metadata
from pointcloud.models import shower_flow


def should_save(epoch):
    if epoch < 10:
        return False
    str_epoch = str(epoch)
    return set(str_epoch[1:]) == {"0"}


def main(config, batch_size=2048, total_epochs=1_000_000_000, shuffle=True):
    # def main(config, batch_size=64, total_epochs=1_000_000_000, shuffle=True):
    """
    Really this is a script, but for ease of testing, it's the main function.
    """
    shower_flow_compiler = shower_flow.versions_dict[config.shower_flow_version]

    cond_dim = get_cond_dim(config, "showerflow")
    inputs_used = showerflow_utils.get_input_mask(config)
    cut_inputs = np.where(~inputs_used)[0]
    input_dim = np.sum(inputs_used)

    # use cuda if avaliable
    if torch.cuda.is_available():
        config.device = "cuda"
    else:
        config.device = "cpu"
    device = torch.device(config.device)

    meta = Metadata(config)
    model, distribution, transforms = shower_flow_compiler(
        num_blocks=config.shower_flow_num_blocks,
        num_inputs=input_dim,
        num_cond_inputs=cond_dim,
        device=device,
    )  # num_cond_inputs
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

    # The energy per layer and the clusters per layer have now either been calculated,
    # or loaded from disk.
    # They should be normalised between 0 and 1 for the training.
    # After this we will visulise a little data for a sanity check.

    if n_events > 10:
        start_event = 5
        clusters = np.load(clusters_per_layer_path, mmap_mode="r")
        energies = np.load(energy_per_layer_path, mmap_mode="r")
        for i in range(start_event, start_event + 5):
            clu = clusters["rescaled_clusters_per_layer"][i]
            e = energies["rescaled_energy_per_layer"][i]
            print(f"Event {i} has {np.sum(clu)} clusters")
            print(f"Event {i} has {np.sum(e)} energy")

    # center of gravity

    cog_path, cog = showerflow_training.get_cog(
        config, showerflow_dir, redo=False, local_batch_size=local_batch_size
    )

    print(f"Cog in x has mean {np.mean(cog[0])} and std {np.std(cog[0])}")
    print(f"Cog in y has mean {np.mean(cog[1])} and std {np.std(cog[1])}")
    print(f"Cog in z has mean {np.mean(cog[2])} and std {np.std(cog[2])}")
    normed_cog = np.copy(cog)
    prenormed_mean = np.mean(normed_cog, axis=1)
    prenormed_std = np.std(normed_cog, axis=1)
    print(f"Found means {prenormed_mean}, stds {prenormed_std}")

    normed_cog = (normed_cog - prenormed_mean[:, np.newaxis]) / prenormed_std[
        :, np.newaxis
    ]

    print(
        f"Normed cog in x has mean {np.mean(normed_cog[0])}"
        f" and std {np.std(normed_cog[0])}"
    )
    print(
        f"Normed cog in y has mean {np.mean(normed_cog[1])}"
        f" and std {np.std(normed_cog[1])}"
    )
    print(
        f"Normed cog in z has mean {np.mean(normed_cog[2])}"
        f" and std {np.std(normed_cog[2])}"
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

    start_points = np.arange(0, n_events, local_batch_size)
    # try it out
    dataset = make_train_ds(0, 5)

    pin_memory = device == "cpu"
    print(f"Pin memory is {pin_memory}, device is {device}")
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )
    # Then we set where we intent to save the created model.
    prefix = time.strftime("%y-%m-%d_cog_e_layer_")
    outpath = os.path.join(showerflow_dir, prefix)
    print(f"Saving to {outpath}")

    _, best_model_path, best_data_path = showerflow_utils.model_save_paths(
        config, config.shower_flow_version, config.shower_flow_num_blocks, cut_inputs
    )
    latest_model_path = best_model_path.replace("best", "latest")
    history_data_path = best_data_path.replace("best_data.txt", "history.npy")
    print(f"Best model will be saved to {best_model_path}")
    print(f"Latest model will be saved to {latest_model_path}")
    print(f"Best data will be saved to {best_data_path}")
    # torch.manual_seed(123)

    lr = 1e-5  # default: 5e-5
    weight_decay = getattr(config, "shower_flow_weight_decay", 0)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    start_over = False
    # epoch_load = 550
    # model.load_state_dict(torch.load(outpath+f'ShowerFlow_{epoch_load}.pth')['model'])
    # optimizer.load_state_dict(torch.load(outpath+f'ShowerFlow_{epoch_load}.pth')['optimizer'])
    if os.path.exists(best_model_path) and not start_over:
        model.load_state_dict(
            torch.load(best_model_path, map_location=config.device, weights_only=False)[
                "model"
            ]
        )
        optimizer.load_state_dict(
            torch.load(best_model_path, map_location=config.device, weights_only=False)[
                "optimizer"
            ]
        )

        # load best_loss
        with open(best_data_path, "r") as f:
            data = f.read().strip().split()
        best_loss = float(data[0])
        epoch_start = int(data[1])
    else:
        best_loss = np.inf
        epoch_start = 0
    print(
        f"A total of {total_epochs} epochs are requested, "
        f"the model had already undergone {epoch_start}"
    )

    detailed_history = getattr(config, "shower_flow_detailed_history", False)
    if os.path.exists(history_data_path):
        history = np.load(history_data_path)
        epoch_nums = history[0].tolist()
        losses = history[1].tolist()
        if detailed_history:
            mean_parameter = history[2].tolist()
            max_parameter = history[3].tolist()
            gradient_norm = history[4].tolist()

    else:
        epoch_nums = []
        losses = []
        if detailed_history:
            mean_parameter = []
            max_parameter = []
            gradient_norm = []
    history_save_list = [epoch_nums, losses]
    if detailed_history:
        history_save_list += [mean_parameter, max_parameter, gradient_norm]

    # The data is now all loaded, and ready to train on.
    # If we had a previous best epoch it's loaded and we will start there.
    model.train()

    batch_len = len(train_loader)
    mean_loss = np.inf
    for epoch in range(epoch_start, total_epochs):
        start_idx = start_points[epoch % len(start_points)]
        dataset = make_train_ds(start_idx, start_idx + local_batch_size)
        if len(dataset) == 0:
            continue
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
        )
        percent = (epoch - epoch_start) / (total_epochs - epoch_start)
        print(
            f"     Epoch number {epoch}; {percent:.0%} complete;"
            f"mean loss {mean_loss:.2}",
            end="\r\r",
        )
        loss_list = losses if detailed_history else []
        for batch_idx, data in enumerate(train_loader):
            context = data[:, :cond_dim].to(device).float()
            input_data = data[:, cond_dim:].to(device).float()
            if batch_idx % 10 == 0:
                print(f"{batch_idx/batch_len:.0%}", end="\r")

            # with additional features in latent space (e.g. Esum)
            optimizer.zero_grad()

            # check if any of the weights are nans
            if torch.stack([torch.isnan(p).any() for p in model.parameters()]).any():
                print("Weights are nan!")
                # load recent model
                # model.load_state_dict(torch.load(latest_model_path)["model"])
                model.load_state_dict(
                    torch.load(best_model_path, weights_only=False)["model"]
                )
                optimizer = optim.Adam(
                    model.parameters(), lr=lr, weight_decay=weight_decay
                )
                if config.shower_flow_optimiser_on_nan == "best":
                    optimizer.load_state_dict(
                        torch.load(best_model_path, weights_only=False)["optimizer"]
                    )
                elif config.shower_flow_optimiser_on_nan == "blank":
                    pass
                else:
                    raise NotImplementedError(
                        f"Unknown optimiser_on_nan {config.shower_flow_optimiser_on_nan}"
                    )
                # change the seed so we don't get back to the same nan
                torch.manual_seed(time.time())
                # print(f'model from {epoch-3} epoch reloaded')
                print("latest model reloaded, optimizer reset")

            nll = -distribution.condition(context).log_prob(input_data)
            loss = nll.mean()
            loss.backward()
            if detailed_history:
                norm = 0
                for p in model.parameters():
                    norm += torch.norm(p.grad).item() ** 2
                gradient_norm.append(norm**0.5)
            optimizer.step()
            distribution.clear_cache()
            loss_list.append(loss.item())
            if detailed_history:
                epoch_nums.append(epoch)
                parameter_list = torch.concatenate(
                    [p.flatten() for p in model.parameters()]
                )
                mean_parameter.append(torch.mean(parameter_list).item())
                max_parameter.append(torch.max(parameter_list).item())

        mean_loss = np.mean(loss_list[-batch_len:])
        if not detailed_history:
            epoch_nums.append(epoch)
            losses.append(mean_loss)

        np.save(history_data_path, np.array(history_save_list))

        if torch.stack(
            [torch.isnan(p).any() for p in model.parameters()]
        ).any():  # save models only if no nan weights
            print("model not saved due to nan weights")
        else:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                latest_model_path,
            )

            if should_save(epoch):
                epoch_path = latest_model_path.replace(
                    "_latest.pth", f"_{epoch:07d}.pth"
                )
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    epoch_path,
                )

            # save best model based on loss
            if mean_loss <= best_loss:
                best_loss = mean_loss
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    best_model_path,
                )
                # save best loss value to txt file
                with open(best_data_path, "w") as f:
                    f.write(f"{best_loss} {epoch}")

    print(f"Training complete, best loss: {best_loss}")
    # ## After training
    #
    # Now the training is complete, there should be a best checkpoint.
    # We will load this in order to generate some evaluation plots.
    # Setting the seed makes sure the evaluation plots are reproducable.
    # load best checkpoint
    model.load_state_dict(torch.load(best_model_path, weights_only=False)["model"])
    optimizer.load_state_dict(
        torch.load(best_model_path, weights_only=False)["optimizer"]
    )

    model.eval()
    print("model loaded")
    # generate
    torch.manual_seed(123)

    E_true_list = []
    samples_list = []
    num_points_list = []
    visible_energy_list = []
    cog_x_list = []
    cog_y_list = []
    cog_z_list = []
    clusters_per_layer_list = []
    e_per_layer_list = []
    for batch_idx, data in enumerate(tqdm(train_loader)):
        context = data[:, :cond_dim].to(device).float()
        input_data = data[:, cond_dim:].to(device).float()
        with torch.no_grad():
            samples = (
                distribution.condition(context)
                .sample(
                    torch.Size(
                        [
                            context.shape[0],
                        ]
                    )
                )
                .cpu()
                .numpy()
            )
        E_true_list.append(context[:, -1].cpu().numpy())
        samples_list.append(samples)
        i = 0
        if "total_clusters" in config.shower_flow_inputs:
            num_points_list.append(input_data[:, i].cpu().numpy())
            i += 1
        if "total_energy" in config.shower_flow_inputs:
            visible_energy_list.append(input_data[:, i].cpu().numpy())
            i += 1
        if "cog_x" in config.shower_flow_inputs:
            cog_x_list.append(input_data[:, i].cpu().numpy())
            i += 1
        if "cog_y" in config.shower_flow_inputs:
            cog_y_list.append(input_data[:, i].cpu().numpy())
            i += 1
        if "cog_z" in config.shower_flow_inputs:
            cog_z_list.append(input_data[:, i].cpu().numpy())
            i += 1
        if "clusters_per_layer" in config.shower_flow_inputs:
            clusters_per_layer_list.append(input_data[:, i : i + 30].cpu().numpy())
            i += 30
        if "energy_per_layer" in config.shower_flow_inputs:
            e_per_layer_list.append(input_data[:, i : i + 30].cpu().numpy())

    raw_E_true = np.concatenate(E_true_list, axis=0)
    samples = np.concatenate(samples_list, axis=0)
    raw_num_points = np.concatenate(num_points_list, axis=0)
    raw_visible_energy = np.concatenate(visible_energy_list, axis=0)
    raw_cog_x = np.concatenate(cog_x_list, axis=0)
    raw_cog_y = np.concatenate(cog_y_list, axis=0)
    raw_cog_z = np.concatenate(cog_z_list, axis=0)
    samples.shape, raw_E_true.shape
    clusters_per_layer = np.concatenate(clusters_per_layer_list, axis=0)
    e_per_layer = np.concatenate(e_per_layer_list, axis=0)
    # sampled
    num_points_sampled = samples[:, 0] * meta.n_pts_rescale
    visible_energy_sampled = samples[:, 1] * meta.vis_eng_rescale
    cog_x_sampled = (samples[:, 2] * meta.std_cog[0]) + meta.mean_cog[0]
    cog_y_sampled = (samples[:, 3] * meta.std_cog[1]) + meta.mean_cog[1]
    cog_z_sampled = (samples[:, 4] * meta.std_cog[2]) + meta.mean_cog[2]
    clusters_per_layer_sampled = samples[:, 5:35]
    e_per_layer_sampled = samples[:, 35:]

    # truth; we need to undo the shifts that were done in the last training batch

    E_true = raw_E_true * meta.incident_rescale
    num_points = raw_num_points * meta.n_pts_rescale
    visible_energy = raw_visible_energy * meta.vis_eng_rescale
    cog_x = (raw_cog_x * meta.std_cog[0]) + meta.mean_cog[0]
    cog_y = (raw_cog_y * meta.std_cog[1]) + meta.mean_cog[1]
    cog_z = (raw_cog_z * meta.std_cog[2]) + meta.mean_cog[2]

    # clip cluster and energies per layer to [0,1]
    clusters_per_layer_sampled = np.clip(clusters_per_layer_sampled, 0, 1)
    e_per_layer_sampled = np.clip(e_per_layer_sampled, 0, 1)
    print(cog_x.min(), cog_x.max())
    print(cog_y.min(), cog_y.max())
    print(cog_z.min(), cog_z.max())
    meta.vis_eng_rescale
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2)

    ax = fig.add_subplot(gs[0])
    h = plt.hist(num_points.flatten(), bins=100, color="lightgrey")
    plt.hist(
        num_points_sampled.flatten(),
        bins=h[1],
        histtype="step",
        lw=2,
        color="tab:orange",
    )

    # same but in log scale
    ax = fig.add_subplot(gs[1])
    h = plt.hist(num_points.flatten(), bins=100, color="lightgrey")
    plt.hist(
        num_points_sampled.flatten(),
        bins=h[1],
        histtype="step",
        lw=2,
        color="tab:orange",
    )
    plt.yscale("log")
    plt.ylim(1, 1e5)

    plt.suptitle("Number of points per event")
    plt.savefig("ShowerFlow_eval_1.png")
    plt.close()

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2)

    ax = fig.add_subplot(gs[0])
    h = plt.hist(visible_energy.flatten(), bins=100, color="lightgrey")
    plt.hist(
        visible_energy_sampled.flatten(),
        bins=h[1],
        histtype="step",
        lw=2,
        color="tab:orange",
    )

    # same but in log scale
    ax = fig.add_subplot(gs[1])
    h = plt.hist(visible_energy.flatten(), bins=100, color="lightgrey")
    plt.hist(
        visible_energy_sampled.flatten(),
        bins=h[1],
        histtype="step",
        lw=2,
        color="tab:orange",
    )
    plt.yscale("log")
    plt.ylim(1, 1e5)

    plt.suptitle("visible energy")
    plt.savefig("ShowerFlow_eval_2.png")
    plt.close()

    # sampling fraction
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2)

    ax = fig.add_subplot(gs[0])
    h = plt.hist(
        visible_energy.flatten() / E_true.flatten(), bins=100, color="lightgrey"
    )
    plt.hist(
        visible_energy_sampled.flatten() / E_true.flatten(),
        bins=h[1],
        histtype="step",
        lw=2,
        color="tab:orange",
    )

    # same but in log scale
    ax = fig.add_subplot(gs[1])
    h = plt.hist(
        visible_energy.flatten() / E_true.flatten(), bins=100, color="lightgrey"
    )
    plt.hist(
        visible_energy_sampled.flatten() / E_true.flatten(),
        bins=h[1],
        histtype="step",
        lw=2,
        color="tab:orange",
    )
    plt.yscale("log")
    plt.ylim(1, 1e5)

    # title over all subplots
    fig.suptitle("Sampling fraction")
    plt.savefig("ShowerFlow_eval_3.png")
    plt.close()

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2)

    ax = fig.add_subplot(gs[0])
    h = plt.hist(cog_x.flatten(), bins=100, color="lightgrey")
    plt.hist(
        cog_x_sampled.flatten(), bins=h[1], histtype="step", lw=2, color="tab:orange"
    )
    plt.xlim(-20, 20)

    # same but in log scale
    ax = fig.add_subplot(gs[1])
    h = plt.hist(cog_x.flatten(), bins=100, color="lightgrey")
    plt.hist(
        cog_x_sampled.flatten(), bins=h[1], histtype="step", lw=2, color="tab:orange"
    )
    plt.yscale("log")
    plt.ylim(1, 1e6)
    plt.xlim(-20, 20)

    plt.suptitle("center of gravity x")
    plt.savefig("ShowerFlow_eval_4.png")
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    h = ax1.hist(cog_y.flatten(), bins=100, color="lightgrey")
    ax1.hist(
        cog_y_sampled.flatten(), bins=h[1], histtype="step", lw=2, color="tab:orange"
    )
    # ax1.xlim(5,30)

    # same but in log scale
    h = ax2.hist(cog_y.flatten(), bins=100, color="lightgrey")
    ax2.hist(
        cog_y_sampled.flatten(), bins=h[1], histtype="step", lw=2, color="tab:orange"
    )
    plt.yscale("log")
    # plt.ylim(1, 1e6)
    # plt.xlim(5,30)

    plt.suptitle("center of gravity y")
    plt.savefig("ShowerFlow_eval_5.png")
    plt.close()

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2)

    ax = fig.add_subplot(gs[0])
    h = plt.hist(cog_z.flatten(), bins=100, color="lightgrey")
    plt.hist(
        cog_z_sampled.flatten(), bins=h[1], histtype="step", lw=2, color="tab:orange"
    )
    # plt.xlim(25,60)

    # same but in log scale
    ax = fig.add_subplot(gs[1])
    h = ax.hist(cog_z.flatten(), bins=100, color="lightgrey")
    ax.hist(
        cog_z_sampled.flatten(), bins=h[1], histtype="step", lw=2, color="tab:orange"
    )
    ax.semilogy()
    # plt.ylim(1, 1e6)
    # plt.xlim(25,60)

    plt.suptitle("center of gravity z")
    plt.savefig("ShowerFlow_eval_6.png")
    plt.close()

    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(6, 5)

    for i in range(30):
        fig.add_subplot(gs[i])
        h = plt.hist(
            clusters_per_layer[:, i].flatten(),
            bins=100,
            color="lightgrey",
            range=[-0.2, 1.2],
        )
        plt.hist(
            clusters_per_layer_sampled[:, i].flatten(),
            bins=h[1],
            histtype="step",
            lw=1,
            color="tab:orange",
        )

    plt.suptitle("clusters per layer")
    plt.savefig("ShowerFlow_eval_7.png")
    plt.close()

    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(6, 5)

    for i in range(30):
        fig.add_subplot(gs[i])
        h = plt.hist(
            clusters_per_layer[:, i].flatten(),
            bins=100,
            color="lightgrey",
            range=[-0.2, 1.2],
        )
        plt.hist(
            clusters_per_layer_sampled[:, i].flatten(),
            bins=h[1],
            histtype="step",
            lw=1,
            color="tab:orange",
        )
        plt.yscale("log")

    plt.suptitle("clusters per layer (log scale)")
    plt.savefig("ShowerFlow_eval_8.png")
    plt.close()

    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(6, 5)

    for i in range(30):
        fig.add_subplot(gs[i])
        h = plt.hist(
            e_per_layer[:, i].flatten(), bins=100, color="lightgrey", range=[-0.2, 1.2]
        )
        plt.hist(
            e_per_layer_sampled[:, i].flatten(),
            bins=h[1],
            histtype="step",
            lw=1,
            color="tab:orange",
        )
        # plt.yscale('log')

    plt.suptitle("energy per layer")
    plt.savefig("ShowerFlow_eval_9.png")
    plt.close()

    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(6, 5)

    for i in range(30):
        fig.add_subplot(gs[i])
        h = plt.hist(
            e_per_layer[:, i].flatten(), bins=100, color="lightgrey", range=[-0.2, 1.2]
        )
        plt.hist(
            e_per_layer_sampled[:, i].flatten(),
            bins=h[1],
            histtype="step",
            lw=1,
            color="tab:orange",
        )
        plt.yscale("log")

    plt.suptitle("energy per layer (log scale)")
    plt.savefig("ShowerFlow_eval_10.png")
    plt.close()
    return best_model_path, best_data_path, history_data_path


if __name__ == "__main__":
    # check user input
    # and if not given get it from the user

    chosen = None
    if len(sys.argv) > 1:
        chosen = sys.argv[1].strip()

    config_choices = {
        "wish": wish,
        "wish_maxwell": wish_maxwell,
        "caloclouds_2": caloclouds_2,
        "caloclouds_2_v2": caloclouds_2_v2,
        "caloclouds_2_v3": caloclouds_2_v3,
        "caloclouds_2_v4": caloclouds_2_v4,
        "caloclouds_3": caloclouds_3,
        "caloclouds_3_simple_shower": caloclouds_3_simple_shower,
        "caloclouds_3_v_simple_shower": caloclouds_3_v_simple_shower,
    }
    while chosen not in config_choices:
        chosen = input(f"Choose a version from {list(config_choices.keys())}:")
        if chosen not in config_choices:
            print("Invalid choice")

    config_kwargs = {
        a.split("=")[0].strip(): a.split("=")[1].strip() for a in sys.argv[2:]
    }
    config = config_choices[chosen].Configs(**config_kwargs)

    main(config)
