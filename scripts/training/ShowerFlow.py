# # Showerflow
#
# Building and training showerflow from data.
#
# Start by setting up notebook enviroment.

import os
import sys
import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from pointcloud import configs as default_configs
from pointcloud.config_varients import caloclouds_2, caloclouds_3, example
from pointcloud.data.conditioning import get_cond_dim
from pointcloud.data.read_write import get_n_events
from pointcloud.models import shower_flow
from pointcloud.utils import showerflow_training, showerflow_utils
from pointcloud.utils.metadata import Metadata
from torch import optim
from tqdm import tqdm


def split_train_val(batch_size, dataset, shuffle, pin_memory):
    perc_training = 0.9
    train_batch, val_batch = (
        int(perc_training * batch_size),
        int((1 - perc_training) * batch_size),
    )
    train_size = int(perc_training * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size]
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def plot_shower_flow_eval(
    showerflowdir,
    E_true,
    num_points,
    num_points_sampled,
    visible_energy,
    visible_energy_sampled,
    cog_x,
    cog_x_sampled,
    cog_y,
    cog_y_sampled,
    cog_z,
    cog_z_sampled,
    clusters_per_layer,
    clusters_per_layer_sampled,
    e_per_layer,
    e_per_layer_sampled,
):
    plot_idx = 1

    def save_and_close(title, filename):
        plt.suptitle(title)
        plt.savefig(showerflowdir + filename)
        plt.close()

    # Number of points
    if num_points is not None and num_points_sampled is not None:
        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(1, 2)

        fig.add_subplot(gs[0])
        h = plt.hist(num_points.flatten(), bins=100, color="lightgrey")
        plt.hist(
            num_points_sampled.flatten(),
            bins=h[1],
            histtype="step",
            lw=2,
            color="tab:orange",
        )

        fig.add_subplot(gs[1])
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

        save_and_close("Number of points per event", f"ShowerFlow_eval_{plot_idx}.png")
    plot_idx += 1

    # Visible energy
    if visible_energy is not None and visible_energy_sampled is not None:
        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(1, 2)

        fig.add_subplot(gs[0])
        h = plt.hist(visible_energy.flatten(), bins=100, color="lightgrey")
        plt.hist(
            visible_energy_sampled.flatten(),
            bins=h[1],
            histtype="step",
            lw=2,
            color="tab:orange",
        )

        fig.add_subplot(gs[1])
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

        save_and_close("visible energy", f"ShowerFlow_eval_{plot_idx}.png")
    plot_idx += 1

    # Sampling fraction
    if visible_energy is not None and visible_energy_sampled is not None:
        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(1, 2)

        fig.add_subplot(gs[0])
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

        fig.add_subplot(gs[1])
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

        save_and_close("Sampling fraction", f"ShowerFlow_eval_{plot_idx}.png")
    plot_idx += 1

    # CoG x
    if cog_x is not None and cog_x_sampled is not None:
        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(1, 2)

        fig.add_subplot(gs[0])
        h = plt.hist(cog_x.flatten(), bins=100, color="lightgrey")
        plt.hist(
            cog_x_sampled.flatten(),
            bins=h[1],
            histtype="step",
            lw=2,
            color="tab:orange",
        )
        plt.xlim(-20, 20)

        fig.add_subplot(gs[1])
        h = plt.hist(cog_x.flatten(), bins=100, color="lightgrey")
        plt.hist(
            cog_x_sampled.flatten(),
            bins=h[1],
            histtype="step",
            lw=2,
            color="tab:orange",
        )
        plt.yscale("log")
        plt.ylim(1, 1e6)
        plt.xlim(-20, 20)

        save_and_close("center of gravity x", f"ShowerFlow_eval_{plot_idx}.png")
    plot_idx += 1

    # CoG y
    if cog_y is not None and cog_y_sampled is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        h = ax1.hist(cog_y.flatten(), bins=100, color="lightgrey")
        ax1.hist(
            cog_y_sampled.flatten(),
            bins=h[1],
            histtype="step",
            lw=2,
            color="tab:orange",
        )

        h = ax2.hist(cog_y.flatten(), bins=100, color="lightgrey")
        ax2.hist(
            cog_y_sampled.flatten(),
            bins=h[1],
            histtype="step",
            lw=2,
            color="tab:orange",
        )
        ax2.set_yscale("log")

        save_and_close("center of gravity y", f"ShowerFlow_eval_{plot_idx}.png")
    plot_idx += 1

    # CoG z
    if cog_z is not None and cog_z_sampled is not None:
        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(1, 2)

        fig.add_subplot(gs[0])
        h = plt.hist(cog_z.flatten(), bins=100, color="lightgrey")
        plt.hist(
            cog_z_sampled.flatten(),
            bins=h[1],
            histtype="step",
            lw=2,
            color="tab:orange",
        )

        ax = fig.add_subplot(gs[1])
        h = ax.hist(cog_z.flatten(), bins=100, color="lightgrey")
        ax.hist(
            cog_z_sampled.flatten(),
            bins=h[1],
            histtype="step",
            lw=2,
            color="tab:orange",
        )
        ax.semilogy()

        save_and_close("center of gravity z", f"ShowerFlow_eval_{plot_idx}.png")
    plot_idx += 1

    # Clusters per layer
    if clusters_per_layer is not None and clusters_per_layer_sampled is not None:
        for log_scale in [False, True]:
            fig = plt.figure(figsize=(12, 12))
            gs = gridspec.GridSpec(6, 5)
            for i in range(30):
                fig.add_subplot(gs[i])
                h = plt.hist(
                    clusters_per_layer[:, i].flatten(),
                    bins=40,
                    color="lightgrey",
                    range=[
                        clusters_per_layer[:, i].flatten().min() - 0.2,
                        clusters_per_layer[:, i].flatten().max() + 0.2,
                    ],
                )
                plt.hist(
                    clusters_per_layer_sampled[:, i].flatten(),
                    bins=h[1],
                    histtype="step",
                    lw=1,
                    color="tab:orange",
                )
                if log_scale:
                    plt.yscale("log")
            title = (
                "clusters per layer (log scale)" if log_scale else "clusters per layer"
            )
            save_and_close(title, f"ShowerFlow_eval_{plot_idx}.png")
            plot_idx += 1
    else:
        plot_idx += 2

    # Energy per layer
    if e_per_layer is not None and e_per_layer_sampled is not None:
        for log_scale in [False, True]:
            fig = plt.figure(figsize=(12, 12))
            gs = gridspec.GridSpec(6, 5)
            for i in range(30):
                fig.add_subplot(gs[i])
                h = plt.hist(
                    e_per_layer[:, i].flatten(),
                    bins=40,
                    color="lightgrey",
                    range=[
                        e_per_layer[:, i].flatten().min() - 0.2,
                        e_per_layer[:, i].flatten().max() + 0.2,
                    ],
                )
                plt.hist(
                    e_per_layer_sampled[:, i].flatten(),
                    bins=h[1],
                    histtype="step",
                    lw=1,
                    color="tab:orange",
                )
                if log_scale:
                    plt.yscale("log")
            title = "energy per layer (log scale)" if log_scale else "energy per layer"
            save_and_close(title, f"ShowerFlow_eval_{plot_idx}.png")
            plot_idx += 1


def should_save(epoch):
    if epoch < 10:
        return False
    str_epoch = str(epoch)
    return set(str_epoch[1:]) == {"0"}


def main(config, batch_size=2048, total_epochs=3_000, shuffle=True):
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
        af_dim=config.af_dim,
        device=device,
    )  # num_cond_inputs

    # print out the number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
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
        config,
        showerflow_dir,
        redo=False,
        local_batch_size=local_batch_size,
    )

    energy_per_layer_path = showerflow_training.get_energy_per_layer(
        config,
        showerflow_dir,
        redo=False,
        local_batch_size=local_batch_size,
    )

    # The energy per layer and the clusters per layer have now either been calculated,
    # or loaded from disk.
    # They should be normalised between 0 and 1 for the training.
    # After this we will visulise a little data for a sanity check.

    # if n_events > 10:
    #     start_event = 5
    #     clusters = np.load(clusters_per_layer_path, mmap_mode="r")
    #     energies = np.load(energy_per_layer_path, mmap_mode="r")
    #     for i in range(start_event, start_event + 5):
    #         clu = clusters["rescaled_clusters_per_layer"][i]
    #         e = energies["rescaled_energy_per_layer"][i]
    #         print(f"Event {i} has {np.sum(clu)} clusters")
    #         print(f"Event {i} has {np.sum(e)} energy")

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
        losses_val = history[2].tolist()
        if detailed_history:
            mean_parameter = history[2].tolist()
            max_parameter = history[3].tolist()
            gradient_norm = history[4].tolist()

    else:
        epoch_nums = []
        losses, losses_val = [], []
        if detailed_history:
            mean_parameter = []
            max_parameter = []
            gradient_norm = []
    history_save_list = [epoch_nums, losses, losses_val]
    if detailed_history:
        history_save_list += [mean_parameter, max_parameter, gradient_norm]

    # The data is now all loaded, and ready to train on.
    # If we had a previous best epoch it's loaded and we will start there.
    model.train()
    if torch.stack([torch.isnan(p).any() for p in model.parameters()]).any():
        print("model has nan weights before training!")

    batch_len = len(train_loader)
    mean_loss = np.inf
    for epoch in range(epoch_start, total_epochs):
        start_idx = start_points[epoch % len(start_points)]
        dataset = make_train_ds(start_idx, start_idx + local_batch_size)
        if len(dataset) == 0:
            continue
        train_loader, val_loader = split_train_val(
            local_batch_size, dataset, shuffle, pin_memory
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
            if torch.isnan(input_data).any():
                print(f"NaN in input_data! Shape: {input_data.shape}")
            if torch.isnan(context).any():
                print(f"NaN in context! Shape: {context.shape}")
            if batch_idx % 10 == 0:
                print(f"{batch_idx / batch_len:.0%}", end="\r")
            # with additional features in latent space (e.g. Esum)
            optimizer.zero_grad()

            # check if any of the weights are nans
            if torch.stack([torch.isnan(p).any() for p in model.parameters()]).any():
                print("Weights are nan!")
                print(torch.isnan(input_data).any())
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

        # ── Validation ──────────────────────────────────────────────────────────
        model.eval()
        val_loss_list = []
        with torch.no_grad():
            for val_data in val_loader:
                context = val_data[:, :cond_dim].to(device).float()
                input_data = val_data[:, cond_dim:].to(device).float()
                assert not torch.isnan(input_data).any(), (
                    f"NaN in input_data! Shape: {input_data.shape}"
                )
                # also check context
                assert not torch.isnan(context).any(), "NaN in context!"
                try:
                    nll = -distribution.condition(context).log_prob(input_data)
                    # ... rest of validation
                except ValueError as e:
                    if epoch == 0:
                        nll = torch.zeros(input_data.shape[0], device=input_data.device)
                    else:
                        print(f"Skipping validation at epoch {epoch}: {e}")
                        continue
                val_loss_list.append(nll.mean().item())
                distribution.clear_cache()
        mean_val_loss = np.mean(val_loss_list)
        model.train()
        # ────────────────────────────────────────────────────────────────────────

        if not detailed_history:
            epoch_nums.append(epoch)
            losses.append(mean_loss)
            losses_val.append(mean_val_loss)

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
                plt.figure(1)
                plt.plot(epoch_nums, losses_val, label="val")
                plt.plot(epoch_nums, losses, label="train")
                plt.legend()
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.savefig(showerflow_dir + "/loss_history.png")
                plt.close()
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

    raw_num_points = (
        np.concatenate(num_points_list, axis=0) if num_points_list else None
    )
    raw_visible_energy = (
        np.concatenate(visible_energy_list, axis=0) if visible_energy_list else None
    )
    raw_cog_x = np.concatenate(cog_x_list, axis=0) if cog_x_list else None
    raw_cog_y = np.concatenate(cog_y_list, axis=0) if cog_y_list else None
    raw_cog_z = np.concatenate(cog_z_list, axis=0) if cog_z_list else None
    samples.shape, raw_E_true.shape
    clusters_per_layer = (
        np.concatenate(clusters_per_layer_list, axis=0)
        if clusters_per_layer_list
        else None
    )
    e_per_layer = np.concatenate(e_per_layer_list, axis=0) if e_per_layer_list else None
    # truth; we need to undo the shifts that were done in the last training batch
    # sampled
    i = 0
    if "total_clusters" in config.shower_flow_inputs:
        num_points = raw_num_points * meta.n_pts_rescale
        num_points_sampled = samples[:, i] * meta.n_pts_rescale
        i += 1
    if "total_energy" in config.shower_flow_inputs:
        visible_energy = raw_visible_energy * meta.vis_eng_rescale
        visible_energy_sampled = samples[:, i] * meta.vis_eng_rescale
        i += 1
    if "cog_x" in config.shower_flow_inputs:
        cog_x = (raw_cog_x * meta.std_cog[0]) + meta.mean_cog[0]
        cog_x_sampled = (samples[:, i] * meta.std_cog[0]) + meta.mean_cog[0]
        i += 1
    if "cog_y" in config.shower_flow_inputs:
        cog_y = (raw_cog_y * meta.std_cog[1]) + meta.mean_cog[1]
        cog_y_sampled = (samples[:, i] * meta.std_cog[1]) + meta.mean_cog[1]
        i += 1
    if "cog_z" in config.shower_flow_inputs:
        cog_z = (raw_cog_z * meta.std_cog[2]) + meta.mean_cog[2]
        cog_z_sampled = (samples[:, i] * meta.std_cog[2]) + meta.mean_cog[2]
        i += 1
    if "clusters_per_layer" in config.shower_flow_inputs:
        clusters_per_layer_sampled = samples[:, i : i + 30]
        # np.clip(samples[:, i : i + 30], 0, 1)
        i += 30
    if "energy_per_layer" in config.shower_flow_inputs:
        e_per_layer_sampled = samples[:, i : i + 30]
        # np.clip(samples[:, i : i + 30], 0, 1)
        i += 30

    # When the flow isn't trained directly on the totals, derive them by
    # summing the (fixed-norm) per-layer values, so we can still compare
    # the model's total points/energy against the truth.
    have_num_points = "total_clusters" in config.shower_flow_inputs
    have_visible_energy = "total_energy" in config.shower_flow_inputs
    if (
        not have_num_points
        and "clusters_per_layer" in config.shower_flow_inputs
        and getattr(config, "shower_flow_fixed_input_norms", False)
    ):
        clusters_per_layer_norm = np.load(
            os.path.join(showerflow_dir, "input_norms.npz")
        )["clusters_per_layer_norm"]
        num_points = clusters_per_layer.sum(axis=1) / clusters_per_layer_norm
        num_points_sampled = (
            clusters_per_layer_sampled.sum(axis=1) / clusters_per_layer_norm
        )
        have_num_points = True
    if (
        not have_visible_energy
        and "energy_per_layer" in config.shower_flow_inputs
        and getattr(config, "shower_flow_fixed_input_norms", False)
    ):
        energy_per_layer_norm = np.load(
            os.path.join(showerflow_dir, "input_norms.npz")
        )["energy_per_layer_norm"]
        visible_energy = e_per_layer.sum(axis=1) / energy_per_layer_norm
        visible_energy_sampled = (
            e_per_layer_sampled.sum(axis=1) / energy_per_layer_norm
        )
        have_visible_energy = True

    E_true = raw_E_true * meta.incident_rescale
    meta.vis_eng_rescale

    plot_shower_flow_eval(
        showerflow_dir + "/",
        E_true,
        num_points if have_num_points else None,
        num_points_sampled if have_num_points else None,
        visible_energy if have_visible_energy else None,
        visible_energy_sampled if have_visible_energy else None,
        cog_x if "cog_x" in config.shower_flow_inputs else None,
        cog_x_sampled if "cog_x" in config.shower_flow_inputs else None,
        cog_y if "cog_y" in config.shower_flow_inputs else None,
        cog_y_sampled if "cog_y" in config.shower_flow_inputs else None,
        cog_z if "cog_z" in config.shower_flow_inputs else None,
        cog_z_sampled if "cog_z" in config.shower_flow_inputs else None,
        clusters_per_layer
        if "clusters_per_layer" in config.shower_flow_inputs
        else None,
        clusters_per_layer_sampled
        if "clusters_per_layer" in config.shower_flow_inputs
        else None,
        e_per_layer if "energy_per_layer" in config.shower_flow_inputs else None,
        e_per_layer_sampled
        if "energy_per_layer" in config.shower_flow_inputs
        else None,
    )
    return best_model_path, best_data_path, history_data_path


if __name__ == "__main__":
    import importlib.util

    chosen = sys.argv[1].strip() if len(sys.argv) > 1 else "default"

    config_kwargs = {
        a.split("=")[0].strip(): a.split("=")[1].strip()
        for a in sys.argv[2:]
        if "=" in a
    }

    if chosen.endswith(".py"):
        # Load any config file by path, e.g.:
        #   python ShowerFlow.py pointcloud/config_varients/caloclouds_3_S2P_hdbscan.py
        spec = importlib.util.spec_from_file_location("_user_config", os.path.abspath(chosen))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        config = mod.Configs(**config_kwargs)
    else:
        config_choices = {
            "caloclouds_2": caloclouds_2,
            "caloclouds_3": caloclouds_3,
            "example": example,
            "default": default_configs,
        }
        if chosen not in config_choices:
            chosen = input(f"Choose a version from {list(config_choices.keys())}: ")
        config = config_choices[chosen].Configs(**config_kwargs)

    main(config)
