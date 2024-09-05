# # Showerflow
#
# Building and training showerflow from data.
#
# Start by setting up notebook enviroment.

from tqdm import tqdm

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
import os
import sys
import time

from pointcloud.config_varients import caloclouds_3
from pointcloud.data.read_write import get_n_events, read_raw_regaxes
from pointcloud.utils import showerflow_training
from pointcloud.utils.metadata import Metadata
from pointcloud.models.shower_flow import (
    compile_HybridTanH_model,
    compile_HybridTanH_alt1,
    compile_HybridTanH_alt2,
)

# check user input
# and if not given get it from the user

versions = {
    "original": compile_HybridTanH_model,
    "alt1": compile_HybridTanH_alt1,
    "alt2": compile_HybridTanH_alt2,
}

chosen = None
if len(sys.argv) > 1:
    chosen = sys.argv[1].strip()

while chosen not in versions.keys():
    chosen = input(f"Choose a version from {versions.keys()}: ")
    if chosen not in versions.keys():
        print("Invalid choice")

num_blocks = None
if len(sys.argv) > 2:
    num_blocks = int(sys.argv[2])

while num_blocks is None:
    try:
        num_blocks = int(input("Choose the number of blocks: "))
        if num_blocks < 1:
            num_blocks = None
            print("Number of blocks must be greater than 0")
    except ValueError:
        print("Invalid choice")


max_input_dims = 65
inputs_used = np.ones(max_input_dims, dtype=bool)
if len(sys.argv) > 3:
    cut_inputs = sys.argv[3].strip()
    for i in range(5):
        if str(i) in cut_inputs:
            inputs_used[i] = False

input_dim = np.sum(inputs_used)
inputs_used_as_binary = "".join(["1" if i else "0" for i in inputs_used])
inputs_used_as_base10 = int(inputs_used_as_binary, 2)

# all user input is suitable

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
configs = caloclouds_3.Configs()
configs.device = device

meta = Metadata(configs)
model, distribution = versions[chosen](
    num_blocks=num_blocks,
    num_inputs=input_dim,  ### when 'condioning' on additional Esum, Nhits etc add them on as inputs rather than
    num_cond_inputs=1,
    device=device,
)  # num_cond_inputs
default_params = {
    "batch_size": 2048,
    "epochs": 10_000_000,
    "shuffle": True,
}
# ## Setup
#
# Starting from default parameters, the model can be customised.
kwargs = {}
params = {}
for param in default_params.keys():
    if param in kwargs.keys():
        params[param] = kwargs[param]
    else:
        params[param] = default_params[param]
# Load the data and check it's properties.
# path = '/beegfs/desy/user/akorol/data/calo-clouds/hdf5/high_granular_grid/train/' \
# '10-90GeV_x36_grid_regular_524k_float32.hdf5'
path = (
    "/home/dayhallh/Data/p22_th90_ph90_en10-100_joined/"
    "p22_th90_ph90_en10-100_seed{}_all_steps.hdf5"
)
path = (
    # "/beegfs/desy/user/dayhallh/data/ILCsoftEvents/p22_th90_ph90_en10-100_joined"
    "/gpfs/dust/maxwell/user/dayhallh/data/ILCsoftEvents/p22_th90_ph90_en10-100_joined"
    "/p22_th90_ph90_en10-100_seed{}_all_steps.hdf5"
)
configs.dataset_path = path
print(f"Using dataset from {configs.dataset_path}")
configs.n_dataset_files = 10
n_events = sum(get_n_events(configs.dataset_path, configs.n_dataset_files))
local_batch_size = 10_000


data_dir = os.path.join(configs.storage_base, "dayhallh/point-cloud-diffusion-data")
# data_dir = "/home/dayhallh/Data/"
print(f"Data dir is {data_dir}")
showerflow_dir = os.path.join(data_dir, "showerFlow")
print(f"Showerflow data will be saved to {showerflow_dir}")
if not os.path.exists(showerflow_dir):
    os.mkdir(showerflow_dir)

print(f"Of the {n_events} avaliable, we are batching into {local_batch_size}")
# ## Data prep
#
# Now some values are needed to aid the showerflow training.

pointsE_path = showerflow_training.get_incident_npts_visible(
    configs, showerflow_dir, redo=False, local_batch_size=local_batch_size
)
# Calculating clusters per layer takes ~ 5 mins, so it's saved between runs.

clusters_per_layer_path = showerflow_training.get_clusters_per_layer(
    configs, showerflow_dir, redo=False, local_batch_size=local_batch_size
)

energy_per_layer_path = showerflow_training.get_energy_per_layer(
    configs, showerflow_dir, redo=False, local_batch_size=local_batch_size
)

# The energy per layer and the clusters per layer have now either been calculated,
# or loaded from disk.
# They should be normalised between 0 and 1 for the training.
# After this we will visulise a little data for a sanity check.

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
    configs, showerflow_dir, redo=False, local_batch_size=local_batch_size
)

print(f"Cog in x has mean {np.mean(cog[0])} and std {np.std(cog[0])}")
print(f"Cog in y has mean {np.mean(cog[1])} and std {np.std(cog[1])}")
print(f"Cog in z has mean {np.mean(cog[2])} and std {np.std(cog[2])}")
normed_cog = np.copy(cog)
prenormed_mean = np.mean(normed_cog, axis=1)
prenormed_std = np.std(normed_cog, axis=1)
print(f"Found means {prenormed_mean}, stds {prenormed_std}")


normed_cog = (normed_cog - prenormed_mean[:, np.newaxis]) / prenormed_std[:, np.newaxis]

print(
    f"Normed cog in x has mean {np.mean(normed_cog[0])} and std {np.std(normed_cog[0])}"
)
print(
    f"Normed cog in y has mean {np.mean(normed_cog[1])} and std {np.std(normed_cog[1])}"
)
print(
    f"Normed cog in z has mean {np.mean(normed_cog[2])} and std {np.std(normed_cog[2])}"
)

# ## Training data
#
# We now have everything needed to trian showerflow,
# we can add it to a pandas dataframe and run the training loop
# As the dataset may be larger than can be carried in the ram,
# this will be repeated for each epoch.

make_train_ds = showerflow_training.train_ds_function_factory(
    pointsE_path, cog_path, clusters_per_layer_path, energy_per_layer_path, configs
)


start_points = np.arange(0, n_events, local_batch_size)
# try it out
dataset = make_train_ds(0, 5)

pin_memory = device == "cpu"
print(f"Pin memory is {pin_memory}, device is {device}")
train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=params["batch_size"],
    shuffle=params["shuffle"],
    pin_memory=pin_memory,
)
# Then we set where we intent to save the created model.
prefix = time.strftime("%y-%m-%d_cog_e_layer_")
outpath = os.path.join(showerflow_dir, prefix)
print(f"Saving to {outpath}")

name_base = f"ShowerFlow_{chosen}_nb{num_blocks}_inputs{inputs_used_as_base10}"
best_model_path = os.path.join(showerflow_dir, f"{name_base}_best.pth")
latest_model_path = os.path.join(showerflow_dir, f"{name_base}_latest.pth")
best_data_path = os.path.join(showerflow_dir, f"{name_base}_best_data.txt")
history_data_path = os.path.join(showerflow_dir, f"{name_base}_history.npy")
print(f"Best model will be saved to {best_model_path}")
print(f"Latest model will be saved to {latest_model_path}")
print(f"Best data will be saved to {best_data_path}")
# torch.manual_seed(123)

lr = 1e-5  # default: 5e-5
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

start_over = False
# epoch_load = 550
# model.load_state_dict(torch.load(outpath+f'ShowerFlow_{epoch_load}.pth')['model'])
# optimizer.load_state_dict(torch.load(outpath+f'ShowerFlow_{epoch_load}.pth')['optimizer'])
if os.path.exists(best_model_path) and not start_over:
    model.load_state_dict(torch.load(best_model_path)["model"])
    optimizer.load_state_dict(torch.load(best_model_path)["optimizer"])

    # load best_loss
    with open(best_data_path, "r") as f:
        data = f.read().strip().split()
    best_loss = float(data[0])
    epoch_start = int(data[1])
else:
    best_loss = np.inf
    epoch_start = 1
print(
    f"A total of {params['epochs']} epochs are requested, "
    f"the model had already undergone {epoch_start}"
)
# The data is now all loaded, and ready to train on.
# If we had a previous best epoch it's loaded and we will start there.
model.train()

losses = []
epoch_nums = []
batch_len = len(train_loader)
mean_loss = np.inf
total_epochs = params["epochs"]
for epoch in range(epoch_start, total_epochs):
    start_idx = start_points[epoch % len(start_points)]
    dataset = make_train_ds(start_idx, start_idx + local_batch_size)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params["batch_size"],
        shuffle=params["shuffle"],
        pin_memory=pin_memory,
    )
    percent = (epoch - epoch_start) / (total_epochs - epoch_start)
    print(
        f"     Epoch number {epoch}; {percent:.0%} complete; mean loss {mean_loss:.2}",
        end="\r\r",
    )
    loss_list = []
    for batch_idx, (
        energy,
        num_points,
        visible_energy,
        cog_x,
        cog_y,
        cog_z,
        clusters_per_layer,
        e_per_layer,
    ) in enumerate(train_loader):
        if batch_idx % 10 == 0:
            print(f"{batch_idx/batch_len:.0%}", end="\r")

        E_true = energy.view(-1, 1).to(device).float()
        num_points = num_points.view(-1, 1).to(device).float()
        energy_sum = visible_energy.view(-1, 1).to(device).float()
        cog_x = cog_x.view(-1, 1).to(device).float()
        cog_y = cog_y.view(-1, 1).to(device).float()
        cog_z = cog_z.view(-1, 1).to(device).float()
        clusters_per_layer = clusters_per_layer.to(device).float()
        e_per_layer = e_per_layer.to(device).float()

        input_data = torch.cat(
            (
                num_points,
                energy_sum,
                cog_x,
                cog_y,
                cog_z,
                clusters_per_layer,
                e_per_layer,
            ),
            1,
        )  # input data structure required for network
        # filter any unused inputs
        input_data = input_data[:, inputs_used].to(device)
        # with additional features in latent space (e.g. Esum)
        optimizer.zero_grad()
        # try to add context for conditioning by concatenating
        context = E_true

        if np.any(np.isnan(input_data.clone().detach().cpu().numpy())):
            print("Nans in the training data!")

        # check if any of the weights are nans
        if torch.stack([torch.isnan(p).any() for p in model.parameters()]).any():
            print("Weights are nan!")
            # load recent model
            model.load_state_dict(torch.load(latest_model_path)["model"])
            optimizer = optim.Adam(model.parameters(), lr=lr)
            optimizer.load_state_dict(torch.load(latest_model_path)['optimizer'])
            # change the seed so we don't get back to the same nan
            torch.manual_seed(time.time()) 
            # print(f'model from {epoch-3} epoch reloaded')
            print("latest model reloaded, optimizer reset")

        nll = -distribution.condition(context).log_prob(input_data)
        loss = nll.mean()
        loss.backward()
        optimizer.step()
        distribution.clear_cache()
        loss_list.append(loss.item())
    mean_loss = np.mean(loss_list)
    epoch_nums.append(epoch)
    losses.append(mean_loss)
    
    np.save(history_data_path, np.array([epoch_nums, losses]))

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
model.load_state_dict(torch.load(best_model_path)["model"])
optimizer.load_state_dict(torch.load(best_model_path)["optimizer"])

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
for batch_idx, (
    energy,
    num_points,
    visible_energy,
    cog_x,
    cog_y,
    cog_z,
    clusters_per_layer,
    e_per_layer,
) in enumerate(tqdm(train_loader)):
    E_true = energy.view(-1, 1).to(device).float()
    E_true = (E_true / 100).float()
    with torch.no_grad():
        samples = (
            distribution.condition(E_true)
            .sample(
                torch.Size(
                    [
                        E_true.shape[0],
                    ]
                )
            )
            .cpu()
            .numpy()
        )
    E_true_list.append(E_true.cpu().numpy())
    samples_list.append(samples)
    num_points_list.append(num_points.cpu().numpy())
    visible_energy_list.append(visible_energy.cpu().numpy())
    cog_x_list.append(cog_x.cpu().numpy())
    cog_y_list.append(cog_y.cpu().numpy())
    cog_z_list.append(cog_z.cpu().numpy())
    clusters_per_layer_list.append(clusters_per_layer.cpu().numpy())
    e_per_layer_list.append(e_per_layer.cpu().numpy())

raw_E_true = np.concatenate(E_true_list, axis=0)
samples = np.concatenate(samples_list, axis=0)
raw_num_points = np.concatenate(num_points_list, axis=0)
raw_visible_energy = np.concatenate(visible_energy_list, axis=0)
raw_cog_x = np.concatenate(cog_x_list, axis=0)
raw_cog_y = np.concatenate(cog_y_list, axis=0)
raw_cog_z = np.concatenate(cog_z_list, axis=0)
raw_clusters_per_layer = np.concatenate(clusters_per_layer_list, axis=0)
raw_e_per_layer = np.concatenate(e_per_layer_list, axis=0)
samples.shape, raw_E_true.shape
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
h2 = plt.hist(
    num_points_sampled.flatten(), bins=h[1], histtype="step", lw=2, color="tab:orange"
)

# same but in log scale
ax = fig.add_subplot(gs[1])
h = plt.hist(num_points.flatten(), bins=100, color="lightgrey")
h2 = plt.hist(
    num_points_sampled.flatten(), bins=h[1], histtype="step", lw=2, color="tab:orange"
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
h2 = plt.hist(
    visible_energy_sampled.flatten(),
    bins=h[1],
    histtype="step",
    lw=2,
    color="tab:orange",
)

# same but in log scale
ax = fig.add_subplot(gs[1])
h = plt.hist(visible_energy.flatten(), bins=100, color="lightgrey")
h2 = plt.hist(
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
h = plt.hist(visible_energy.flatten() / E_true.flatten(), bins=100, color="lightgrey")
h2 = plt.hist(
    visible_energy_sampled.flatten() / E_true.flatten(),
    bins=h[1],
    histtype="step",
    lw=2,
    color="tab:orange",
)

# same but in log scale
ax = fig.add_subplot(gs[1])
h = plt.hist(visible_energy.flatten() / E_true.flatten(), bins=100, color="lightgrey")
h2 = plt.hist(
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
h2 = plt.hist(
    cog_x_sampled.flatten(), bins=h[1], histtype="step", lw=2, color="tab:orange"
)
plt.xlim(-20, 20)

# same but in log scale
ax = fig.add_subplot(gs[1])
h = plt.hist(cog_x.flatten(), bins=100, color="lightgrey")
h2 = plt.hist(
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
h2 = ax1.hist(
    cog_y_sampled.flatten(), bins=h[1], histtype="step", lw=2, color="tab:orange"
)
# ax1.xlim(5,30)

# same but in log scale
h = ax2.hist(cog_y.flatten(), bins=100, color="lightgrey")
h2 = ax2.hist(
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
h2 = plt.hist(
    cog_z_sampled.flatten(), bins=h[1], histtype="step", lw=2, color="tab:orange"
)
# plt.xlim(25,60)

# same but in log scale
ax = fig.add_subplot(gs[1])
h = plt.hist(cog_z.flatten(), bins=100, color="lightgrey")
h2 = plt.hist(
    cog_z_sampled.flatten(), bins=h[1], histtype="step", lw=2, color="tab:orange"
)
plt.yscale("log")
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
    h2 = plt.hist(
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
    h2 = plt.hist(
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
    h2 = plt.hist(
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
    h2 = plt.hist(
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
