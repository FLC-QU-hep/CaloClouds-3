# # Showerflow

# 

# Building and training showerflow from data.

# 

# Start by setting up notebook enviroment.
%load_ext autoreload

%autoreload 2



from IPython.display import display, HTML

import random

display(HTML("<style>.container { width:90% !important; }</style>"))



import pandas as pd

from tqdm import tqdm



import torch

import torch.nn as nn

import torch.nn.functional as F

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec


import numpy as np

import os

import time

import torch.utils.data

from torch import optim

from torch.nn import functional as F

from torchvision.utils import save_image



import time



from pointcloud.config_varients import caloclouds_3

from pointcloud.data.read_write import get_n_events, read_raw_regaxes

from pointcloud.utils.detector_map import floors_ceilings

from pointcloud.utils.metadata import Metadata

from pointcloud.models.shower_flow import compile_HybridTanH_model





#device = torch.device('cuda:0')

device = torch.device('cpu')

configs = caloclouds_3.Configs()

meta = Metadata(configs)
model, distribution = compile_HybridTanH_model(num_blocks=10, 

                                           num_inputs=65, ### when 'condioning' on additional Esum, Nhits etc add them on as inputs rather than 

                                           num_cond_inputs=1, device=device)  # num_cond_inputs

default_params = {

    "batch_size" : 2048,

    "epochs" : 2000,

    "shuffle" : True,

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
#path = '/beegfs/desy/user/akorol/data/calo-clouds/hdf5/high_granular_grid/train/10-90GeV_x36_grid_regular_524k_float32.hdf5'

path = "/home/dayhallh/Data/p22_th90_ph90_en10-100_joined/p22_th90_ph90_en10-100_seed{}_all_steps.hdf5"

path = "/beegfs/desy/user/dayhallh/data/ILCsoftEvents/p22_th90_ph90_en10-100_joined/p22_th90_ph90_en10-100_seed{}_all_steps.hdf5"

configs.dataset_path = path

configs.n_dataset_files = 10

n_events_used = 100_000

energy, events = read_raw_regaxes(configs, pick_events=slice(0, n_events_used))

n_events = sum(get_n_events(configs.dataset_path, configs.n_dataset_files))


redo_pointsE = False



#data_dir = configs.storage_base

data_dir = "/home/dayhallh/Data/"

showerflow_dir = os.path.join(data_dir, "showerFlow")

if not os.path.exists(showerflow_dir):

    os.mkdir(showerflow_dir)



pointsE_path = os.path.join(showerflow_dir, "pointsE.npz")

if os.path.exists(pointsE_path) and not redo_pointsE:

    print("Using precaluclated energies and counts")

else:

    print("Recalculating energies and counts")

    floors, ceilings = floors_ceilings(meta.layer_bottom_pos_raw, meta.cell_thickness_raw) 

    energy = np.zeros(n_events)

    num_points = np.zeros(n_events)

    visible_energy = np.zeros(n_events)

    for start_idx in range(0, n_events, local_batch_size):

        print(f"{start_idx/n_events:.0%}", end='\r')

        my_slice = slice(start_idx, start_idx + local_batch_size)

        energies_batch, events_batch = read_raw_regaxes(configs, pick_events=my_slice)

        energy[my_slice] = energies_batch

        num_points[my_slice] = (events[:, :, 3] > 0).sum(axis=1)

        visible_energy[my_slice] = (events[:, :, 3]).sum(axis=1)

    np.savez(pointsE_path, energy=energy, num_points=num_points, visible_energy=visible_energy)

    print(f"Energy goes from {visible_energy.min()}, to {visible_energy.max()}")

    print(f"Num points goes from {num_points.min()}, to {num_points.max()}")

    del energy

    del num_points

    del visible_energy    

print(f"Of the {n_events} avaliable, we are batching into {local_batch_size}")
# ## Data prep

# 

# Now some values are needed to aid the showerflow training.

# Calculating clusters per layer takes ~ 5 mins, so it's saved between runs.
redo_clusters = False



#data_dir = configs.storage_base

data_dir = "/home/dayhallh/Data/"

data_dir = "../../point-cloud-diffusion-data/"

showerflow_dir = os.path.join(data_dir, "showerFlow")

if not os.path.exists(showerflow_dir):

    os.mkdir(showerflow_dir)



clusters_per_layer_path = os.path.join(showerflow_dir, "clusters_per_layer.npy")

if os.path.exists(clusters_per_layer_path) and not redo_clusters:

    print("Using precaluclated clusters per layer")

else:

    print("Recalculating clusters per layer")

    floors, ceilings = floors_ceilings(meta.layer_bottom_pos_raw, meta.cell_thickness_raw) 

    clusters_per_layer = np .zeros((n_events, len(floors)))

    for start_idx in range(0, n_events, local_batch_size):

        print(f"{start_idx/n_events:.0%}", end='\r')

        my_slice = slice(start_idx, start_idx + local_batch_size)

        _, events_batch = read_raw_regaxes(configs, pick_events=my_slice)

        mask = events_batch[:, :, 3] > 0

        clusters_here = [((events_batch[:, :, 1] < c) & (events_batch[:, :, 1] > f) & mask).sum(axis=1) for f, c in zip(floors, ceilings)]

        clusters_per_layer[my_slice] = np.vstack(clusters_here).T

    rescales = clusters_per_layer/clusters_per_layer.max(axis=1)[:, np.newaxis]

    np.savez(clusters_per_layer_path, clusters_per_layer=clusters_per_layer, rescaled_clusters_per_layer=rescales)

    assert np.all(~np.isnan(clusters_per_layer))

    del clusters_per_layer

    del rescales



redo_energy = False

energy_per_layer_path = os.path.join(showerflow_dir, "energy_per_layer.npz")

if os.path.exists(energy_per_layer_path) and not redo_energy:

    print("Using precaluclated energy per layer")

else:

    print("Recalculating energy per layer")

    floors, ceilings = floors_ceilings(meta.layer_bottom_pos_raw, meta.cell_thickness_raw) 

    energy_per_layer = np .zeros((n_events, len(floors)))

    for start_idx in range(0, n_events, local_batch_size):

        print(f"{start_idx/n_events:.0%}", end='\r')

        my_slice = slice(start_idx, start_idx + local_batch_size)

        _, events_batch = read_raw_regaxes(configs, pick_events=my_slice)

        energy_here = [(events_batch[..., 3]*(events_batch[:, :, 1] < c) * (events_batch[:, :, 1] > f)).sum(axis=1) for f, c in zip(floors, ceilings)]

        energy_per_layer[my_slice] = np.vstack(energy_here).T

    rescaled = energy_per_layer/energy_per_layer.max(axis=1)[:, np.newaxis]

    np.savez(energy_per_layer_path, energy_per_layer=energy_per_layer, rescaled_energy_per_layer=rescaled)

    assert np.all(~np.isnan(energy_per_layer))

    del energy_per_layer

    del rescaled

    

# The energy per layer and the clusters per layer have now either been calculated, or loaded from disk.

# They should be normalised between 0 and 1 for the training.

# After this we will visulise a little data for a sanity check.
# normalize cluster and energy per layer to [0,1]

clusters_per_layer = clusters_per_layer / clusters_per_layer.max(axis=1).reshape(len(clusters_per_layer), 1)

e_per_layer = energy_per_layer

e_per_layer = e_per_layer / e_per_layer.max(axis=1).reshape(len(e_per_layer), 1)


hist, (clusters_ax, energy_ax) = plt.subplots(1, 2, figsize=(10, 4))

start_event = 5

clusters = np.load(clusters_per_layer_path, mmap_mode='r')

energies = np.load(energy_per_layer_path, mmap_mode='r')

for i in range(start_event, start_event + 5):

    clu = clusters["rescaled_clusters_per_layer"][i]

    e = energies["rescaled_energy_per_layer"][i]

    clusters_ax.hist(np.arange(30), weights=clu, histtype='step', bins=30)

    energy_ax.hist(np.arange(30), weights=e, histtype='step', bins=30)

energy_ax.set_xlabel("layer")

clusters_ax.set_xlabel("layer")

energy_ax.set_ylabel("energy")

clusters_ax.set_ylabel("clusters")

energy_ax.legend()

# center of gravity 



def get_cog(x,y,z,e):

    sum_e = e.sum(axis=1)

    return np.sum((x * e), axis=1) / sum_e, np.sum((y * e), axis=1) / sum_e, np.sum((z * e), axis=1) / sum_e



cog = get_cog(

    events[..., 0],

    events[..., 1],

    events[..., 2],

    events[..., 3],

)

fig, (unnormed_ax, normed_ax) = plt.subplots(2, 1, figsize=(10,8))

kw_args = dict(bins = 50, alpha=0.5, desnity=True)

unnormed_ax.hist(cog[0], label='x', **kw_args)

unnormed_ax.hist(cog[1], label='y', **kw_args)

unnormed_ax.hist(cog[2], label='z', **kw_args)



unnormed_ax.set_xlabel("Center of Gravity")

unnormed_ax.legend()



normed_cog = np.copy(cog)

prenormed_mean = np.mean(normed_cog, axis=1)

prenormed_std = np.std(normed_cog, axis=1)

print(f"Found means {prenormed_mean}, stds {prenormed_std}")





normed_cog = (normed_cog - prenormed_mean[:, np.newaxis])/prenormed_std[:, np.newaxis]





cog_norm_path = os.path.join(showerflow_dir, "cog_norm")

np.savez(cog_norm_path, mean_cog=prenormed_mean, std_cog=prenormed_std)



kw_args["bins"] = np.linspace(-30, 30, 50)

normed_ax.hist(normed_cog[0], label='x', **kw_args)

normed_ax.hist(normed_cog[1], label='y', **kw_args)

normed_ax.hist(normed_cog[2], label='z', **kw_args)



normed_ax.set_xlabel("Normalised Center of Gravity")

normed_ax.legend()

print(prenormed_mean, prenormed_std)
# ## Training data

# 

# We now have everything needed to trian showerflow, we can add it to a pandas dataframe and run the training loop

# As the dataset may be larger than can be carried in the ram, this will be repeated for each epoch.




def make_train_ds(start_idx, end_idx):

    my_slice = slice(start_idx, end_idx)

    df = pd.DataFrame([])

    # mem-map the files to avoid loading all data

    pointE = np.load(pointsE_path , mmap_mode='r')

    df['energy'] = pointE["energy"][my_slice].copy().reshape(-1)   # normalisation done in training loop

    

    df['num_points'] = pointE["num_points"][my_slice].copy() / n_pts_rescale 

    df['visible_energy'] = pointE["visible_energy"][my_slice].copy() / vis_eng_rescale

    

    normed_cog = np.load(cog_path, mmap_mode='r')[my_slice].copy()

    normed_cog = (normed_cog - mean_cog)/std_cog

    

    df['cog_x'] = normed_cog[:, 0]

    df['cog_y'] = normed_cog[:, 1]

    df['cog_z'] = normed_cog[:, 2]

    

    clusters = np.load(clusters_per_layer_path, mmap_mode='r')

    df['clusters_per_layer'] = clusters["rescaled_clusters_per_layer"][my_slice].tolist()

    energies = np.load(energy_per_layer_path, mmap_mode='r')

    df['e_per_layer'] = energies["rescaled_energy_per_layer"][my_slice].tolist()

    

    for series_name, series in df.items():

        series = np.vstack(series.to_numpy())

        assert np.all(~np.isnan(series)), series_name

        

    dataset = torch.utils.data.TensorDataset(

        torch.tensor(df.energy.values), 

        torch.tensor(df.num_points.values), 

        torch.tensor(df.visible_energy.values),

        torch.tensor(df.cog_x.values),

        torch.tensor(df.cog_y.values),

        torch.tensor(df.cog_z.values),

        torch.tensor(df.clusters_per_layer),

        torch.tensor(df.e_per_layer),

    )

    return dataset



start_points = np.arange(0, n_events, local_batch_size)

# try it out

dataset = make_train_ds(0, 5)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=params["batch_size"], shuffle=params["shuffle"], pin_memory=True)
# Then we set where we intent to save the created model.
# clusters_per_layer = clusters_per_layer / 400



df = pd.DataFrame([])

df['energy'] = energy[:].reshape(-1)   # normalisation done in training loop



df['num_points'] = num_points / 5000

df['visible_energy'] = visible_energy / 2.5 



df['cog_x'] = normed_cog[0]

df['cog_y'] = normed_cog[1]

df['cog_z'] = normed_cog[2]



df['clusters_per_layer'] = clusters_per_layer.tolist()

df['e_per_layer'] = e_per_layer.tolist()
dataset = torch.utils.data.TensorDataset(

    torch.tensor(df.energy.values), 

    torch.tensor(df.num_points.values), 

    torch.tensor(df.visible_energy.values),

    torch.tensor(df.cog_x.values),

    torch.tensor(df.cog_y.values),

    torch.tensor(df.cog_z.values),

    torch.tensor(df.clusters_per_layer),

    torch.tensor(df.e_per_layer),

)



train_loader = torch.utils.data.DataLoader(dataset, batch_size=params["batch_size"], shuffle=params["shuffle"], pin_memory=True)
prefix = time.strftime("%y-%m-%d_cog_e_layer_")

outpath = os.path.join(showerflow_dir, prefix)

print(f"Saving to {outpath}")



best_model_path = os.path.join(showerflow_dir, "ShowerFlow_best.pth")

latest_model_path = os.path.join(showerflow_dir, "ShowerFlow_latest.pth")

best_data_path = os.path.join(showerflow_dir, "ShowerFlow_best_data.txt")
# torch.manual_seed(123)



lr = 1e-5   # default: 5e-5

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)



start_over = False

# epoch_load = 550

# model.load_state_dict(torch.load(outpath+f'ShowerFlow_{epoch_load}.pth')['model'])

# optimizer.load_state_dict(torch.load(outpath+f'ShowerFlow_{epoch_load}.pth')['optimizer'])

if os.path.exists(best_model_path) and not start_over:

    model.load_state_dict(torch.load(best_model_path)['model'])

    optimizer.load_state_dict(torch.load(best_model_path)['optimizer'])



    # load best_loss

    with open(best_data_path, 'r') as f:

        data = f.read().strip().split()

    best_loss = float(data[0])

    epoch_start = int(data[1])

else:

    best_loss = np.inf

    epoch_start = 1

print(f"A total of {params['epochs']} epochs are requested, the model had already undergone {epoch_start}")
# The data is now all loaded, and ready to train on. If we had a previous best epoch it's loaded and we will start there.
model.train()



losses = []

batch_len = len(train_loader)

mean_loss = np.inf

total_epochs = params["epochs"]

for epoch in range(epoch_start, total_epochs):

    start_idx = start_points[epoch%len(start_points)]

    dataset = make_train_ds(start_idx, start_idx+local_batch_size)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=params["batch_size"], shuffle=params["shuffle"], pin_memory=True)

    

    print(f"     Epoch number {epoch}; {(epoch-epoch_start)/(total_epochs-epoch_start):.0%} complete; mean loss {mean_loss:.2}", end='\r\r')

    loss_list = []

    for batch_idx, (energy, num_points, visible_energy, cog_x, cog_y, cog_z, clusters_per_layer, e_per_layer) in enumerate(train_loader):

        if batch_idx % 10 == 0:

            print(f"{batch_idx/batch_len:.0%}", end='\r')

            



        E_true = energy.view(-1, 1).to(device).float()

        num_points = num_points.view(-1, 1).to(device).float()

        energy_sum = visible_energy.view(-1,1).to(device).float()

        cog_x = cog_x.view(-1, 1).to(device).float()

        cog_y = cog_y.view(-1, 1).to(device).float()

        cog_z = cog_z.view(-1, 1).to(device).float()

        clusters_per_layer = clusters_per_layer.to(device).float()

        e_per_layer = e_per_layer.to(device).float()

   

        # normalise conditional labels

        E_true = (E_true/incident_rescale).float()



        input_data = torch.cat((num_points, energy_sum, cog_x, cog_y, cog_z, clusters_per_layer, e_per_layer), 1)   #### input data structure required for network with additional features in latent space (e.g. Esum)

        optimizer.zero_grad()

        # try to add context for conditioning by concatenating 

        context = E_true

        

        if np.any(np.isnan(input_data.clone().detach().cpu().numpy())) == True:

            print('Nans in the training data!')

            

        #### check if any of the weights are nans

        if torch.stack([torch.isnan(p).any() for p in model.parameters()]).any():

            print('Weights are nan!')

            # load recent model

            model.load_state_dict(torch.load(latest_model_path)['model'])

            optimizer = optim.Adam(model.parameters(), lr=lr)

            #optimizer.load_state_dict(torch.load(outpath+f'ShowerFlow_latest.pth')['optimizer'])

            # print(f'model from {epoch-3} epoch reloaded')

            print(f'latest model reloaded, optimizer resetted')

                           

        nll = -distribution.condition(context).log_prob(input_data)

        loss = nll.mean()

        loss.backward()

        optimizer.step() 

        distribution.clear_cache()

        loss_list.append(loss.item())

    mean_loss = np.mean(loss_list)

    losses.append(mean_loss)



    if torch.stack([torch.isnan(p).any() for p in model.parameters()]).any(): # save models only if no nan weights

        print('model not saved due to nan weights')

    else:

        torch.save(

            {'model': model.state_dict(),

            'optimizer': optimizer.state_dict(),},

            latest_model_path

        )



        # save best model based on loss

        if mean_loss <= best_loss:

            best_loss = mean_loss

            torch.save(

                {'model': model.state_dict(),

                'optimizer': optimizer.state_dict(),},

                best_model_path

            )

            # save best loss value to txt file

            with open(best_data_path, 'w') as f:

                f.write(f"{best_loss} {epoch}")

        



# ## After training

# 

# Now the training is complete, there should be a best checkpoint. We will load this in order to generate some evaluation plots. Setting the seed makes sure the evaluation plots are reproducable.
# load best checkpoint

model.load_state_dict(torch.load(best_model_path)['model'])

optimizer.load_state_dict(torch.load(best_model_path)['optimizer'])



model.eval()

print('model loaded')
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

for batch_idx, (energy, num_points, visible_energy, cog_x, cog_y, cog_z, clusters_per_layer, e_per_layer) in enumerate(tqdm(train_loader)):

    E_true = energy.view(-1, 1).to(device).float()

    E_true = (E_true/100).float()

    with torch.no_grad():

        samples = distribution.condition(E_true).sample(torch.Size([E_true.shape[0], ])).cpu().numpy()

    E_true_list.append(E_true.cpu().numpy())

    samples_list.append(samples)

    num_points_list.append(num_points.cpu().numpy())

    visible_energy_list.append(visible_energy.cpu().numpy())

    cog_x_list.append(cog_x.cpu().numpy())  

    cog_y_list.append(cog_y.cpu().numpy())

    cog_z_list.append(cog_z.cpu().numpy())

    clusters_per_layer_list.append(clusters_per_layer.cpu().numpy())

    e_per_layer_list.append(e_per_layer.cpu().numpy())



E_true = np.concatenate(E_true_list, axis=0)

samples = np.concatenate(samples_list, axis=0)

num_points = np.concatenate(num_points_list, axis=0)

visible_energy = np.concatenate(visible_energy_list, axis=0)

cog_x = np.concatenate(cog_x_list, axis=0)

cog_y = np.concatenate(cog_y_list, axis=0)

cog_z = np.concatenate(cog_z_list, axis=0)

clusters_per_layer = np.concatenate(clusters_per_layer_list, axis=0)

e_per_layer = np.concatenate(e_per_layer_list, axis=0)
samples.shape, E_true.shape
# sampled 

num_points_sampled = samples[:, 0] * n_pts_rescale

visible_energy_sampled = samples[:, 1] * vis_eng_rescale

cog_x_sampled = (samples[:, 2] * std_cog[0]) + mean_cog[0]

cog_y_sampled = (samples[:, 3] * std_cog[1]) + mean_cog[1]

cog_z_sampled = (samples[:, 4] * std_cog[2]) + mean_cog[2]

clusters_per_layer_sampled = samples[:, 5:35]

e_per_layer_sampled = samples[:, 35:]



# truth; we need to undo the shifts that were done in the last training batch



E_true = E_true * incident_rescale

num_points = num_points * n_pts_rescale

visible_energy = visible_energy * vis_eng_rescale

cog_x = (cog_x * std_cog[0]) + mean_cog[0]

cog_y = (cog_y * std_cog[1]) + mean_cog[1]

cog_z = (cog_z * std_cog[2]) + mean_cog[2]





# clip cluster and energies per layer to [0,1]

clusters_per_layer_sampled = np.clip(clusters_per_layer_sampled, 0, 1)

e_per_layer_sampled = np.clip(e_per_layer_sampled, 0, 1)
print(cog_x.min(), cog_x.max())

print(cog_y.min(), cog_y.max())

print(cog_z.min(), cog_z.max())
fig = plt.figure(figsize=(12,6))

gs = gridspec.GridSpec(1,2)



ax = fig.add_subplot(gs[0])

h = plt.hist(num_points.flatten(), bins=100, color='lightgrey')

h2 = plt.hist(num_points_sampled.flatten(), bins=h[1], histtype='step', lw=2, color='tab:orange')



# same but in log scale

ax = fig.add_subplot(gs[1])

h = plt.hist(num_points.flatten(), bins=100, color='lightgrey')

h2 = plt.hist(num_points_sampled.flatten(), bins=h[1], histtype='step', lw=2, color='tab:orange')

plt.yscale('log')

plt.ylim(1, 1e5)



plt.suptitle('Number of points per event')

plt.show()
fig = plt.figure(figsize=(12,6))

gs = gridspec.GridSpec(1,2)



ax = fig.add_subplot(gs[0])

h = plt.hist(visible_energy.flatten(), bins=100, color='lightgrey')

h2 = plt.hist(visible_energy_sampled.flatten(), bins=h[1], histtype='step', lw=2, color='tab:orange')



# same but in log scale

ax = fig.add_subplot(gs[1])

h = plt.hist(visible_energy.flatten(), bins=100, color='lightgrey')

h2 = plt.hist(visible_energy_sampled.flatten(), bins=h[1], histtype='step', lw=2, color='tab:orange')

plt.yscale('log')

plt.ylim(1, 1e5)



plt.suptitle('visible energy')

plt.show()
# sampling fraction

fig = plt.figure(figsize=(12,6))

gs = gridspec.GridSpec(1,2)



ax = fig.add_subplot(gs[0])

h = plt.hist(visible_energy.flatten()/E_true.flatten(), bins=100, color='lightgrey')

h2 = plt.hist(visible_energy_sampled.flatten()/E_true.flatten(), bins=h[1], histtype='step', lw=2, color='tab:orange')



# same but in log scale

ax = fig.add_subplot(gs[1])

h = plt.hist(visible_energy.flatten()/E_true.flatten(), bins=100, color='lightgrey')

h2 = plt.hist(visible_energy_sampled.flatten()/E_true.flatten(), bins=h[1], histtype='step', lw=2, color='tab:orange')

plt.yscale('log')

plt.ylim(1, 1e5)



# title over all subplots

fig.suptitle('Sampling fraction')

plt.show()
fig = plt.figure(figsize=(12,6))

gs = gridspec.GridSpec(1,2)



ax = fig.add_subplot(gs[0])

h = plt.hist(cog_x.flatten(), bins=100, color='lightgrey')

h2 = plt.hist(cog_x_sampled.flatten(), bins=h[1], histtype='step', lw=2, color='tab:orange')

plt.xlim(-20,20)



# same but in log scale

ax = fig.add_subplot(gs[1])

h = plt.hist(cog_x.flatten(), bins=100, color='lightgrey')

h2 = plt.hist(cog_x_sampled.flatten(), bins=h[1], histtype='step', lw=2, color='tab:orange')

plt.yscale('log')

plt.ylim(1, 1e6)

plt.xlim(-20,20)



plt.suptitle('center of gravity x')

plt.show()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))



h = ax1.hist(cog_y.flatten(), bins=100, color='lightgrey')

h2 = ax1.hist(cog_y_sampled.flatten(), bins=h[1], histtype='step', lw=2, color='tab:orange')

#ax1.xlim(5,30)



# same but in log scale

h = ax2.hist(cog_y.flatten(), bins=100, color='lightgrey')

h2 = ax2.hist(cog_y_sampled.flatten(), bins=h[1], histtype='step', lw=2, color='tab:orange')

plt.yscale('log')

#plt.ylim(1, 1e6)

#plt.xlim(5,30)



plt.suptitle('center of gravity y')

plt.show()
fig = plt.figure(figsize=(12,6))

gs = gridspec.GridSpec(1,2)



ax = fig.add_subplot(gs[0])

h = plt.hist(cog_z.flatten(), bins=100, color='lightgrey')

h2 = plt.hist(cog_z_sampled.flatten(), bins=h[1], histtype='step', lw=2, color='tab:orange')

#plt.xlim(25,60)



# same but in log scale

ax = fig.add_subplot(gs[1])

h = plt.hist(cog_z.flatten(), bins=100, color='lightgrey')

h2 = plt.hist(cog_z_sampled.flatten(), bins=h[1], histtype='step', lw=2, color='tab:orange')

plt.yscale('log')

#plt.ylim(1, 1e6)

#plt.xlim(25,60)



plt.suptitle('center of gravity z')

plt.show()
fig = plt.figure(figsize=(12,12))

gs = gridspec.GridSpec(6,5)



for i in range(30):

    fig.add_subplot(gs[i])

    h = plt.hist(clusters_per_layer[:,i].flatten(), bins=100, color='lightgrey', range=[-0.2,1.2])

    h2 = plt.hist(clusters_per_layer_sampled[:,i].flatten(), bins=h[1], histtype='step', lw=1, color='tab:orange')



plt.suptitle('clusters per layer')

plt.show()
fig = plt.figure(figsize=(12,12))

gs = gridspec.GridSpec(6,5)



for i in range(30):

    fig.add_subplot(gs[i])

    h = plt.hist(clusters_per_layer[:,i].flatten(), bins=100, color='lightgrey', range=[-0.2,1.2])

    h2 = plt.hist(clusters_per_layer_sampled[:,i].flatten(), bins=h[1], histtype='step', lw=1, color='tab:orange')

    plt.yscale('log')



plt.suptitle('clusters per layer (log scale)')

plt.show()
fig = plt.figure(figsize=(12,12))

gs = gridspec.GridSpec(6,5)



for i in range(30):

    fig.add_subplot(gs[i])

    h = plt.hist(e_per_layer[:,i].flatten(), bins=100, color='lightgrey', range=[-0.2,1.2])

    h2 = plt.hist(e_per_layer_sampled[:,i].flatten(), bins=h[1], histtype='step', lw=1, color='tab:orange')

    # plt.yscale('log')



plt.suptitle('energy per layer')

plt.show()
fig = plt.figure(figsize=(12,12))

gs = gridspec.GridSpec(6,5)



for i in range(30):

    fig.add_subplot(gs[i])

    h = plt.hist(e_per_layer[:,i].flatten(), bins=100, color='lightgrey', range=[-0.2,1.2])

    h2 = plt.hist(e_per_layer_sampled[:,i].flatten(), bins=h[1], histtype='step', lw=1, color='tab:orange')

    plt.yscale('log')



plt.suptitle('energy per layer (log scale)')

plt.show()


