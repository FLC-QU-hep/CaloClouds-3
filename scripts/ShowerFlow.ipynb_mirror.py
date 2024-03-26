%load_ext autoreload

%autoreload 2



from IPython.core.display import display, HTML

import random

display(HTML("<style>.container { width:90% !important; }</style>"))
import pandas as pd

from tqdm import tqdm
import torch

import torch.nn as nn

import torch.nn.functional as F

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
# from functools import partial



# import torch



# from pyro.nn import AutoRegressiveNN, ConditionalAutoRegressiveNN



# from pyro.distributions import constraints

# from pyro.distributions.conditional import ConditionalTransformModule

# from pyro.distributions.torch_transform import TransformModule

# from pyro.distributions.util import copy_docs_from

# from pyro.distributions.transforms import SplineCoupling

# from pyro.distributions.transforms.spline import ConditionalSpline







import numpy as np

# import torch

import torch.utils.data

from torch import optim

from torch.nn import functional as F

# from torchvision import datasets, transforms

from torchvision.utils import save_image

# from data_utils.data_loader import HDF5Dataset, MaxwellBatchLoader, MaxwellBatchLoaderFullyCond



# from torch.nn.parallel import DistributedDataParallel as DDP

# from torch.multiprocessing import Process



#import ML_models.models as models

#import ML_models.functions as functions

import time

import h5py



# from custom_pyro import ConditionalAffineCouplingTanH



# from pyro.nn import ConditionalDenseNN, DenseNN, ConditionalAutoRegressiveNN

# import pyro.distributions as dist

# import pyro.distributions.transforms as T



from pointcloud.models.shower_flow import compile_HybridTanH_model



device = torch.device('cuda:0')

model, distribution = compile_HybridTanH_model(num_blocks=10, 

                                           num_inputs=65, ### when 'condioning' on additional Esum, Nhits etc add them on as inputs rather than 

                                           num_cond_inputs=1, device=device)  # num_cond_inputs
default_params = {

    "batch_size" : 2048,

    "epochs" : 1000,

    "shuffle" : True,

}
kwargs = {}

params = {}

for param in default_params.keys():



    if param in kwargs.keys():

        params[param] = kwargs[param]

    else:

        params[param] = default_params[param]

path = '/beegfs/desy/user/akorol/data/calo-clouds/hdf5/high_granular_grid/train/10-90GeV_x36_grid_regular_524k_float32.hdf5'

energy = h5py.File(path, 'r')['energy'][:]

events = h5py.File(path, 'r')['events'][:]



num_points = (events[:][:, -1] > 0).sum(axis=1)

visible_energy = (events[:][:, -1]).sum(axis=1)
visible_energy.min(), .max()
# calc cluster number and cluster energy per layer   - TAKES ABOUT 5 MINUTES, so just run once and save



# clusters_per_layer = [((events[:, 1, :] < i+1) & (events[:, 1, :] > i)).sum(axis=1) for i in range(30)]

# clusters_per_layer = np.vstack(clusters_per_layer)

# clusters_per_layer = np.moveaxis(clusters_per_layer, 0, -1)





# e_per_layer = []

# for i in tqdm(range(30)):

#     layer_mask = (events[:, 1, :] < i+1) & (events[:, 1, :] > i)

#     e_per_layer.append( (events[:, -1, :] * layer_mask).sum(axis=1) )

# e_per_layer = np.vstack(e_per_layer)

# e_per_layer = np.moveaxis(e_per_layer, 0, -1)
outdir = '/beegfs/desy/user/buhmae/6_PointCloudDiffusion/dataset/tmp/'



# save 

# np.save(outdir+'clusters_per_layer.npy', clusters_per_layer)

# np.save(outdir+'e_per_layer.npy', e_per_layer)



# load

clusters_per_layer = np.load(outdir+'clusters_per_layer.npy')

e_per_layer = np.load(outdir+'e_per_layer.npy')
# normalize cluster and energy per layer to [0,1]

clusters_per_layer = clusters_per_layer / clusters_per_layer.max(axis=1).reshape(len(clusters_per_layer), 1)

e_per_layer = e_per_layer / e_per_layer.max(axis=1).reshape(len(e_per_layer), 1)
plt.figure(figsize=(6,6))

for i in range(500000, 500005):

    plt.hist(np.arange(30), weights=clusters_per_layer[i, :], label=f'energy: {energy[i]}', histtype='step', bins=30)

plt.legend()

plt.show()



plt.figure(figsize=(6,6))

for i in range(500000, 500000+5):

    plt.hist(np.arange(30), weights=e_per_layer[i, :], label=f'energy: {energy[i]}', histtype='step', bins=30)

plt.legend()

plt.show()
# center of gravity 



def get_cog(x,y,z,e):

    return np.sum((x * e), axis=1) / e.sum(axis=1), np.sum((y * e), axis=1) / e.sum(axis=1), np.sum((z * e), axis=1) / e.sum(axis=1)



cog = get_cog(

    events[:, 0],

    events[:, 1],

    events[:, 2],

    events[:, 3],

)

plt.figure(figsize=(10,10))

plt.hist(cog[0] / 25, bins=100, range=(-10, 10), label='x')

plt.hist((cog[1] - 15) / 15, bins=100)

plt.hist((cog[2] - 40) / 20, bins=100)

# plt.xlim(-10, 10)

plt.show()
# clusters_per_layer = clusters_per_layer / 400



df = pd.DataFrame([])

df['energy'] = energy[:].reshape(-1)   # normalisation done in training loop



df['num_points'] = num_points / 5000

df['visible_energy'] = visible_energy / 2.5 



df['cog_x'] = cog[0] / 25

df['cog_y'] = (cog[1] - 15) / 15

df['cog_z'] = (cog[2] - 40) / 20



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

batch = next(iter(train_loader))
for item in batch:

    print(item.shape)
output_path = '/beegfs/desy/user/buhmae/6_PointCloudDiffusion/shower_flow/'

prefix = '220714_cog_e_layer_'

outpath = output_path + prefix
# torch.manual_seed(123)



lr = 1e-5   # default: 5e-5

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)



# epoch_load = 550

# model.load_state_dict(torch.load(outpath+f'ShowerFlow_{epoch_load}.pth')['model'])

# optimizer.load_state_dict(torch.load(outpath+f'ShowerFlow_{epoch_load}.pth')['optimizer'])

model.load_state_dict(torch.load(outpath+f'ShowerFlow_best.pth')['model'])



# load best_loss

with open(outpath+f'ShowerFlow_best_loss.txt', 'r') as f:

    best_loss = float(f.read())
epoch_start = 284   # default = 1
model.train()



losses = []

for epoch in range(epoch_start, params["epochs"]+1):

    input_list = []

    loss_list = []

#     for batch_idx, (mu, logvar, e, theta, e_sum) in enumerate(train_loader):

    for batch_idx, (energy, num_points, visible_energy, cog_x, cog_y, cog_z, clusters_per_layer, e_per_layer) in enumerate(tqdm(train_loader)):



        E_true = energy.view(-1, 1).to(device).float()

        num_points = num_points.view(-1, 1).to(device).float()

        energy_sum = visible_energy.view(-1,1).to(device).float()

        cog_x = cog_x.view(-1, 1).to(device).float()

        cog_y = cog_y.view(-1, 1).to(device).float()

        cog_z = cog_z.view(-1, 1).to(device).float()

        clusters_per_layer = clusters_per_layer.to(device).float()

        e_per_layer = e_per_layer.to(device).float()

   

        # normalise conditional labels

        E_true = (E_true/100).float()



        input_data = torch.cat((num_points, energy_sum, cog_x, cog_y, cog_z, clusters_per_layer, e_per_layer), 1)   #### input data structure required for network with additional features in latent space (e.g. Esum)

        

        #input_data = torch.cat((z), 1)

        #input_data = z



        optimizer.zero_grad()



        # try to add context for conditioning by concatenating 

        context = E_true

        #print(theta_true.size())

        

        #nll = -distribution.condition(E_true).log_prob(input_data) ## solution   # does this work, or do need spearate .condition for each label?

        

        if np.any(np.isnan(input_data.clone().detach().cpu().numpy())) == True:

            print('Nans in the training data!')

            

        #### check if any of the weights are nans

        if torch.stack([torch.isnan(p).any() for p in model.parameters()]).any():

            print('Weights are nan!')

            # load recent model

            model.load_state_dict(torch.load(outpath+f'ShowerFlow_latest.pth')['model'])

            optimizer = optim.Adam(model.parameters(), lr=lr)

            #optimizer.load_state_dict(torch.load(outpath+f'ShowerFlow_latest.pth')['optimizer'])

            # print(f'model from {epoch-3} epoch reloaded')

            print(f'latest model reloaded, optimizer resetted')

                           

        nll = -distribution.condition(context).log_prob(input_data)

        loss = nll.mean()

        #print(loss.item())

        loss.backward()



        optimizer.step() 



        distribution.clear_cache()

        

        loss_list.append(loss.item())

        # input_list.append(input_data.detach().cpu().numpy())



    print(epoch, np.mean(loss_list))

    losses.append(np.mean(loss_list))

    if epoch == 1:

        best_loss = np.mean(loss_list)



    if torch.stack([torch.isnan(p).any() for p in model.parameters()]).any(): # save models only if no nan weights

        print('model not saved due to nan weights')

    else:

        torch.save(

            {'model': model.state_dict(),

            'optimizer': optimizer.state_dict(),},

            outpath+f'ShowerFlow_latest.pth'

        )



        if epoch%10 == 0:

            torch.save(

                {'model': model.state_dict(),

                'optimizer': optimizer.state_dict(),},

                outpath+f'ShowerFlow_{epoch}.pth'

            )



        # save best model based on loss

        if np.mean(loss_list) <= best_loss:

            best_loss = np.mean(loss_list)

            torch.save(

                {'model': model.state_dict(),

                'optimizer': optimizer.state_dict(),},

                outpath+f'ShowerFlow_best.pth'

            )

            # save best loss value to txt file

            with open(outpath+f'ShowerFlow_best_loss.txt', 'w') as f:

                f.write(str(best_loss))

            print('best model saved, with loss: ', best_loss)

        # print('model saved')

        






# load best checkpoint

model.load_state_dict(torch.load(outpath+f'ShowerFlow_best.pth')['model'])

optimizer.load_state_dict(torch.load(outpath+f'ShowerFlow_best.pth')['optimizer'])



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
# df['num_points'] = num_points / 5000

# df['visible_energy'] = visible_energy / 2.5 



# df['cog_x'] = cog[0]

# df['cog_y'] = cog[1] / 2 - 7.5

# df['cog_z'] = cog[2] - 40



# df['clusters_per_layer'] = clusters_per_layer.tolist()

# df['e_per_layer'] = e_per_layer.tolist()
# sampled 

num_points_sampled = samples[:, 0] * 5000

visible_energy_sampled = samples[:, 1] * 2.5

cog_x_sampled = samples[:, 2] * 25

cog_y_sampled = samples[:, 3] * 15 + 15

cog_z_sampled = samples[:, 4] * 20 + 40

clusters_per_layer_sampled = samples[:, 5:35]

e_per_layer_sampled = samples[:, 35:]



# truth

E_true = E_true * 100

num_points = num_points * 5000

visible_energy = visible_energy * 2.5

cog_x = cog_x * 25

cog_y = cog_y * 15 + 15

cog_z = cog_z * 20 + 40





# clip cluster and energies per layer to [0,1]

clusters_per_layer_sampled = np.clip(clusters_per_layer_sampled, 0, 1)

e_per_layer_sampled = np.clip(e_per_layer_sampled, 0, 1)
print(cog_x.min(), cog_x.max())

print(cog_y.min() - 15, cog_y.max() - 15)

print(cog_z.min() - 40 , cog_z.max() - 40)
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
fig = plt.figure(figsize=(12,6))

gs = gridspec.GridSpec(1,2)



ax = fig.add_subplot(gs[0])

h = plt.hist(cog_y.flatten(), bins=100, color='lightgrey')

h2 = plt.hist(cog_y_sampled.flatten(), bins=h[1], histtype='step', lw=2, color='tab:orange')

plt.xlim(5,30)



# same but in log scale

ax = fig.add_subplot(gs[1])

h = plt.hist(cog_y.flatten(), bins=100, color='lightgrey')

h2 = plt.hist(cog_y_sampled.flatten(), bins=h[1], histtype='step', lw=2, color='tab:orange')

plt.yscale('log')

plt.ylim(1, 1e6)

plt.xlim(5,30)



plt.suptitle('center of gravity y')

plt.show()
fig = plt.figure(figsize=(12,6))

gs = gridspec.GridSpec(1,2)



ax = fig.add_subplot(gs[0])

h = plt.hist(cog_z.flatten(), bins=100, color='lightgrey')

h2 = plt.hist(cog_z_sampled.flatten(), bins=h[1], histtype='step', lw=2, color='tab:orange')

plt.xlim(25,60)



# same but in log scale

ax = fig.add_subplot(gs[1])

h = plt.hist(cog_z.flatten(), bins=100, color='lightgrey')

h2 = plt.hist(cog_z_sampled.flatten(), bins=h[1], histtype='step', lw=2, color='tab:orange')

plt.yscale('log')

plt.ylim(1, 1e6)

plt.xlim(25,60)



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




