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



from functools import partial



import torch



from pyro.nn import AutoRegressiveNN, ConditionalAutoRegressiveNN



from pyro.distributions import constraints

from pyro.distributions.conditional import ConditionalTransformModule

from pyro.distributions.torch_transform import TransformModule

from pyro.distributions.util import copy_docs_from

from pyro.distributions.transforms import SplineCoupling

from pyro.distributions.transforms.spline import ConditionalSpline







import numpy as np

import torch

import torch.utils.data

from torch import nn, optim

from torch.nn import functional as F

from torchvision import datasets, transforms

from torchvision.utils import save_image

#from data_utils.data_loader import HDF5Dataset, MaxwellBatchLoader, MaxwellBatchLoaderFullyCond





#import ML_models.models as models

#import ML_models.functions as functions

import time

import h5py



from custom_pyro import ConditionalAffineCouplingTanH



from pyro.nn import ConditionalDenseNN, DenseNN, ConditionalAutoRegressiveNN

import pyro.distributions as dist

import pyro.distributions.transforms as T



from models.shower_flow import compile_HybridTanH_model, compile_HybridTanH_model_CaloC





device = torch.device('cuda:0')
# num_inputs: visible_energy 1 + num_points (total) 1 + e_per_layer (normalized [0, 1]) 45, num_point_per_layer (normalized [0, 1]) 45 = 92

model, distribution = compile_HybridTanH_model_CaloC(num_blocks = 10, 

                                           num_inputs=92, ### when 'condioning' on additional Esum, Nhits etc add them on as inputs rather than 

                                           num_cond_inputs=1, device=device)  # num_cond_inputs
default_params = {

    "batch_size" : 512,

    "epochs" : 500, # 138 epochs with lr 5e-5, then to 300 epochs with lr 1e-5 and to 500 with 5e-6

    "shuffle" : True,

}
kwargs = {}

params = {}

for param in default_params.keys():



    if param in kwargs.keys():

        params[param] = kwargs[param]

    else:

        params[param] = default_params[param]
path = '/beegfs/desy/user/akorol/data/calo-challange/dataset_3_xyz_tarin.hdf5'

energy = h5py.File(path, 'r')['energy'][:]

events = h5py.File(path, 'r')['events'][-93060:]



num_points = (events[:][:, -1] > 0).sum(axis=1)

visible_energy = (events[:][:, -1]).sum(axis=1)
events.shape
# path =  '/beegfs/desy/user/valentel/CaloTransfer/data/calo-challenge/dataset_3_xyz_tarin.hdf5'

path = '/beegfs/desy/user/valentel/CaloTransfer/data/calo-challenge/dataset_3_3_10-90GeV.hdf5'



# chosen_size = 99000

# percentage = 0.5



# size = int ( chosen_size * percentage)

# idx = np.random.choice(np.arange((1e5 - chosen_size), 1e5), size=size, replace=False)

# # idx = np.sort(idx).astype(int)





energy = h5py.File(path, 'r')['incident_energies'][ : ]

events = h5py.File(path, 'r')['showers'][ : ]



# num_points = (events[:][:, -1] > 0).sum(axis=1)

# visible_energy = (events[:][:, -1]).sum(axis=1)





# if shape is ( n , 40500)

# Calculate the number of non-zero data points per event

num_points = (events > 0).sum(axis=1)



# Calculate the total sum of data points per event

visible_energy = events.sum(axis=1)



print('Number of non-zero points per event:', num_points)

print('Total visible energy per event:', visible_energy)

# for validation

# visible_energy_true = (events[:][:, -1]).sum(axis=1)

# num_points_true = (events[:][:, -1] > 0).sum(axis=1)



# if shape is ( n , 40500 )

visible_energy_true = ( events > 0 ).sum(axis=1)

num_points_true = (events > 0).sum(axis=1)



energy_true = h5py.File(path, 'r')['incident_energies'][:]
plt.hist(num_points, bins=np.logspace(np.log(1e-7), np.log(num_points.max()), 200, base=np.e),)

plt.xscale('log')

plt.yscale('log')

plt.xlim(xmin = num_points.min(), xmax = num_points.max())

print('min and max points: ',num_points.min(), num_points.max())

plt.show()
import os 

e_per_layer_path = './utils/per_layer/evaluation_10-90GeV_clusters_per_layer.npy'

clusters_per_layer_path = './utils/per_layer/evaluation_10_90GeV_e_per_layer.npy'





if not os.path.exists(e_per_layer_path) and not os.path.exists(clusters_per_layer_path):

    # takes some time to run this, calculate it onces and then save as numpy

    clusters_per_layer = []

    for i in tqdm(range(45)):

        clusters_per_layer.append( ((events[:, 2, :] < i+1) & (events[:, 2, :] > i)).sum(axis=1) )



    clusters_per_layer = np.vstack(clusters_per_layer)

    clusters_per_layer = np.moveaxis(clusters_per_layer, 0, -1)



    e_per_layer = []

    for i in tqdm(range(45)):

        layer_mask = (events[:, 2, :] < i+1) & (events[:, 2, :] > i)

        e_per_layer.append( (events[:, -1, :] * layer_mask).sum(axis=1) )

        

    e_per_layer = np.vstack(e_per_layer)

    e_per_layer = np.moveaxis(e_per_layer, 0, -1)



    # save 

    np.save(e_per_layer_path, e_per_layer)

    np.save(clusters_per_layer_path, clusters_per_layer)
e_per_layer_path = './utils/per_layer/evaluation_10-90GeV_clusters_per_layer.npy'

clusters_per_layer_path = './utils/per_layer/evaluation_10_90GeV_e_per_layer.npy'





# Only perform calculations if the files do not already exist

if not os.path.exists(e_per_layer_path) and not os.path.exists(clusters_per_layer_path):

    # Assume there are 45 "layers", each containing an equal portion of the 40500 features

    num_layers = 45

    layer_width = 40500 // num_layers  # Calculate the number of features per layer

    

    # Initialize lists to store results for each layer

    clusters_per_layer = []

    e_per_layer = []



    # Iterate over each layer and calculate the required metrics

    for i in tqdm(range(num_layers)):

        # Calculate the start and end indices for each layer

        start_idx = i * layer_width

        end_idx = start_idx + layer_width



        # Calculate the number of non-zero points in each layer

        layer_data = events[:, start_idx:end_idx]

        clusters_per_layer.append(np.count_nonzero(layer_data, axis=1))



        # Calculate the sum of points in each layer

        e_per_layer.append(np.sum(layer_data, axis=1))



    # Convert results from lists to arrays and save them

    np.save(clusters_per_layer_path, np.array(clusters_per_layer).T)

    np.save(e_per_layer_path, np.array(e_per_layer).T)
# load precalculated

e_per_layer = np.load(e_per_layer_path)

clusters_per_layer = np.load(clusters_per_layer_path)

print('e per layer shape: ', e_per_layer.shape)

print('cluster per layer shape: ', clusters_per_layer.shape)
plt.plot(clusters_per_layer[300]/clusters_per_layer[300].max())

plt.show()
clusters_per_layer = clusters_per_layer / clusters_per_layer.max(axis=1).reshape(len(clusters_per_layer), 1)

e_per_layer = e_per_layer / e_per_layer.max(axis=1).reshape(len(e_per_layer), 1)
plt.hist(visible_energy)

plt.show()

plt.hist(visible_energy, bins=np.logspace(np.log(1e-7), np.log(visible_energy.max()), 200, base=np.e))

plt.xscale('log')

plt.yscale('log')

plt.xlim(xmin=visible_energy.min(), xmax=visible_energy.max() )

plt.ylim(ymax=1e4)





plt.show()
plt.figure(figsize=(6,6))

for i in range(len(clusters_per_layer)-5, len(clusters_per_layer)):

    plt.hist(np.arange(45), weights=clusters_per_layer[i, :], label=f'energy: {energy[i]}', histtype='step', bins=45)

plt.legend()

plt.show()



plt.figure(figsize=(6,6))

for i in range(len(e_per_layer)-5, len(e_per_layer)):

    plt.hist(np.arange(45), weights=e_per_layer[i, :], label=f'energy: {energy[i]}', histtype='step', bins=45)

plt.legend()

plt.show()
df = pd.DataFrame([])

df['energy'] = ( np.log(energy/energy.min()) / np.log(energy.max()/energy.min()) ).reshape(-1)

df['visible_energy'] = ( np.log(visible_energy/visible_energy.min()) / np.log(visible_energy.max()/visible_energy.min()) ).reshape(-1)

df['num_points'] = ( np.log((num_points+1)/num_points.min()) / np.log(num_points.max()/num_points.min()) ).reshape(-1)



df['clusters_per_layer'] = clusters_per_layer.tolist()

df['e_per_layer'] = e_per_layer.tolist()



df
visible_energy.min() * (np.e ** (df.visible_energy.values * np.log(visible_energy.max() / visible_energy.min())))

visible_energy
num_points.min() * (np.e ** (df.num_points.values * np.log(num_points.max() / num_points.min()))) - 1
num_points
dataset = torch.utils.data.TensorDataset(

    torch.tensor(df.energy.values), 

    torch.tensor(df.visible_energy.values),

    torch.tensor(df.num_points.values),

    torch.tensor(df.e_per_layer),

    torch.tensor(df.clusters_per_layer)

    )
train_loader = torch.utils.data.DataLoader(dataset, batch_size=params["batch_size"], shuffle=params["shuffle"], pin_memory=True)

batch = next(iter(train_loader))
# items in batch: (insident energy, visible_energy, num_points, e_per_layer, clusters_per_layer)

for item in batch:

    print(item.shape)
import comet_ml

import logging

from configs import Configs



# Set the logging level for comet_ml

logging.getLogger("comet_ml").setLevel(logging.ERROR)



cfg = Configs()



with open('./utils/comet_api_key.txt', 'r') as file:

    key = file.read()



# Log in to Comet (replace 'your-api-key' and 'your-project-name' with your actual Comet API key and project name)

comet_logger = comet_ml.Experiment(

    api_key=key,

    project_name='showerflow',

    auto_metric_logging=False,

    workspace="lorenzovalente3",

)



lr = 1e-6



model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)



if cfg.uda:

    model, distribution = compile_HybridTanH_model(num_blocks=10, 

                                           num_inputs=65, ### when 'condioning' on additional Esum, Nhits etc add them on as inputs rather than 

                                           num_cond_inputs=1, device=device)  # num_cond_inputs

    path_showerflow = '/beegfs/desy/user/valentel/6_PointCloudDiffusion/shower_flow/220714_cog_e_layer_ShowerFlow_best.pth'

    model.load_state_dict(torch.load(path_showerflow)['model'])

    print('Pre-trained model loaded!')



model.train()



torch.manual_seed(41)



losses = []

print('Start training ...')

for epoch in range(301):

    input_list = []



    for batch_idx, (energy, visible_energy, num_points, e_per_layer, clusters_per_layer) in enumerate(tqdm(train_loader)):

        E_true = energy.view(-1, 1).to(device).float()

        energy_sum = visible_energy.view(-1, 1).to(device).float()

        num_points = num_points.view(-1, 1).to(device).float()

        e_per_layer = e_per_layer.to(device).float()

        clusters_per_layer = clusters_per_layer.to(device).float()



        input_data = torch.cat((energy_sum, num_points, e_per_layer, clusters_per_layer), 1)



        optimizer.zero_grad()



        context = E_true

        if np.any(np.isnan(input_data.clone().detach().cpu().numpy())) == True:

            print('Nans in the training data!')



        nll = -distribution.condition(context).log_prob(input_data)

        loss = nll.mean()

        loss.backward()

        optimizer.step()



        distribution.clear_cache()

        input_list.append(input_data.detach().cpu().numpy())



    print(epoch, loss.item())

    losses.append(loss.item())



    # Log to COMET ML

    with open('./utils/comet_api_key.txt', 'r') as file:

        key = file.read()



    if cfg.log_comet:

        experiment = comet_ml.Experiment(

            api_key=key,

            project_name='showerflow',

            auto_metric_logging=False,

            workspace="lorenzovalente3",

        )

        experiment.log_parameters(cfg.__dict__)

        comet_logger.log_metric("loss", loss.item(), step=epoch)



        # Save model every 10 epochs

        if epoch % 10 == 0:

            torch.save(

                {'model': model.state_dict()},

                f'/beegfs/desy/user/valentel/logs/shower_flow/nonsmeared_50%/ShowerFlow_new_f_{epoch}.pth'

            )



# Close Comet logger at the end

comet_logger.end()

experiment.end()  # Assuming experiment is defined in the loop and you want to end it outside the loop

!ls /beegfs/desy/user/valentel/logs/shower_flow/smeared_50%_data
# checkpoint = torch.load('/beegfs/desy/user/akorol/logs/ShowerFlow_new_303.pth')

checkpoint = torch.load('/beegfs/desy/user/valentel/logs/shower_flow/smeared_10%/ShowerFlow_new_f_300.pth')
model.load_state_dict(checkpoint['model'])
model.eval().to(device)
low_log = np.log10(1000)  # convert to log space

high_log = np.log10(1000000)  # convert to log space

uniform_samples = np.random.uniform(low_log, high_log, len(energy))



# apply exponential function (base 10)

log_uniform_samples = np.power(10, uniform_samples)

log_uniform_samples = ( np.log(log_uniform_samples/log_uniform_samples.min()) / np.log(log_uniform_samples.max()/log_uniform_samples.min()) ).reshape(-1)

cond_E = torch.tensor(log_uniform_samples).view(len(energy), 1).to(device).float()



# energy_sum, e_per_layer, clusters_per_layer

with torch.no_grad():

    samples = distribution.condition(cond_E).sample(torch.Size([len(energy), ])).cpu().numpy() #.detach().numpy

    

# energy_sum = samples[:, 0] * 800000

# e_ins = cond_E.cpu().numpy().reshape(-1)*1000*1000
print(energy)

print(energy.shape)
samples = np.column_stack([visible_energy, num_points, e_per_layer, clusters_per_layer, energy ])

# TODO: save the samples to a file

np.save("/beegfs/desy/user/valentel/CaloTransfer/calo-challenge/utils/per_layer/10_90GeV_samples.npy", samples)



print(samples.shape)  # This prints the shape of the resulting array.
# del samples



showers = np.load("/beegfs/desy/user/valentel/CaloTransfer/calo-challenge/utils/per_layer/10_90GeV_samples.npy")

print(showers.shape)

print(showers[: , 92])

print(showers[: , -1])

print(showers[: , -2])

print(showers[: , 47:-2])
print(samples.shape)

print(samples[0])
model
print(visible_energy, num_points, e_per_layer, clusters_per_layer)

print( ' + - ' * 30 )

print(visible_energy.shape, 

      num_points.shape,

        e_per_layer.shape,

          clusters_per_layer.shape)
energy_sum = visible_energy_true.min() * (np.e ** (samples[:, 0] * np.log(visible_energy_true.max() / visible_energy_true.min())))
energy.min(), energy.max()
e_ins = cond_E.cpu().numpy().reshape(-1)

e_ins = (energy.min() + 1e-7 ) * (np.e ** (e_ins * np.log(energy.max() / (energy.min() + 1e-7))))
e_ins = cond_E.cpu().numpy().reshape(-1)

# e_ins = energy.min() * (np.e ** (e_ins * np.log(energy.max().numpy() / energy.min().numpy())))

# energy = cond_E.cpu().numpy()

e_ins = (energy.min() + 1e-7 ) * (np.e ** (e_ins * np.log(energy.max() /(energy.min() + 1e-7 ))))
plt.figure(figsize=(10,10))

# h = plt.hist(visible_energy_true/energy_true.reshape(-1), bins=100, range=(0.5, 1.5), color='lightgrey', density=1)

# plt.hist(energy_sum/e_ins, bins=h[1], histtype='step', lw=2, color='tab:orange', density=1)

h = plt.hist(visible_energy_true/energy_true.reshape(-1), bins=100, range=(0.5, 1.5), color='lightgrey')

h2 = plt.hist(energy_sum/e_ins, bins=h[1], histtype='step', lw=2, color='tab:orange')

plt.show()
# separation power

ratios = (h[0] - h2[0])**2 / (h[0] + h2[0])



S2 = ratios[~np.isnan(ratios)].sum()/2

print(S2/200000)
plt.figure(figsize=(6, 6))

plt.scatter(cond_E.cpu().numpy().reshape(-1), samples[:, 0], s=0.01, alpha=0.1)

plt.xlabel('incident energy', fontsize=15)

plt.ylabel('visible energy', fontsize=15)

# plt.xlim(-150,3900)

plt.ylim(0,1)

plt.title('Shower Flow, CaloChallenge')

plt.show()
plt.figure(figsize=(6, 6))

plt.scatter(df.energy.values, df.visible_energy.values, s=0.01, alpha=0.1)

plt.xlabel('incident energy', fontsize=15)

plt.ylabel('visible energy', fontsize=15)

# plt.xlim(-150,3900)

plt.ylim(0,1)

plt.title('Data, CaloChallenge')

plt.show()

def invers_transform_energy(energy):

    energy_min, energy_max = 944.6402, 811494.44 # min max energy in the dataset

    return energy_min * (np.e ** (energy * np.log(energy_max / energy_min)))



def invers_transform_points(n_points):

    points_min, points_max = 201, 19206 # min max number of points in the dataset

    return points_min * (np.e ** (n_points * np.log(points_max / points_min)))
plt.figure(figsize=(6, 6))

h = plt.hist(invers_transform_energy(df.visible_energy.values)/1000, bins=100, color='lightgray')

plt.hist(invers_transform_energy(samples[:, 0])/1000, bins=h[1], histtype='step', color='tab:orange', lw=1.2)



plt.yscale('log')

plt.xlabel('E_sum [GeV]', fontsize=15)

plt.ylabel('# events', fontsize=15)

plt.show()
plt.figure(figsize=(6, 6))

h = plt.hist(np.arange(45), weights=e_per_layer.mean(axis=0), bins=45, color='lightgray')

plt.hist(np.arange(45), weights=samples[:, 2:47].mean(axis=0), bins=h[1], histtype='step', lw=1.5, color='tab:orange')

plt.yscale('log')

plt.xlabel('layer', fontsize=15)

plt.ylabel('E_mean [GeV]', fontsize=15)

plt.show()
plt.figure(figsize=(6, 6))

h = plt.hist(np.arange(45), weights=clusters_per_layer.mean(axis=0), bins=45, color='lightgray')

plt.hist(np.arange(45), weights=samples[:, 47:].mean(axis=0), bins=h[1], histtype='step', lw=1.5, color='tab:orange')

plt.yscale('log')

plt.xlabel('layer', fontsize=15)

plt.ylabel('# points_mean', fontsize=15)

plt.show()
### plt.figure(figsize=(7, 6))

plt.scatter(e_per_layer.sum(axis=1)/1000, clusters_per_layer.sum(axis=1), s=0.1, c=invers_transform_energy(df.energy)/1000)

# plt.scatter(samples[:, 1:46].sum(axis=1)*40, samples[:, 46:].sum(axis=1)*600, color='tab:orange', s=0.1, alpha=0.1)

plt.xlabel('E_sum [GeV]', fontsize=15)

plt.ylabel('# points', fontsize=15)

plt.title('CaloChallenge Data')

plt.colorbar()

plt.show()

num_points_gen = invers_transform_points(samples[:, 1])

vis_energy_gen = invers_transform_energy(samples[:, 0])
plt.figure(figsize=(7, 6))

plt.scatter(vis_energy_gen/1000, num_points_gen, c=invers_transform_energy(cond_E.cpu().numpy().reshape(-1))/1000, s=0.1)

plt.xlabel('E_sum [GeV]', fontsize=15)

plt.ylabel('# points', fontsize=15)

plt.title('Shower Flow')

plt.colorbar()

plt.show()

