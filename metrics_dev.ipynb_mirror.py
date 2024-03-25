%load_ext autoreload

%autoreload 2



import math

import argparse

import h5py

import importlib

import numpy as np

import torch

import time

import sys

from torch.utils.data import DataLoader

from torch.nn.utils import clip_grad_norm_



from models.vae_flow import *

from models.flow import add_spectral_norm, spectral_norm_power_iteration

from configs import Configs

from utils.plotting import get_plots, get_features

from utils.detector_map import get_projections, create_map

from tqdm import tqdm



import k_diffusion as K



import utils.metrics as metrics

import utils.plotting as plotting



from utils.metadata import Metadata





cfg = Configs()

metadata = Metadata(cfg)



print(cfg.__dict__)
importlib.reload(plotting)

# path = '/beegfs/desy/user/akorol/data/calo-clouds/hdf5/high_granular_grid/validation/10-90GeV_x36_grid_regular.hdf5'

# path = '/beegfs/desy/user/akorol/data/calo-clouds/hdf5/high_granular_grid/validation/50GeV_x36_grid_regular_2k_Z4.hdf5'

path = '/beegfs/desy/user/akorol/data/calo-clouds/hdf5/all_steps/validation/photon-showers_10-90GeV_A90_Zpos4.slcio.hdf5'

real_showers, real_energy = generate_for_metrics.get_g4_data(path)

dataset_class = dataset.dataset_class_from_config(Configs())

real_showers[:, :, -1] = real_showers[:, :, -1] * dataset_class.energy_scale   # GeV to MeV

print(real_showers.shape)
MAP = create_map(configs=cfg)

real_events = get_projections(real_showers[:], MAP, configs=cfg)

len(real_events)

len(real_events[0])
importlib.reload(plotting)

plt_config = plotting.PltConfigs()



st = time.time()

dict = plotting.get_features(plt_config, MAP, metadata.half_cell_size, real_events[:])

print(time.time()-st)
print(dict.keys())



print(dict['binned_layer_e'].shape)
dict['e_radial_lists'][3].shape
dict['occ'][3]
dict['occ_layers'].shape
dict['e_layers_distibution'].shape
# # caluclate percentile edges for layer wise occupancy
importlib.reload(metrics)



percentile_edges_layer_occupancy = metrics.percentile_edges_layer_occupancy(dict['occ_layers'])



print(percentile_edges_layer_occupancy)

print(len(percentile_edges_layer_occupancy))
# # binned occupancy
occ_layers = dict['occ_layers']



binned_layer_occ = metrics.binned_layer_occupcany(occ_layers, bin_edges = percentile_edges_layer_occupancy)



print(binned_layer_occ.shape)
# # binned e layers
e_layers_distibution = dict['e_layers_distibution']



binned_layer_e = metrics.binned_layer_energy(e_layers_distibution, bin_edges = percentile_edges_layer_occupancy)



print(binned_layer_e.shape)
# # percentile edges for raidal occpuancy
occ_radial = dict['e_radial_lists'][0]



importlib.reload(metrics)

percentile_edges_radial_occupancy = metrics.percentile_edges_radial_occupancy(occ_radial)



print(percentile_edges_radial_occupancy)
# # binned occ radial (for testing)
importlib.reload(metrics)



binned_radial_occ_sum = metrics.binned_radial_occupancy_sum(occ_radial, bin_edges = percentile_edges_radial_occupancy)



print(binned_radial_occ_sum.shape)

print(binned_radial_occ_sum / binned_radial_occ_sum.sum())
# # binned e radial distributions
importlib.reload(metrics)



binned_radial_e = metrics.binned_radial_energy(dict['e_radial_lists'], bin_edges = percentile_edges_radial_occupancy)



print(binned_radial_e.shape)

