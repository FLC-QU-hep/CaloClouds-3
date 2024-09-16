from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))
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

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pointcloud.data.dataset import PointCloudDataset

from pointcloud.models.common import reparameterize_gaussian
from pointcloud.models.flow import add_spectral_norm, spectral_norm_power_iteration
from pointcloud.configs import Configs
from tqdm import tqdm


cfg = Configs()

print(cfg.__dict__)

seed_all(seed = cfg.seed)
# from pointcloud.models.allCond_epicVAE_nflow_PointDiff import AllCond_epicVAE_nFlow_PointDiff
# from pointcloud.models.epicVAE_nflows_kDiffusion import epicVAE_nFlow_kDiffusion

import pointcloud.models.epicVAE_nflows_kDiffusion as mdls
import pointcloud.models.allCond_epicVAE_nflow_PointDiff as mdls2
importlib.reload(mdls)
importlib.reload(mdls2)


# cfg.sched_mode = 'quardatic'
# cfg.num_steps = 100
# cfg.residual = True
# cfg.latent_dim = 256
# model = mdls2.AllCond_epicVAE_nFlow_PointDiff(cfg).to(cfg.device)
# checkpoint = torch.load('/beegfs/desy/user/akorol/logs/point-cloud/AllCond_epicVAE_nFlow_PointDiff_100s_MSE_loss_smired_possitions_quardatic2023_04_06__16_34_39/ckpt_0.000000_837000.pt') # quadratic
# model.load_state_dict(checkpoint['state_dict'])




# cfg.model['sigma_data'] = 0.5
# cfg.residual = False
# cfg.dropout_rate = 0.0
# cfg.latent_dim = 256
# checkpoint = torch.load(cfg.logdir + '/' + 'kCaloClouds_2023_05_24__14_54_09/ckpt_0.000000_500000.pt')
# checkpoint = torch.load(cfg.logdir + '/' + 'kCaloClouds_2023_05_31__17_57_11/ckpt_0.000000_1700000.pt')

# cfg.model['sigma_data'] = 0.08
# checkpoint = torch.load(cfg.logdir + '/' + 'kCaloClouds_2023_06_01__13_28_09/ckpt_0.000000_340000.pt')  # too wide cog_Y, too low hit_E

# RAdam optimizer - sigma_data=0.5, residual=False, lr=2e-3, dropout_rate=0.1
# cfg.model['sigma_data'] = 0.5
# cfg.residual = False
# cfg.model['dropout_rate'] = 0.1
# checkpoint = torch.load(cfg.logdir + '/' + 'kCaloClouds_2023_06_02__16_43_14/ckpt_0.000000_258000.pt')

# # RAdam optimizer, 2M iterations, scheduler from 300k-2M, EMApower 0.6667 - sigma_data=0.25
# cfg.model['sigma_data'] = 0.25
# cfg.residual = False
# cfg.model['dropout_rate'] = 0.0
# checkpoint = torch.load(cfg.logdir + '/' + 'kCaloClouds_2023_06_02__15_55_11/ckpt_0.000000_1000000.pt')

# # RAdam optimizer - sigma_data=0.5, residual=False, lr=1e-4, dropout_rate=0.0
# cfg.model['sigma_data'] = 0.5
# cfg.residual = False
# cfg.model['dropout_rate'] = 0.0
# checkpoint = torch.load(cfg.logdir + '/' + 'kCaloClouds_2023_06_02__16_40_41/ckpt_0.000000_1000000.pt')

# # RAdam optimizer - sigma_data=0.5, residual=True, lr=2e-3, dropout_rate=0.0
# cfg.model['sigma_data'] = 0.5
# cfg.residual = True
# cfg.model['dropout_rate'] = 0.0
# checkpoint = torch.load(cfg.logdir + '/' + 'kCaloClouds_2023_06_02__16_36_04/ckpt_0.000000_920000.pt')

# # RAdam optimizer - sigma_data=0.5, residual=False, lr=2e-3, dropout_rate=0.1
# cfg.model['sigma_data'] = 0.5
# cfg.residual = False
# cfg.model['dropout_rate'] = 0.1
# checkpoint = torch.load(cfg.logdir + '/' + 'kCaloClouds_2023_06_03__10_24_31/ckpt_0.000000_737000.pt')

# # "sigma_data" : [0.08, 0.35, 0.08, 0.5]
# cfg.model['sigma_data'] = [0.08, 0.35, 0.08, 0.5]
# checkpoint = torch.load(cfg.logdir + '/' + 'kCaloClouds_2023_06_05__17_27_15/ckpt_0.000000_300000.pt')

# # "sigma_data = 1
# cfg.model['sigma_data'] = 1.
# checkpoint = torch.load(cfg.logdir + '/' + 'kCaloClouds_2023_06_05__17_28_38/ckpt_0.000000_280000.pt')

# # ""sigma_data = 0.1
# cfg.model['sigma_data'] = 0.1
# cfg.dropout_rate = 0.0
# checkpoint = torch.load(cfg.logdir + '/' + 'kCaloClouds_2023_06_06__13_34_50/ckpt_0.000000_375000.pt')

# # "sigma_data = 0.5, dropout=0.05
# cfg.model['sigma_data'] = 0.5
# cfg.dropout_rate = 0.05
# checkpoint = torch.load(cfg.logdir + '/' + 'kCaloClouds_2023_06_06__14_21_41/ckpt_0.000000_320000.pt')

# # "sigma_data = 0.5, dropout=0.025
# cfg.model['sigma_data'] = 0.5
# cfg.dropout_rate = 0.025
# cfg.latent_dim = 256
# checkpoint = torch.load(cfg.logdir + '/' + 'kCaloClouds_2023_06_07__14_36_04/ckpt_0.000000_410000.pt')

# # "sigma_data = 0.5, dropout=0.01
# cfg.model['sigma_data'] = 0.5
# cfg.dropout_rate = 0.01
# cfg.latent_dim = 256
# checkpoint = torch.load(cfg.logdir + '/' + 'kCaloClouds_2023_06_07__14_35_01/ckpt_0.000000_375000.pt')

# # "sigma_data = 0.5, dropout=0.0, latent=32
# cfg.model['sigma_data'] = 0.5
# cfg.dropout_rate = 0.0
# cfg.latent_dim = 32
# checkpoint = torch.load(cfg.logdir + '/' + 'kCaloClouds_2023_06_07__17_36_22/ckpt_0.000000_410000.pt')

# # "sigma_data = 0.5, dropout=0.0, latent=8
# cfg.model['sigma_data'] = 0.5
# cfg.dropout_rate = 0.0
# cfg.latent_dim = 8
# checkpoint = torch.load(cfg.logdir + '/' + 'kCaloClouds_2023_06_09__14_59_24/ckpt_0.000000_500000.pt')

# # "sigma_data = 0.5, dropout=0.0, latent=2
# cfg.model['sigma_data'] = 0.5
# cfg.dropout_rate = 0.0
# cfg.latent_dim = 2
# checkpoint = torch.load(cfg.logdir + '/' + 'kCaloClouds_2023_06_09__19_23_03/ckpt_0.000000_500000.pt')

# baseline with latent_dim = 32, max_iter 500k, kld_weight=1e-5
cfg.model['sigma_data'] = 0.5
cfg.dropout_rate = 0.0
cfg.latent_dim = 32
checkpoint = torch.load(cfg.logdir + '/' + 'kCaloClouds_2023_06_15__14_54_11/ckpt_0.000000_410000.pt')

# # baseline with latent_dim = 32, max_iter 500k, kld_weight=1e-6
# cfg.model['sigma_data'] = 0.5
# cfg.dropout_rate = 0.0
# cfg.latent_dim = 32
# checkpoint = torch.load(cfg.logdir + '/' + 'kCaloClouds_2023_06_15__16_12_53/ckpt_0.000000_410000.pt')


model = mdls.epicVAE_nFlow_kDiffusion(cfg).to(cfg.device)
model.load_state_dict(checkpoint['others']['model_ema'])
# model.load_state_dict(checkpoint['state_dict'])

#  load model with torch load with name "model_ema


# checkpoint = torch.load('/beegfs/desy/user/akorol/logs/point-cloud/AllCond_epicVAE_nFlow_PointDiff_100s2023_03_29__14_39_04/ckpt_0.000000_570000.pt') #worst cog x
# checkpoint = torch.load('/beegfs/desy/user/akorol/logs/point-cloud/AllCond_epicVAE_nFlow_PointDiff_100s2023_03_29__14_39_04/ckpt_0.000000_748000.pt') #best cog x

# checkpoint = torch.load('/beegfs/desy/user/akorol/logs/point-cloud/AllCond_epicVAE_nFlow_PointDiff_100s_MSE_loss_smired_possitions_sigmoid2023_04_06__16_35_47/ckpt_0.000000_849000.pt') # sigmoid 
# checkpoint = torch.load('/beegfs/desy/user/akorol/logs/point-cloud/AllCond_epicVAE_nFlow_PointDiff_100s_MSE_loss_smired_possitions_quardatic2023_04_06__16_34_39/ckpt_0.000000_837000.pt') # quadratic



model.eval()

print('model loaded')
# TODO update dataloader such that not multiple events are taken again

cfg.val_bs = 32
cfg.workers = 1

# dataset_path =  '/beegfs/desy/user/akorol/data/calo-clouds/hdf5/high_granular_grid/validation/50GeV_x36_grid_regular_2k_Z4.hdf5'

train_dset = PointCloudDataset(
    file_path=cfg.dataset_path,
    bs=cfg.val_bs,
    quantized_pos=cfg.quantized_pos
)

dataloader = DataLoader(
    train_dset,
    batch_size=1,
    num_workers=cfg.workers,
    shuffle=cfg.shuffle
)
# generate from data loop
max_events = 2000

encoded_z_list, flow_z_list = [], []
mu_list, logvar_list = [], []

# max_iters = int(len(dataloader) / cfg.val_bs)
max_iters = int(max_events / cfg.val_bs + 1)
for _ in tqdm(range(max_iters)):
    batch = next(iter(dataloader))
    x = batch['event'][0].float().to(cfg.device)  # B,N,d
    e = batch['energy'][0].float().to(cfg.device)  # B,1
    n = batch['points'][0].float().to(cfg.device)  # B,1
    # conditioning feature vector --> scale featuers to [-1,1] and concat
    if cfg.norm_cond:
        e = e / 100 * 2 -1   # max incident energy: 100 GeV
        n = n / cfg.max_points * 2  - 1
    cond_feats = torch.cat([e,n], -1)  # B,2

    with torch.no_grad():
        z_mu, z_logvar = model.encoder(x, cond_feats)
        encoded_z = reparameterize_gaussian(mean=z_mu, logvar=z_logvar)  # (B, F)
        
        # w = torch.randn([cfg.val_bs, cfg.latent_dim], device=cfg.device)
        # flow_z = model_flow(w, cond_feats, reverse=True).view(cfg.val_bs, -1)
        flow_z = model.flow.sample(context=cond_feats, num_samples=1).view(cfg.val_bs, cfg.latent_dim)

    encoded_z_list.append(encoded_z.cpu().numpy())
    flow_z_list.append(flow_z.cpu().numpy())
    mu_list.append(z_mu.cpu().numpy())
    logvar_list.append(z_logvar.cpu().numpy())

encoded_z = np.vstack(encoded_z_list)[:max_events]
flow_z = np.vstack(flow_z_list)[:max_events]
mu = np.vstack(mu_list)[:max_events]
logvar = np.vstack(logvar_list)[:max_events]
print(encoded_z.shape, flow_z.shape)
print(encoded_z.min(), flow_z.min())
print(encoded_z.max(), flow_z.max())

# calculate KLD per latent dimension
kdlloss = KLDloss()
kld = [kdlloss(torch.from_numpy(mu[:,i]), torch.from_numpy(logvar[:,i])) for i in range(cfg.latent_dim)]
kld = np.array(kld)

# sort kld from high to low
sortmask = np.argsort(kld)[::-1]
kld = kld[sortmask]

print('KLD sorted: ', kld)
print('KLD sum:', kld.sum())

encoded_z = encoded_z[:,sortmask]
flow_z = flow_z[:,sortmask]
mu = mu[:,sortmask]
logvar = logvar[:,sortmask]

fig = plt.figure(figsize=(30, 30), facecolor='none', dpi=150)
gs = GridSpec(9,8)

normal = np.random.normal(size=(max_events, 1))

# for i in tqdm(range(encoded_z.shape[1])):
for i in tqdm(range(32)):

    ax = fig.add_subplot(gs[i])
    h0 = ax.hist(encoded_z[:,i], bins=50, histtype='step', label='encoded')
    h1 = ax.hist(flow_z[:,i], bins=h0[1], histtype='step', label='flow')
    h2 = ax.hist(normal, bins=h0[1], histtype='stepfilled', label='normal', alpha=0.1, color='k')
    ax.set_xlabel(str(i))

    if i == 0: 
        ax.legend(loc='best', fontsize=14)

    ax.set_yscale('log')

plt.show()
print(mu.mean(0))

print(logvar.mean(0))

