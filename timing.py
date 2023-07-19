import sys
sys.path.append('/beegfs/desy/user/akorol/projects/point-cloud-diffusion/')

from tqdm import tqdm

import torch
import time
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from models.vae_flow import *
from models.flow import add_spectral_norm, spectral_norm_power_iteration
from configs import Configs

from utils.plotting import get_projections, get_plots, MAP, offset, layer_bottom_pos, cell_thickness, Xmax, Xmin, Zmax, Zmin

import utils.gen_utils as gen_utils

import k_diffusion as K

from models.shower_flow import compile_HybridTanH_model
import models.epicVAE_nflows_kDiffusion as mdls
import models.allCond_epicVAE_nflow_PointDiff as mdls2


########################
### PARAMS #############
########################

caloclouds = 'edm'   # 'ddpm, 'edm', 'cm'

cfg = Configs()
cfg.device = 'cpu'  # 'cuda' or 'cpu'
#use single thread
torch.set_num_threads(1)

# min and max energy of the generated events
min_e = 10
max_e = 100

num = 2000 # total number of generated events

bs = 1 # batch size   # optimized: bs= 16(cm), 64(edm), 64(ddpm) for GPU, bs=512 for CPU (multi-threaded), bs=1 for CPU (single-threaded)

iterations = 25 # number of iterations for timing

cfg.num_steps = 13
cfg.sampler = 'heun'   # default 'heun'
cfg.s_churn =  0.0     # stochasticity, default 0.0  (if s_churn more than num_steps, it will be clamped to max value)
cfg.s_noise = 1.0    # default 1.0   # noise added when s_churn > 0
cfg.sigma_max = 80.0 #  5.3152e+00  # default 80.0
cfg.sigma_min = 0.002   # default 0.002
cfg.rho = 7. # default 7.0

# CaloClouds
# coef_real = np.array([ 2.57988645e-09, -2.94056522e-05,  3.42194568e-01,  5.34968378e+01])
# coef_fake = np.array([ 3.85057207e-09, -4.16463897e-05,  4.19800713e-01,  5.82246858e+01])

# kCaloClouds_2023_05_24__14_54_09_heun18
# coef_real = np.array([ 2.39735048e-09, -2.69842295e-05,  2.96136986e-01,  4.89770787e+01])
# coef_fake = np.array([ 4.45753201e-09, -4.26483492e-05,  4.03632976e-01,  6.31063427e+01])

# # kCaloClouds_2023_05_24__14_54_09_heun13
# coef_real = np.array([ 2.39735048e-09, -2.69842295e-05,  2.96136986e-01,  4.89770787e+01])
# coef_fake = np.array([ 5.72940149e-09, -4.76120436e-05,  4.37720799e-01,  5.97962496e+01])

n_scaling = True


########################
########################
########################


def main(cfg, min_e, max_e, num, bs, iterations):

    num_blocks = 10
    flow, distribution = compile_HybridTanH_model(num_blocks, 
                                            num_inputs=65, ### when 'condioning' on additional Esum, Nhits etc add them on as inputs rather than 
                                            num_cond_inputs=1, device=cfg.device)  # num_cond_inputs

    #checkpoint = torch.load('/beegfs/desy/user/akorol/chekpoints/ECFlow/EFlow+CFlow_138.pth')
    checkpoint = torch.load('/beegfs/desy/user/buhmae/6_PointCloudDiffusion/shower_flow/220714_cog_e_layer_ShowerFlow_best.pth', map_location=torch.device(cfg.device))   # trained about 350 epochs
    flow.load_state_dict(checkpoint['model'])
    flow.eval().to(cfg.device)


    if caloclouds == 'ddpm':
        kdiffusion=False   # EDM vs DDPM diffusion
        # caloclouds baseline
        cfg.sched_mode = 'quardatic'
        cfg.num_steps = 100
        cfg.residual = True
        cfg.latent_dim = 256
        model = mdls2.AllCond_epicVAE_nFlow_PointDiff(cfg).to(cfg.device)
        checkpoint = torch.load('/beegfs/desy/user/akorol/logs/point-cloud/AllCond_epicVAE_nFlow_PointDiff_100s_MSE_loss_smired_possitions_quardatic2023_04_06__16_34_39/ckpt_0.000000_837000.pt', map_location=torch.device(cfg.device)) # quadratic
        model.load_state_dict(checkpoint['state_dict'])
        coef_real = np.array([ 2.50244046e-09, -2.82685784e-05,  3.15731003e-01,  5.08123555e+01])
        coef_fake = np.array([ 3.72975819e-09, -3.87472364e-05,  3.80314204e-01,  5.30334567e+01])

    elif caloclouds == 'edm':
        kdiffusion=True   # EDM vs DDPM diffusion
        # caloclouds EDM
        # # # baseline with lat_dim = 0, max_iter 10M, lr=1e-4 fixed, dropout_rate=0.0, ema_power=2/3 (long training)            USING THIS TRAINING
        cfg.dropout_rate = 0.0
        cfg.latent_dim = 0
        checkpoint = torch.load(cfg.logdir + '/' + 'kCaloClouds_2023_06_29__23_08_31/ckpt_0.000000_2000000.pt', map_location=torch.device(cfg.device))    # max 5200000
        model = mdls.epicVAE_nFlow_kDiffusion(cfg).to(cfg.device)
        model.load_state_dict(checkpoint['others']['model_ema'])
        coef_real = np.array([ 2.50244046e-09, -2.82685784e-05,  3.15731003e-01,  5.08123555e+01])
        coef_fake = np.array([ 5.08021809e-09, -5.26101363e-05,  4.74959822e-01,  5.34314449e+01])

    elif caloclouds == 'cm':
        kdiffusion=True   # EDM vs DDPM diffusion
        # condsistency model
        # long baseline with lat_dim = 0, max_iter 1M, lr=1e-4 fixed, num_steps=18, bs=256, simga_max=80, epoch=2M, EMA
        cfg.dropout_rate = 0.0
        cfg.latent_dim = 0
        checkpoint = torch.load(cfg.logdir + '/' + 'CD_2023_07_07__16_32_09/ckpt_0.000000_1000000.pt', map_location=torch.device(cfg.device))   # max 1200000
        model = mdls.epicVAE_nFlow_kDiffusion(cfg, distillation = True).to(cfg.device)
        model.load_state_dict(checkpoint['others']['model_ema'])
        coef_real = np.array([ 2.50244046e-09, -2.82685784e-05,  3.15731003e-01,  5.08123555e+01])
        coef_fake = np.array([ 4.29894066e-09, -4.61132724e-05,  4.40193379e-01,  6.23006887e+01])

    else:
        raise ValueError('caloclouds must be one of: ddpm, edm, cm')

    model.eval()

    times_per_shower = []
    for _ in range(iterations):

        s_t = time.time()
        fake_showers = gen_utils.gen_showers_batch(model, distribution, min_e, max_e, num, bs, kdiffusion=kdiffusion, config=cfg, coef_real=coef_real, coef_fake=coef_fake, n_scaling=n_scaling)
        t = time.time() - s_t
        print(fake_showers.shape)
        print('total time [seconds]: ', t)
        print('time per shower [ms]: ', t / num * 1000)
        times_per_shower.append(t / num)

    print('mean time per shower [ms]: ', np.mean(times_per_shower) * 1000)
    print('std time per shower [ms]: ', np.std(times_per_shower) * 1000)




if __name__ == '__main__':
    main(cfg, min_e, max_e, num, bs, iterations)