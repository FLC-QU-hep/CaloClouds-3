from comet_ml import Experiment

import sys
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
# from tqdm.auto import tqdm

from utils.dataset import *
from utils.misc import *
# from utils.data import *
# from models.vae_gaussian import *
from models.vae_flow import *
from models.flow import add_spectral_norm, spectral_norm_power_iteration
from models.allCond_epicVAE_nflow_PointDiff import AllCond_epicVAE_nFlow_PointDiff
from models.epicVAE_nflows_kDiffusion import epicVAE_nFlow_kDiffusion
from configs import Configs

import k_diffusion as K


cfg = Configs()
seed_all(seed = cfg.seed)
start_time = time.localtime()

# Comet online logging
if cfg.log_comet:
    experiment = Experiment(
        # api_key=key,
        project_name=cfg.comet_project, auto_metric_logging=False,
        # workspace="akorol",
    )
    experiment.log_parameters(cfg.__dict__)
    experiment.set_name(cfg.name+time.strftime('%Y_%m_%d__%H_%M_%S', start_time))

# Logging
log_dir = get_new_log_dir(cfg.logdir, prefix=cfg.name, postfix='_' + cfg.tag if cfg.tag is not None else '', start_time=start_time)
ckpt_mgr = CheckpointManager(log_dir)

# Datasets and loaders
if cfg.dataset == 'x36_grid' or cfg.dataset ==  'clustered':
    train_dset = PointCloudDataset(
        file_path=cfg.dataset_path,
        bs=cfg.train_bs,
        quantized_pos=cfg.quantized_pos
    )
elif cfg.dataset == 'gettig_high':
    train_dset = PointCloudDatasetGH(
        file_path=cfg.dataset_path,
        bs=cfg.train_bs,
        quantized_pos=cfg.quantized_pos
    )
dataloader = DataLoader(
    train_dset,
    batch_size=1,
    num_workers=cfg.workers,
    shuffle=cfg.shuffle
)

# Model
model = epicVAE_nFlow_kDiffusion(cfg).to(cfg.device)
model_teacher = epicVAE_nFlow_kDiffusion(cfg).to(cfg.device)
model_ema_target = epicVAE_nFlow_kDiffusion(cfg).to(cfg.device)

# load model
checkpoint = torch.load(cfg.logdir + '/' + cfg.model_path)    # EDM BASELINE first training
model.load_state_dict(checkpoint['others']['model_ema'])
model_teacher.load_state_dict(checkpoint['others']['model_ema'])
model_ema_target.load_state_dict(checkpoint['others']['model_ema'])
print('Model loaded from: ', cfg.logdir + '/' + cfg.model_path)

# set model status
model.train()    # student ("online") model which is actually trained
model_teacher.eval()  # teacher model used as score function in ODE solver
model_ema_target.eval() # target model for sampling from consistency model, updated as EMA of student model

# Optimizer
if cfg.optimizer == 'Adam':
    optimizer = torch.optim.Adam(   # Consistency Model was trained with Rectified Adam, in k-diffusion AdamW is used, in EDM normal Adam
            [
            {'params': model.diffusion.parameters()},
            ], 
            lr=cfg.lr,  
            weight_decay=cfg.weight_decay
        )
elif cfg.optimizer == 'RAdam':
    optimizer = torch.optim.RAdam(   # Consistency Model was trained with Rectified Adam, in k-diffusion AdamW is used, in EDM normal Adam
            [
            {'params': model.diffusion.parameters()},
            ], 
            lr=cfg.lr,  
            weight_decay=cfg.weight_decay
        )
else: 
    raise NotImplementedError('Optimizer not implemented')
print('optimizer used: ', cfg.optimizer, 'only diffusion parameters are optimized')
print('no learning rate scheduler implemented')

# get time step boundaries
sigmas = K.sampling.get_sigmas_karras(cfg.num_steps, cfg.sigma_min, cfg.sigma_max, rho=7., device=cfg.device)

print('sigmas: ', sigmas)

breakpoint()
