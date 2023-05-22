from comet_ml import Experiment

import math
import argparse
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from utils.dataset import *
from utils.misc import *
# from utils.data import *
# from models.vae_gaussian import *
from models.vae_flow import *
from models.flow import add_spectral_norm, spectral_norm_power_iteration
from models.allCond_epicVAE_nflow_PointDiff import AllCond_epicVAE_nFlow_PointDiff
from configs import Configs

cfg = Configs()

# with open('comet_api_key.txt', 'r') as file:
    # key = file.read()

experiment = Experiment(
    # api_key=key,
    project_name=cfg.comet_project,
    # workspace="akorol",
)

seed_all(seed = cfg.seed)

start_time = time.localtime()

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
val_dset = []


# Model
if cfg.model == 'flow':
    model = FlowVAE(cfg).to(cfg.device)
elif cfg.model == 'AllCond_epicVAE_nFlow_PointDiff':
    model = AllCond_epicVAE_nFlow_PointDiff(cfg).to(cfg.device)


if cfg.spectral_norm:
    add_spectral_norm(model)

# Optimizer and scheduler
if cfg.model == 'flow':

    optimizer = torch.optim.Adam(model.parameters(), 
        lr=cfg.lr, 
        weight_decay=cfg.weight_decay
    )

    scheduler = get_linear_scheduler(
        optimizer,
        start_epoch=cfg.sched_start_epoch,
        end_epoch=cfg.sched_end_epoch,
        start_lr=cfg.lr,
        end_lr=cfg.end_lr
    )

elif cfg.model == 'AllCond_epicVAE_nFlow_PointDiff':

    optimizer = torch.optim.Adam(
            [
            {'params': model.encoder.parameters()}, 
            {'params': model.diffusion.parameters()},
            ], 
            lr=cfg.lr,  
            weight_decay=cfg.weight_decay
        )
    optimizer_flow = torch.optim.Adam(
        [
        {'params': model.flow.parameters()}, 
        ], 
        lr=cfg.lr,  
        weight_decay=cfg.weight_decay
    )

    scheduler = get_linear_scheduler(optimizer, start_epoch=cfg.sched_start_epoch, end_epoch=cfg.sched_end_epoch, start_lr=cfg.lr, end_lr=cfg.end_lr)
    scheduler_flow = get_linear_scheduler(optimizer_flow, start_epoch=cfg.sched_start_epoch, end_epoch=cfg.sched_end_epoch, start_lr=cfg.lr, end_lr=cfg.end_lr)


# Train, validate and test
def train(batch, it):
    # Load data
    x = batch['event'][0].float().to(cfg.device)
    e = batch['energy'][0].float().to(cfg.device)
    n = batch['points'][0].float().to(cfg.device)
    # Reset grad and model state
    optimizer.zero_grad()
    if cfg.model == 'AllCond_epicVAE_nFlow_PointDiff':
        optimizer_flow.zero_grad()
    model.train()
    if cfg.spectral_norm:
        spectral_norm_power_iteration(model, n_power_iterations=1)

    # Forward
    if cfg.model == 'flow':
        loss = model.get_loss(x, kl_weight=cfg.kl_weight, writer=experiment, it=it)
    
    # Backward and optimize
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()
        scheduler.step()

    elif cfg.model == 'AllCond_epicVAE_nFlow_PointDiff':
        if cfg.norm_cond:
            e = e / 100 * 2 -1   # max incident energy: 100 GeV
            n = n / cfg.max_points * 2  - 1
        cond_feats = torch.cat([e,n], -1) 
        loss, loss_flow = model.get_loss(x, cond_feats, kl_weight=cfg.kl_weight, writer=experiment, it=it, kld_min=cfg.kld_min)

     # Backward and optimize
        loss.backward()
        loss_flow.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()
        optimizer_flow.step()
        scheduler.step()
        scheduler_flow.step()

    if it % 10 == 0:
        print('[Train] Iter %04d | Loss %.6f | Grad %.4f | KLWeight %.4f' % (
            it, loss.item(), orig_grad_norm, cfg.kl_weight
        ))
        experiment.log_metric('train/loss', loss, it)
        experiment.log_metric('train/loss_flow', loss_flow, it)
        experiment.log_metric('train/kl_weight', cfg.kl_weight, it)
        experiment.log_metric('train/lr', optimizer.param_groups[0]['lr'], it)
        experiment.log_metric('train/lr_flow', optimizer_flow.param_groups[0]['lr'], it)
        experiment.log_metric('train/grad_norm', orig_grad_norm, it)

# Main loop
print('Start training...')

stop = False
it = 1
while not stop:
    for batch in dataloader:
        it += 1
        train(batch, it)
        if it % cfg.val_freq == 0 or it == cfg.max_iters:
            if cfg.model == 'flow':
                opt_states = {
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }
            elif cfg.model == 'AllCond_epicVAE_nFlow_PointDiff':
                opt_states = {
                    'optimizer': optimizer.state_dict(),
                    'optimizer_flow': optimizer_flow.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scheduler_flow': scheduler_flow.state_dict(),
                }
            ckpt_mgr.save(model, cfg, 0, others=opt_states, step=it)
        if it >= cfg.max_iters:
            stop = True
            break
