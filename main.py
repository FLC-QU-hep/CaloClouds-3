from comet_ml import Experiment

import math
import argparse
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from utils.dataset import *
from utils.misc import *
from utils.data import *
from models.vae_gaussian import *
from models.vae_flow import *
from models.flow import add_spectral_norm, spectral_norm_power_iteration
from configs import Configs

cfg = Configs()

with open('comet_api_key.txt', 'r') as file:
    key = file.read()

experiment = Experiment(
    api_key=key,
    project_name="point-cloud",
    workspace="akorol",
)

experiment.log_parameters(cfg.__dict__)


seed_all()

# Logging
log_dir = get_new_log_dir(cfg.logdir, prefix='GEN_', postfix='_' + cfg.tag if cfg.tag is not None else '')
ckpt_mgr = CheckpointManager(log_dir)

# Datasets and loaders
train_dset = PointCloudDataset(
    file_path=cfg.dataset_path,
    bs=cfg.train_bs
)
dataloader = DataLoader(
    train_dset,
    batch_size=1,
    num_workers=cfg.workers,
    shuffle=cfg.shuffle
)
val_dset = []


# Model
model = FlowVAE(cfg).to(cfg.device)

if cfg.spectral_norm:
    add_spectral_norm(model)

# Optimizer and scheduler
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

# Train, validate and test
def train(batch, it):
    # Load data
    x = batch['event'][0].float().to(cfg.device)
    # Reset grad and model state
    optimizer.zero_grad()
    model.train()
    if cfg.spectral_norm:
        spectral_norm_power_iteration(model, n_power_iterations=1)

    # Forward
    kl_weight = cfg.kl_weight
    loss = model.get_loss(x, kl_weight=kl_weight, writer=experiment, it=it)

    # Backward and optimize
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
    optimizer.step()
    scheduler.step()

    if it % 10 == 0:
        print('[Train] Iter %04d | Loss %.6f | Grad %.4f | KLWeight %.4f' % (
            it, loss.item(), orig_grad_norm, kl_weight
        ))
    experiment.log_metric('train/loss', loss, it)
    experiment.log_metric('train/kl_weight', kl_weight, it)
    experiment.log_metric('train/lr', optimizer.param_groups[0]['lr'], it)
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
            opt_states = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            ckpt_mgr.save(model, cfg, 0, others=opt_states, step=it)
        if it >= cfg.max_iters:
            stop = True
            break
