from comet_ml import Experiment

import time
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from pointcloud.utils import misc, training
from pointcloud.data import dataset
from pointcloud.models.diffusion import Diffusion

# from configs import Configs
from pointcloud.configs import Configs

import k_diffusion as K

cfg = Configs()
misc.seed_all(seed=cfg.seed)
start_time = time.localtime()


if cfg.log_comet:
    with open("comet_api_key.txt", "r") as file:
        key = file.read()
    experiment = Experiment(
        project_name=cfg.comet_project,
        auto_metric_logging=False,
        api_key=key,
    )
    experiment.log_parameters(cfg.__dict__)
    experiment.set_name(cfg.name + time.strftime("%Y_%m_%d__%H_%M_%S", start_time))

    # Log the code
    experiment.log_code()

# Logging
log_dir = misc.get_new_log_dir(
    cfg.logdir,
    prefix=cfg.name,
    postfix="_" + cfg.tag if cfg.tag is not None else "",
    start_time=start_time,
)
ckpt_mgr = misc.CheckpointManager(log_dir)

# Datasets and loaders
if not cfg.quantized_pos:
    print("Warning, as we use angular datasets, the assumption is that data is fuzzed on disk"
          " config.quantized_pos will be ignored")

train_dset = dataset.PointCloudAngular(file_path=cfg.dataset_path, bs=cfg.train_bs)
dataloader = DataLoader(
    train_dset, batch_size=1, num_workers=cfg.workers, shuffle=cfg.shuffle
)
val_dset = []

# Model
model = Diffusion(cfg).to(cfg.device)
model_ema = Diffusion(cfg).to(cfg.device)
# checkpoint = torch.load(cfg.logdir+cfg.model_path,
#                       map_location=torch.device(cfg.device))
# model.load_state_dict(checkpoint['state_dict'])

# initiate EMA (exponential moving average) model
model_ema.load_state_dict(model.state_dict())
model_ema.eval().requires_grad_(False)
assert cfg.ema_type == "inverse"
ema_sched = K.utils.EMAWarmup(power=cfg.ema_power, max_value=cfg.ema_max_value)

# Sigma (time step) distibution --> lognormal distribution, so minimum value is 0
sample_density = K.config.make_sample_density(cfg.__dict__["model"])

# Optimizer and scheduler
if cfg.optimizer == "Adam":
    # Consistency Model was trained with Rectified Adam,
    # in k-diffusion AdamW is used, in EDM normal Adam
    optimizer = torch.optim.Adam(
        [
            {"params": model.diffusion.parameters()},
        ],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
elif cfg.optimizer == "RAdam":
    optimizer = torch.optim.RAdam(
        [
            {"params": model.diffusion.parameters()},
        ],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
else:
    raise NotImplementedError("Optimizer not implemented")

scheduler = training.get_linear_scheduler(
    optimizer,
    start_epoch=cfg.sched_start_epoch,
    end_epoch=cfg.sched_end_epoch,
    start_lr=cfg.lr,
    end_lr=cfg.end_lr,
)


# Train, validate and test
def train(batch, it):
    # Load data
    x = batch["event"][0].float().to(cfg.device)  # B, N, 4
    e = batch["energy"][0].float().to(cfg.device)  # B, 1
    p = batch["p_norm_local"][0].float().to(cfg.device)  # B, 3
    # layer_num = batch['layer_num'][0] # B, N
    # Reset grad and model state
    optimizer.zero_grad()
    model.train()

    # Forward
    if cfg.norm_cond:
        # e = e / 100 * 2 -1   # assumse max incident energy: 100 GeV
        e = e / 127  # same as shower flow model
    cond_feats = torch.cat([e, p], -1)  # B, 2

    noise = torch.randn_like(x)  # noise for forward diffusion
    sigma = sample_density([x.shape[0]], device=x.device)  # time steps

    if cfg.log_comet:
        loss, loss_flow = model.get_loss(
            x,
            noise,
            sigma,
            cond_feats,
            kl_weight=cfg.kl_weight,
            writer=experiment,
            it=it,
            kld_min=cfg.kld_min,
        )
    else:
        loss, loss_flow = model.get_loss(
            x,
            noise,
            sigma,
            cond_feats,
            kl_weight=cfg.kl_weight,
            writer=None,
            it=it,
            kld_min=cfg.kld_min,
        )

    # Backward and optimize
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
    optimizer.step()
    scheduler.step()

    # Update EMA model
    ema_decay = ema_sched.get_value()
    K.utils.ema_update(model, model_ema, ema_decay)
    ema_sched.step()

    if it % cfg.log_iter == 0:
        print(
            "[Train] Iter %04d | Loss %.6f | Grad %.4f | KLWeight %.4f | EMAdecay %.4f"
            % (it, loss.item(), orig_grad_norm, cfg.kl_weight, ema_decay)
        )
        if cfg.log_comet:
            experiment.log_metric("train/loss", loss, it)
            experiment.log_metric("train/lr", optimizer.param_groups[0]["lr"], it)
            experiment.log_metric("train/grad_norm", orig_grad_norm, it)
            experiment.log_metric("train/ema_decay", ema_decay, it)


# Main loop
print("Start training...")

stop = False
it = 1
start_time = time.time()
while not stop:
    for batch in dataloader:
        it += 1
        train(batch, it)
        if it % cfg.val_freq == 0 or it == cfg.max_iters:
            opt_states = {
                "model_ema": model_ema.state_dict(),  # save the EMA model
                "ema_sched": ema_sched.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            ckpt_mgr.save(model, cfg, 0, others=opt_states, step=it)
            if cfg.log_comet:
                experiment.log_image(
                    log_dir + f"/iter_{it}_1d_hist.png", name="1d_hist"
                )
                experiment.log_image(
                    log_dir + f"/iter_{it}_energy_hist.png", name="energy_hist"
                )
        if it >= cfg.max_iters:
            stop = True
            break
print("training done in %.2f seconds" % (time.time() - start_time))
