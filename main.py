# in the public repo this is training.py
# this trains the teacher model
from comet_ml import Experiment

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

import k_diffusion
import time

from utils.dataset import PointCloudDataset, PointCloudDatasetGH
from utils.misc import seed_all, get_new_log_dir, CheckpointManager
from models.common import get_linear_scheduler
from models.vae_flow import VAEFlow
from models.allCond_epicVAE_nflow_PointDiff import AllCond_epicVAE_nFlow_PointDiff
from models.epicVAE_nflows_kDiffusion import epicVAE_nFlow_kDiffusion
from configs import Configs


def get_comet_experiment(cfg):
    if cfg.log_comet:
        with open("comet_api_key.txt", "r") as file:
            key = file.read()
        experiment = Experiment(
            api_key=key,
            project_name=cfg.comet_project,
            auto_metric_logging=False,
            workspace=cfg.comet_workspace,
        )
        experiment.log_parameters(cfg.__dict__)
        experiment.set_name(cfg.name + time.strftime("%Y_%m_%d__%H_%M_%S", start_time))
    else:
        experiment = None
    return experiment


def get_ckp_mgr(cfg, start_time):
    log_dir = get_new_log_dir(
        cfg.logdir,
        prefix=cfg.name,
        postfix="_" + cfg.tag if cfg.tag is not None else "",
        start_time=start_time,
    )
    ckpt_mgr = CheckpointManager(log_dir)
    return ckpt_mgr


def get_dataloader(cfg):
    if cfg.dataset == "x36_grid" or cfg.dataset == "clustered":
        train_dset = PointCloudDataset(
            file_path=cfg.dataset_path, bs=cfg.train_bs, quantized_pos=cfg.quantized_pos
        )
    elif cfg.dataset == "gettig_high":
        train_dset = PointCloudDatasetGH(
            file_path=cfg.dataset_path, bs=cfg.train_bs, quantized_pos=cfg.quantized_pos
        )
    dataloader = DataLoader(
        train_dset, batch_size=1, num_workers=cfg.workers, shuffle=cfg.shuffle
    )
    return dataloader


val_dset = []


def get_model(cfg):
    if cfg.model_name == "flow":
        model = VAEFlow(cfg).to(cfg.device)
    elif cfg.model_name == "AllCond_epicVAE_nFlow_PointDiff":
        model = AllCond_epicVAE_nFlow_PointDiff(cfg).to(cfg.device)
        model_ema = AllCond_epicVAE_nFlow_PointDiff(cfg).to(cfg.device)
    elif cfg.model_name == "epicVAE_nFlow_kDiffusion":
        model = epicVAE_nFlow_kDiffusion(cfg).to(cfg.device)
        model_ema = epicVAE_nFlow_kDiffusion(cfg).to(cfg.device)
    # initiate EMA (exponential moving average) model
    model_ema.load_state_dict(model.state_dict())
    model_ema.eval().requires_grad_(False)
    assert cfg.ema_type == "inverse"
    ema_sched = k_diffusion.utils.EMAWarmup(
        power=cfg.ema_power, max_value=cfg.ema_max_value
    )
    return model, model_ema, ema_sched


def get_sample_density(cfg):
    sample_density = k_diffusion.config.make_sample_density(cfg.__dict__["model"])
    return sample_density


def get_optimiser_schedular(cfg, model):
    if cfg.model_name == "flow":
        optimizer_flow = None
        scheduler_flow = None
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = get_linear_scheduler(
            optimizer,
            start_epoch=cfg.sched_start_epoch,
            end_epoch=cfg.sched_end_epoch,
            start_lr=cfg.lr,
            end_lr=cfg.end_lr,
        )

    elif (
        cfg.model_name == "AllCond_epicVAE_nFlow_PointDiff"
        or cfg.model_name == "epicVAE_nFlow_kDiffusion"
    ):
        if cfg.optimizer == "Adam":
            if cfg.latent_dim > 0:
                optimizer = torch.optim.Adam(  # Consistency Model was trained with Rectified Adam, in k-diffusion AdamW is used, in EDM normal Adam
                    [
                        {"params": model.encoder.parameters()},
                        {"params": model.diffusion.parameters()},
                    ],
                    lr=cfg.lr,
                    weight_decay=cfg.weight_decay,
                )
                optimizer_flow = torch.optim.Adam(
                    [
                        {"params": model.flow.parameters()},
                    ],
                    lr=cfg.lr,
                    weight_decay=cfg.weight_decay,
                )
            else:
                optimizer = torch.optim.Adam(  # Consistency Model was trained with Rectified Adam, in k-diffusion AdamW is used, in EDM normal Adam
                    [
                        {"params": model.diffusion.parameters()},
                    ],
                    lr=cfg.lr,
                    weight_decay=cfg.weight_decay,
                )
                optimizer_flow = None
        elif cfg.optimizer == "RAdam":
            if cfg.latent_dim > 0:
                optimizer = torch.optim.RAdam(  # Consistency Model was trained with Rectified Adam, in k-diffusion AdamW is used, in EDM normal Adam
                    [
                        {"params": model.encoder.parameters()},
                        {"params": model.diffusion.parameters()},
                    ],
                    lr=cfg.lr,
                    weight_decay=cfg.weight_decay,
                )
                optimizer_flow = torch.optim.RAdam(
                    [
                        {"params": model.flow.parameters()},
                    ],
                    lr=cfg.lr,
                    weight_decay=cfg.weight_decay,
                )
            else:
                optimizer = torch.optim.RAdam(  # Consistency Model was trained with Rectified Adam, in k-diffusion AdamW is used, in EDM normal Adam
                    [
                        {"params": model.diffusion.parameters()},
                    ],
                    lr=cfg.lr,
                    weight_decay=cfg.weight_decay,
                )
                optimizer_flow = None
        else:
            raise NotImplementedError("Optimizer not implemented")

        scheduler = get_linear_scheduler(
            optimizer,
            start_epoch=cfg.sched_start_epoch,
            end_epoch=cfg.sched_end_epoch,
            start_lr=cfg.lr,
            end_lr=cfg.end_lr,
        )
        if cfg.latent_dim > 0:
            scheduler_flow = get_linear_scheduler(
                optimizer_flow,
                start_epoch=cfg.sched_start_epoch,
                end_epoch=cfg.sched_end_epoch,
                start_lr=cfg.lr,
                end_lr=cfg.end_lr,
            )
        else:
            scheduler_flow = None
    return optimizer, scheduler, optimizer_flow, scheduler_flow


def main(cfg=Configs()):
    # Prepare logging, models, optimizers, schedulers, dataloaders from configs
    seed_all(seed=cfg.seed)
    start_time = time.localtime()
    experiment = get_comet_experiment(cfg)
    ckpt_mgr = get_ckp_mgr(cfg, start_time)
    dataloader = get_dataloader(cfg)
    model, model_ema, ema_sched = get_model(cfg)
    sample_density = get_sample_density(cfg)
    optimizer, scheduler, optimizer_flow, scheduler_flow = get_optimiser_schedular(
        cfg, model
    )

    # Train, validate and test
    def train(batch, it):
        # Load data
        x = batch["event"][0].float().to(cfg.device)  # B, N, 4
        e = batch["energy"][0].float().to(cfg.device)  # B, 1
        n = batch["points"][0].float().to(cfg.device)  # B, 1
        # Reset grad and model state
        optimizer.zero_grad()
        if (
            cfg.model_name == "AllCond_epicVAE_nFlow_PointDiff"
            or cfg.model_name == "epicVAE_nFlow_kDiffusion"
        ):
            if cfg.latent_dim > 0:
                optimizer_flow.zero_grad()
        model.train()

        # Forward
        if cfg.model_name == "flow":
            loss = model.get_loss(x, kl_weight=cfg.kl_weight, writer=experiment, it=it)
            # Backward and optimize
            loss.backward()
            orig_grad_norm = clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()

        elif cfg.model_name == "AllCond_epicVAE_nFlow_PointDiff":
            if cfg.norm_cond:
                e = e / 100 * 2 - 1  # assumse max incident energy: 100 GeV
                n = n / cfg.max_points * 2 - 1
            cond_feats = torch.cat([e, n], -1)  # B, 2
            if cfg.log_comet:
                loss, loss_flow = model.get_loss(
                    x,
                    cond_feats,
                    kl_weight=cfg.kl_weight,
                    writer=experiment,
                    it=it,
                    kld_min=cfg.kld_min,
                )
            else:
                loss, loss_flow = model.get_loss(
                    x,
                    cond_feats,
                    kl_weight=cfg.kl_weight,
                    writer=None,
                    it=it,
                    kld_min=cfg.kld_min,
                )

        elif cfg.model_name == "epicVAE_nFlow_kDiffusion":
            if cfg.norm_cond:
                e = e / 100 * 2 - 1  # assumse max incident energy: 100 GeV
                n = n / cfg.max_points * 2 - 1
            cond_feats = torch.cat([e, n], -1)  # B, 2

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
            if cfg.latent_dim > 0:
                loss_flow.backward()
            orig_grad_norm = clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            if cfg.latent_dim > 0:
                optimizer_flow.step()
                scheduler_flow.step()

            # Update EMA model
            ema_decay = ema_sched.get_value()
            k_diffusion.utils.ema_update(model, model_ema, ema_decay)
            ema_sched.step()

        if it % cfg.log_iter == 0:
            print(
                "[Train] Iter %04d | Loss %.6f | Grad %.4f | KLWeight %.4f | EMAdecay %.4f"
                % (it, loss.item(), orig_grad_norm, cfg.kl_weight, ema_decay)
            )
            if cfg.log_comet:
                experiment.log_metric("train/loss", loss, it)
                experiment.log_metric("train/kl_weight", cfg.kl_weight, it)
                experiment.log_metric("train/lr", optimizer.param_groups[0]["lr"], it)
                experiment.log_metric("train/grad_norm", orig_grad_norm, it)
                experiment.log_metric("train/ema_decay", ema_decay, it)
                if cfg.latent_dim > 0:
                    experiment.log_metric("train/loss_flow", loss_flow, it)
                    experiment.log_metric(
                        "train/lr_flow", optimizer_flow.param_groups[0]["lr"], it
                    )

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
                if cfg.model_name == "flow":
                    opt_states = {
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    }
                elif (
                    cfg.model_name == "AllCond_epicVAE_nFlow_PointDiff"
                    or cfg.model_name == "epicVAE_nFlow_kDiffusion"
                ):
                    opt_states = {
                        "model_ema": model_ema.state_dict(),  # save the EMA model
                        "ema_sched": ema_sched.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        # 'optimizer_flow': optimizer_flow.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        # 'scheduler_flow': scheduler_flow.state_dict(),
                    }
                ckpt_mgr.save(model, cfg, 0, others=opt_states, step=it)
            if it >= cfg.max_iters:
                stop = True
                break
    print("training done in %.2f seconds" % (time.time() - start_time))


if __name__ == "__main__":
    main()
