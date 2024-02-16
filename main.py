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


def get_comet_experiment(config):
    if config.log_comet:
        with open("comet_api_key.txt", "r") as file:
            key = file.read()
        experiment = Experiment(
            api_key=key,
            project_name=config.comet_project,
            auto_metric_logging=False,
            workspace=config.comet_workspace,
        )
        experiment.log_parameters(config.__dict__)
        experiment.set_name(
            config.name + time.strftime("%Y_%m_%d__%H_%M_%S", start_time)
        )
    else:
        experiment = None
    return experiment


def get_ckp_mgr(config, start_time):
    log_dir = get_new_log_dir(
        config.logdir,
        prefix=config.name,
        postfix="_" + config.tag if config.tag is not None else "",
        start_time=start_time,
    )
    ckpt_mgr = CheckpointManager(log_dir)
    return ckpt_mgr


def get_dataloader(config):
    if config.dataset == "x36_grid" or config.dataset == "clustered":
        train_dset = PointCloudDataset(
            file_path=config.dataset_path,
            bs=config.train_bs,
            quantized_pos=config.quantized_pos,
        )
    elif config.dataset == "gettig_high":
        train_dset = PointCloudDatasetGH(
            file_path=config.dataset_path,
            bs=config.train_bs,
            quantized_pos=config.quantized_pos,
        )
    dataloader = DataLoader(
        train_dset, batch_size=1, num_workers=config.workers, shuffle=config.shuffle
    )
    return dataloader


val_dset = []


def get_model(config):
    if config.model_name == "flow":
        model = VAEFlow(config).to(config.device)
    elif config.model_name == "AllCond_epicVAE_nFlow_PointDiff":
        model = AllCond_epicVAE_nFlow_PointDiff(config).to(config.device)
        model_ema = AllCond_epicVAE_nFlow_PointDiff(config).to(config.device)
    elif config.model_name == "epicVAE_nFlow_kDiffusion":
        model = epicVAE_nFlow_kDiffusion(config).to(config.device)
        model_ema = epicVAE_nFlow_kDiffusion(config).to(config.device)
    # initiate EMA (exponential moving average) model
    model_ema.load_state_dict(model.state_dict())
    model_ema.eval().requires_grad_(False)
    assert config.ema_type == "inverse"
    ema_sched = k_diffusion.utils.EMAWarmup(
        power=config.ema_power, max_value=config.ema_max_value
    )
    return model, model_ema, ema_sched


def get_sample_density(config):
    sample_density = k_diffusion.config.make_sample_density(config.__dict__["model"])
    return sample_density


def get_optimiser_schedular(config, model):
    if config.model_name == "flow":
        optimizer_flow = None
        scheduler_flow = None
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        scheduler = get_linear_scheduler(
            optimizer,
            start_epoch=config.sched_start_epoch,
            end_epoch=config.sched_end_epoch,
            start_lr=config.lr,
            end_lr=config.end_lr,
        )

    elif (
        config.model_name == "AllCond_epicVAE_nFlow_PointDiff"
        or config.model_name == "epicVAE_nFlow_kDiffusion"
    ):
        if config.optimizer == "Adam":
            if config.latent_dim > 0:
                optimizer = torch.optim.Adam(  # Consistency Model was trained with Rectified Adam, in k-diffusion AdamW is used, in EDM normal Adam
                    [
                        {"params": model.encoder.parameters()},
                        {"params": model.diffusion.parameters()},
                    ],
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                )
                optimizer_flow = torch.optim.Adam(
                    [
                        {"params": model.flow.parameters()},
                    ],
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                )
            else:
                optimizer = torch.optim.Adam(  # Consistency Model was trained with Rectified Adam, in k-diffusion AdamW is used, in EDM normal Adam
                    [
                        {"params": model.diffusion.parameters()},
                    ],
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                )
                optimizer_flow = None
        elif config.optimizer == "RAdam":
            if config.latent_dim > 0:
                optimizer = torch.optim.RAdam(  # Consistency Model was trained with Rectified Adam, in k-diffusion AdamW is used, in EDM normal Adam
                    [
                        {"params": model.encoder.parameters()},
                        {"params": model.diffusion.parameters()},
                    ],
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                )
                optimizer_flow = torch.optim.RAdam(
                    [
                        {"params": model.flow.parameters()},
                    ],
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                )
            else:
                optimizer = torch.optim.RAdam(  # Consistency Model was trained with Rectified Adam, in k-diffusion AdamW is used, in EDM normal Adam
                    [
                        {"params": model.diffusion.parameters()},
                    ],
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                )
                optimizer_flow = None
        else:
            raise NotImplementedError("Optimizer not implemented")

        scheduler = get_linear_scheduler(
            optimizer,
            start_epoch=config.sched_start_epoch,
            end_epoch=config.sched_end_epoch,
            start_lr=config.lr,
            end_lr=config.end_lr,
        )
        if config.latent_dim > 0:
            scheduler_flow = get_linear_scheduler(
                optimizer_flow,
                start_epoch=config.sched_start_epoch,
                end_epoch=config.sched_end_epoch,
                start_lr=config.lr,
                end_lr=config.end_lr,
            )
        else:
            scheduler_flow = None
    return optimizer, scheduler, optimizer_flow, scheduler_flow


def main(config=Configs()):
    # Prepare logging, models, optimizers, schedulers, dataloaders from configs
    seed_all(seed=config.seed)
    start_time = time.localtime()
    experiment = get_comet_experiment(config)
    ckpt_mgr = get_ckp_mgr(config, start_time)
    dataloader = get_dataloader(config)
    model, model_ema, ema_sched = get_model(config)
    sample_density = get_sample_density(config)
    optimizer, scheduler, optimizer_flow, scheduler_flow = get_optimiser_schedular(
        config, model
    )

    # Train, validate and test
    def train(batch, it):
        # Load data
        x = batch["event"][0].float().to(config.device)  # B, N, 4
        e = batch["energy"][0].float().to(config.device)  # B, 1
        n = batch["points"][0].float().to(config.device)  # B, 1
        # Reset grad and model state
        optimizer.zero_grad()
        if (
            config.model_name == "AllCond_epicVAE_nFlow_PointDiff"
            or config.model_name == "epicVAE_nFlow_kDiffusion"
        ):
            if config.latent_dim > 0:
                optimizer_flow.zero_grad()
        model.train()

        # Forward
        if config.model_name == "flow":
            loss = model.get_loss(
                x, kl_weight=config.kl_weight, writer=experiment, it=it
            )
            # Backward and optimize
            loss.backward()
            orig_grad_norm = clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()

        elif config.model_name == "AllCond_epicVAE_nFlow_PointDiff":
            if config.norm_cond:
                e = e / 100 * 2 - 1  # assumse max incident energy: 100 GeV
                n = n / config.max_points * 2 - 1
            cond_feats = torch.cat([e, n], -1)  # B, 2
            if config.log_comet:
                loss, loss_flow = model.get_loss(
                    x,
                    cond_feats,
                    kl_weight=config.kl_weight,
                    writer=experiment,
                    it=it,
                    kld_min=config.kld_min,
                )
            else:
                loss, loss_flow = model.get_loss(
                    x,
                    cond_feats,
                    kl_weight=config.kl_weight,
                    writer=None,
                    it=it,
                    kld_min=config.kld_min,
                )

        elif config.model_name == "epicVAE_nFlow_kDiffusion":
            if config.norm_cond:
                e = e / 100 * 2 - 1  # assumse max incident energy: 100 GeV
                n = n / config.max_points * 2 - 1
            cond_feats = torch.cat([e, n], -1)  # B, 2

            noise = torch.randn_like(x)  # noise for forward diffusion
            sigma = sample_density([x.shape[0]], device=x.device)  # time steps

            if config.log_comet:
                loss, loss_flow = model.get_loss(
                    x,
                    noise,
                    sigma,
                    cond_feats,
                    kl_weight=config.kl_weight,
                    writer=experiment,
                    it=it,
                    kld_min=config.kld_min,
                )
            else:
                loss, loss_flow = model.get_loss(
                    x,
                    noise,
                    sigma,
                    cond_feats,
                    kl_weight=config.kl_weight,
                    writer=None,
                    it=it,
                    kld_min=config.kld_min,
                )

            # Backward and optimize
            loss.backward()
            if config.latent_dim > 0:
                loss_flow.backward()
            orig_grad_norm = clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            if config.latent_dim > 0:
                optimizer_flow.step()
                scheduler_flow.step()

            # Update EMA model
            ema_decay = ema_sched.get_value()
            k_diffusion.utils.ema_update(model, model_ema, ema_decay)
            ema_sched.step()

        if it % config.log_iter == 0:
            print(
                "[Train] Iter %04d | Loss %.6f | Grad %.4f | KLWeight %.4f | EMAdecay %.4f"
                % (it, loss.item(), orig_grad_norm, config.kl_weight, ema_decay)
            )
            if config.log_comet:
                experiment.log_metric("train/loss", loss, it)
                experiment.log_metric("train/kl_weight", config.kl_weight, it)
                experiment.log_metric("train/lr", optimizer.param_groups[0]["lr"], it)
                experiment.log_metric("train/grad_norm", orig_grad_norm, it)
                experiment.log_metric("train/ema_decay", ema_decay, it)
                if config.latent_dim > 0:
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
            if it % config.val_freq == 0 or it == config.max_iters:
                if config.model_name == "flow":
                    opt_states = {
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    }
                elif (
                    config.model_name == "AllCond_epicVAE_nFlow_PointDiff"
                    or config.model_name == "epicVAE_nFlow_kDiffusion"
                ):
                    opt_states = {
                        "model_ema": model_ema.state_dict(),  # save the EMA model
                        "ema_sched": ema_sched.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        # 'optimizer_flow': optimizer_flow.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        # 'scheduler_flow': scheduler_flow.state_dict(),
                    }
                ckpt_mgr.save(model, config, 0, others=opt_states, step=it)
            if it >= config.max_iters:
                stop = True
                break
    ckpt_mgr.save(model, config, 0, others=opt_states, step=it)
    print("training done in %.2f seconds" % (time.time() - start_time))


if __name__ == "__main__":
    main()
