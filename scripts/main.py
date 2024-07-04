# in the public repo this is training.py
# this trains the teacher model
import torch
from torch.nn.utils import clip_grad_norm_

import k_diffusion
import time

from pointcloud.utils.misc import seed_all
from pointcloud.utils.training import (
    get_comet_experiment,
    get_ckp_mgr,
    get_dataloader,
    get_sample_density,
    get_optimiser_schedular,
    get_pretrained,
)
from pointcloud.configs import Configs

from pointcloud.models.vae_flow import VAEFlow
from pointcloud.models.allCond_epicVAE_nflow_PointDiff import (
    AllCond_epicVAE_nFlow_PointDiff,
)
from pointcloud.models.epicVAE_nflows_kDiffusion import epicVAE_nFlow_kDiffusion
from pointcloud.models.wish import Wish, accumulate_and_load_wish


def get_model(config):
    if config.model_name == "flow":
        model = VAEFlow(config).to(config.device)
        model_ema = None
    elif config.model_name == "AllCond_epicVAE_nFlow_PointDiff":
        model = AllCond_epicVAE_nFlow_PointDiff(config).to(config.device)
        model_ema = AllCond_epicVAE_nFlow_PointDiff(config).to(config.device)
    elif config.model_name == "epicVAE_nFlow_kDiffusion":
        model = epicVAE_nFlow_kDiffusion(config).to(config.device)
        model_ema = epicVAE_nFlow_kDiffusion(config).to(config.device)
    elif config.model_name == "wish":
        model, _ = accumulate_and_load_wish(config)
        model_ema = Wish(config)
    else:
        raise NotImplementedError(
            f"Model {config.model_name} not implemented, known models: "
            "flow, AllCond_epicVAE_nFlow_PointDiff, epicVAE_nFlow_kDiffusion, wish"
        )
    if model_ema is not None:
        # initiate EMA (exponential moving average) model
        model_ema.load_state_dict(model.state_dict())
        model_ema.eval().requires_grad_(False)
        assert config.ema_type == "inverse"
        ema_sched = k_diffusion.utils.EMAWarmup(
            power=config.ema_power, max_value=config.ema_max_value
        )
    else:
        ema_sched = None
    return model, model_ema, ema_sched


def main(config=Configs()):
    # Prepare logging, models, optimizers, schedulers, dataloaders from configs
    seed_all(seed=config.seed)
    start_time = time.localtime()
    experiment = get_comet_experiment(config, start_time)
    ckpt_mgr = get_ckp_mgr(config, start_time)
    dataloader = get_dataloader(config)
    model, model_ema, ema_sched = get_model(config)
    sample_density = get_sample_density(config)
    optimizer, scheduler, optimizer_flow, scheduler_flow = get_optimiser_schedular(
        config, model
    )

    #loading a pretrained model, if the path is provided
    model = get_pretrained(config, model)

    # set variable for printing to avoid errors due to models that don't use them
    if not hasattr(config, "kl_weight"):
        config.kl_weight = "unset"

    # Train, validate and test
    def train(batch, it):
        # Load data
        x = batch["event"][0].float().to(config.device)  # B, N, 4
        e = batch["energy"][0].float().to(config.device)  # B, 1
        n = batch["points"][0].float().to(config.device)  # B, 1

        print(f"Shape of x: {x.shape}, shape of e: {e.shape}, shape of n: {n.shape}")
        # Reset grad and model state
        optimizer.zero_grad()
        if (
            config.model_name == "AllCond_epicVAE_nFlow_PointDiff"
            or config.model_name == "epicVAE_nFlow_kDiffusion"
        ):
            if config.latent_dim > 0:
                optimizer_flow.zero_grad()
        model.train()

        loss = None
        loss_flow = None

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

        elif config.model_name == "wish":
            loss = model.get_loss(batch, experiment, it)
            loss.backward()
            orig_grad_norm = clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            # Update EMA model
            ema_decay = ema_sched.get_value()
            k_diffusion.utils.ema_update(model, model_ema, ema_decay)
            ema_sched.step()

        if it % config.log_iter == 0:
            print(
                f"[Train] Iter {it:04d} | Loss {loss.item():.6f} "
                + f"|Grad {orig_grad_norm} | KLWeight {config.kl_weight} "
                + f"| EMAdecay {ema_decay}"
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
        return loss, loss_flow

    # Main loop
    print("Start training...")

    stop = False
    # for reasons I totaly don't understand this must be 1, not 0????
    it = 1
    start_time = time.time()
    while not stop:
        for batch in dataloader:
            it += 1
            loss, loss_flow = train(batch, it)
            if loss is not None:
                loss = loss.item()
            if it % config.val_freq == 0 or it == config.max_iters:
                if config.model_name == "flow":
                    opt_states = {
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    }
                elif config.model_name in [
                    "AllCond_epicVAE_nFlow_PointDiff",
                    "epicVAE_nFlow_kDiffusion",
                    "wish",
                ]:
                    opt_states = {
                        "model_ema": model_ema.state_dict(),  # save the EMA model
                        "ema_sched": ema_sched.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        # 'optimizer_flow': optimizer_flow.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        # 'scheduler_flow': scheduler_flow.state_dict(),
                    }
                    if loss_flow is not None:
                        opt_states["loss_flow"] = loss_flow.item()
                ckpt_mgr.save(model, config, loss, others=opt_states, step=it)
            if it >= config.max_iters:
                stop = True
                break
    ckpt_mgr.save(model, config, loss, others=opt_states, step=it)
    print("training done in %.2f seconds" % (time.time() - start_time))


if __name__ == "__main__":
    from pointcloud.config_varients import caloclouds_3

    main(caloclouds_3.Configs())
