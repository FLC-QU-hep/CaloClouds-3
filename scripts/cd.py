# in the public repo this is distillation.py
# this trains the student model
from comet_ml import Experiment

import os
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import time

from pointcloud.data.dataset import PointCloudDataset, PointCloudDatasetGH
from pointcloud.data.conditioning import get_cond_feats, normalise_cond_feats
from pointcloud.utils.misc import seed_all, get_new_log_dir, CheckpointManager
from pointcloud.utils.training import get_comet_experiment, get_ckp_mgr, get_dataloader
from pointcloud.models.epicVAE_nflows_kDiffusion import epicVAE_nFlow_kDiffusion
from pointcloud.configs import Configs


def get_models(config):
    model = epicVAE_nFlow_kDiffusion(config, distillation=True).to(config.device)
    model_ema_target = epicVAE_nFlow_kDiffusion(config, distillation=True).to(
        config.device
    )
    model_teacher = epicVAE_nFlow_kDiffusion(config, distillation=False).to(
        config.device
    )

    # load model
    checkpoint = torch.load(os.path.join(config.logdir, config.model_path), weights_only=False)
    if config.use_ema_trainer:
        model.load_state_dict(checkpoint["others"]["model_ema"])
        model_ema_target.load_state_dict(checkpoint["others"]["model_ema"])
        model_teacher.load_state_dict(checkpoint["others"]["model_ema"])
    else:
        model.load_state_dict(checkpoint["state_dict"])
        model_ema_target.load_state_dict(checkpoint["state_dict"])
        model_teacher.load_state_dict(checkpoint["state_dict"])
    print("Model loaded from: ", config.logdir + "/" + config.model_path)
    return model, model_ema_target, model_teacher


def main(config=Configs()):
    seed_all(seed=config.seed)
    start_time = time.localtime()

    # Comet online logging
    experiment = get_comet_experiment(config, start_time)
    ckpt_mgr = get_ckp_mgr(config, start_time)

    # Datasets and loaders
    dataloader = get_dataloader(config)

    # Model
    model, model_ema_target, model_teacher = get_models(config)

    if config.cm_random_init:
        print(
            "randomly initializing diffusion parameters"
            + "for online and ema (target) model"
        )
        random_model = epicVAE_nFlow_kDiffusion(config, distillation=True).to(
            config.device
        )
        model.diffusion.load_state_dict(random_model.diffusion.state_dict())
        model_ema_target.diffusion.load_state_dict(random_model.diffusion.state_dict())
        del random_model

    # set model status
    model.diffusion.requires_grad_(
        True
    )  # student ("online") model which is actually trained
    model.diffusion.train()
    if config.latent_dim > 0:
        model.encoder.requires_grad_(False)  # encoder is not trained
        model.encoder.eval()
    model_ema_target.requires_grad_(
        False
    )  # target model for sampling from consistency model,
    # updated as EMA of student model
    model_ema_target.train()  # traget model needs to be in same state as online model,
    # but does not require gradients
    model_teacher.requires_grad_(
        False
    )  # teacher model used as score function in ODE solver
    model_teacher.eval()

    # Optimizer
    if config.optimizer == "Adam":
        # Consistency Model was trained with Rectified Adam,
        # in k-diffusion AdamW is used, in EDM normal Adam
        optimizer = torch.optim.Adam(
            [
                {"params": model.diffusion.parameters()},
            ],
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "RAdam":
        # Consistency Model was trained with Rectified Adam,
        # in k-diffusion AdamW is used, in EDM normal Adam
        optimizer = torch.optim.RAdam(
            [
                {"params": model.diffusion.parameters()},
            ],
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    else:
        raise NotImplementedError("Optimizer not implemented")
    print(
        "optimizer used: ", config.optimizer, "only diffusion parameters are optimized"
    )
    print("no learning rate scheduler implemented")

    # Train
    def train(batch, it):

        # Load data
        x = batch["event"][0].float().to(config.device)  # B, N, 4
        # Reset grad
        optimizer.zero_grad()
        model.zero_grad()

        cond_feats = get_cond_feats(config, batch, "diffusion")
        cond_feats = normalise_cond_feats(config, cond_feats, "diffusion")
        cond_feats = torch.tensor(cond_feats).to(config.device).float()


        # noise = torch.randn_like(x)    # noise for forward diffusion
        # sigma = sample_density([x.shape[0]], device=x.device)  # time steps

        loss = model.get_cd_loss(x, cond_feats, model_teacher, model_ema_target, config)

        # Backward and optimize
        loss.backward()
        orig_grad_norm = clip_grad_norm_(
            model.diffusion.parameters(), config.max_grad_norm
        )
        optimizer.step()

        # Update EMA target model
        # (necessary for CD, not the same as an EMA decay of the online model itself)
        # update only diffusion parameters
        mu = config.start_ema
        for targ, src in zip(
            model_ema_target.diffusion.parameters(), model.diffusion.parameters()
        ):
            targ.detach().mul_(mu).add_(src, alpha=1 - mu)

        # TODO also add EMA model of online model with lower decay rate, i.e. 0.9999
        # (might perfrom better than target model or last online model for sampling?)

        # Logging
        if it % config.log_iter == 0:
            print(
                "[Train] Iter %04d | Loss %.6f | Grad %.4f | KLWeight %.4f"
                % (it, loss.item(), orig_grad_norm, config.kl_weight)
            )
            if config.log_comet:
                experiment.log_metric("train/loss", loss, it)
                experiment.log_metric("train/kl_weight", config.kl_weight, it)
                experiment.log_metric("train/lr", optimizer.param_groups[0]["lr"], it)
                experiment.log_metric("train/grad_norm", orig_grad_norm, it)

    # training loop
    # Main loop
    print("Start training...")

    it = 1
    start_time = time.time()
    st = time.time()
    while it <= config.max_iters:
        for batch in dataloader:
            it += 1
            train(batch, it)
            if it % config.log_iter == 0:
                print(
                    "Time for %d iterations: %.2f" % (config.log_iter, time.time() - st)
                )
                st = time.time()
            if it % config.val_freq == 0 or it == config.max_iters:
                opt_states = {
                    # save the EMA model
                    "model_ema": model_ema_target.state_dict(),
                    # save the teacher model
                    "model_teacher": model_teacher.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                ckpt_mgr.save(model, config, 0, others=opt_states, step=it)

    print("training done in %.2f seconds" % (time.time() - start_time))


if __name__ == "__main__":
    main()
