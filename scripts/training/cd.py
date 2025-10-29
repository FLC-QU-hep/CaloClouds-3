from comet_ml import Experiment

import torch
import time
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from pointcloud.utils import misc
from pointcloud.data import dataset
from pointcloud.models.diffusion import Diffusion
from configs import Configs


def main(config=Configs()):
    misc.seed_all(seed=config.seed)
    start_time = time.localtime()

    if config.log_comet:
        with open("comet_api_key.txt", "r") as file:
            key = file.read()
        experiment = Experiment(
            project_name=config.comet_project,
            auto_metric_logging=False,
            api_key=key,
        )
        experiment.log_parameters(config.__dict__)
        experiment.set_name(
            config.name + time.strftime("%Y_%m_%d__%H_%M_%S", start_time)
        )
    else:
        experiment = None

    # Logging
    log_dir = misc.get_new_log_dir(
        config.logdir,
        prefix=config.name,
        postfix="_" + config.tag if config.tag is not None else "",
        start_time=start_time,
    )
    ckpt_mgr = misc.CheckpointManager(log_dir)

    train_dset = dataset.PointCloudAngular(
        file_path=config.dataset_path, bs=config.train_bs
    )
    dataloader = DataLoader(
        train_dset, batch_size=1, num_workers=config.workers, shuffle=config.shuffle
    )

    # Model
    model = Diffusion(config, distillation=True).to(config.device)
    model_ema_target = Diffusion(config, distillation=True).to(config.device)
    model_teacher = Diffusion(config, distillation=False).to(config.device)

    # load model
    checkpoint = torch.load(
        config.logdir + "/" + config.model_path,
        map_location=torch.device(config.device),
    )
    if config.use_ema_trainer:
        model.load_state_dict(checkpoint["others"]["model_ema"])
        model_ema_target.load_state_dict(checkpoint["others"]["model_ema"])
        model_teacher.load_state_dict(checkpoint["others"]["model_ema"])
    else:
        model.load_state_dict(checkpoint["state_dict"])
        model_ema_target.load_state_dict(checkpoint["state_dict"])
        model_teacher.load_state_dict(checkpoint["state_dict"])
    print("Model loaded from: ", config.logdir + "/" + config.model_path)

    if config.cm_random_init:
        print(
            "randomly initializing diffusion parameters for online and ema (target) model"
        )
        random_model = Diffusion(config, distillation=True).to(config.device)
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
        optimizer = torch.optim.RAdam(  # Consistency Model was trained with Rectified Adam, in k-diffusion AdamW is used, in EDM normal Adam
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

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
    #                                                        T_max=15000)

    # Train
    def train(batch, it):
        # Load data
        x = batch["event"][0].float().to(config.device)  # B, N, 4
        e = batch["energy"][0].float().to(config.device)  # B, 1
        p = batch["p_norm_local"][0].float().to(config.device)  # B, 3
        # num_points = batch['num_points'][0].max().item() # B, 1
        # Reset grad
        optimizer.zero_grad()
        model.zero_grad()

        # get condition features
        if config.norm_cond:
            # e = e / 100 * 2 -1   # assumse max incident energy: 100 GeV
            e = e / 127  # same as new SF
        cond_feats = torch.cat([e, p], -1)  # B, 2

        # teacher_showers, x_T = model_teacher.sample(cond_feats, num_points, config, retun_noise=True)
        # student_showers = model.sample(cond_feats, num_points, config, x_T=x_T)
        # loss = torch.sqrt(F.mse_loss(student_showers, teacher_showers))
        # # loss = F.huber_loss(student_showers, teacher_showers, delta=0.2)

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

        # scheduler.step()

        ## TODO also add EMA model of online model with lower decay rate, i.e. 0.9999
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
