import os
import re
import time

import k_diffusion as K
import torch

# from configs import Configs
from pointcloud.configs import Configs
from pointcloud.data import dataset
from pointcloud.models.diffusion import Diffusion
from pointcloud.utils import misc, training
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader


def _name_from_dataset_path(dataset_path):
    """Derive a run name from the dataset's parent folder, e.g. a folder
    'cc3input_hdbscan_ms3_mcs10' yields the name 'hdbscan_ms3_mcs10'."""
    folder_name = os.path.basename(os.path.dirname(dataset_path))
    match = re.match(r"cc3input_(.+)", folder_name)
    return match.group(1) if match else folder_name


# Train, validate and test
def train(batch, it, **setup):
    cfg = setup["cfg"]
    # Load data
    x = batch["event"][0].float().to(cfg.device)  # B, N, 4
    e = batch["energy"][0].float().to(cfg.device)  # B, 1
    p = batch["p_norm_local"][0].float().to(cfg.device)  # B, 3
    # layer_num = batch['layer_num'][0] # B, N
    # Reset grad and model state
    optimizer = setup["optimizer"]
    scheduler = setup["scheduler"]
    optimizer.zero_grad()
    ema_sched = setup["ema_sched"]
    model = setup["model"]
    model_ema = setup["model_ema"]
    model.train()

    # Forward
    if cfg.norm_cond:
        # e = e / 100 * 2 -1   # assumse max incident energy: 100 GeV
        e = e / 127  # same as shower flow model
    cond_feats = torch.cat([e, p], -1)  # B, 2

    noise = torch.randn_like(x)  # noise for forward diffusion
    sample_density = setup["sample_density"]
    sigma = sample_density([x.shape[0]], device=x.device)  # time steps

    experiment = setup["experiment"]
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
        if experiment is not None:
            experiment.log_metric("train/loss", loss, it)
            experiment.log_metric("train/lr", optimizer.param_groups[0]["lr"], it)
            experiment.log_metric("train/grad_norm", orig_grad_norm, it)
            experiment.log_metric("train/ema_decay", ema_decay, it)


def main(cfg=Configs()):
    misc.seed_all(seed=cfg.seed)
    start_time = time.localtime()

    cfg.name = _name_from_dataset_path(cfg.dataset_path) + "_"

    # Logging: when resuming, reuse the folder of the run being resumed
    # (same checkpoint dir, same Comet experiment) instead of starting a new one.
    if cfg.resume_path:
        log_dir = os.path.join(cfg.logdir, os.path.dirname(cfg.resume_path))
    else:
        log_dir = misc.get_new_log_dir(
            cfg.logdir,
            prefix=cfg.name,
            postfix="_" + cfg.tag if cfg.tag is not None else "",
            start_time=start_time,
        )
    ckpt_mgr = misc.CheckpointManager(log_dir)

    comet_key_file = os.path.join(log_dir, "comet_experiment_key.txt")
    if cfg.log_comet:
        from comet_ml import Experiment, ExistingExperiment

        with open("comet_api_key.txt", "r") as file:
            key = file.read().strip()

        if cfg.resume_path and os.path.isfile(comet_key_file):
            with open(comet_key_file, "r") as file:
                experiment_key = file.read().strip()
            experiment = ExistingExperiment(
                api_key=key,
                experiment_key=experiment_key,
                project_name=cfg.comet_project,
            )
            print(f"Resuming Comet experiment {experiment_key}")
        else:
            experiment = Experiment(
                project_name=cfg.comet_project,
                auto_metric_logging=False,
                api_key=key,
            )
            experiment.log_parameters(cfg.__dict__)
            experiment.set_name(os.path.basename(log_dir))

            # Log the code
            experiment.log_code()

            with open(comet_key_file, "w") as file:
                file.write(experiment.get_key())
    else:
        experiment = None

    # Datasets and loaders
    if not cfg.quantized_pos:
        print(
            "Warning, as we use angular datasets, the assumption is that data is fuzzed on disk"
            " config.quantized_pos will be ignored"
        )

    train_dset = dataset.PointCloudAngular(file_path=cfg.dataset_path, bs=cfg.train_bs)
    # the first time it creates the dataloader it creates on the fly the metrics to normalize.
    # so first, use a huge batch size and a single worker to properly save the metrics
    dataloader = DataLoader(train_dset, batch_size=10000, num_workers=0, shuffle=False)
    dataloader = DataLoader(
        train_dset, batch_size=1, num_workers=cfg.workers, shuffle=cfg.shuffle
    )

    # Model
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Diffusion(cfg).to(cfg.device)
    model_ema = Diffusion(cfg).to(cfg.device)

    resume_ckpt = None
    start_it = 1
    if cfg.resume_path:
        resume_ckpt_file = os.path.join(cfg.logdir, cfg.resume_path)
        print(f"Resuming training from checkpoint: {resume_ckpt_file}")
        resume_ckpt = torch.load(
            resume_ckpt_file, map_location=torch.device(cfg.device), weights_only=False
        )
        model.load_state_dict(resume_ckpt["state_dict"])
        # checkpoint files are named "ckpt_<score>_<step>.pt" (see misc.CheckpointManager.save)
        _, _, step_part = os.path.basename(cfg.resume_path).split("_")
        start_it = int(step_part.split(".")[0])
        print(f"Resuming at iteration {start_it}")

    # initiate EMA (exponential moving average) model
    if resume_ckpt is not None:
        model_ema.load_state_dict(resume_ckpt["others"]["model_ema"])
    else:
        model_ema.load_state_dict(model.state_dict())
    model_ema.eval().requires_grad_(False)
    assert cfg.ema_type == "inverse"
    ema_sched = K.utils.EMAWarmup(power=cfg.ema_power, max_value=cfg.ema_max_value)
    if resume_ckpt is not None:
        ema_sched.load_state_dict(resume_ckpt["others"]["ema_sched"])

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
    if resume_ckpt is not None:
        optimizer.load_state_dict(resume_ckpt["others"]["optimizer"])
        scheduler.load_state_dict(resume_ckpt["others"]["scheduler"])

    setup = {
        "model": model,
        "model_ema": model_ema,
        "ema_sched": ema_sched,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "sample_density": sample_density,
        "cfg": cfg,
        "experiment": experiment,
    }

    print("device: ", cfg.device)

    # Main loop
    print("Start training...")

    stop = False
    it = start_it
    start_time = time.time()
    while not stop:
        for batch in dataloader:
            it += 1
            train(batch, it, **setup)
            if it % cfg.val_freq == 0 or it == cfg.max_iters:
                opt_states = {
                    "model_ema": model_ema.state_dict(),  # save the EMA model
                    "ema_sched": ema_sched.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }
                ckpt_mgr.save(model, cfg, 0, others=opt_states, step=it)
                if cfg.log_comet:
                    for img_name, comet_name in [
                        (f"iter_{it}_1d_hist.png", "1d_hist"),
                        (f"iter_{it}_energy_hist.png", "energy_hist"),
                    ]:
                        img_path = os.path.join(log_dir, img_name)
                        if os.path.isfile(img_path):
                            experiment.log_image(img_path, name=comet_name)
            if it >= cfg.max_iters:
                stop = True
                break
    print("training done in %.2f seconds" % (time.time() - start_time))


if __name__ == "__main__":
    main()
