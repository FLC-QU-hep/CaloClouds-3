"""
Utility functions extracted from the training scripts
"""
from comet_ml import Experiment

import torch
from torch.utils.data import DataLoader

import k_diffusion
import time


from ..data.dataset import dataset_class_from_config
from .misc import get_new_log_dir, CheckpointManager
from ..models.common import get_linear_scheduler


def get_comet_experiment(config, start_time):
    if config.log_comet:
        with open("comet_api_key.txt", "r") as file:
            key = file.read().strip()
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
    dataset_class = dataset_class_from_config(config)
    print(f"Using dataset class: {dataset_class}")
    train_dset = dataset_class(
        file_path=config.dataset_path,
        bs=config.train_bs,
        quantized_pos=config.quantized_pos,
        n_files=config.n_dataset_files,
    )
    dataloader = DataLoader(
        train_dset, batch_size=1, num_workers=config.workers, shuffle=config.shuffle
    )
    return dataloader


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

    elif config.model_name in [
        "AllCond_epicVAE_nFlow_PointDiff",
        "epicVAE_nFlow_kDiffusion",
        "wish",
    ]:
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
            elif config.model_name == "wish":
                optimizer = torch.optim.RAdam(  # Consistency Model was trained with Rectified Adam, in k-diffusion AdamW is used, in EDM normal Adam
                    [{"params": model.parameters()}],
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                )
                optimizer_flow = None
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
