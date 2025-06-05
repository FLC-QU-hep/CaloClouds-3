"""
Module to test the training module
"""
import time
import torch
from helpers import config_creator

from pointcloud.utils import training
from pointcloud.utils.misc import CheckpointManager

from pointcloud.models import wish
from pointcloud.evaluation import generate


def test_get_comet_experiment():
    # We don't actually care about this,
    # it just has to not throw an error
    config = config_creator.make()
    training.get_comet_experiment(config, time.localtime())


def test_get_ckp_mgr(tmpdir):
    config = config_creator.make(my_tmpdir=tmpdir)
    ckp_mgr = training.get_ckp_mgr(config, time.localtime())
    assert isinstance(ckp_mgr, CheckpointManager)


def test_get_dataloader():
    for config_type in ["wish", "default"]:
        config = config_creator.make(config_type)
        dataloader = training.get_dataloader(config)
        assert isinstance(dataloader, torch.utils.data.DataLoader)


def test_get_sample_density():
    config = config_creator.make()
    sample_density = training.get_sample_density(config)
    sigma = sample_density([4], device=config.device)
    assert isinstance(sigma, torch.Tensor)


def test_get_optimiser_schedular():
    combinations = [
        ("wish", "Adam", 0),
        ("wish", "RAdam", 0),
        ("wish", "Adam", 1),
        ("wish", "RAdam", 1),
        ("default", "Adam", 0),
        ("default", "RAdam", 0),
        ("default", "Adam", 1),
        ("default", "RAdam", 1),
    ]

    for name, optim, latent_dim in combinations:
        config = config_creator.make(name)
        config.optimizer = optim
        config.latent_dim = latent_dim
        if name == "wish":
            model = wish.Wish(config)
        elif name == "default":
            model, _, _, _ = generate.load_diffusion_model(
                config, "cm", model_path="test/example_cm_model.pt"
            )
        (
            optimiser,
            scheduler,
            opimiser_flow,
            scheduler_flow,
        ) = training.get_optimiser_schedular(config, model)
        # check we can zero grad
        optimiser.zero_grad()
        if opimiser_flow is not None:
            opimiser_flow.zero_grad()
        # we can't step without calculating loss, but the function should be there
        assert hasattr(optimiser.step, "__call__")
        if opimiser_flow is not None:
            assert hasattr(opimiser_flow.step, "__call__")
        # the optimisers should also have param groups and state_dict
        assert optimiser.param_groups
        optimiser.state_dict()
        if opimiser_flow is not None:
            assert opimiser_flow.param_groups
            opimiser_flow.state_dict()
        # schedulars just step and have state dicts
        assert hasattr(scheduler.step, "__call__")
        scheduler.state_dict()
        if scheduler_flow is not None:
            assert hasattr(scheduler_flow.step, "__call__")
            scheduler_flow.state_dict()


def test_get_pretrained():
    for config_type in ["wish", "config_calotransf"]:
        config = config_creator.make(config_type)
        if config_type == "wish":
            model = wish.Wish(config)
        else:
            model, _, _, _ = generate.load_diffusion_model(
                config, "cm", model_path="test/example_cm_model.pt"
            )
        found = training.get_pretrained(config, model)
        assert isinstance(found, type(model))
