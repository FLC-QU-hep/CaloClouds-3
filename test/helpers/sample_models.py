import torch

from . import sample_accumulator

from pointcloud.models.shower_flow import compile_HybridTanH_model
from pointcloud.models.wish import Wish
from pointcloud.utils.stats_accumulator import HighLevelStats


def write_fake_flow_model(config, file_path):
    """
    The flow models are too large to be stored in the repo, so we write a fake one.
    """
    flow, distribution = compile_HybridTanH_model(
        num_blocks=10,
        num_inputs=65,
        num_cond_inputs=1,
        device=config.device,
    )
    torch.save({"model": flow.state_dict()}, file_path)


def make_fake_wish_model(configs):
    """
    Make a wish model with somehwat functional settings.
    """
    configs.fit_attempts = 2
    wish_model = Wish(configs)
    acc = sample_accumulator.make(add_varients=True)
    hls = HighLevelStats(acc, wish_model.poly_degree)
    wish_model.set_from_stats(hls)
    return wish_model


def write_fake_wish_model(config, file_path):
    model = make_fake_wish_model(config)
    model.save(file_path)
