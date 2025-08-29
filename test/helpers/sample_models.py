import torch

from pointcloud.models.shower_flow import compile_HybridTanH_model


def write_fake_flow_model(config, file_path):
    """
    The flow models are too large to be stored in the repo, so we write a fake one.
    """
    flow, distribution, transforms = compile_HybridTanH_model(
        num_blocks=10,
        num_inputs=65,
        num_cond_inputs=1,
        device=config.device,
    )
    torch.save({"model": flow.state_dict()}, file_path)

