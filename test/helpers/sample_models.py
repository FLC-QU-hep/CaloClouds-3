import torch

from pointcloud.models.shower_flow import versions_dict
from pointcloud.data.conditioning import get_cond_dim
from pointcloud.utils.showerflow_utils import get_input_mask


def write_fake_flow_model(config, file_path):
    """
    The flow models are too large to be stored in the repo, so we write a fake one.
    """
    factory = versions_dict[config.shower_flow_version]
    flow, distribution, transforms = factory(
        num_blocks=config.shower_flow_num_blocks,
        num_inputs=get_input_mask(config).sum(),
        num_cond_inputs=get_cond_dim(config, "showerflow"),
        device=config.device,
    )
    torch.save({"model": flow.state_dict()}, file_path)

