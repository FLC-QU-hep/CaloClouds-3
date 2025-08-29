import numpy as np
import torch

from pointcloud.configs import Configs
from pointcloud.models.diffusion import Diffusion
from pointcloud.models import shower_flow
from pointcloud.utils import showerflow_utils
from pointcloud.data.conditioning import get_cond_features_names, get_cond_dim


def load_flow_model(
    config: Configs,
    model_path: str = "/beegfs/desy/user/buhmae/6_PointCloudDiffusion/"
    "shower_flow/220714_cog_e_layer_ShowerFlow_best.pth",
) -> tuple[torch.nn.ModuleList, torch.distributions.Distribution]:
    """
    Load the flow model to be used in the generation of showers.

    Parameters
    ----------
    config : Configs
        The configuration object.
    model_path : str, optional
        The path to the model checkpoint to be loaded, by default
        use a known good checkpoint.

    Returns
    -------
    flow : torch.nn.ModuleList
        Actually unused in the generation, but is the shower_flow model.
    distribution : torch.distributions.Distribution
        Will be used in gen_utils to sample from the flow model.
        After being condition on the shower energy, this can be
        sampled to find the moments of the shower plus the
        energy and number of points in each layer.
    """
    if config.model_name == "wish":
        return None, None

    version = config.shower_flow_version
    inputs_used = showerflow_utils.get_input_mask(config)
    input_dim = np.sum(inputs_used)

    flow, distribution, transforms = shower_flow.versions_dict[version](
        num_blocks=config.shower_flow_num_blocks,
        # when 'condioning' on additional Esum,
        # Nhits etc add them on as inputs rather than
        # adding 30 e layers
        num_inputs=input_dim,
        num_cond_inputs=get_cond_dim(config, "showerflow"),
        device=config.device,
    )  # num_cond_inputs
    checkpoint = torch.load(model_path, map_location=config.device, weights_only=False)
    flow.load_state_dict(checkpoint["model"])
    flow.eval().to(config.device)
    return flow, distribution, transforms


def load_diffusion_model(
    config: Configs, model_name: str, model_path: str = None
) -> tuple[torch.nn.Module, np.array, np.array, None]:
    """
    Load a diffusion model to be used in the generation of showers.

    Parameters
    ----------
    config : Configs
        The configuration object.
    model_name : str
        Type of model to be loaded. Must be one of 'ddpm', 'edm', 'cm'.
    model_path : str, optional
        The path to the model checkpoint to be loaded, if not provided
        and a connectiont to /beegfs/desy is available, a known good
        checkpoint will be used.

    Returns
    -------
    model : torch.nn.Module
        The diffusion model to be used in the generation of showers.
    coef_real : np.array
        Coefficients of a polynomal used to rescale the number of clusters.
        scale_factor = poly_fn_fake(poly_fn_real(num_clusters)) / num_clusters
    coef_fake : np.array
        Coefficients of a polynomal used to rescale the number of clusters.
        scale_factor = poly_fn_fake(poly_fn_real(num_clusters)) / num_clusters
    n_splines : None
        Alternative method of rescaling the number of clusters, could be loaded
        but is not used in this implementation.
    """
    # things common to all diffusion models
    coef_real = np.array(
        [2.42091454e-09, -2.72191705e-05, 2.95613817e-01, 4.88328360e01]
    )  # fixed coeff at 0.1 threshold
    coef_fake = np.array(
        [-7.68614180e-07, 2.49613388e-03, 1.00790407e00, 1.63126644e02]
    )
    checkpoint = torch.load(
        model_path,
        map_location=torch.device(config.device),
    )  # max 5200000
    model_class = get_model_class(config)
    model = model_class(config).to(config.device)
    model.load_state_dict(checkpoint["others"]["model_ema"])
    n_splines = None

    model.eval()
    return model, coef_real, coef_fake, n_splines


def get_model_class(config):
    if config.model_name == "Diffusion":
        m_class = Diffusion
    else:
        raise NotImplementedError(
            f"Model {config.model_name} not implemented, known models: " "Diffuson"
        )
    return m_class
