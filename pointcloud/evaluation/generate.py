"""
Generate fake showers used for plotting scripts.
"""
import os
import numpy as np
import torch
import time

from ..models.shower_flow import compile_HybridTanH_model
from ..configs import Configs
from ..utils import gen_utils
from ..data.read_write import regularise_event_axes

from ..models import epicVAE_nflows_kDiffusion as mdls
from ..models import allCond_epicVAE_nflow_PointDiff as mdls2
from ..models.wish import Wish


def make_params_dict() -> dict:
    """
    Generate the default param dict for the generation of showers.
    It is expected that the user will modify this function or
    the output dict to suit their needs.

    Returns
    -------
    params_dict : dict
        Dictionary containing the parameters for the generation of showers.
    """
    params_dict = {}

    # SINGLE ENERGY GENERATION
    # params_dict["min_energy_list"] = [10, 50, 90]
    # params_dict["max_energy_list"] = [10, 50, 90]
    # params_dict["n_events"] = 2000
    # params_dict[
    #     "out_path"
    # ] = "/beegfs/desy/user/buhmae/6_PointCloudDiffusion/output/singleE/"

    # FULL SPECTRUM GENERATION
    params_dict["min_energy_list"] = [10]
    params_dict["max_energy_list"] = [90]
    params_dict["n_events"] = 40_000
    params_dict[
        "out_path"
    ] = "/beegfs/desy/user/buhmae/6_PointCloudDiffusion/output/full/"

    # COMMMON PARAMETERS
    params_dict["caloclouds_list"] = ["ddpm", "edm", "cm"]  # 'ddpm, 'edm', 'cm'
    params_dict["seed_list"] = [12345, 123456, 1234567]
    # params_dict["caloclouds_list"] = ['cm']   # 'ddpm, 'edm', 'cm'
    # params_dict["seed_list"] = [1234]
    params_dict["n_scaling"] = True  # default True
    params_dict["batch_size"] = 16
    params_dict["prefix"] = ""  # default ''
    return params_dict


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
    flow, distribution, transforms = compile_HybridTanH_model(
        num_blocks=10,
        # when 'condioning' on additional Esum,
        # Nhits etc add them on as inputs rather than
        # adding 30 e layers
        #  num_inputs=32,
        #  num_inputs=35,
        num_inputs=65,
        num_cond_inputs=1,
        device=config.device,
    )  # num_cond_inputs
    # checkpoint = torch.load('/beegfs/desy/user/akorol'
    #                         '/chekpoints/ECFlow/EFlow+CFlow_138.pth')
    # checkpoint = torch.load('/beegfs/desy/user/buhmae'
    #                         '/6_PointCloudDiffusion/shower_flow/'
    #                         '220706_cog_ShowerFlow_350.pth')
    # checkpoint = torch.load('/beegfs/desy/user/buhmae'
    #                         '/6_PointCloudDiffusion/shower_flow/'
    #                         '220707_cog_ShowerFlow_500.pth')  # max 730
    # checkpoint = torch.load('/beegfs/desy/user/buhmae'
    #                         '/6_PointCloudDiffusion/shower_flow/'
    #                         '220713_cog_e_layer_ShowerFlow_best.pth')
    checkpoint = torch.load(model_path)  # trained about 350 epochs
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
    config.dropout_rate = 0.0

    # caloclouds baseline
    if model_name == "ddpm":
        # config = Configs()
        config.sched_mode = "quardatic"
        config.num_steps = 100
        config.residual = True
        config.latent_dim = 256
        model = mdls2.AllCond_epicVAE_nFlow_PointDiff(config).to(config.device)
        if model_path is None:
            model_path = (
                "/beegfs/desy/user/akorol/logs/point-cloud/"
                "AllCond_epicVAE_nFlow_PointDiff_100s_MSE_"
                "loss_smired_possitions_quardatic2023_04_06__16_34_39"
                "/ckpt_0.000000_837000.pt"
            )
        checkpoint = torch.load(
            model_path,
            map_location=torch.device(config.device),
        )  # quadratic
        model.load_state_dict(checkpoint["state_dict"])
        coef_fake = np.array(
            [-2.03879741e-06, 4.93529413e-03, 5.11518795e-01, 3.14176987e02]
        )
        # joblib.load('/beegfs/desy/user/buhmae/6_PointCloudDiffusion/n_spline/spline_ddpm.joblib')
        n_splines = None

    # caloclouds EDM
    elif model_name == "edm":
        # config = Configs()
        config.num_steps = 13
        config.sampler = "heun"  # default 'heun'
        # stochasticity, default 0.0
        # (if s_churn more than num_steps, it will be clamped to max value)
        config.s_churn = 0.0
        config.s_noise = 1.0  # default 1.0   # noise added when s_churn > 0
        config.sigma_max = 80.0  # 5.3152e+00  # default 80.0
        config.sigma_min = 0.002  # default 0.002
        config.rho = 7.0  # default 7.0
        # baseline with lat_dim = 0, max_iter 10M, lr=1e-4 fixed,
        # dropout_rate=0.0, ema_power=2/3 (long training)  USING THIS TRAINING
        config.latent_dim = 0
        config.residual = False
        if model_path is None:
            model_path = (
                config.logdir
                + "/kCaloClouds_2023_06_29__23_08_31/ckpt_0.000000_2000000.pt"
            )
        checkpoint = torch.load(
            model_path,
            map_location=torch.device(config.device),
        )  # max 5200000
        model = mdls.epicVAE_nFlow_kDiffusion(config).to(config.device)
        model.load_state_dict(checkpoint["others"]["model_ema"])
        coef_fake = np.array(
            [-7.68614180e-07, 2.49613388e-03, 1.00790407e00, 1.63126644e02]
        )
        # joblib.load('/beegfs/desy/user/buhmae/6_PointCloudDiffusion/n_spline/spline_edm.joblib')
        n_splines = None

    # condsistency model
    elif model_name == "cm":
        # config = Configs()
        config.num_steps = 1
        config.sigma_max = 80.0  # 5.3152e+00  # default 80.0
        # long baseline with lat_dim = 0, max_iter 1M, lr=1e-4 fixed,
        # num_steps=18, bs=256, simga_max=80, epoch=2M, EMA
        config.latent_dim = 0
        config.residual = False
        if model_path is None:
            model_path = (
                config.logdir + "/CD_2023_07_07__16_32_09/ckpt_0.000000_1000000.pt"
            )
        checkpoint = torch.load(
            model_path,
            map_location=torch.device(config.device),
        )  # max 1200000
        model = mdls.epicVAE_nFlow_kDiffusion(config, distillation=True).to(
            config.device
        )
        model.load_state_dict(checkpoint["others"]["model_ema"])
        coef_fake = np.array(
            [-9.02997505e-07, 2.82747963e-03, 1.01417267e00, 1.64829018e02]
        )
        # joblib.load('/beegfs/desy/user/buhmae/6_PointCloudDiffusion/n_spline/spline_cm.joblib')
        n_splines = None

    elif model_name == "wish":
        if model_path is None:
            model_path = (
                "/home/henry/training/point-cloud-diffusion-logs/"
                + "wish/from_rescaled_hls_v4.pt"
            )
        model = Wish.load(model_path)
        n_splines = None
        coef_real = None
        coef_fake = None
    else:
        raise ValueError(
            f"model_name {model_name} unknown"
            + "model_name must be one of: ddpm, edm, cm, wish"
        )
    model.eval()
    return model, coef_real, coef_fake, n_splines


def generate_showers(
    config, min_energy, max_energy, params_dict, flow_model, diffusion_model
):
    """
    Generate a batch of showers, and time to process.
    This function is a wrapper around gen_utils.gen_showers_batch that
    does the timing and prints the time per shower.

    Parameters
    ----------
    config : Configs
        The configuration object.
    min_energy : float
        Minimum energy of the showers to be generated.
    max_energy : float
        Maximum energy of the showers to be generated.
    params_dict : dict
        Dictionary containing the parameters for the generation of showers.
        As produced by make_params_dict.
    flow_model : tuple[torch.nn.ModuleList, torch.distributions.Distribution]
        The flow model to be used in the generation of showers.
        As produced by load_flow_model.
    diffusion_model : tuple[torch.nn.Module, np.array, np.array, None]
        The diffusion model to be used in the generation of showers.
        As produced by load_diffusion_model.

    Returns
    -------
    showers : np.array (params_dict["n_events"], config.max_points, 4)
        The generated showers. The third dimension is (x, y, z, e)
    cond_E : np.array (params_dict["n_events"], 1)
        The energy of the incident particle that conditions the showers.

    """
    # unpack params
    flow, distribution, transforms = flow_model
    model, coef_real, coef_fake, n_splines = diffusion_model
    s_t = time.time()
    showers, cond_E = gen_utils.gen_showers_batch(
        model,
        distribution,
        min_energy,
        max_energy,
        params_dict["n_events"],
        bs=params_dict["batch_size"],
        config=config,
        coef_real=coef_real,
        coef_fake=coef_fake,
        n_scaling=params_dict["n_scaling"],
        n_splines=n_splines,
    )
    t = time.time() - s_t
    print("time per shower: (s)", t / params_dict["n_events"])
    return showers, cond_E


def write_showers(config, params_dict, flow_model_path=None, diffusion_model_path=None):
    if flow_model_path is not None:
        flow_model = load_flow_model(config, flow_model_path)
    else:
        flow_model = load_flow_model(config)
    for model_n in range(len(params_dict["caloclouds_list"])):
        model_name = params_dict["caloclouds_list"][model_n]
        seed = params_dict["seed_list"][model_n]

        if diffusion_model_path is not None:
            diffusion_model = load_diffusion_model(
                config, model_name, diffusion_model_path
            )
        else:
            diffusion_model = load_diffusion_model(config, model_name)
        print("f{model_name} model loaded")

        for energy_n in range(len(params_dict["min_energy_list"])):
            min_energy = params_dict["min_energy_list"][energy_n]
            max_energy = params_dict["max_energy_list"][energy_n]

            # GENERATE EVENTS
            fake_showers, cond_E = generate_showers(
                config, min_energy, max_energy, params_dict, flow_model, diffusion_model
            )

            # save fake showers
            file_path = os.path.join(
                params_dict["out_path"],
                f"{params_dict['prefix']}{min_energy}-{max_energy}GeV_"
                + f"{params_dict['n_events']}_{model_name}_seed{seed}.npz",
            )
            np.savez(file_path, fake_showers=fake_showers, energy=cond_E)

            print(f"fake showers (energy in MeV) saved in {file_path}")


def load_np_showers(file_path: str) -> np.array or tuple[np.array, np.array]:
    """
    Flexible loading method for showers saved in np.save or np.savez format.
    Can deal with a file continaing showers and energies, or just showers.
    Will ensure axis order is [n_events, max_points, 4] for showers.

    Parameters
    ----------
    file_path : str
        The path to the file containing the fake showers.

    Returns
    -------
    showers : np.array (n_events, max_points, 4)
        The showers. The third dimension is (x, y, z, e).
    cond_E : np.array (n_events, 1) or None
        The energy of the input particle that conditions the showers,
        if it is present in the file.
    """
    data = np.load(file_path)
    try:
        showers = data["fake_showers"]
        cond_E = data["energy"]
    except (KeyError, IndexError):
        showers = data
        cond_E = None
    showers = regularise_event_axes(showers)
    return showers, cond_E


def main():
    config = Configs()
    params_dict = make_params_dict()
    write_showers(config, params_dict)


if __name__ == "__main__":
    main()
