#!/usr/bin/env python
"""
Compile a CaloClouds ShowerFlow + diffusion model pair into one TorchScript file.

The compiled module is called as

    model(energy, theta, phi) -> Tensor

with the incident energy in GeV and the angles in degrees, all in local coordinates.
It returns the 30 points-per-layer counts followed by (x [mm], y [mm], layer, e [MeV])
for every generated point, flattened into one vector.

Edit the settings below, then run the script from anywhere.
"""
import inspect
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import k_diffusion
import pyro.distributions as dist

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
CALOCLOUDS_VERSION = 3  # 2 or 3

# Checkout of https://github.com/FLC-QU-hep/CaloClouds-3, next to this script's folder
REPO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "point-cloud-diffusion")

MODEL_PATHS = {
    3: dict(
        showerflow=(
            "/data/dust/user/dayhallh/point-cloud-diffusion-data/showerFlow/sim-E1261AT600AP180-180/"
            "ShowerFlow_alt1_nb2_inputs1152921504606846975_fnorms_dhist_try4_best.pth"
        ),
        diffusion="/data/dust/group/ilc/sft-ml/model_weights/CaloClouds/CC3/ckpt_0.000000_6135000.pt",
    ),
    2: dict(
        showerflow=(
            "/data/dust/user/dayhallh/point-cloud-diffusion-data/showerFlow/highGran_g40_p22_th90_ph90_en10-100/"
            "ShowerFlow_original_nb10_inputs36893488147419103231_dhist_best.pth"
        ),
        diffusion="/data/dust/group/ilc/sft-ml/model_weights/CaloClouds/CC2/ckpt_0.000000_1000000.pt",
    ),
}

# Only used to locate the dataset metadata in pointcloud/metadata/; the files need not exist.
DATASET_PATH = (
    "/data/dust/user/akorol/data/AngularShowers_RegularDetector/hdf5_for_CC/"
    "sim-E1261AT600AP180-180_file_{}slcio.hdf5"
)

# "auto" uses the one-step consistency sampler for consistency-distilled checkpoints (they carry
# their EDM teacher in others["model_teacher"]) and the multi-step Heun (EDM) sampler otherwise.
DIFFUSION_SAMPLER = "auto"  # "auto", "consistency" or "edm"

# CC3 calibration, as baked into the working CC3 paper compilation (same diffusion weights).
# These are tuned per model pair: re-derive them for a new ShowerFlow with
# pointcloud/evaluation/calculate_scale_factors.py.
CC3_CALIBRATION = dict(
    n_points_factor=0.64,  # occupancy
    energy_factor=1.00609194411797,  # energy scale
    scale_xy=1.04,  # lateral width
    max_negative_layer_energies=7,  # redraw ShowerFlow samples with more negative layer energies
)

OUT_DIR = "/home/dayhallh/training/physical_recos/inference_out"
MAKE_PLOTS = True

sys.path.insert(0, REPO_PATH)
from pointcloud.config_varients import caloclouds_3  # noqa: E402
from pointcloud.models import shower_flow  # noqa: E402
from pointcloud.utils import showerflow_utils  # noqa: E402
from pointcloud.utils.metadata import Metadata  # noqa: E402

N_LAYERS = 30
EIR = 1804.7  # ECAL inner radius (front face of the ECAL) from the IP, mm


# ---------------------------------------------------------------------------
# ShowerFlow
# ---------------------------------------------------------------------------
def load_showerflow(path, caloclouds_version):
    """
    Build the ShowerFlow architecture encoded in the file name and load its weights.

    The block layout must come from the file: 'original' and 'alt1' have identical weight
    names, so loading into the wrong layout succeeds silently and gives a different flow.

    Returns the flow transforms, the config (for the dataset metadata) and the numbers of
    inputs and conditioning inputs.
    """
    configs = caloclouds_3.Configs()
    configs.device = "cpu"
    configs.dataset_path = DATASET_PATH
    if caloclouds_version == 2:
        configs.shower_flow_cond_features = ["energy"]
    (
        configs.shower_flow_version,
        configs.shower_flow_num_blocks,
        _,
        configs.shower_flow_inputs,
        configs.shower_flow_fixed_input_norms,
    ) = showerflow_utils.config_params_from_showerflow_path(path)
    n_inputs = int(np.sum(showerflow_utils.get_input_mask(configs)))
    n_cond = int(np.sum(showerflow_utils.get_cond_mask(configs)))

    state = torch.load(path, map_location="cpu", weights_only=False)["model"]
    constructor = shower_flow.versions_dict[configs.shower_flow_version]
    kwargs = dict(
        num_blocks=configs.shower_flow_num_blocks,
        num_inputs=n_inputs,
        num_cond_inputs=n_cond,
        device="cpu",
    )
    if "af_dim" in inspect.signature(constructor).parameters:  # repo versions since May 2026
        # the coupling hypernets are num_inputs * af_dim wide
        kwargs["af_dim"] = state["0.nn.layers.0.weight"].shape[0] // n_inputs
    flow_model, _, flow_transforms = constructor(**kwargs)
    flow_model.load_state_dict(state)
    flow_model.eval()
    flow_model.requires_grad_(False)
    print(
        f"ShowerFlow: {configs.shower_flow_version}, {configs.shower_flow_num_blocks} blocks, "
        f"{n_inputs} inputs, {n_cond} conditioning inputs"
    )
    return flow_transforms, configs, n_inputs, n_cond


class ShowerFlowSampler(nn.Module):
    """
    Draws one ShowerFlow sample, shape (1, n_inputs), for a (1, n_cond) context.
    The flow is a pyro distribution, which TorchScript cannot compile, so this is traced.
    """

    def __init__(self, flow_transforms, n_inputs):
        super().__init__()
        self.flow_transforms = flow_transforms
        self.n_inputs = n_inputs

    def forward(self, context):
        with torch.no_grad():
            base = dist.Normal(torch.zeros(self.n_inputs), torch.ones(self.n_inputs))
            flow = dist.ConditionalTransformedDistribution(base, self.flow_transforms)
            return flow.condition(context).expand([1]).to_event(1).rsample()


def random_context(rng):
    """(1, 4) context: energy / 127 and the direction, inside the CC3 training range."""
    theta = np.deg2rad(rng.uniform(0, 60))
    phi = np.deg2rad(rng.uniform(-180, 180))
    return torch.tensor(
        [[rng.uniform(1, 126) / 127, np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)]],
        dtype=torch.float32,
    )


def has_frozen_scalar_view(traced):
    """True if the trace recorded a view(..., []) with a data-dependent shape."""
    for node in traced.inlined_graph.nodes():
        if node.kind() == "aten::view":
            size = node.inputsAt(1).node()
            if size.kind() == "prim::ListConstruct" and size.inputsSize() == 0:
                return True
            if size.kind() == "prim::Constant" and size.output().toIValue() == []:
                return True
    return False


def trace_showerflow(flow_transforms, n_inputs, n_cond, max_attempts=50, n_smoke_tests=500):
    """
    torch.jit.trace the ShowerFlow sampler, rejecting traces that froze a data-dependent shape.

    A trace records the path of one random draw. If exactly one spline input lies outside the
    spline bound (+-3) while tracing, pyro's masked assignment is recorded with a fixed
    view(..., []), and the compiled model later fails with "shape '[]' is invalid for input of
    size N" whenever a draw has a different number of out-of-bound values. That happens in
    about half of all traces.
    (check_trace is off: it compares two independent random draws, so it can never pass.)
    """
    sampler = ShowerFlowSampler(flow_transforms, n_inputs)
    example = torch.tensor([[50 / 127, 0.0, 0.0, 1.0]])[:, :n_cond]
    for attempt in range(max_attempts):
        traced = torch.jit.trace(sampler, example, check_trace=False)
        if not has_frozen_scalar_view(traced):
            break
        print(f"ShowerFlow trace {attempt} froze a data-dependent shape, re-tracing")
    else:
        raise RuntimeError("Could not obtain a shape-generic trace of the ShowerFlow")

    rng = np.random.default_rng(1)
    with torch.no_grad():
        for _ in range(n_smoke_tests):
            traced(random_context(rng)[:, :n_cond])
    return traced


# ---------------------------------------------------------------------------
# Diffusion model
# ---------------------------------------------------------------------------
class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super().__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = nn.Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        return self._layer(x) * gate + bias


class PointwiseNet_kDiffusion(nn.Module):
    """Per-point network; hidden_1 is 32 for CC3 and 128 for CC2."""

    def __init__(self, point_dim, context_dim, hidden_1, time_dim=64, fourier_scale=16):
        super().__init__()
        hidden_2 = hidden_1 * 2
        hidden_3 = hidden_2 * 2
        ctx_dim = context_dim + time_dim
        self.act = nn.functional.leaky_relu
        self.layers = nn.ModuleList(
            [
                ConcatSquashLinear(point_dim, hidden_1, ctx_dim),
                ConcatSquashLinear(hidden_1, hidden_2, ctx_dim),
                ConcatSquashLinear(hidden_2, hidden_3, ctx_dim),
                ConcatSquashLinear(hidden_3, hidden_2, ctx_dim),
                ConcatSquashLinear(hidden_2, hidden_1, ctx_dim),
                ConcatSquashLinear(hidden_1, point_dim, ctx_dim),
            ]
        )
        self.timestep_embed = nn.Sequential(
            k_diffusion.layers.FourierFeatures(1, time_dim, std=fourier_scale),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, x: torch.Tensor, sigma: torch.Tensor, context: torch.Tensor):
        """x: (B, N, 4) points, sigma: (B,) noise level, context: (B, C)."""
        batch_size = x.size(0)
        sigma = sigma.view(batch_size, 1, 1)
        context = context.view(batch_size, 1, -1)
        c_noise = sigma.log() / 4  # EDM / k-diffusion time conditioning
        time_emb = self.act(self.timestep_embed(c_noise))
        ctx_emb = torch.cat([time_emb, context], dim=-1)
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)
        return out


class ConsistencyDenoiser(nn.Module):
    """Karras preconditioning with the consistency-model boundary condition at sigma_min."""

    def __init__(self, inner_model, sigma_data=0.5, sigma_min=0.002):
        super().__init__()
        self.inner_model = inner_model
        self.sigma_data = torch.tensor([sigma_data] * 4)  # one per point feature
        self.sigma_min = sigma_min

    def get_scalings(self, sigma):
        sigma_data = self.sigma_data.expand(sigma.shape[0], -1)
        c_skip = sigma_data**2 / ((sigma - self.sigma_min) ** 2 + sigma_data**2)
        c_out = (sigma - self.sigma_min) * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def forward(self, input, sigma: float, context):
        sigma_t = torch.tensor([sigma] * input.shape[0], dtype=torch.float32).to(input.device).unsqueeze(1)
        c_skip, c_out, c_in = [x.unsqueeze(1) for x in self.get_scalings(sigma_t)]
        return self.inner_model(input * c_in, sigma_t, context) * c_out + input * c_skip


class ConsistencySampler(nn.Module):
    """One-step sampling of a consistency-distilled diffusion model."""

    def __init__(self, net, sigma_max=80.0, sigma_min=0.002, sigma_data=0.5):
        super().__init__()
        self.features = 4
        self.sigma_max = sigma_max
        self.diffusion = ConsistencyDenoiser(net, sigma_data=sigma_data, sigma_min=sigma_min)

    def forward(self, cond_feats, num_points: int):
        x_T = torch.randn([cond_feats.size(0), num_points, self.features], device=cond_feats.device)
        return self.diffusion(x_T * self.sigma_max, self.sigma_max, cond_feats)


class EDMDenoiser(nn.Module):
    """Karras (EDM) preconditioning, for undistilled diffusion models."""

    def __init__(self, inner_model, sigma_data=0.5):
        super().__init__()
        self.inner_model = inner_model
        self.sigma_data = torch.tensor([sigma_data] * 4)

    def get_scalings(self, sigma):
        sigma_data = self.sigma_data.expand(sigma.shape[0], -1)
        sigma = sigma.unsqueeze(-1)
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
        c_out = sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def forward(self, input, sigma: torch.Tensor, context):
        c_skip, c_out, c_in = [x.unsqueeze(1) for x in self.get_scalings(sigma)]
        return self.inner_model(input * c_in, sigma, context) * c_out + input * c_skip


class EDMSampler(nn.Module):
    """Stochastic Heun sampler (Karras et al. 2022, algorithm 2) for undistilled models."""

    def __init__(self, net, num_steps=15, s_churn=0.0, s_noise=1.0, sigma_max=80.0, sigma_min=0.002, rho=7.0):
        super().__init__()
        self.features = 4
        self.num_steps = num_steps
        self.s_churn = s_churn
        self.s_noise = s_noise
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.rho = rho
        self.diffusion = EDMDenoiser(net)

    def get_sigmas(self, device: torch.device):
        """Karras noise schedule, with a final zero appended."""
        ramp = torch.linspace(0, 1, self.num_steps)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        return torch.cat([sigmas, sigmas.new_zeros([1])]).to(device)

    def forward(self, cond_feats, num_points: int):
        x = torch.randn([cond_feats.size(0), num_points, self.features], device=cond_feats.device)
        x = x * self.sigma_max
        sigmas = self.get_sigmas(cond_feats.device)
        s_in = x.new_ones([x.shape[0]])
        for i in range(len(sigmas) - 1):
            sigma, sigma_next = sigmas[i], sigmas[i + 1]
            gamma = min(self.s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            eps = torch.randn_like(x) * self.s_noise
            sigma_hat = sigma * (gamma + 1)
            if gamma > 0:
                x = x + eps * ((sigma_hat**2 - sigma**2) ** 0.5)
            d = (x - self.diffusion(x, sigma_hat * s_in, cond_feats)) / sigma_hat
            dt = sigma_next - sigma_hat
            if sigma_next == 0:  # Euler step
                x = x + d * dt
            else:  # Heun step
                x_2 = x + d * dt
                d_2 = (x_2 - self.diffusion(x_2, sigma_next * s_in, cond_feats)) / sigma_next
                x = x + (d + d_2) / 2 * dt
        return x


EDM_SETTINGS = {3: dict(s_churn=10.0, s_noise=0.97), 2: dict(s_churn=0.0, s_noise=1.0)}


def load_diffusion(path, caloclouds_version, sampler="auto"):
    """Build the sampler that matches the checkpoint and load its EMA weights."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    others = checkpoint["others"]
    is_consistency = "model_teacher" in others
    if sampler == "auto":
        sampler = "consistency" if is_consistency else "edm"
    if sampler == "consistency" and not is_consistency:
        print(f"WARNING: {path} has no model_teacher; is it really consistency-distilled?")

    context_dim, hidden_1 = (4, 32) if caloclouds_version == 3 else (2, 128)
    net = PointwiseNet_kDiffusion(point_dim=4, context_dim=context_dim, hidden_1=hidden_1)
    if sampler == "consistency":
        model, weights = ConsistencySampler(net), "model_ema"
    else:
        model = EDMSampler(net, **EDM_SETTINGS[caloclouds_version])
        # a consistency-distilled checkpoint keeps its undistilled EDM teacher separately
        weights = "model_teacher" if is_consistency else "model_ema"
    model.load_state_dict(others[weights])
    model.eval()
    print(f"Diffusion: {sampler} sampler with the {weights} weights of {os.path.basename(path)}")
    return model


# ---------------------------------------------------------------------------
# Full models
# ---------------------------------------------------------------------------
class InferrerCC3(nn.Module):
    """ShowerFlow (points and energy per layer) + diffusion (points within layers) for CC3."""

    def __init__(
        self,
        denoiser,
        shower_flow,
        metadata,
        n_points_factor=1.0,
        energy_factor=1.0,
        scale_xy=1.0,
        max_negative_layer_energies=7,
        max_redraws=100,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.shower_flow = shower_flow
        self.n_layers = N_LAYERS

        # Conditioning and ShowerFlow normalisation (metadata rescales.npy)
        self.energy_max = float(metadata.incident_rescale)  # GeV
        self.max_num_points = float(metadata.n_pts_rescale)
        self.vis_energy_max = float(metadata.vis_eng_rescale) * 1000  # MeV

        # Diffusion normalisation (metadata box_edges.npy), inverted from
        # PointCloudAngular.normalize_xyze: x_n = (x - mean) / std / 2, e_n = (log e - mean) / std / 2
        self.Xmean = float(metadata.mean_xs)
        self.Xstd = float(metadata.std_xs)
        self.Ymean = float(metadata.mean_ys)
        self.Ystd = float(metadata.std_ys)
        self.Emean = float(metadata.mean_log_es)
        self.Estd = float(metadata.std_log_es)

        # Calibration
        self.n_points_factor = float(n_points_factor)
        self.energy_factor = float(energy_factor)
        self.scale_xy = float(scale_xy)

        # Points outside the training box are dropped (local x = global z, local y = global x)
        self.Xabs = float(metadata.Zmax_global)
        self.Yabs = float(metadata.Xmax_global)
        # ShowerFlow samples with too many negative layer energies are redrawn
        self.max_negative_layer_energies = int(max_negative_layer_energies)
        self.max_redraws = int(max_redraws)

        # Depth of each layer centre behind the ECAL front face, for the axis of inclined showers
        layer_bottom = torch.tensor(metadata.layer_bottom_pos_global, dtype=torch.float32)
        self.dist_to_layers = layer_bottom - EIR + float(metadata.cell_thickness_global) / 2
        self.z_positions = torch.arange(self.n_layers, dtype=torch.float)

    def get_direction(self, theta, phi):
        theta = theta / 180.0 * torch.pi
        phi = phi / 180.0 * torch.pi
        px = torch.cos(phi) * torch.sin(theta)
        py = torch.sin(phi) * torch.sin(theta)
        pz = torch.cos(theta)
        return torch.tensor([float(px.item()), float(py.item()), float(pz.item())])

    def get_shift(self, theta, phi, layer: int):
        """Lateral position of the shower axis in a layer."""
        theta = theta / 180.0 * torch.pi
        phi = phi / 180.0 * torch.pi
        r = self.dist_to_layers[layer] / torch.cos(theta)
        return r * torch.sin(theta) * torch.cos(phi), r * torch.sin(theta) * torch.sin(phi)

    def forward(self, energy, theta, phi):
        """
        energy: incident energy [GeV]; theta, phi: incident direction [deg], local coordinates.
        Returns the points-per-layer counts, then (x [mm], y [mm], layer, e [MeV]) per point.
        """
        device = energy.device
        n = self.n_layers
        direction = self.get_direction(theta, phi).unsqueeze(0).to(device)
        context = torch.cat(((energy / self.energy_max).view(1, 1), direction), 1)

        with torch.no_grad():
            samples = self.shower_flow(context).to(device)
            n_redraws = 0
            while (
                bool((samples[:, -n:] < 0).sum() > self.max_negative_layer_energies)
                and n_redraws < self.max_redraws
            ):
                samples = self.shower_flow(context).to(device)
                n_redraws += 1

        # ShowerFlow outputs end with 30 point counts and 30 energies, normalised to 1/30 of the maximum
        clusters_per_layer = (
            torch.clip(samples[:, -2 * n : -n], 0, None) * self.n_points_factor * self.max_num_points / n
        ).int()
        e_per_layer = torch.clip(samples[:, -n:], 0, None) * self.vis_energy_max / n
        num_clusters = clusters_per_layer.sum().int()

        with torch.no_grad():
            fake_shower = self.denoiser(context, num_clusters)

        # Assign layers by the rank of the generated z
        z_flow = torch.repeat_interleave(self.z_positions, clusters_per_layer[0], dim=0)
        idx_dm = torch.argsort(fake_shower[0, :, 2])
        fake_shower[0, :, 2][idx_dm] = z_flow

        # Undo the training normalisation: x, y to mm, energy out of log space
        fake_shower[:, :, 0] = fake_shower[:, :, 0] * self.Xstd * 2 * self.scale_xy + self.Xmean
        fake_shower[:, :, 1] = fake_shower[:, :, 1] * self.Ystd * 2 * self.scale_xy + self.Ymean
        fake_shower[:, :, 3] = torch.exp(fake_shower[:, :, 3] * self.Estd * 2 + self.Emean)

        in_box = (torch.abs(fake_shower[:, :, 0]) <= self.Xabs) & (torch.abs(fake_shower[:, :, 1]) <= self.Yabs)
        fake_shower[:, :, 0][~in_box] = 0.0
        fake_shower[:, :, 1][~in_box] = 0.0
        fake_shower[:, :, 3][~in_box] = 0.0

        for j in range(n):
            mask = fake_shower[0, :, 2] == self.z_positions[j]
            # scale the layer to the ShowerFlow energy (clamp: a layer may be emptied by the box cut)
            layer_e = fake_shower[0, :, -1][mask]
            fake_shower[0, :, -1][mask] = layer_e / torch.clamp(layer_e.sum(), min=1e-12) * e_per_layer[0, j]
            # follow the axis of inclined showers
            x_shift, y_shift = self.get_shift(theta, phi, j)
            fake_shower[0, :, 0][mask] = (fake_shower[0, :, 0][mask] + x_shift).view(-1)
            fake_shower[0, :, 1][mask] = (fake_shower[0, :, 1][mask] + y_shift).view(-1)

        fake_shower[:, :, 3] = fake_shower[:, :, 3] * self.energy_factor
        return torch.concat((clusters_per_layer.view(-1), fake_shower[0].view(-1)))


def polyval(coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Polynomial with coefficients from the highest power down (like numpy.polyval)."""
    terms = [coeffs[-i - 1] * x**i for i in range(len(coeffs))]
    return torch.sum(torch.stack(terms))


class InferrerCC2(nn.Module):
    """
    CC2 counterpart, logic unchanged from the previous script apart from applying the CC2 axis
    rotation exactly once. Not validated: the polynomial point-count coefficients and the CoG
    values were still marked TODO, and the conditioning conventions should be checked against
    the CC2 ShowerFlow and diffusion training.
    """

    def __init__(self, denoiser, shower_flow):
        super().__init__()
        self.denoiser = denoiser
        self.shower_flow = shower_flow

        self.energy_max = 100  # GeV
        self.max_num_points = 5000
        self.vis_energy_max = 2.5 * 1000  # MeV
        self.real_coeff = torch.Tensor([0, 1, 0, 0])  # TODO: placeholder point-count scaling
        self.fake_coeff = torch.Tensor([0, 1, 0, 0])
        self.energy_factor = 1.0
        self.Xmin, self.Xmax = -240, 140
        self.Ymin, self.Ymax = -200, 200
        self.mean_cog = torch.tensor([-4.06743696e-03, -2.27790998e-01, 1.10137465e01])  # TODO: check
        self.std_cog = torch.tensor([1.24559791, 0.95357278, 2.59475371])
        self.n_layers = N_LAYERS
        self.z_positions = torch.arange(self.n_layers, dtype=torch.float)

    def scale_n_clusters(self, num_clusters):
        num_clusters = torch.clip(num_clusters, 0.0, self.max_num_points)
        num_clusters = polyval(self.fake_coeff, polyval(self.real_coeff, num_clusters))
        return torch.clip(num_clusters, 0.0, self.max_num_points).int()

    def get_cog(self, x, y, e):
        return torch.sum(x * e, dim=1) / e.sum(dim=1), torch.sum(y * e, dim=1) / e.sum(dim=1)

    def forward(self, energy, theta, phi):
        """energy [GeV]; theta and phi are not used by CC2."""
        device = energy.device
        context = (energy / self.energy_max).view(1, 1) * 2 - 1
        with torch.no_grad():
            samples = self.shower_flow(context).to(device)

        e = torch.ones((1, 1)).to(device) * energy
        e = e / 100 * 2 - 1  # max incident energy: 100 GeV

        # ShowerFlow outputs: total points, total energy, CoG x, y, z, 30 point counts, 30 energies.
        # After the CC2 axis rotation below, x is the old z and y the old x.
        cog_x = samples[:, 4] * self.std_cog[0] + self.mean_cog[0]
        cog_y = samples[:, 2] * self.std_cog[1] + self.mean_cog[1]

        total_clusters = self.scale_n_clusters(samples[:, 0])
        clusters_per_layer = torch.clip(samples[:, -60:-30], 0, 1)
        clusters_per_layer = (clusters_per_layer * total_clusters / clusters_per_layer.sum(-1)).int()
        total_clusters = clusters_per_layer.sum()
        total_energy = torch.clip(samples[:, 1], 0, None) * 1000 * self.vis_energy_max * self.energy_factor
        e_per_layer = torch.clip(samples[:, -30:], 0, None)
        e_per_layer = e_per_layer * total_energy / e_per_layer.sum(-1)

        cond_feats = torch.cat([e, samples[:, [0]]], -1)
        with torch.no_grad():
            fake_shower = self.denoiser(cond_feats, total_clusters)

        # CC2 axis rotation (the only place it is applied)
        fake_shower[:, :, [0, 1, 2]] = fake_shower[:, :, [2, 0, 1]]

        z_flow = torch.repeat_interleave(self.z_positions, clusters_per_layer[0], dim=0)
        idx_dm = torch.argsort(fake_shower[0, :, 2])
        fake_shower[0, :, 2][idx_dm] = z_flow

        # from [-1, 1] to the bounding box, then move to the ShowerFlow CoG
        fake_shower[:, :, 0] = (fake_shower[:, :, 0] + 1) / 2 * (self.Xmax - self.Xmin) - self.Xmax
        fake_shower[:, :, 1] = (fake_shower[:, :, 1] + 1) / 2 * (self.Ymax - self.Ymin) - self.Ymax
        cog = self.get_cog(fake_shower[:, :, 0], fake_shower[:, :, 1], fake_shower[:, :, 3])
        fake_shower[:, :, 0] = fake_shower[:, :, 0] - (cog[0] - cog_x).unsqueeze(1)
        fake_shower[:, :, 1] = fake_shower[:, :, 1] - (cog[1] - cog_y).unsqueeze(1)

        for j in range(self.n_layers):
            mask = fake_shower[0, :, 2] == self.z_positions[j]
            fake_shower[0, :, -1][mask] = (
                fake_shower[0, :, -1][mask] / fake_shower[0, :, -1][mask].sum() * e_per_layer[0, j]
            )

        return torch.concat((clusters_per_layer.view(-1), fake_shower[0].view(-1)))


# ---------------------------------------------------------------------------
# Checks and plots
# ---------------------------------------------------------------------------
def generate(model, energy, theta, phi, n_showers):
    """Concatenated (n_points, 4) array of x, y, layer, e from n_showers showers."""
    showers = []
    with torch.no_grad():
        for _ in range(n_showers):
            out = model(torch.tensor(float(energy)), torch.tensor(float(theta)), torch.tensor(float(phi)))
            showers.append(out[N_LAYERS:].numpy().reshape(-1, 4))
    return np.concatenate(showers)


def sanity_check(model, caloclouds_version, n_showers=30):
    """Print headline numbers; for CC3 also check that inclined showers follow their axis."""
    x, y, _, e = generate(model, 50, 0, 0, n_showers).T
    rms = np.sqrt(np.average(x**2 + y**2, weights=e))
    print(
        f"50 GeV, normal incidence: <points> {len(e) / n_showers:.0f}, "
        f"<sum E> {e.sum() / n_showers:.0f} MeV, lateral RMS {rms:.1f} mm"
    )
    if caloclouds_version == 3:
        theta, phi = 30.0, 45.0
        x, y, layer, e = generate(model, 50, theta, phi, n_showers).T
        axis = model.dist_to_layers.numpy()[layer.astype(int)] * np.tan(np.deg2rad(theta))
        along = x * np.cos(np.deg2rad(phi)) + y * np.sin(np.deg2rad(phi))
        offset = np.average(along - axis, weights=e)
        print(f"50 GeV, theta={theta:.0f}, phi={phi:.0f}: mean offset of hits from the axis {offset:.2f} mm")


def make_plots(model, out_dir, name, energy=50, n_showers=30):
    """Energy-weighted maps of showers at normal incidence."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    x, y, layer, e = generate(model, energy, 0, 0, n_showers).T

    def save(fig, label):
        fig.savefig(os.path.join(out_dir, f"{name}_{label}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(y, e, s=1, color="#2a78d6")
    ax.set(yscale="log", xlabel="y [mm]", ylabel="hit energy [MeV]", title=f"{energy} GeV, hit energies")
    save(fig, "hit_energy_vs_y")

    # x-y map over the central 10-90 % of the energy, padded by 20 %
    def weighted_range(v):
        order = np.argsort(v)
        cdf = np.cumsum(e[order]) / e.sum()
        lo, hi = np.interp([0.1, 0.9], cdf, v[order])
        return lo - 0.2 * (hi - lo), hi + 0.2 * (hi - lo)

    x_range, y_range = weighted_range(x), weighted_range(y)
    H, _, _ = np.histogram2d(x, y, range=(x_range, y_range), bins=100, weights=e)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    image = ax.imshow(
        H, origin="lower", extent=(*y_range, *x_range), cmap="Blues",
        norm=LogNorm(vmin=H[H > 0].min() * 100, vmax=H.max()),
    )
    fig.colorbar(image, ax=ax, label="energy [MeV]")
    ax.set(xlabel="y [mm]", ylabel="x [mm]", title=f"{energy} GeV, {n_showers} showers")
    save(fig, "xy_energy")

    # Relative asymmetries of the x-y map: transposed and mirrored in y
    with np.errstate(divide="ignore", invalid="ignore"):
        for label, other in [("asymmetry_transpose", H.T), ("asymmetry_mirror_y", H[:, ::-1])]:
            fig, ax = plt.subplots(figsize=(6.5, 5.5))
            image = ax.imshow(
                np.where(H > 0, (H - other) / H, np.nan), origin="lower", extent=(*y_range, *x_range),
                cmap="RdBu_r", vmin=-0.1, vmax=0.1,
            )
            fig.colorbar(image, ax=ax, label="relative difference")
            ax.set(xlabel="y [mm]", ylabel="x [mm]", title=label.replace("_", " "))
            save(fig, label)

    H, _, _ = np.histogram2d(x, layer, bins=(np.linspace(*x_range, 100), np.arange(N_LAYERS + 1)), weights=e)
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(
        H, origin="lower", aspect="auto", extent=(0, N_LAYERS, *x_range), cmap="Blues",
        norm=LogNorm(vmin=H[H > 0].min() * 100, vmax=H.max()),
    )
    fig.colorbar(image, ax=ax, label="energy [MeV]")
    ax.set(xlabel="layer", ylabel="x [mm]", title=f"{energy} GeV, longitudinal view")
    save(fig, "x_vs_layer")


# ---------------------------------------------------------------------------
def output_name(paths, caloclouds_version):
    diffusion_dir = os.path.basename(os.path.dirname(paths["diffusion"]))
    diffusion_ckpt = os.path.basename(paths["diffusion"]).replace(".pt", "")
    showerflow = os.path.basename(paths["showerflow"]).replace(".pth", "").replace("ShowerFlow_", "")
    return f"{diffusion_dir}_{diffusion_ckpt}_{showerflow}_cc{caloclouds_version}"


def main():
    torch.set_num_threads(1)  # single-threaded, as in production
    paths = MODEL_PATHS[CALOCLOUDS_VERSION]
    for path in paths.values():
        assert os.path.exists(path), f"{path} does not exist"

    flow_transforms, configs, n_inputs, n_cond = load_showerflow(paths["showerflow"], CALOCLOUDS_VERSION)
    flow = trace_showerflow(flow_transforms, n_inputs, n_cond)
    diffusion = load_diffusion(paths["diffusion"], CALOCLOUDS_VERSION, DIFFUSION_SAMPLER)

    if CALOCLOUDS_VERSION == 3:
        inferrer = InferrerCC3(diffusion, flow, Metadata(configs), **CC3_CALIBRATION)
    else:
        inferrer = InferrerCC2(diffusion, flow)

    os.makedirs(OUT_DIR, exist_ok=True)
    name = output_name(paths, CALOCLOUDS_VERSION)
    save_path = os.path.join(OUT_DIR, name + ".pt")
    torch.jit.save(torch.jit.script(inferrer), save_path)
    print(f"Saved {save_path}")

    compiled = torch.jit.load(save_path)
    sanity_check(compiled, CALOCLOUDS_VERSION)
    if MAKE_PLOTS:
        make_plots(compiled, OUT_DIR, name)


if __name__ == "__main__":
    main()
