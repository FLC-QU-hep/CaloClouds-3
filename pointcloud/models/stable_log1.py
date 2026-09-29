"""
A numerically-stabilized log-scale shower-flow variant ("log1_stable").

It needs per-dataset log-space mean/std, so it cannot be a static entry of
shower_flow.versions_dict; register()/ensure_registered() add it to the
in-memory dict at runtime (shower_flow.py on disk is untouched).

Fixes the NaN/inf of the original "log1", caused by the empty-layer point mass
at log(1e-6): (1) LOG_OFFSET instead of a 1e-6 epsilon, (2) z-scoring of the 60
log-space values, (3) a clamp before exp(). The full analysis, measurements and
outstanding work are in scripts/training/showerflow_search/README.md
("Why log1_stable").
"""

import functools
import os

import numpy as np
import pyro.distributions.transforms as T
import torch
from torch.distributions import constraints

from pointcloud.models.shower_flow import HybridTanH_factory

# Offset added before the log and subtracted after the exp. Must be the same
# in the transform and compute_log_stats(), hence one constant. 1e-2 minimizes
# the |z|>5 tail (offset scan in the showerflow_search README).
LOG_OFFSET = 0.01


class ClampedSafeExpTransform(T.Transform):
    """Same as shower_flow.SafeExpTransform, but clamps the pre-exp value so
    exp() can never overflow to inf regardless of how large the upstream
    (randomly-initialized / partially-trained) coupling stack's output gets.

    CAVEAT - this is not a true bijector once the clamp engages. Where
    |x| > clamp, _call is constant, so the real log|det| is -inf rather than
    the +-clamp that log_abs_det_jacobian reports, _inverse does not invert
    _call, and the declared bijective = True is a lie. Any batch in which the
    clamp actually fires therefore has a silently wrong density. A hard clamp
    cannot be a bijector; making this correct means a smooth saturating map
    (and its matching Jacobian), or removing the clamp once the atom is dealt
    with properly and it is no longer needed as a safety net.
    """

    domain = constraints.real
    codomain = constraints.nonnegative
    bijective = True
    sign = +1

    def __init__(self, eps=LOG_OFFSET, clamp=20.0, cache_size=1):
        super().__init__(cache_size=cache_size)
        self.eps = eps
        self.clamp = clamp

    def _call(self, x):
        return torch.exp(torch.clamp(x, -self.clamp, self.clamp)) - self.eps

    def _inverse(self, y):
        return torch.log(y + self.eps)

    def log_abs_det_jacobian(self, x, y):
        return torch.clamp(x, -self.clamp, self.clamp)


class StableLog1Factory(HybridTanH_factory):
    def add_log_standardize(self, loc=None, scale=None, **kwargs):
        transform = T.AffineTransform(loc.to(self.device), scale.to(self.device))
        self.transforms.append(transform)

    def add_partial_log_clamped(self, log_clamp=20.0, **kwargs):
        # mirrors HybridTanH_factory.add_partial_log, swapping in the
        # clamped exp transform.
        log_len = 60
        prefix_len = self.num_inputs - log_len
        exp_transform = ClampedSafeExpTransform(clamp=log_clamp)
        transform = T.CatTransform(
            [T.identity_transform, exp_transform], dim=-1, lengths=[prefix_len, log_len]
        )
        self.transforms.append(transform)


def compile_HybridTanH_log1_stable(
    num_blocks, num_inputs, num_cond_inputs, af_dim, device, loc, scale, log_clamp=20.0
):
    factory = StableLog1Factory(num_inputs, num_cond_inputs, af_dim, device)

    transform_pattern = [
        "affine_coupling",
        "permutation",
        "affine_coupling",
        "permutation",
        "affine_coupling",
        "permutation",
        "spline_coupling",
        "permutation",
        "affine_coupling",
        "permutation",
        "affine_coupling",
        "permutation",
        "affine_coupling",
        "permutation",
    ]
    transform_pattern = transform_pattern * num_blocks
    transform_pattern += ["log_standardize", "partial_log_clamped"]

    model, flow_dist, transforms = factory.create(
        1, transform_pattern, count_bins=8, loc=loc, scale=scale, log_clamp=log_clamp
    )
    return model, flow_dist, transforms


def compute_log_stats(cache_dir, eps=LOG_OFFSET):
    """Per-dimension (60,) mean/std of log(fixed-norm-scaled input + eps),
    computed from the same cached arrays ShowerFlow.main() itself loads -
    i.e. exactly the values that would reach the flow as input_data's last
    60 columns for a config with shower_flow_inputs =
    ["clusters_per_layer", "energy_per_layer"] and shower_flow_fixed_input_norms
    = True (clusters block first, then energy block - matches
    _train_ds_function_factory's append order).
    """
    clusters_npz = np.load(os.path.join(cache_dir, "clusters_per_layer.npz"))
    energy_npz = np.load(os.path.join(cache_dir, "energy_per_layer.npz"))
    norms = np.load(os.path.join(cache_dir, "input_norms.npz"))

    clusters = clusters_npz["clusters_per_layer"] * norms["clusters_per_layer_norm"]
    energy = energy_npz["energy_per_layer"] * norms["energy_per_layer_norm"]
    combined = np.concatenate([clusters, energy], axis=1)  # (N, 60)

    log_combined = np.log(combined + eps)
    loc = log_combined.mean(axis=0)
    scale = np.clip(log_combined.std(axis=0), 1e-3, None)
    return torch.tensor(loc, dtype=torch.float32), torch.tensor(scale, dtype=torch.float32)


_registered_cache_dirs = set()


def register(cache_dir, device, log_clamp=20.0):
    """Compute stats from cache_dir and register "log1_stable" into the
    live pointcloud.models.shower_flow.versions_dict for this process.
    Idempotent per cache_dir - safe to call once per trial process.
    """
    from pointcloud.models import shower_flow as shower_flow_module

    loc, scale = compute_log_stats(cache_dir)
    compiler = functools.partial(
        compile_HybridTanH_log1_stable, loc=loc, scale=scale, log_clamp=log_clamp
    )
    shower_flow_module.versions_dict["log1_stable"] = compiler
    _registered_cache_dirs.add(cache_dir)
    return loc, scale


REQUIRED_CACHE_FILES = (
    "clusters_per_layer.npz",
    "energy_per_layer.npz",
    "input_norms.npz",
)


def ensure_registered(cache_dir, device=None, log_clamp=20.0):
    """register() but a no-op if this cache_dir is already registered.

    Raises a readable error when the cached arrays the standardization stats
    are derived from do not exist yet, rather than letting np.load fail deep
    inside flow construction. Those caches are written during dataset prep
    (showerflow_training.get_clusters_per_layer / get_energy_per_layer), so a
    brand-new dataset must be prepped once before log1_stable can be built.
    """
    if cache_dir in _registered_cache_dirs:
        return None
    missing = [
        f for f in REQUIRED_CACHE_FILES
        if not os.path.exists(os.path.join(cache_dir, f))
    ]
    if missing:
        raise RuntimeError(
            f"shower_flow_version='log1_stable' needs cached per-layer arrays in "
            f"{cache_dir} to compute its log-space standardization stats, but "
            f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} missing. "
            f"Run the dataset prep once (or point the config at a prepared "
            f"dataset) before training/loading this version."
        )
    return register(cache_dir, device, log_clamp=log_clamp)
