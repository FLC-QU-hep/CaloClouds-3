"""
A numerically-stabilized log-scale shower-flow variant.

Lives here (not in shower_flow.py) because it needs per-dataset log-space
mean/std, so it cannot be a plain entry in the static versions_dict - it is
registered at runtime instead, see register()/ensure_registered() below.
shower_flow.py itself is never edited.

Root cause of the NaN/inf seen in the original "log1". NOTE: an earlier
version of this note claimed "most of a shower's 30 layers have no hits".
That is wrong - measured on the cached hdbscan_ms3_mcs10 arrays, exact zeros
are only 7.2% of the 60 values, concentrated in the shower tail (layer 29 is
41% zero, layer 0 is 11%, the middle layers are ~0%). The problem is not how
many zeros there are, it is that they form a point mass, and a continuous
flow cannot fit an atom: to place finite probability on a delta it has to
drive its density to infinity there, which sends the log-det term to +-inf.

log(x + eps) with a tiny eps is what turns a mild zero-inflation into a
fatal one. With eps=1e-6 the atom lands at log(1e-6) = -13.8 and, after the
z-scoring below, sits 6.2 sigma (median) to 14.6 sigma (max) away from the
non-zero body - z range [-14.57, +2.13]. In linear space (alt1) that same
atom sits at 0, at the edge of the data, which is why alt1 trains cleanly
and every log run here hit NaN training batches.

Fix, all confined to this file:
  1. LOG_OFFSET (below) instead of a 1e-6 epsilon, which pulls the atom in
     to a z range of [-5.46, +8.72].
  2. z-score standardize the 60 log-space values (per-dimension mean/std,
     computed once from the cached clusters_per_layer.npz / energy_per_layer.npz
     / input_norms.npz - the same "fixed norm" inputs the real training
     pipeline uses) via an extra affine bijector inserted right before the
     log/exp step. Note this cannot remove the atom on its own: an affine map
     of a point mass is still a point mass, which is why (1) was needed.
  3. clamp the pre-exp() value (ClampedSafeExpTransform, a local copy of
     shower_flow.SafeExpTransform with a clamp added) as a hard safety net
     against overflow, independent of whether (2) is perfectly calibrated.

Still outstanding: the real fix for the atom is dequantization (the cluster
counts are integers, so log(n + u) with u ~ U(0,1) removes it outright), or
factorizing out an explicit per-layer emptiness mask. The clusters and energy
zero-masks are identical - a layer with no hits has no energy - so that mask
is one 30-dim Bernoulli, not two. See also the log_abs_det_jacobian caveat on
ClampedSafeExpTransform below.

Registered into pointcloud.models.shower_flow.versions_dict at runtime
under the key "log1_stable" (register()) - this mutates the in-memory dict
of the imported module for this process only; it does not write to
shower_flow.py on disk.
"""

import functools
import os

import numpy as np
import pyro.distributions.transforms as T
import torch
from torch.distributions import constraints

from pointcloud.models.shower_flow import HybridTanH_factory

# Offset added before the log, and subtracted after the exp. Must be the same
# value in both the transform and compute_log_stats(), or the standardization
# stats no longer describe the values the transform actually sees - hence the
# single constant rather than two defaults that can drift apart.
#
# Chosen by measuring the z-scored log inputs on the cached hdbscan_ms3_mcs10
# arrays. Larger pulls the zero-atom in towards the data body, but also lets a
# sampled layer undershoot to -LOG_OFFSET:
#
#   offset   z range           frac |z|>5   worst undershoot
#   1e-6     [-14.57, +2.13]   0.00398      -0.00 hits
#   1e-3     [ -7.85, +4.71]   0.00132      -0.13 hits
#   1e-2     [ -5.46, +8.72]   0.00020      -1.25 hits   <- best |z|>5
#   1e-1     [ -3.49, +22.1]   0.00045      -12.5 hits
#
# 1e-2 minimizes the tail fraction; -1.25 hits of possible undershoot is
# negligible against Wasserstein distances of 40-250 hits.
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
