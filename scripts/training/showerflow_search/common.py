"""
Shared helpers for the (temporary) shower-flow hyperparameter search.

Everything here reuses the real training code in scripts/training/ShowerFlow.py
and pointcloud/utils/showerflow_*.py rather than re-implementing the training
loop, so results stay consistent with normal (non-search) shower-flow runs.
"""

import importlib.util
import os
import sys

import numpy as np
import torch

# scripts/training/ShowerFlow.py relies on a `pointcloud` symlink sitting next
# to it (scripts/training/pointcloud -> ../../pointcloud/) so that running it
# as `python scripts/training/ShowerFlow.py` can `import pointcloud`. This
# search lives one directory deeper and has no such symlink, so put the real
# repo root on sys.path explicitly instead of adding another symlink.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def import_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_config(config_path):
    """Load a Configs() instance from a pointcloud/config_varients/*.py file path."""
    mod = import_module_from_path("_search_base_config", config_path)
    return mod.Configs()


def trial_dir(search_root, trial_id):
    return os.path.join(search_root, "trials", f"trial_{trial_id:04d}")


def apply_trial_overrides(config, af_dim, num_blocks, version, search_root, trial_id):
    """
    Point this trial at its own private shower_flow_data_dir so checkpoints
    from different (af_dim, version) combinations never collide - the
    checkpoint filename encodes num_blocks and version, but not af_dim, so
    two trials sharing a directory could silently load/overwrite each
    other's weights.
    """
    config.af_dim = af_dim
    config.shower_flow_num_blocks = num_blocks
    config.shower_flow_version = version
    data_dir = trial_dir(search_root, trial_id)
    # pointcloud.utils.showerflow_utils.get_data_dir() silently ignores
    # shower_flow_data_dir and falls back to storage_base or /home/{user}/Data
    # if the path doesn't already exist - create it now so every trial
    # actually lands in its own directory instead of colliding in a shared
    # fallback location.
    os.makedirs(data_dir, exist_ok=True)
    config.shower_flow_data_dir = data_dir
    return config


def link_cache_files(cache_dir, showerflow_dir):
    """
    Symlink the expensive-to-compute cached per-event arrays (clusters per
    layer, energy per layer, cog, incident E / n points, direction, and the
    fixed input norms) from a shared cache dir into this trial's own
    showerflow_dir, so ShowerFlow.main()'s redo=False checks see them as
    already computed and skip the ~5-10 minute recompute per trial.

    cache_dir must be an existing showerFlow/<dataset> dir for the SAME
    dataset_path as the trial config (e.g. a previous full training run's
    output dir), so the cached arrays actually match.
    """
    os.makedirs(showerflow_dir, exist_ok=True)
    cache_files = [
        "clusters_per_layer.npz",
        "energy_per_layer.npz",
        "cog.npy",
        "pointsE.npz",
        "direction.npy",
        "input_norms.npz",
    ]
    for fname in cache_files:
        src = os.path.join(cache_dir, fname)
        dst = os.path.join(showerflow_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)


def wasserstein_metrics(config, best_model_path, device, seed=123):
    """
    Reload the best checkpoint for `config` and recompute the sampled vs.
    truth marginals for num_points and visible_energy (mirroring the eval
    block at the end of ShowerFlow.main), then score them with the
    Wasserstein (earth-mover) distance - unlike the scalar NLL loss, this
    directly measures the mismatch visible in the ShowerFlow_eval_*.png
    plots.
    """
    from scipy.stats import wasserstein_distance

    from pointcloud.data.conditioning import get_cond_dim
    from pointcloud.data.read_write import get_n_events
    from pointcloud.models import shower_flow
    from pointcloud.utils import showerflow_training, showerflow_utils
    from pointcloud.utils.metadata import Metadata

    showerflow_utils.ensure_version_registered(config)
    shower_flow_compiler = shower_flow.versions_dict[config.shower_flow_version]
    cond_dim = get_cond_dim(config, "showerflow")
    inputs_used = showerflow_utils.get_input_mask(config)
    cut_inputs = np.where(~inputs_used)[0]
    input_dim = np.sum(inputs_used)

    model, distribution, transforms = shower_flow_compiler(
        num_blocks=config.shower_flow_num_blocks,
        num_inputs=input_dim,
        num_cond_inputs=cond_dim,
        af_dim=config.af_dim,
        device=device,
    )
    model.load_state_dict(
        torch.load(best_model_path, map_location=device, weights_only=False)["model"]
    )
    model.to(device)
    model.eval()

    meta = Metadata(config)
    showerflow_dir = showerflow_utils.get_showerflow_dir(config)

    n_events = np.sum(get_n_events(config.dataset_path, config.n_dataset_files))
    local_batch_size = min(50_000, n_events)

    pointsE_path = showerflow_training.get_incident_npts_visible(
        config, showerflow_dir, redo=False, local_batch_size=local_batch_size
    )
    direction_path = None
    if "p_norm_local" in config.shower_flow_cond_features:
        direction_path = showerflow_training.get_gun_direction(
            config, showerflow_dir, redo=False, local_batch_size=local_batch_size
        )
    clusters_per_layer_path = showerflow_training.get_clusters_per_layer(
        config, showerflow_dir, redo=False, local_batch_size=local_batch_size
    )
    energy_per_layer_path = showerflow_training.get_energy_per_layer(
        config, showerflow_dir, redo=False, local_batch_size=local_batch_size
    )
    cog_path, _cog = showerflow_training.get_cog(
        config, showerflow_dir, redo=False, local_batch_size=local_batch_size
    )

    make_train_ds = showerflow_training.train_ds_function_factory(
        pointsE_path,
        cog_path,
        clusters_per_layer_path,
        energy_per_layer_path,
        config,
        direction_path=direction_path,
    )
    eval_dataset = make_train_ds(0, local_batch_size)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=2048, shuffle=False
    )

    torch.manual_seed(seed)
    clusters_per_layer_list, e_per_layer_list = [], []
    clusters_per_layer_sampled_list, e_per_layer_sampled_list = [], []
    with torch.no_grad():
        for data in eval_loader:
            context = data[:, :cond_dim].to(device).float()
            input_data = data[:, cond_dim:].to(device).float()
            samples = (
                distribution.condition(context)
                .sample(torch.Size([context.shape[0]]))
                .cpu()
                .numpy()
            )
            distribution.clear_cache()

            i = 0
            if "total_clusters" in config.shower_flow_inputs:
                i += 1
            if "total_energy" in config.shower_flow_inputs:
                i += 1
            if "cog_x" in config.shower_flow_inputs:
                i += 1
            if "cog_y" in config.shower_flow_inputs:
                i += 1
            if "cog_z" in config.shower_flow_inputs:
                i += 1
            if "clusters_per_layer" in config.shower_flow_inputs:
                clusters_per_layer_list.append(input_data[:, i : i + 30].cpu().numpy())
                clusters_per_layer_sampled_list.append(samples[:, i : i + 30])
                i += 30
            if "energy_per_layer" in config.shower_flow_inputs:
                e_per_layer_list.append(input_data[:, i : i + 30].cpu().numpy())
                e_per_layer_sampled_list.append(samples[:, i : i + 30])

    clusters_per_layer = np.concatenate(clusters_per_layer_list, axis=0)
    clusters_per_layer_sampled = np.concatenate(clusters_per_layer_sampled_list, axis=0)
    e_per_layer = np.concatenate(e_per_layer_list, axis=0)
    e_per_layer_sampled = np.concatenate(e_per_layer_sampled_list, axis=0)

    norms = np.load(os.path.join(showerflow_dir, "input_norms.npz"))
    clusters_per_layer_norm = norms["clusters_per_layer_norm"]
    energy_per_layer_norm = norms["energy_per_layer_norm"]

    num_points = clusters_per_layer.sum(axis=1) / clusters_per_layer_norm
    num_points_sampled = clusters_per_layer_sampled.sum(axis=1) / clusters_per_layer_norm
    visible_energy = e_per_layer.sum(axis=1) / energy_per_layer_norm
    visible_energy_sampled = e_per_layer_sampled.sum(axis=1) / energy_per_layer_norm

    return {
        "n_eval_events": int(len(num_points)),
        "w_num_points": float(wasserstein_distance(num_points, num_points_sampled)),
        "w_visible_energy": float(
            wasserstein_distance(visible_energy, visible_energy_sampled)
        ),
    }
