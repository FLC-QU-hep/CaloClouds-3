import numpy as np
import torch

from ..models.load import get_model_class
from ..config_varients.wish import Configs
from ..utils.gen_utils import get_shower
from ..utils.metadata import Metadata
from ..data.read_write import read_raw_regaxes
from ..utils.detector_map import floors_ceilings


def events_to_hits_per_layer(events, config):
    """
    Create the hits per layer distribution needed to generate showers from
    caloclouds.

    Parameters
    ----------
    events : np.array (bs, max_num_hits, 4)
        The events to generate the hits per layer distribution from.
        The last dimension is (x, y, z, e).
    config : pointcloud.configs.Configs
        The configuration object.
    
    Returns
    -------
    hits_per_layer : np.array (bs, num_layers)
        Number of hits per layer in the events.

    """
    meta = Metadata(config)
    floors, ceilings = floors_ceilings(
        meta.layer_bottom_pos_raw, meta.cell_thickness_raw
    )
    mask = events[:, :, 3] > 0
    hits = [
        ((events[:, :, 1] < c) & (events[:, :, 1] > f) & mask).sum(axis=1)
        for f, c in zip(floors, ceilings)
    ]
    hits_per_layer = np.vstack(hits).T
    hits_per_layer = hits_per_layer.astype(int)
    return hits_per_layer



def gen_raw_caloclouds(
    model,
    hits_per_layer,
    cond_E_batch,
    config,
):
    """
    Generate a batch of showers using any of the v1 style models
    according to the given incident energies.

    Parameters
    ----------
    model : torch.nn.Module
        The model to generate showers from.
    hits_per_layer : np.array (bs, num_layers)
        Number of hits per layer in the events.
    cond_E_batch : np.array (bs, 1)
        The incident energies of the showers.
    config : pointcloud.configs.Configs
        The configuration object.

    Returns
    -------
    points : np.array (bs, max_points, 4)
        The generated showers. The third dimension is (x, y, z, e)
    """
    bs = cond_E_batch.size(0)
    max_num_clusters = hits_per_layer.sum(axis=1).max()
    cond_N = (
        torch.Tensor(hits_per_layer.sum(axis=1)).to(config.device).unsqueeze(-1)
    )

    fake_showers = get_shower(
        model,
        max_num_clusters,
        cond_E_batch,
        cond_N,
        bs=bs,
        config=config,
    )
    points = fake_showers.detach().numpy()
    return points


def process_events(model_path, configs, n_events):
    """
    Run caloclouds using hits per layer drawn from G4 events.

    Parameters
    ----------
    model_path : str
        The path to the caloclouds model.
    configs : pointcloud.configs.Configs
        The configuration object.
    n_events : int
        The number of events to process.

    Returns
    -------
    hits_per_layer : np.array (n_events, num_layers)
        Number of hits per layer in the events.
    points : np.array (n_events, max_points, 4)
        The generated showers. The third dimension is (x, y, z, e)
    """
    model = get_model_class(configs)(configs).to(configs.device)
    model.load_state_dict(
        torch.load(model_path, map_location=configs.device)["state_dict"]
    )

    meta = Metadata(configs)
    n_layers = len(meta.layer_bottom_pos_raw)

    hits_per_layer = np.zeros((n_events, n_layers), dtype=int)
    max_points = 10_000
    points = np.zeros((n_events, max_points, 4), dtype=float)

    batch_len = 100
    batch_starts = np.arange(0, n_events, batch_len)
    n_batches = np.ceil(n_events / batch_len)

    for b, start in enumerate(batch_starts):
        print(f"{b/n_batches:.1%}", end="\r", flush=True)
        energy, events = read_raw_regaxes(
            configs, pick_events=slice(start, start + batch_len)
        )
        cond_E_batch = torch.Tensor(energy).to(configs.device).unsqueeze(-1)
        hits_per_layer_batch = events_to_hits_per_layer(events, configs)
        hits_per_layer[start : start + batch_len] = hits_per_layer_batch
        points_batch = gen_raw_caloclouds(
            model, hits_per_layer_batch, cond_E_batch, configs
        )
        pnts_found = points_batch.shape[1]
        points[start : start + batch_len, :pnts_found] = points_batch[:, :max_points]
    return hits_per_layer, points
