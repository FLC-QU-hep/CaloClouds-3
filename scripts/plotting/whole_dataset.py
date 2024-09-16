from pointcloud.configs import Configs
from pointcloud.utils import metadata

from pointcloud.models import wish
from pointcloud.data.dataset import PointCloudDataset
from pointcloud.data.read_write import read_raw_regaxes

import numpy as np
from matplotlib import pyplot as plt
import torch

cfg = Configs()
meta = metadata.Metadata(cfg)
layers_of_intrest = np.arange(0, 29, 3)
n_layers = len(meta.layer_bottom_pos_global)
batch_size = 100
max_batches = 100
total_size = batch_size * max_batches
normed_layer_bottom = np.linspace(-1, 1, n_layers + 1)[:n_layers]


# get a "big_batch"
def get_data(how="direct", batch_size=batch_size, max_batches=max_batches, dataset=None):
    if dataset is None:
        print("Creating dataset")
        dataset = PointCloudDataset(
            file_path=cfg.dataset_path, bs=batch_size, quantized_pos=cfg.quantized_pos
        )

    print("Reading data")
    if how == "dataloader":
        from torch.utils.data import DataLoader

        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        for i, batch in enumerate(dataloader):
            print(f"{i/max_batches:.1%}", end="\r")
            if i == 0:
                big_batch = {key: [batch[key]] for key in batch}
            else:
                for key in big_batch:
                    big_batch[key].append(batch[key])
            if i >= max_batches - 1:
                break
        print()
    elif how == "dataset":
        start_idxs = np.linspace(0, len(dataset) - batch_size - 1, max_batches).astype(
            int
        )
        big_batch = {"event": [], "energy": [], "points": []}
        for i in start_idxs:
            print(f"{i/start_idxs[-1]:.1%}", end="\r")
            data = dataset[i]
            for key in data:
                big_batch[key].append(torch.Tensor(data[key][np.newaxis, ...]))
        print()
    elif how == "direct":
        energy, events = read_raw_regaxes(cfg, total_size=total_size)
        dataset.normalize_xyze(events)

        big_batch = {
            "event": torch.Tensor(events[np.newaxis, ...]),
            "energy": torch.Tensor(energy[np.newaxis, ...]),
        }
    print("Done")
    if how in ["dataloader", "dataset"]:
        big_batch["energy"] = torch.cat(big_batch["energy"]), dim=1)
        big_batch["points"] = torch.cat(big_batch["points"])
        full_size = torch.zeros(
            (1, batch_size * max_batches, int(big_batch["points"].max().item()), 4)
        )
        for i, event in enumerate(big_batch["event"]):
            full_size[
                0, batch_size * i : batch_size * (i + 1), : event.shape[-2], :
            ] = event[0]
        big_batch["event"] = full_size

    return big_batch


# set up axes
def get_axes(layers_of_intrest=layers_of_intrest):
    plt.close()
    plt.ion()
    double_row = len(layers_of_intrest) > 10
    if double_row:
        row_len = np.ceil(len(layers_of_intrest) / 2).astype(int)
        fig, ax_arr = plt.subplots(2, row_len, sharex=True, sharey=True, figsize=(12, 6))
        ordered_ax = ax_arr.T.flatten()
    else:
        row_len = len(layers_of_intrest)
        fig, ordered_ax = plt.subplots(1, row_len, sharex=True, sharey=True, figsize=(12, 3))

    for i, layer in enumerate(layers_of_intrest):
        ordered_ax[i].set_title(f"layer {layer}")
    return fig, ordered_ax


def plot_batch(
    batch,
    ordered_ax,
    layers_of_intrest=layers_of_intrest,
    normed_layer_bottom=normed_layer_bottom,
    value_name='n_points',
    **scatter_kwargs
):
    layer_colours = plt.cm.viridis(np.linspace(0, 1, len(layers_of_intrest)))
    cell_thickness_global = 0.5 * (normed_layer_bottom[1] - normed_layer_bottom[0])
    #wish.hist_batch_backbone(
    wish.scatter_batch_backbone(
    #wish.plot_batch_backbone_layer_correlations(
        ordered_ax,
        layer_colours,
        batch,
        normed_layer_bottom,
        cell_thickness_global,
        layers_of_intrest,
        value_name=value_name,
        **scatter_kwargs
    )
    ordered_ax[0].set_ylabel(value_name.replace("_", " ").capitalize())
    ordered_ax[int(len(ordered_ax) / 2)].set_xlabel("Incidient energy")
    plt.tight_layout()


#direct_batch = get_data("direct")
import h5py
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
sample_path = os.path.join(this_dir, "../../../point-cloud-diffusion-data/even_batch_10k.h5")
batch_file = h5py.File(sample_path)
direct_batch = {k: torch.Tensor(v[:]) for k, v in batch_file.items()}
#fig, ordered_ax = get_axes()
#plot_batch(direct_batch, ordered_ax)
