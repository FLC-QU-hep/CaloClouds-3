import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from pointcloud.data.read_write import read_raw_regaxes, regularise_event_axes


def get_cfg():
    from config import Configs

    # set a test config
    cfg = Configs()
    # no logging for tests, as we would need a comet key
    cfg.log_comet = False
    # cfg.dataset_path = '/mnt' + cfg.dataset_path
    #cfg.dataset_path = "/home/henry/training/local_data/varied_20_sample.hdf5"
    cfg.max_iters = 2
    cfg.workers = 1
    try:
        os.mkdir("temp_logs")
    except FileExistsError:
        pass
    cfg.device = "cpu"
    return cfg


def plot_points(energies, xs, ys, zs, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
    normed_energies = (
        7 * (energies - energies.min()) / (energies.max() - energies.min())
    )**2
    ax.scatter(xs, ys, zs, c=energies, alpha=0.6, s=normed_energies, cmap="viridis")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    return ax


def plot_data_iteration(event, energy, event_n=0):
    event = regularise_event_axes(event)
    energy = np.squeeze(energy)
    # make a 3d plot of the x, y, z points in the events array
    energies = event[event_n, :, 3]
    mask = energies > 0
    energies = energies[mask]
    xs = event[event_n, mask, 0]
    ys = event[event_n, mask, 1]
    zs = event[event_n, mask, 2]
    ax = plot_points(energies, xs, ys, zs)
    # write a title
    observed_energy = energies.sum()
    n_points = len(energies)
    energy_in = energy[event_n]
    ax.set_title(
        f"Evt: {event_n}, $n_{{pts}}$: {n_points}, $E_{{in}}$: {energy_in:.2f}, $E_{{vis}}$: {observed_energy:.2f}"
    )


def plot_model_energy(model_name, model, energy_in):
    xs, ys, zs, energies = model.inference(energy_in)
    energies = np.array(energies)/1000
    ax = plot_points(energies, xs, ys, zs)
    # set the x and z limits from -100 to 100
    ax.set_xlim(-1, 1)
    ax.set_zlim(-1, 1)

    # write a title
    observed_energy = np.sum(energies)
    n_points = len(xs)
    ax.set_title(
        f"{model_name}, $n_{{pts}}$: {n_points}, $E_{{in}}$: {energy_in:.2f}, $E_{{vis}}$: {observed_energy:.2f}"
    )


def save_fig(file_start, version=1):
    this_script_path = os.path.dirname(os.path.realpath(__file__))
    if isinstance(file_start, int):
        file_start = f"event{file_start}"
    file_name = os.path.join(
        this_script_path,
        "../..",
        "point-cloud-diffusion-images",
        "per-event",
        f"{file_start}_rawPts_v{version}.png",
    )
    file_dir = os.path.dirname(file_name)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    plt.savefig(file_name)

this_dir = os.path.dirname(os.path.abspath(__file__))
sample_path = os.path.join(this_dir, "../../../point-cloud-diffusion-data/even_batch_10k.h5")

def main(event_n=None):
    plt.ion()
    cfg = get_cfg()
    n_events = 10
    energies, events = read_raw_regaxes(cfg, total_size=n_events)
    #batch_file = h5py.File(sample_path)
    #energies = batch_file["energy"][:]
    #events = batch_file["event"][:]
    if event_n is None:
        for i in range(n_events):
            event = events[event_n]
            energy = energies[event_n]
            plot_data_iteration(event, energy, event_n=event_n)
            version = "raw"
            save_fig(event_n, version=version)
    else:
        event = events[event_n]
        energy = energy[event_n]
        plot_data_iteration(event, energy, event_n=event_n)
        version = "raw"
        save_fig(event_n, version=version)
