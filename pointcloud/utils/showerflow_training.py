# # Showerflow
#
# utilty functions for scripts/Showerflow.py

import pandas as pd
import torch
import numpy as np
import os

from pointcloud.utils.metadata import Metadata
from pointcloud.data.read_write import get_n_events, read_raw_regaxes
from pointcloud.utils.detector_map import floors_ceilings


def get_incident_npts_visible(
    configs, showerflow_dir, redo=False, local_batch_size=10_000
):
    meta = Metadata(configs)
    pointsE_path = os.path.join(showerflow_dir, "pointsE.npz")
    if os.path.exists(pointsE_path) and not redo:
        print("Using precaluclated energies and counts", flush=True)
    else:
        print("Recalculating energies and counts", flush=True)
        n_events = sum(get_n_events(configs.dataset_path, configs.n_dataset_files))
        floors, ceilings = floors_ceilings(
            meta.layer_bottom_pos_raw, meta.cell_thickness_raw
        )
        energy = np.zeros(n_events)
        num_points = np.zeros(n_events)
        visible_energy = np.zeros(n_events)
        for start_idx in range(0, n_events, local_batch_size):
            print(f"{start_idx/n_events:.0%}", end="\r")
            my_slice = slice(start_idx, start_idx + local_batch_size)
            energies_batch, events_batch = read_raw_regaxes(
                configs, pick_events=my_slice
            )
            energy[my_slice] = energies_batch
            num_points[my_slice] = (events_batch[:, :, 3] > 0).sum(axis=1)
            visible_energy[my_slice] = (events_batch[:, :, 3]).sum(axis=1)
        np.savez(
            pointsE_path,
            energy=energy,
            num_points=num_points,
            visible_energy=visible_energy,
        )
        print(f"Energy goes from {visible_energy.min()}, to {visible_energy.max()}")
        print(f"Num points goes from {num_points.min()}, to {num_points.max()}")
    return pointsE_path


def get_clusters_per_layer(
    configs, showerflow_dir, redo=False, local_batch_size=10_000
):
    clusters_per_layer_path = os.path.join(showerflow_dir, "clusters_per_layer.npz")
    if os.path.exists(clusters_per_layer_path) and not redo:
        print("Using precaluclated clusters per layer", flush=True)
    else:
        print("Recalculating clusters per layer", flush=True)  
        floors, ceilings = floors_ceilings(
            meta.layer_bottom_pos_raw, meta.cell_thickness_raw
        )
        clusters_per_layer = np.zeros((n_events, len(floors)))
        for start_idx in range(0, n_events, local_batch_size):
            print(f"{start_idx/n_events:.0%}", end="\r")
            my_slice = slice(start_idx, start_idx + local_batch_size)
            _, events_batch = read_raw_regaxes(configs, pick_events=my_slice)
            mask = events_batch[:, :, 3] > 0
            clusters_here = [
                ((events_batch[:, :, 1] < c) & (events_batch[:, :, 1] > f) & mask).sum(
                    axis=1
                )
                for f, c in zip(floors, ceilings)
            ]
            clusters_per_layer[my_slice] = np.vstack(clusters_here).T
        rescales = clusters_per_layer / clusters_per_layer.max(axis=1)[:, np.newaxis]
        np.savez(
            clusters_per_layer_path,
            clusters_per_layer=clusters_per_layer,
            rescaled_clusters_per_layer=rescales,
        )
        assert np.all(~np.isnan(clusters_per_layer))
    return clusters_per_layer_path


def get_energy_per_layer(configs, showerflow_dir, redo=False, local_batch_size=10_000):
    energy_per_layer_path = os.path.join(showerflow_dir, "energy_per_layer.npz")
    if os.path.exists(energy_per_layer_path) and not redo:
        print("Using precaluclated energy per layer", flush=True)
    else:
        print("Recalculating energy per layer", flush=True)
        floors, ceilings = floors_ceilings(
            meta.layer_bottom_pos_raw, meta.cell_thickness_raw
        )
        energy_per_layer = np.zeros((n_events, len(floors)))
        for start_idx in range(0, n_events, local_batch_size):
            print(f"{start_idx/n_events:.0%}", end="\r")
            my_slice = slice(start_idx, start_idx + local_batch_size)
            _, events_batch = read_raw_regaxes(configs, pick_events=my_slice)
            energy_here = [
                (
                    events_batch[..., 3]
                    * (events_batch[:, :, 1] < c)
                    * (events_batch[:, :, 1] > f)
                ).sum(axis=1)
                for f, c in zip(floors, ceilings)
            ]
            energy_per_layer[my_slice] = np.vstack(energy_here).T
        rescaled = energy_per_layer / energy_per_layer.max(axis=1)[:, np.newaxis]
        np.savez(
            energy_per_layer_path,
            energy_per_layer=energy_per_layer,
            rescaled_energy_per_layer=rescaled,
        )
        assert np.all(~np.isnan(energy_per_layer))
    return energy_per_layer_path


def cog_from_kinematics(x, y, z, e):
    sum_e = e.sum(axis=1)
    return (
        np.sum((x * e), axis=1) / sum_e,
        np.sum((y * e), axis=1) / sum_e,
        np.sum((z * e), axis=1) / sum_e,
    )


def get_cog(configs, showerflow_dir, redo=False, local_batch_size=10_000):
    cog_path = os.path.join(showerflow_dir, "cog.npy")
    if os.path.exists(cog_path) and not redo:
        print("Using precaluclated cog", flush=True)
        my_slice = slice(0, local_batch_size)
        _, events_batch = read_raw_regaxes(configs, pick_events=my_slice)
        cog_batch = cog_from_kinematics(
            events_batch[..., 0],
            events_batch[..., 1],
            events_batch[..., 2],
            events_batch[..., 3],
        )
    else:
        print("Recalculating cog", flush=True)
        cog = np.zeros((n_events, 3))
        for start_idx in range(0, n_events, local_batch_size):
            print(f"{start_idx/n_events:.0%}", end="\r")
            my_slice = slice(start_idx, start_idx + local_batch_size)
            _, events_batch = read_raw_regaxes(configs, pick_events=my_slice)
            cog_batch = cog_from_kinematics(
                events_batch[..., 0],
                events_batch[..., 1],
                events_batch[..., 2],
                events_batch[..., 3],
            )
            cog[my_slice] = np.vstack(cog_here).T
        np.save(cog_path, cog)
        assert np.all(~np.isnan(cog))
    return cog_path, cog_batch


def train_ds_function_factory(
    pointsE_path, cog_path, clusters_per_layer_path, energy_per_layer_path, configs
):
    meta = Metadata(configs)
    device = configs.device

    def make_train_ds(start_idx, end_idx):
        my_slice = slice(start_idx, end_idx)
        df = pd.DataFrame([])
        # mem-map the files to avoid loading all data
        pointE = np.load(pointsE_path, mmap_mode="r")
        df["energy"] = (
            pointE["energy"][my_slice].copy().reshape(-1) / meta.incident_rescale
        )

        df["num_points"] = pointE["num_points"][my_slice].copy() / meta.n_pts_rescale
        df["visible_energy"] = (
            pointE["visible_energy"][my_slice].copy() / meta.vis_eng_rescale
        )

        normed_cog = np.load(cog_path, mmap_mode="r")[my_slice].copy()
        normed_cog = (normed_cog - meta.mean_cog) / meta.std_cog

        df["cog_x"] = normed_cog[:, 0]
        df["cog_y"] = normed_cog[:, 1]
        df["cog_z"] = normed_cog[:, 2]

        clusters = np.load(clusters_per_layer_path, mmap_mode="r")
        df["clusters_per_layer"] = clusters["rescaled_clusters_per_layer"][
            my_slice
        ].tolist()
        energies = np.load(energy_per_layer_path, mmap_mode="r")
        df["e_per_layer"] = energies["rescaled_energy_per_layer"][my_slice].tolist()

        for series_name, series in df.items():
            series = np.vstack(series.to_numpy())
            assert np.all(~np.isnan(series)), series_name

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(df.energy.values).to(device),
            torch.tensor(df.num_points.values).to(device),
            torch.tensor(df.visible_energy.values).to(device),
            torch.tensor(df.cog_x.values).to(device),
            torch.tensor(df.cog_y.values).to(device),
            torch.tensor(df.cog_z.values).to(device),
            torch.tensor(df.clusters_per_layer).to(device),
            torch.tensor(df.e_per_layer).to(device),
        )
        return dataset

    return make_train_ds
