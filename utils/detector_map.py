from tqdm import tqdm
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from scipy.stats import binned_statistic

import utils.metrics as metrics



half_cell_size = 5.0883331298828125 / 2
cell_thickness = 0.5250244140625

layer_bottom_pos = np.array(
    [
        1811.34020996,
        1814.46508789,
        1823.81005859,
        1826.93505859,
        1836.2800293,
        1839.4050293,
        1848.75,
        1851.875,
        1861.2199707,
        1864.3449707,
        1873.68994141,
        1876.81494141,
        1886.16003418,
        1889.28503418,
        1898.63000488,
        1901.75500488,
        1911.09997559,
        1914.22497559,
        1923.56994629,
        1926.69494629,
        1938.14001465,
        1943.36499023,
        1954.81005859,
        1960.03503418,
        1971.47998047,
        1976.70495605,
        1988.15002441,
        1993.375,
        2004.81994629,
        2010.04504395,
    ]
)

Ymin = 1811
Xmin = -200
Xmax = 200
# Xmin = -260
# Xmax = 340

Zmin = -160
Zmax = 240
# Zmin = -300
# Zmax = 300


def create_map(X=None, Y=None, Z=None, dm=1):
    """
    X, Y, Z: np.array
        ILD coordinates of sencors hited with muons
    dm: int (1, 2, 3, 4, 5) dimension split multiplicity
    """

    if X is None:
        X, Y, Z, E = confine_to_box(*load_muon_map())

    offset = half_cell_size * 2 / (dm)

    layers = []
    for l in tqdm(range(len(layer_bottom_pos))):  # loop over layers
        idx = np.where(
            (Y <= (layer_bottom_pos[l] + cell_thickness * 1.5))
            & (Y >= layer_bottom_pos[l] - cell_thickness / 2)
        )

        xedges = np.array([])
        zedges = np.array([])

        unique_X = np.unique(X[idx])
        unique_Z = np.unique(Z[idx])

        xedges = np.append(xedges, unique_X[0] - half_cell_size)
        xedges = np.append(xedges, unique_X[0] + half_cell_size)

        for i in range(len(unique_X) - 1):  # loop over X coordinate cell centers
            if abs(unique_X[i] - unique_X[i + 1]) > half_cell_size * 1.9:
                xedges = np.append(xedges, unique_X[i + 1] - half_cell_size)
                xedges = np.append(xedges, unique_X[i + 1] + half_cell_size)

                for of_m in range(dm):
                    xedges = np.append(
                        xedges, unique_X[i + 1] - half_cell_size + offset * of_m
                    )  # for higher granularity

        for z in unique_Z:  # loop over Z coordinate cell centers
            zedges = np.append(zedges, z - half_cell_size)
            zedges = np.append(zedges, z + half_cell_size)

            for of_m in range(dm):
                zedges = np.append(
                    zedges, z - half_cell_size + offset * of_m
                )  # for higher granularity

        zedges = np.unique(zedges)
        xedges = np.unique(xedges)

        xedges = [
            xedges[i]
            for i in range(len(xedges) - 1)
            if abs(xedges[i] - xedges[i + 1]) > 1e-3
        ] + [xedges[-1]]
        zedges = [
            zedges[i]
            for i in range(len(zedges) - 1)
            if abs(zedges[i] - zedges[i + 1]) > 1e-3
        ] + [zedges[-1]]

        H, xedges, zedges = np.histogram2d(X[idx], Z[idx], bins=(xedges, zedges))
        layers.append({"xedges": xedges, "zedges": zedges, "grid": H})

    return layers, offset


def get_projections(
    showers, MAP, layer_bottom_pos, return_cell_point_cloud=False, max_num_hits=6000
):
    events = []
    for shower in tqdm(showers):
        layers = []

        x_coord, y_coord, z_coord, e_coord = shower

        for l in range(len(MAP)):
            idx = np.where(
                (y_coord <= (layer_bottom_pos[l] + 1))
                & (y_coord >= layer_bottom_pos[l] - 0.5)
            )

            xedges = MAP[l]["xedges"]
            zedges = MAP[l]["zedges"]
            H_base = MAP[l]["grid"]

            H, xedges, zedges = np.histogram2d(
                x_coord[idx], z_coord[idx], bins=(xedges, zedges), weights=e_coord[idx]
            )
            if not cfg.include_artifacts:
                H[H_base == 0] = 0

            layers.append(H)

        events.append(layers)

    if not return_cell_point_cloud:
        return events

    else:
        cell_point_clouds = []
        for event in tqdm(events):
            point_cloud = []
            for l, layer in enumerate(event):
                xedges = MAP[l]["xedges"]
                zedges = MAP[l]["zedges"]

                x_indx, z_indx = np.where(layer > 0)

                cell_energy = layer[layer > 0]
                cell_coordinate_x = xedges[x_indx] + half_cell_size
                cell_coordinate_y = np.repeat(
                    layer_bottom_pos[l] + cell_thickness / 2, len(x_indx)
                )
                cell_coordinate_z = zedges[z_indx] + half_cell_size

                point_cloud.append(
                    [
                        cell_coordinate_x,
                        cell_coordinate_y,
                        cell_coordinate_z,
                        cell_energy,
                    ]
                )

            point_cloud = np.concatenate(point_cloud, axis=1)
            zeros_to_concat = np.zeros((4, max_num_hits - len(point_cloud[0])))
            point_cloud = np.concatenate((point_cloud, zeros_to_concat), axis=1)

            cell_point_clouds.append([point_cloud])

        cell_point_clouds = np.vstack(cell_point_clouds)

        return events, cell_point_clouds


def load_muon_map(folder="10-90GeV_x36_grid_regular_524k"):
    # start by checking which dataset we are using
    from configs import Configs
    cfg = Configs()
    dataset_filebase = os.path.basename(cfg.dataset_path).rsplit(".", 1)[0]
    if folder is None:
        if dataset_filebase in ["10-90GeV_x36_grid_regular_524k",
                                "10-90GeV_x36_grid_regular_524k_float32"]:
            folder = "10-90GeV_x36_grid_regular_524k"
        else:
            raise NotImplementedError(f"Cannot recognise the dataset at {cfg.dataset_path}")
    # use a relative path
    this_dir = os.path.dirname(__file__)
    data_dir = os.path.join(this_dir, "../metadata/muon_map/", folder)

    X = np.load(data_dir + "/X.npy")
    Z = np.load(data_dir + "/Z.npy")
    Y = np.load(data_dir + "/Y.npy")
    E = np.load(data_dir + "/E.npy")

    return X, Y, Z, E


def confine_to_box( X, Y, Z, E):
    inbox_idx = np.where(
        (Y > Ymin) & (X < Xmax) & (X > Xmin) & (Z < Zmax) & (Z > Zmin)
    )[0]
    # inbox_idx = np.where(Y > Ymin)[0]

    X = X[inbox_idx]
    Z = Z[inbox_idx]
    Y = Y[inbox_idx]
    E = E[inbox_idx]
    return X, Y, Z, E



