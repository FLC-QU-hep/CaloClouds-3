from tqdm import tqdm
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from scipy.stats import binned_statistic

from .metadata import Metadata
from configs import Configs


def create_map(
    X=None,
    Y=None,
    Z=None,
    layer_bottom_pos=None,
    half_cell_size=None,
    cell_thickness=None,
    dm=1,
    configs=Configs(),
):
    """
    Using metadata about the detector, create a map of the cells in the detector.

    Parameters
    ----------
    X, Y, Z : np.array
        ILD coordinates of sensors as shown by muon hits
    layer_bottom_pos : np.array
        Array of the bottom positions of the layers.
    half_cell_size : float
        Half the size of the cells in the detector,
        perpendicular to the radial direction
    cell_thickness : float
        Thickness of the cells in the detector, in the radial direction
    dm : int
        dimension split multiplicity
        can be (1, 2, 3, 4, 5)
    configs : Configs (optional)
        Configs object containing the configuration of the dataset
        Only used if X, Y, Z or layer_bottom_pos are not given.
        A default Configs object is created if not given.

    Returns
    -------
    layers : list
        list of dictionaries, each containing the grid of cells for a layer
    offset : float
        The cell size divided by the dimension split multiplicity
    """

    if None in [X, Y, Z, layer_bottom_pos, half_cell_size, cell_thickness]:
        metadata = Metadata(configs)
        layer_bottom_pos = metadata.layer_bottom_pos
        half_cell_size = metadata.half_cell_size
        cell_thickness = metadata.cell_thickness
        X, Y, Z, E = confine_to_box(*metadata.load_muon_map(), metadata=metadata)

    offset = half_cell_size * 2 / (dm)

    layers = []
    for l in range(len(layer_bottom_pos)):  # loop over layers
        # layers are well seperated, so take a 0.5 buffer either side
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


def normalise_map(MAP):
    """
    Normalise the map of cells in the detector so that it
    has cells for the output of a dataset whose x, y, z coordinates
    are in the range [-1, 1]

    Parameters
    ----------
    MAP : list
        list of dictionaries, each containing the grid of cells for a layer
        As returned by create_map

    Returns
    -------
    MAP : list
        list of dictionaries, each containing the grid of cells for a layer
        As returned by create_map, but normalised
    """
    for l in range(len(MAP)):
        xedges = MAP[l]["xedges"]
        zedges = MAP[l]["zedges"]
        xedges = (xedges - xedges[0]) / (xedges[-1] - xedges[0]) * 2 - 1
        zedges = (zedges - zedges[0]) / (zedges[-1] - zedges[0]) * 2 - 1
    return MAP


def split_to_layers(points, layer_bottom_pos):
    """
    Yield points by their layer

    Parameters
    ----------
    points : np.array (N, 4)
        2D array containing hits of the points,
        in coordinates (x, y, z, energy)
    layer_bottom_pos : np.array
        Array of the bottom positions of the layers.

    Yields
    ------
    np.array
        2D array containing hits of the points on this layer,
        in coordinates (x, y, z, energy)
    """
    y_coord = points[:, 1]
    for bottom in layer_bottom_pos:
        idx = np.where((y_coord <= (bottom + 1)) & (y_coord >= bottom - 0.5))
        yield points[idx]


def points_to_cells(points, MAP, layer_bottom_pos=None, include_artifacts=False):
    """
    Project the generated pointss onto the detector map.

    Parameters
    ----------
    points : np.array (N, 4)
        2D array containing hits of the points,
        in coordinates (x, y, z, energy)
    MAP : list
        list of dictionaries, each containing the grid of cells for a layer
        As returned by create_map
    layer_bottom_pos : np.array
        Array of the bottom positions of the layers.
    include_artifacts : bool
        Should the projection include hits in cells that don't have
        sensitive material?
        (default is False)

    Returns
    -------
    layers : list
        list of 2D arrays, each containing the energy deposited in each cell of the layer
        with the arangement specified by MAP
        in the style of a histogram

    """
    layers = []

    for l, points in enumerate(split_to_layers(points, layer_bottom_pos)):
        x_coord, y_coord, z_coord, e_coord = points.T

        layers.append(
            layer_to_cells(x_coord, z_coord, e_coord, MAP[l], include_artifacts)
        )

    return layers


def layer_to_cells(x_coord, z_coord, e_coord, MAP_layer, include_artifacts=False):
    """
    Project this layer of points onto the detector map.

    Parameters
    ----------
    x_coord : np.array
        x coordinates of the points in this layer
    z_coord : np.array
        z coordinates of the points in this layer
    e_coord : np.array
        energy of the points in this layer
    MAP_layer : dict
        dictionary containing the grid of cells for a layer
        As returned by create_map
    include_artifacts : bool
        Should the projection include hits in cells that don't have
        sensitive material?
        (default is False)

    Returns
    -------
    layers : list
        list of 2D arrays, each containing the energy deposited in each cell of the layer
        with the arangement specified by MAP
        in the style of a histogram

    """
    x_coord, y_coord, z_coord, e_coord = points.T

    xedges = MAP_layer["xedges"]
    zedges = MAP_layer["zedges"]
    H_base = MAP_layer["grid"]

    H, xedges, zedges = np.histogram2d(
        x_coord, z_coord, bins=(xedges, zedges), weights=e_coord
    )
    if not include_artifacts:
        H[H_base == 0] = 0

    return H


def cells_to_points(
    layers, MAP, layer_bottom_pos, half_cell_size, cell_thickness, length_to_pad=None
):
    """
    Given the energy deposited in each cell of the detector, convert
    back to a list of points, discreetised by the cell layout.

    Parameters
    ----------
    layers : list
        list of 2D arrays, each containing the energy deposited in each cell of the layer
        with the arangement specified by MAP
        in the style of a histogram
    MAP : list
        list of dictionaries, each containing the grid of cells for a layer
        As returned by create_map
    layer_bottom_pos : np.array
        Array of the bottom positions of the layers.
    half_cell_size : float
        Half the size of the cells in the detector,
        perpendicular to the radial direction
    cell_thickness : float
        Thickness of the cells in the detector, in the radial direction
    length_to_pad : int (optional)
        Required length of the output array. If the number of hits is less than
        this, the output will be padded with zeros.
        If not given, the output will be the length of the number of non-zero hits.

    Returns
    -------
    points : np.array
        2D array containing hits of the points,
        in coordinates (x, y, z, energy)
    """

    points = []
    for l, layer in enumerate(layers):
        xedges = MAP[l]["xedges"]
        zedges = MAP[l]["zedges"]

        x_indx, z_indx = np.where(layer > 0)

        cell_energy = layer[layer > 0]
        cell_coordinate_x = xedges[x_indx] + half_cell_size
        cell_coordinate_y = np.repeat(
            layer_bottom_pos[l] + cell_thickness / 2, len(x_indx)
        )
        cell_coordinate_z = zedges[z_indx] + half_cell_size

        points.append(
            [
                cell_coordinate_x,
                cell_coordinate_y,
                cell_coordinate_z,
                cell_energy,
            ]
        )

    points = np.concatenate(points, axis=1)
    if length_to_pad is not None:
        zeros_to_concat = np.zeros((4, length_to_pad - len(points[0])))
        points = np.concatenate((points, zeros_to_concat), axis=1)

    return points


def get_projections(
    showers,
    MAP,
    layer_bottom_pos=None,
    return_cell_point_cloud=False,
    max_num_hits=None,
    configs=Configs(),
):
    """
    Project all events onto the detector map, and optionally
    convert back to a discreetised point cloud.

    Parameters
    ----------
    showers : iterable
        iterable of 2D arrays, each containing the hits of a shower
        in coordinates (x, y, z, energy)
    MAP : list
        list of dictionaries, each containing the grid of cells for a layer
        As returned by create_map
    layer_bottom_pos : np.array (optional)
        Array of the bottom positions of the layers.
        If not given, the config will be used to generate a Metadata object
        and the layer_bottom_pos will be taken from there.
    return_cell_point_cloud : bool
        Return a second variable containing the deiscreetised point cloud
    max_num_hits : int (optional)
        Padding length for the output point cloud,
        must be greater than the number of active cells in the
        largest shower.
        If not given the value will be taken from the max_points
        attribute of the Configs object.
    configs : Configs (optional)
        Configs object containing the configuration of the dataset
        Only used if layer_bottom_pos is not given.
        A default Configs object is created if not given.

    Returns
    -------
    events : list
        list of 2D arrays, each containing the energy deposited in each cell of the layer
        with the arangement specified by MAP
        in the style of a histogram
    cell_point_clouds : np.array (optional)
        3D array containing hits of the points,
        in coordinates (x, y, z, energy)
    """
    if max_num_hits is None:
        max_num_hits = configs.max_points

    if layer_bottom_pos is None:
        metadata = Metadata(configs)
        layer_bottom_pos = metadata.layer_bottom_pos

    events = []
    for shower in tqdm(showers):
        layers = points_to_cells(shower, MAP, layer_bottom_pos, cfg.include_artifacts)
        events.append(layers)

    if not return_cell_point_cloud:
        return events

    cell_point_clouds = []
    for event in tqdm(events):
        point_cloud = cells_to_points(event, MAP, layer_bottom_pos, max_num_hits)
        cell_point_clouds.append([point_cloud])

    cell_point_clouds = np.vstack(cell_point_clouds)

    return events, cell_point_clouds


def confine_to_box(X, Y, Z, E, metadata=Metadata()):
    inbox_idx = np.where(
        (Y > metadata.Ymin)
        & (X < metadata.Xmax)
        & (X > metadata.Xmin)
        & (Z < metadata.Zmax)
        & (Z > metadata.Zmin)
    )[0]
    # inbox_idx = np.where(Y > Ymin)[0]

    X = X[inbox_idx]
    Z = Z[inbox_idx]
    Y = Y[inbox_idx]
    E = E[inbox_idx]
    return X, Y, Z, E
