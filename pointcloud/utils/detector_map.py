from tqdm import tqdm
import numpy as np

from ..configs import Configs

from .metadata import Metadata
from ..data.read_write import local_to_global


def get_offset(half_cell_size_global, dm):
    """
    Get the offset for the cell map.

    Parameters
    ----------
    half_cell_size_global : float
        Half the size of the cells in the detector,
        perpendicular to the radial direction
    dm : int
        dimension split multiplicity used for the cell map
        can be (1, 2, 3, 4, 5)

    Returns
    -------
    offset : float
        The cell size divided by the dimension split multiplicity
    """
    return half_cell_size_global * 2 / dm


def create_map(
    X=None,
    Y=None,
    Z=None,
    layer_bottom_pos=None,
    half_cell_size_global=None,
    cell_thickness_global=None,
    dm=1,
    configs=Configs(),
):
    """
    Using metadata about the detector, create a map of the cells in the detector.
    Will be in global coordinates i.e. detector coordinates.

    Parameters
    ----------
    X, Y, Z : np.array
        ILD coordinates of sensors as shown by muon hits
    layer_bottom_pos : np.array
        Array of the bottom positions of the layers.
    half_cell_size_global : float
        Half the size of the cells in the detector,
        perpendicular to the radial direction
    cell_thickness_global : float
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

    metadata = Metadata(configs)
    if not all(hasattr(item, "__iter__") for item in [X, Y, Z]):
        X, Y, Z, E = confine_to_box(*metadata.load_muon_map(), metadata=metadata)

    if layer_bottom_pos is None:
        layer_bottom_pos = metadata.layer_bottom_pos_global
    if half_cell_size_global is None:
        half_cell_size_global = metadata.half_cell_size_global
    if cell_thickness_global is None:
        cell_thickness_global = metadata.cell_thickness_global

    offset = get_offset(half_cell_size_global, dm)

    layers = []
    for layer_n in range(len(layer_bottom_pos)):  # loop over layers
        # layers are well seperated, so take a 0.5 buffer either side
        idx = np.where(
            (Y <= (layer_bottom_pos[layer_n] + cell_thickness_global * 1.5))
            & (Y >= layer_bottom_pos[layer_n] - cell_thickness_global / 2)
        )

        xedges = np.array([])
        zedges = np.array([])

        unique_X = np.unique(X[idx])
        unique_Z = np.unique(Z[idx])

        xedges = np.append(xedges, unique_X[0] - half_cell_size_global)
        xedges = np.append(xedges, unique_X[0] + half_cell_size_global)

        for i in range(len(unique_X) - 1):  # loop over X coordinate cell centers
            if abs(unique_X[i] - unique_X[i + 1]) > half_cell_size_global * 1.9:
                xedges = np.append(xedges, unique_X[i + 1] - half_cell_size_global)
                xedges = np.append(xedges, unique_X[i + 1] + half_cell_size_global)

                for of_m in range(dm):
                    xedges = np.append(
                        xedges, unique_X[i + 1] - half_cell_size_global + offset * of_m
                    )  # for higher granularity

        for z in unique_Z:  # loop over Z coordinate cell centers
            zedges = np.append(zedges, z - half_cell_size_global)
            zedges = np.append(zedges, z + half_cell_size_global)

            for of_m in range(dm):
                zedges = np.append(
                    zedges, z - half_cell_size_global + offset * of_m
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
    for layer in range(len(MAP)):
        xedges = MAP[layer]["xedges"]
        zedges = MAP[layer]["zedges"]
        xedges = (xedges - xedges[0]) / (xedges[-1] - xedges[0]) * 2 - 1
        zedges = (zedges - zedges[0]) / (zedges[-1] - zedges[0]) * 2 - 1
        MAP[layer]["xedges"] = xedges
        MAP[layer]["zedges"] = zedges
    return MAP


def floors_ceilings(layer_bottom_pos, cell_thickness_global, percent_buffer=0.5):
    """
    Find top and bottom coordinates for the layers in the detector.

    Parameters
    ----------
    layer_bottom_pos : np.array
        Array of the bottom positions of the layers.
    cell_thickness_global : float
        Thickness of the cells in the detector, in the radial direction
    percent_buffer : float
        Percentage beyond the thickness of the cell to include hits
        in the layer. Won't extent the layer beyond the bottom of
        the next layer.
        (default is 0.5)

    Returns
    -------
    layer_floors : np.array
        Array of the bottom positions of the layers.
    layer_ceilings : np.array
        Array of the top positions of the layers.
    """
    # naive calculation of the layer floors and ceilings
    layer_floors = layer_bottom_pos - percent_buffer * cell_thickness_global
    layer_ceilings = layer_bottom_pos + (1 + percent_buffer) * cell_thickness_global
    # Unless the cells are thicker than the layers, (which they shouldn't be)
    # the true ceiling for each layer is the bottom of the layer plus the thickness
    true_ceilings = np.minimum(
        (layer_bottom_pos + cell_thickness_global)[:-1], layer_bottom_pos[1:]
    )
    # we dont' want any extention to cross the midpoint between the true
    # ceiling and the bottom of the next layer
    mid_points = 0.5 * (true_ceilings + layer_bottom_pos[1:])
    # now enforce not crossing those midpoints
    layer_floors[1:] = np.maximum(layer_floors[1:], mid_points)
    layer_ceilings[:-1] = np.minimum(layer_ceilings[:-1], mid_points)
    return layer_floors, layer_ceilings


def split_to_layers(
    points, layer_bottom_pos, cell_thickness_global, percent_buffer=0.5, layer_axis=2
):
    """
    Yield points by their layer

    Parameters
    ----------
    points : np.array (N, 4)
        2D array containing hits of the points,
        in coordinates (x, y, z, energy)
    layer_bottom_pos : np.array
        Array of the bottom positions of the layers.
    cell_thickness_global : float
        Thickness of the cells in the detector, in the radial direction
    percent_buffer : float
        Percentage beyond the thickness of the cell to include hits
        in the layer. Won't extent the layer beyond the bottom of
        the next layer.
        (default is 0.5)
    layer_axis : int
        The index of the last axis that corrisponds to
        the direction crossing through the layers.
        (default is 2)

    Yields
    ------
    np.array
        2D array containing hits of the points on this layer,
        in coordinates (x, y, z, energy)
    """
    z_coord = points[:, layer_axis]

    layer_floors, layer_ceilings = floors_ceilings(
        layer_bottom_pos, cell_thickness_global, percent_buffer
    )
    for floor, ceiling in zip(layer_floors, layer_ceilings):
        mask = (z_coord >= floor) & (z_coord < ceiling)
        yield points[mask]


def points_to_cells(
    points, MAP, layer_bottom_pos, cell_thickness_global, include_artifacts=False
):
    """
    Project the generated pointss onto the detector map.

    Parameters
    ----------
    points : np.array (N, 4)
        2D array containing hits of the points,
        in global coordinates (x, y, z, energy)
    MAP : list
        list of dictionaries, each containing the grid of cells for a layer
        In global coordinates
        As returned by create_map
    layer_bottom_pos : np.array
        Array of the bottom positions of the layers.
    cell_thickness_global : float
        Thickness of the cells in the detector, in the radial direction
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

    for layer, points in enumerate(
        split_to_layers(points, layer_bottom_pos, cell_thickness_global, layer_axis=1)
    ):
        x_coord, y_coord, z_coord, e_coord = points.T

        layers.append(
            layer_to_cells(x_coord, z_coord, e_coord, MAP[layer], include_artifacts)
        )

    return layers


def layer_to_cells(x_coord, y_coord, e_coord, MAP_layer, include_artifacts=False):
    """
    Project this layer of points onto the detector map.

    Parameters
    ----------
    x_coord : np.array
        shower local x coordinates of the points in this layer
    y_coord : np.array
        shower local y coordinates of the points in this layer
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
        This is arranged as in the global coordinates of the detector

    """
    xedges = MAP_layer["xedges"]
    zedges = MAP_layer["zedges"]
    H_base = MAP_layer["grid"]

    # flip x and y due to the rotation of the local coordinates
    H, xedges, zedges = np.histogram2d(
        y_coord, x_coord, bins=(xedges, zedges), weights=e_coord
    )
    if not include_artifacts:
        H[H_base == 0] = 0

    return H


def perpendicular_cell_centers(MAP, half_cell_size_global):
    """
    Given the cell map, return the centers of the cells, perpendicular
    to the radial direction.

    Parameters
    ----------
    MAP : list
        list of dictionaries, each containing the grid of cells for a layer
        in global coordinates
        As returned by create_map
    half_cell_size_global : float
        Half the size of the cells in the detector,
        perpendicular to the radial direction

    Returns
    -------
    centers : list of (np.array, np.array)
        2D array containing hits of the points,
        in global coordinates (x, y, z, energy)
    """
    centers = []
    centers = []
    for MAP_layer in MAP:
        x = MAP_layer["xedges"] + half_cell_size_global
        y = MAP_layer["zedges"] + half_cell_size_global
        centers.append((x, y))
    return centers


def cells_to_points(
    layers,
    MAP,
    layer_bottom_pos,
    half_cell_size_global,
    cell_thickness_global,
    length_to_pad=None,
):
    """
    Given the energy deposited in each cell of the detector, convert
    back to a list of points, discreetised by the cell layout.

    Parameters
    ----------
    layers : list
        list of 2D arrays, each containing the energy deposited in each
        cell of the layer with the arangement specified by MAP
        in the style of a histogram
    MAP : list
        list of dictionaries, each containing the grid of cells for a layer
        in global coordinates
        As returned by create_map
    layer_bottom_pos : np.array
        Array of the bottom positions of the layers.
    half_cell_size_global : float
        Half the size of the cells in the detector,
        perpendicular to the radial direction
    cell_thickness_global : float
        Thickness of the cells in the detector, in the radial direction
    length_to_pad : int (optional)
        Required length of the output array. If the number of hits is less than
        this, the output will be padded with zeros.
        If not given, the output will be the length of the number of non-zero hits.

    Returns
    -------
    points : np.array (n_hits, 4)
        2D array containing hits of the points,
        in global coordinates (x, y, z, energy)
    """

    points = []
    perpendicular_centers = perpendicular_cell_centers(MAP, half_cell_size_global)
    for ln, layer in enumerate(layers):
        xedges = MAP[ln]["xedges"]
        zedges = MAP[ln]["zedges"]

        x_indx, z_indx = np.where(layer > 0)

        cell_energy = layer[layer > 0]
        cell_coordinate_x = perpendicular_centers[ln][0][x_indx]
        cell_coordinate_y = np.repeat(
            layer_bottom_pos[ln] + cell_thickness_global / 2, len(x_indx)
        )
        cell_coordinate_z = perpendicular_centers[ln][1][z_indx]

        # rever the rotation to get shower local coordinates
        points.append(
            [
                cell_coordinate_x,
                cell_coordinate_y,
                cell_coordinate_z,
                cell_energy,
            ]
        )

    points = np.concatenate(points, axis=1).T
    if length_to_pad is not None:
        zeros_to_concat = np.zeros((length_to_pad - points.shape[0], 4))
        points = np.vstack((points, zeros_to_concat))

    return points


def get_projections(
    showers,
    MAP,
    layer_bottom_pos=None,
    half_cell_size_global=None,
    cell_thickness_global=None,
    return_cell_point_cloud=False,
    include_artifacts=False,
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
        in shower local coordinates (x, y, z, energy)
    MAP : list
        list of dictionaries, each containing the grid of cells for a layer
        As returned by create_map, in detector global coordinates
    layer_bottom_pos : np.array (optional)
        Array of the bottom positions of the layers.
        If not given, the config will be used to generate a Metadata object
        and the layer_bottom_pos will be taken from there.
    half_cell_size_global : float (optional)
        Half the size of the cells in the detector, in the radial direction.
        If not given, the config will be used to generate a Metadata object
        and the half_cell_size_global will be taken from there.
    cell_thickness_global : float (optional)
        Thickness of the cells in the detector, in the radial direction
        If not given, the config will be used to generate a Metadata object
        and the layer_bottom_pos will be taken from there.
    return_cell_point_cloud : bool
        Return a second variable containing the deiscreetised point cloud
    include_artifacts : bool
        Retain hits that fall in cells that don't have sensitive material
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
        list of 2D arrays, each containing the energy deposited
        in each cell of the layer with the arangement specified by MAP
        in the style of a histogram
    cell_point_clouds : np.array (n_hits, 4) (optional)
        3D array containing hits of the points,
        in coordinates (x, y, z, energy)
    """
    if max_num_hits is None:
        max_num_hits = configs.max_points

    metadata = Metadata(configs)
    if layer_bottom_pos is None:
        layer_bottom_pos = metadata.layer_bottom_pos_global
    if half_cell_size_global is None:
        half_cell_size_global = metadata.half_cell_size
    if cell_thickness_global is None:
        cell_thickness_global = metadata.cell_thickness

    events = []
    points = local_to_global(showers, metadata.orientation, metadata.orientation_global)
    for shower in tqdm(points):
        layers = points_to_cells(
            shower, MAP, layer_bottom_pos, cell_thickness_global, include_artifacts
        )
        events.append(layers)

    if not return_cell_point_cloud:
        return events

    cell_point_clouds = []
    for event in tqdm(events):
        point_cloud = cells_to_points(
            event,
            MAP,
            layer_bottom_pos,
            half_cell_size_global,
            cell_thickness_global,
            max_num_hits,
        )
        cell_point_clouds.append(point_cloud)

    try:
        cell_point_clouds = np.stack(cell_point_clouds)
    except ValueError:  # to prevent us choking on empty input
        cell_point_clouds = np.empty((0, 0, 4))

    return events, cell_point_clouds


def confine_to_box(X, Y, Z, E, metadata=Metadata()):
    """
    Remove hits that fall outside the box specified by the metadata.

    Parameters
    ----------
    X, Y, Z : array like (N)
        hit global detector coordinates
    E : array like (N)
        hit energies
    metadata : Metadata (optional)
        Metadata object containing the configuration of the dataset
        If not given, a default Metadata object is created.

    Returns
    -------
    X, Y, Z, E : array like (M <= N)
        hit coordinates and energies, confined to the box

    """
    inbox_idx = np.where(
        (Y > metadata.Ymin_global)
        & (X < metadata.Xmax_global)
        & (X > metadata.Xmin_global)
        & (Z < metadata.Zmax_global)
        & (Z > metadata.Zmin_global)
    )[0]
    # inbox_idx = np.where(Y > Ymin)[0]

    X = X[inbox_idx]
    Z = Z[inbox_idx]
    Y = Y[inbox_idx]
    E = E[inbox_idx]
    return X, Y, Z, E
