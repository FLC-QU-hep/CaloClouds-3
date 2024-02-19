""" Module to test the detector_map module. """
# get the folder above on the path
import sys
from pathlib import Path

path_root1 = Path(__file__).parents[1]
sys.path.append(str(path_root1))

import numpy as np
import numpy.testing as npt

from utils import detector_map


# TODO test dm parameter
def test_create_map():
    # check that a default call returns somthing that looks like a map
    layers, offset = detector_map.create_map()
    assert isinstance(offset, float)
    assert offset > 0

    assert len(layers) > 0
    assert_layer_like(layers[0])
    assert_layer_like(layers[1])
    assert_layer_like(layers[-1])

    # We should be able to give a custom set of points
    # and generate a regular map from that

    nx = 6
    nz = 4
    n_layers = 3
    x_points = np.linspace(0, 6, nx)
    X = np.tile(x_points, (1, nz, n_layers)).flatten().flatten()
    z_points = np.linspace(-2, 2, nz)
    Z = np.tile(z_points, (nx, 1, n_layers)).flatten().flatten()
    y_points = np.linspace(0, 4, n_layers)
    Y = np.tile(y_points, (nx, nz, 1)).flatten().flatten()

    half_cell_size = 0.5
    cell_thickness = 0.1
    layer_bottom_pos = np.linspace(0, 4, n_layers) - cell_thickness / 2

    layers, offset = detector_map.create_map(
        X, Y, Z, layer_bottom_pos, half_cell_size, cell_thickness
    )
    npt.assert_allclose(offset, 1.0)
    # all layers should be identical
    for i in range(n_layers - 1):
        npt.assert_allclose(layers[i]["xedges"], layers[-1]["xedges"])
        npt.assert_allclose(layers[i]["zedges"], layers[-1]["zedges"])
        npt.assert_allclose(layers[i]["grid"], layers[-1]["grid"])
    # the layers should contain the points
    x_edges_0 = layers[0]["xedges"]
    assert np.all(x_points >= x_edges_0[:-1])
    assert np.all(x_points <= x_edges_0[1:])
    z_edges_0 = layers[0]["zedges"]
    assert np.all(z_points >= z_edges_0[:-1])
    assert np.all(z_points <= z_edges_0[1:])


def assert_layer_like(layer):
    assert "xedges" in layer
    assert "zedges" in layer
    assert "grid" in layer

    nx = len(layer["xedges"]) - 2
    nz = len(layer["zedges"]) - 1
    assert layer["grid"].shape == (nx, nz)

    assert np.all(layer["grid"] >= 0)


def test_normalise_map():
    x_edges_before = np.linspace(0, 5, 5)
    z_edges_before = np.linspace(-5, 5, 5)
    MAP = [
        {"xedges": x_edges_before, "zedges": z_edges_before, "grid": np.ones((4, 4))}
    ]
    MAP = detector_map.normalise_map(MAP)
    x_edges_after = MAP[0]["xedges"]
    z_edges_after = MAP[0]["zedges"]
    npt.assert_allclose(x_edges_after, np.linspace(-1, 1, 5))
    npt.assert_allclose(z_edges_after, np.linspace(-1, 1, 5))


def test_split_to_layers():
    manual_points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
    layer_bottom_pos = np.array([-1, 1.5])
    layered_data = list(detector_map.split_to_layers(manual_points, layer_bottom_pos))
    assert len(layered_data) == 2
    assert np.all(layered_data[0] == manual_points[:2])
    assert np.all(layered_data[1] == manual_points[2:])


def test_layer_to_cells():
    x_edges_before = np.linspace(0, 5, 5)
    z_edges_before = np.linspace(-5, 5, 5)
    MAP = [
        {"xedges": x_edges_before, "zedges": z_edges_before, "grid": np.ones((4, 4))}
    ]

    # shouldn't choke on empty input
    cells = detector_map.layer_to_cells(MAP, [], [], [])
    expected_cells = np.zeros((5, 5))
    npt.assert_allclose(cells, expected_cells)

    # simple sample
    x_points = [0.5, 0.5, 4.5, 5.5]
    z_points = [0.5, 0.5, 4.0, 4.5]
    e_points = [1, 1, 3, 1]

    cells = detector_map.layer_to_cells(MAP, x_points, z_points, e_points)

    expected_cells = np.zeros((5, 5))
    expected_cells[2, 0] = 2.0
    expected_cells[4, 4] = 3.0
    npt.assert_allclose(cells, expected_cells)


def test_points_to_cells():
    x_edges_before = np.linspace(0, 5, 5)
    z_edges_before = np.linspace(-5, 5, 5)
    MAP = [
        {"xedges": x_edges_before, "zedges": z_edges_before, "grid": np.ones((4, 4))},
        {"xedges": x_edges_before, "zedges": z_edges_before, "grid": np.ones((4, 4))}
    ]
    layer_bottom_pos = np.array([0, 1])

    # shouldn't choke on empty input
    cells = detector_map.points_to_cells([], MAP, layer_bottom_pos)
    expected_cells = np.zeros((2, 5, 5))
    npt.assert_allclose(cells, expected_cells)

    # simple sample
    x_points = [0.5, 0.5, 4.5, 5.5, 0.5]
    y_points = [0.5, 0.5, 0.5, 0.5, 1.5]
    z_points = [0.5, 0.5, 4.0, 4.5, 0.5]
    e_points = [1, 1, 3, 1, 5]

    cells = detector_map.layer_to_cells(MAP, x_points, z_points, e_points)

    expected_cells = np.zeros((2, 5, 5))
    expected_cells[0, 2, 0] = 2.0
    expected_cells[0, 4, 4] = 3.0
    expected_cells[1, 2, 0] = 5.0
    npt.assert_allclose(cells, expected_cells)


def test_cells_to_points():
    MAP = [
        {"xedges": x_edges_before, "zedges": z_edges_before, "grid": np.ones((4, 4))},
        {"xedges": x_edges_before, "zedges": z_edges_before, "grid": np.ones((4, 4))}
    ]
    layer_bottom_pos = np.array([0, 1])
    cells = np.zeros((2, 5, 5))
    cells[0, 2, 0] = 2.0
    cells[0, 4, 4] = 3.0
    cells[1, 2, 0] = 5.0
    pass  # TODO


def test_get_projections():
    pass


def test_confine_to_box():
    pass
