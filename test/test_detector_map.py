""" Module to test the detector_map module. """
# get the folder above on the path
import numpy as np
import numpy.testing as npt

from pointcloud.utils import detector_map


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
    z_points = np.linspace(-2, 2, nz)
    y_points = np.linspace(0, 4, n_layers)
    X, Y, Z = [], [], []
    for z in z_points:
        for x in x_points:
            for y in y_points:
                X.append(x)
                Y.append(y)
                Z.append(z)
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)

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
    assert np.all(x_points[0] >= x_edges_0[0])
    assert np.all(x_points[-1] <= x_edges_0[-1])
    z_edges_0 = layers[0]["zedges"]
    assert np.all(z_points[0] >= z_edges_0[0])
    assert np.all(z_points[-1] <= z_edges_0[-1])


def assert_layer_like(layer):
    assert "xedges" in layer
    assert "zedges" in layer
    assert "grid" in layer

    nx = len(layer["xedges"]) - 1
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
    layer_bottom_pos = np.array([-1, 2])
    cell_thickness = 1
    layered_data = list(
        detector_map.split_to_layers(manual_points, layer_bottom_pos, cell_thickness)
    )
    assert len(layered_data) == 2
    assert np.all(layered_data[0] == manual_points[:1])
    assert np.all(layered_data[1] == manual_points[2:])


def test_layer_to_cells():
    x_edges_before = np.linspace(0, 5, 6)
    z_edges_before = np.linspace(-5, 5, 6)
    MAP_layer = {
        "xedges": x_edges_before,
        "zedges": z_edges_before,
        "grid": np.ones((5, 5)),
    }

    # shouldn't choke on empty input
    cells = detector_map.layer_to_cells([], [], [], MAP_layer)
    expected_cells = np.zeros((5, 5))
    npt.assert_allclose(cells, expected_cells)

    # simple sample
    x_points = [0.5, 0.5, 4.5, 5.5]
    z_points = [0.5, 0.5, 4.0, 4.5]
    e_points = [1, 1, 3, 1]

    cells = detector_map.layer_to_cells(x_points, z_points, e_points, MAP_layer)

    expected_cells = np.zeros((5, 5))
    expected_cells[0, 2] = 2.0
    expected_cells[4, 4] = 3.0
    npt.assert_allclose(cells, expected_cells)


def test_points_to_cells():
    n_across = 5
    x_edges_before = np.linspace(0, 5, n_across + 1)
    z_edges_before = np.linspace(-5, 5, n_across + 1)
    MAP = [
        {
            "xedges": x_edges_before,
            "zedges": z_edges_before,
            "grid": np.ones((n_across, n_across)),
        },
        {
            "xedges": x_edges_before,
            "zedges": z_edges_before,
            "grid": np.ones((n_across, n_across)),
        },
    ]
    layer_bottom_pos = np.array([0, 1])
    cell_thickness = 0.5

    # shouldn't choke on empty input
    empty = np.empty((0, 4))
    cells = detector_map.points_to_cells(empty, MAP, layer_bottom_pos, cell_thickness)
    expected_cells = np.zeros((2, 5, 5))
    npt.assert_allclose(cells, expected_cells)

    # simple sample
    x_points = [0.5, 0.5, 4.5, 5.5, 0.5]
    y_points = [0.2, 0.2, 0.2, 0.2, 1.2]
    z_points = [0.5, 0.5, 4.0, 4.5, 0.5]
    e_points = [1, 1, 3, 1, 5]
    points = np.vstack((x_points, y_points, z_points, e_points)).T

    cells = detector_map.points_to_cells(points, MAP, layer_bottom_pos, cell_thickness)

    expected_cells = np.zeros((2, 5, 5))
    expected_cells[0, 0, 2] = 2.0
    expected_cells[0, 4, 4] = 3.0
    expected_cells[1, 0, 2] = 5.0
    npt.assert_allclose(cells, expected_cells)


def test_cells_to_points():
    half_cell_size = 0.5
    cell_thickness = 0.1
    n_across = 5
    x_edges_before = np.linspace(0, 5, n_across + 1)
    z_edges_before = np.linspace(-5, 5, n_across + 1)
    MAP = [
        {
            "xedges": x_edges_before,
            "zedges": z_edges_before,
            "grid": np.ones((n_across, n_across)),
        },
        {
            "xedges": x_edges_before,
            "zedges": z_edges_before,
            "grid": np.ones((n_across, n_across)),
        },
    ]
    layer_bottom_pos = np.array([0, 1])
    layers = [np.zeros((n_across, n_across)) for _ in layer_bottom_pos]

    # shouldn't choke on empty input
    points = detector_map.cells_to_points(
        layers, MAP, layer_bottom_pos, half_cell_size, cell_thickness
    )
    assert points.shape == (0, 4)

    # if told to pad, the output from empty input should eb all 0
    points = detector_map.cells_to_points(
        layers, MAP, layer_bottom_pos, half_cell_size, cell_thickness, length_to_pad=4
    )
    assert len(points) == 4
    npt.assert_allclose(points, np.zeros((4, 4)))

    # add 3 fake hits
    layers[0][2, 0] = 2.0
    layers[0][4, 4] = 3.0
    layers[1][2, 0] = 5.0

    points = detector_map.cells_to_points(
        layers, MAP, layer_bottom_pos, half_cell_size, cell_thickness
    )
    assert points.shape == (3, 4)
    # we don't know which order the points will be in
    energy_order = np.argsort(points[:, 3])
    expected_points = np.array(
        [[2.5, 0.05, -4.5, 2.0], [4.5, 0.05, 3.5, 3.0], [2.5, 1.05, -4.5, 5.0]]
    )
    npt.assert_allclose(points[energy_order], expected_points)


def test_get_projections():
    half_cell_size = 0.5
    cell_thickness = 0.1
    # Mostly composed of other functions, so just a smoke test
    x_edges_before = np.linspace(0, 5, 5)
    z_edges_before = np.linspace(-5, 5, 5)
    MAP = [
        {"xedges": x_edges_before, "zedges": z_edges_before, "grid": np.ones((4, 4))},
        {"xedges": x_edges_before, "zedges": z_edges_before, "grid": np.ones((4, 4))},
    ]
    layer_bottom_pos = np.array([0, 1])

    # shouldn't choke on empty input
    events = detector_map.get_projections(
        [], MAP, layer_bottom_pos, half_cell_size, cell_thickness, False
    )
    assert len(events) == 0
    events, cell_point_clouds = detector_map.get_projections(
        [], MAP, layer_bottom_pos, half_cell_size, cell_thickness, True, 10
    )
    assert len(events) == 0
    assert cell_point_clouds.shape == (0, 0, 4)

    # one simple event
    shower1 = np.array(
        [[0, 0.05, 0.5, 2.0], [3.75, 0.05, 4.5, 3.0], [0, 1.05, 0.5, 5.0]]
    )
    events = detector_map.get_projections(
        [shower1], MAP, layer_bottom_pos, half_cell_size, cell_thickness, False
    )
    assert len(events) == 1
    events, cell_point_clouds = detector_map.get_projections(
        [shower1], MAP, layer_bottom_pos, half_cell_size, cell_thickness, True,
        max_num_hits=10
    )
    assert len(events) == 1
    assert cell_point_clouds.shape == (1, 10, 4)


def test_confine_to_box():
    class FakeMetadata:
        def __init__(self):
            self.Ymin = 1
            self.Xmin = 0
            self.Xmax = 5
            self.Zmin = -5
            self.Zmax = 5

    fake_metadata = FakeMetadata()

    # shouldn't choke on empty input
    empty = np.empty(0)
    found_x, found_y, found_z, found_e = detector_map.confine_to_box(
        empty, empty, empty, empty, fake_metadata
    )
    assert len(found_x) == 0
    assert len(found_y) == 0
    assert len(found_z) == 0
    assert len(found_e) == 0

    # simple sample
    X = np.array([0, 3.75, 0.1])
    Y = np.array([0.05, 0.05, 1.05])
    Z = np.array([0.5, 4.5, 0.5])
    E = np.array([2.0, 3.0, 5.0])
    found_x, found_y, found_z, found_e = detector_map.confine_to_box(
        X, Y, Z, E, fake_metadata
    )
    npt.assert_allclose(found_x, [0.1])
    npt.assert_allclose(found_y, [1.05])
    npt.assert_allclose(found_z, [0.5])
    npt.assert_allclose(found_e, [5.0])
