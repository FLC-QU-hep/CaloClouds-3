""" Module to test the data/trees module. """

import numpy as np
import numpy.testing as npt
from unittest.mock import patch

from pointcloud.config_varients import default
from pointcloud.data import trees

from helpers.mock_read_write import mock_read_raw_regaxes, mock_get_n_events
from helpers import sample_trees


def test_Tree():
    root_xy = np.array([0.0, 0.0])
    empty_tree = sample_trees.empty(root_xy)
    assert empty_tree.n_layers == 3
    assert empty_tree.max_skips == 6
    npt.assert_array_equal(empty_tree.occupied_layers, [0])
    assert empty_tree.total_points == 1
    npt.assert_allclose(empty_tree.xy, [root_xy])
    npt.assert_allclose(empty_tree.energy, [1.0])
    assert len(empty_tree.edges) == 0
    npt.assert_array_equal(empty_tree.layer, [0])

    tree = sample_trees.simple(root_xy)
    assert tree.n_layers == 5
    assert tree.max_skips == 2
    npt.assert_array_equal(tree.occupied_layers, [0, 1, 2, 4])
    assert tree.total_points == 7
    npt.assert_allclose(
        tree.xy,
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.3],
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ],
    )
    npt.assert_allclose(tree.energy, [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06])
    npt.assert_array_equal(tree.edges, [[0, 1], [1, 2], [1, 3], [2, 4], [3, 5], [2, 6]])
    npt.assert_array_equal(tree.layer, [0, 1, 2, 2, 4, 4, 4])

    # test the project_positions method
    found_idxs, found_positions = tree.project_positions(2)
    npt.assert_array_equal(found_idxs, [1])
    npt.assert_allclose(found_positions, [[0.0, 0.0]])

    found_idxs, found_positions = tree.project_positions(3)
    order = np.argsort(found_idxs)
    npt.assert_array_equal(found_idxs[order], [1, 2, 3])
    npt.assert_allclose(found_positions[order], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.9]])


def test_check_connected():
    tree = sample_trees.empty()
    tree.total_points = 5
    tree.edges = np.array([[0, 1], [1, 2], [2, 3], [2, 4]])
    is_connected, idxs = trees.check_connected(tree)
    assert is_connected
    npt.assert_array_equal(sorted(idxs), [0, 1, 2, 3, 4])

    tree.edges = np.array([[0, 1], [2, 3]])
    tree.total_points = 4
    is_connected, idxs = trees.check_connected(tree)
    assert not is_connected
    npt.assert_array_equal(sorted(idxs), [0, 1])


def test_akt_distance():
    empty_distance = trees.akt_distance(
        np.array([]), np.array([]), np.array([]), np.array([])
    )
    assert len(empty_distance) == 0

    one_parent = trees.akt_distance(
        np.array([[0.0, 0.0]]), np.array([0]), np.array([]), np.array([])
    )
    assert len(one_parent.flatten()) == 0

    one_child = trees.akt_distance(
        np.array([]), np.array([]), np.array([[0.0, 0.0]]), np.array([0])
    )
    assert len(one_child.flatten()) == 0

    one_each = trees.akt_distance(
        np.array([[0.0, 0.0]]), np.array([0]), np.array([[0.0, 0.0]]), np.array([0])
    )
    npt.assert_allclose(one_each, [[0.0]])

    change_e = trees.akt_distance(
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([1.0, 2.0]),
        np.array([[-1.0, 0.0]]),
        np.array([1.0]),
    )
    npt.assert_allclose(change_e, [[1.0, 1.0]])


# both get_n_events and read_raw_regaxis are tested elsewhere
# here we will mock the calls to them
@patch("pointcloud.data.trees.get_n_events", new=mock_get_n_events)
@patch("pointcloud.data.trees.read_raw_regaxes", new=mock_read_raw_regaxes)
def test_DataAsTrees():
    config = default.Configs()
    data_as_trees = trees.DataAsTrees(config, max_trees_in_memory=2)
    assert len(data_as_trees) == 3

    all_trees = data_as_trees.get([0])
    assert len(all_trees) == 1
    tree_0 = all_trees[0]
    # test the get method after a tree has been cached
    all_trees = data_as_trees.get([0, 1])
    assert len(all_trees) == 2
    tree_0_copy = all_trees[0]
    # the tree should be the same object
    assert tree_0 is tree_0_copy

    # test the get method after a tree has been evicted
    all_trees = data_as_trees.get([1, 2])
    assert len(all_trees) == 2

    all_trees = data_as_trees.get([0, 1])
    assert len(all_trees) == 2
    tree_0_copy2 = all_trees[0]
    # this time the tree should be a different object
    assert tree_0 is not tree_0_copy2

    # but it should have the same data
    order_v1 = np.argsort(tree_0.energy)
    children_v1 = order_v1[tree_0.edges[:, 1]]
    parents_v1 = order_v1[tree_0.edges[:, 0]]
    child_ordered = parents_v1[children_v1.argsort()]

    order_v2 = np.argsort(tree_0_copy2.energy)
    children_v2 = order_v2[tree_0_copy2.edges[:, 1]]
    parents_v2 = order_v2[tree_0_copy2.edges[:, 0]]
    child_ordered_copy = parents_v2[children_v2.argsort()]

    npt.assert_array_equal(child_ordered, child_ordered_copy)
