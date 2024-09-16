"""
Tests for the utils/bin_standard_metrics.py module.
"""
import os
import numpy as np
import numpy.testing as npt
from unittest.mock import patch

from helpers.mock_read_write import mock_read_raw_regaxes, mock_get_n_events
from helpers.sample_models import write_fake_flow_model, make_fake_wish_model
from helpers import config_creator


from pointcloud.config_varients.wish import Configs as WishConfigs
from pointcloud.models.wish import Wish
from pointcloud.utils import stats_accumulator
from pointcloud.evaluation.bin_standard_metrics import (
    try_mkdir,
    BinnedData,
    sample_g4,
    sample_model,
    sample_accumulator,
    get_path,
    get_wish_models,
    get_caloclouds_models,
)


def make_default_binned(xyz_limits=None):
    if xyz_limits is None:
        xyz_limits = [[-1, 1], [-1, 1], [-1, 1]]
    layer_bottom_pos = np.linspace(-1, 1, 30)
    binned = BinnedData("test", xyz_limits, 10, layer_bottom_pos, 0.05, [0.0, 0.0])
    return binned


def test_try_mkdir(tmpdir):
    new_dir_name = "test_dir"
    new_dir = os.path.join(tmpdir, new_dir_name)
    # check it doesn't exist
    assert not os.path.exists(new_dir)
    # create it
    try_mkdir(new_dir)
    # check it exists
    assert os.path.exists(new_dir)
    # check it's a directory
    assert os.path.isdir(new_dir)
    # check it can be savely created again
    try_mkdir(new_dir)
    assert os.path.exists(new_dir)


def test_BinnedData(tmpdir):
    # test init
    binned = make_default_binned()
    expected_hists = [
        ["sum hits", "radius [mm]"],
        ["sum energy [MeV]", "radius [mm]"],
        ["sum hits", "layers"],
        ["sum energy [MeV]", "layers"],
        ["number of cells", "visible cell energy [MeV]"],
        ["number of showers", "number of hits"],
        ["number of showers", "energy sum [MeV]"],
        ["number of showers", "center of gravity X [mm]"],
        ["number of showers", "center of gravity Z [mm]"],
        ["number of showers", "center of gravity Y [mm]"],
        ["mean energy [MeV]", "radius [mm]"],
        ["mean energy [MeV]", "layers"],
    ]
    n_expected_hists = len(expected_hists)
    assert len(binned.y_labels) == n_expected_hists
    assert len(binned.x_labels) == n_expected_hists
    assert len(binned.bins) == n_expected_hists
    assert len(binned.counts) == n_expected_hists
    # test hist_idx
    for y_label, x_label in expected_hists:
        assert binned.hist_idx(y_label, x_label) is not None
    assert binned.hist_idx("something", "something") is None
    # test add_hist
    binned.add_hist("test y", "test x", 0, 10, 10, False)
    assert binned.hist_idx("test y", "test x") is not None
    n_expected_hists += 1
    assert len(binned.y_labels) == n_expected_hists
    assert len(binned.x_labels) == n_expected_hists
    assert len(binned.bins) == n_expected_hists
    assert len(binned.counts) == n_expected_hists
    # test add_events
    # add no events
    fake_events = np.zeros((0, 10, 4))
    binned.add_events(fake_events)
    for hist in range(n_expected_hists):
        assert np.all(binned.counts[hist] == 0)
    # add two events with one hit a 0, 0, 0
    fake_events = np.zeros((2, 10, 4))
    fake_events[:, 0, 3] = 1
    binned.add_events(fake_events)
    hits_vs_radius = binned.hist_idx(*expected_hists[0])
    assert np.sum(binned.counts[hits_vs_radius]) == 2
    assert np.sum(binned.counts[hits_vs_radius] > 0) == 1
    energy_vs_radius = binned.hist_idx(*expected_hists[1])
    assert np.sum(binned.counts[energy_vs_radius]) == 2 / 10
    assert np.sum(binned.counts[energy_vs_radius] > 0) == 1
    hits_vs_layers = binned.hist_idx(*expected_hists[2])
    hist = binned.counts[hits_vs_layers]
    center = hist.shape[0] // 2 - 1
    assert hist[center] == 2
    assert np.all(hist[:center] == 0)
    assert np.all(hist[center + 1 :] == 0)
    energy_vs_layers = binned.hist_idx(*expected_hists[3])
    hist = binned.counts[energy_vs_layers]
    assert hist[center] == 2 / 10
    assert np.all(hist[:center] == 0)
    assert np.all(hist[center + 1 :] == 0)
    # for all the rest, just check exactly one bin got 2 counts
    for i in range(4, len(expected_hists) - 2):
        hist_idx = binned.hist_idx(*expected_hists[i])
        hist = binned.counts[hist_idx]
        assert np.sum(hist) == 2
        assert np.sum(hist > 0) == 1
    # test get_gunshift
    gunshift = binned.get_gunshift()
    assert np.allclose(gunshift, np.zeros((1, 1, 4)))
    # test recompute_mean_energies
    binned.recompute_mean_energies()
    for i in range(2):
        i += len(expected_hists) - 2
        energy_vs_radius = binned.hist_idx(*expected_hists[i])
        hist = binned.counts[energy_vs_radius]
        assert np.sum(hist) == 1 / 10
        assert np.sum(hist > 0) == 1
    # test total_showers
    assert binned.total_showers() == 2
    # test dummy_xs
    for i in range(len(expected_hists)):
        bins = binned.bins[i]
        dummy = binned.dummy_xs(i)
        assert len(bins) == len(dummy) + 1
        assert bins[0] < dummy[0]
        assert bins[-1] > dummy[-1]
    # test rescaled_events
    fake_events = np.zeros((1, 10, 4))
    fake_events[0, :, 0] = np.linspace(-1, 1, 10)
    fake_events[0, :, 2] = np.linspace(-1, 1, 10)
    fake_events[0, :, 3] = np.linspace(0, 1, 10)
    rescaled = binned.rescaled_events(fake_events)
    npt.assert_allclose(
        rescaled[0, :, 0],
        np.linspace(binned.true_xyz_limits[0][0], binned.true_xyz_limits[0][1], 10),
    )
    y_center = 0.5 * (binned.true_xyz_limits[1][0] + binned.true_xyz_limits[1][1])
    npt.assert_allclose(rescaled[0, :, 1], np.zeros(10) + y_center)
    npt.assert_allclose(
        rescaled[0, :, 2],
        np.linspace(binned.true_xyz_limits[2][0], binned.true_xyz_limits[2][1], 10),
    )
    npt.assert_allclose(rescaled[0, :, 3], np.linspace(0, 0.1, 10))
    # test normed - just try one
    unnormed = binned.counts[0]
    normed = binned.normed(0)
    npt.assert_allclose(normed, unnormed / 2)
    # test save
    save_path = os.path.join(tmpdir, "test.npz")
    binned.save(save_path)
    assert os.path.exists(save_path)
    # test load
    new_binned = BinnedData.load(save_path)
    npt.assert_allclose(new_binned.true_xyz_limits, binned.true_xyz_limits)
    npt.assert_allclose(
        new_binned.layer_bottom_pos, binned.layer_bottom_pos
    )
    for i in range(len(expected_hists)):
        npt.assert_allclose(new_binned.counts[i], binned.counts[i])
        npt.assert_allclose(new_binned.bins[i], binned.bins[i])


# both get_n_events and read_raw_regaxis are tested elsewhere
# here we will mock the calls to them
@patch("pointcloud.data.trees.get_n_events", new=mock_get_n_events)
@patch("pointcloud.data.trees.read_raw_regaxes", new=mock_read_raw_regaxes)
def test_sample_g4():
    binned = make_default_binned([[-100, 100], [-100, 100], [-100, 100]])
    config = config_creator.make("default")
    # test with no events
    sample_g4(config, binned, 0)
    for counts in binned.counts[:-2]:
        assert np.all(counts == 0)
    # test with one event
    sample_g4(config, binned, 1)
    # check something got added
    for i, counts in enumerate(binned.counts[:-2]):
        assert np.any(
            counts > 0
        ), f"hist {i} is empty, {binned.y_labels[i]}, {binned.x_labels[i]}"


def test_sample_model():
    # test with wish
    configs = config_creator.make("wish")
    wish_model = make_fake_wish_model(configs)

    binned = make_default_binned([[-100, 100], [-100, 100], [-100, 100]])
    # sample no events
    sample_model(configs, binned, 0, wish_model)
    for counts in binned.counts:
        assert np.all(counts == 0)
    # sample many events
    energies, events = sample_model(configs, binned, 100, wish_model)
    # check something got added
    for i, counts in enumerate(binned.counts[:6]):
        assert np.any(
            counts > 0
        ), f"hist {i} is empty, {binned.y_labels[i]}, {binned.x_labels[i]}"


def test_sample_accumulator():
    # test with an empty accumulator
    configs = config_creator.make()
    acc = stats_accumulator.StatsAccumulator()
    binned = make_default_binned()
    sample_accumulator(configs, binned, acc, 0)
    for counts in binned.counts[:4]:
        assert np.all(counts == 0)
    sample_accumulator(configs, binned, acc, 100)
    for counts in binned.counts[:4]:
        assert np.all(counts == 0)
    fake_events = np.zeros((1, 10, 4))
    fake_events[0, :, 3] = 1
    acc.add([1], np.ones(1), fake_events)
    sample_accumulator(configs, binned, acc, 1)
    assert np.any(binned.counts[0] > 0)


def test_get_path(tmpdir):
    configs = config_creator.make()
    configs.logdir = str(tmpdir)
    name = "test me"
    path = get_path(configs, name)
    assert isinstance(path, str)
    dir_name = os.path.dirname(path)
    assert os.path.exists(dir_name)


def test_get_wish_models(tmpdir):
    # make some fake models to read
    configs = config_creator.make("wish")
    wish_model = make_fake_wish_model(configs)
    save_path = os.path.join(str(tmpdir), "wish_model_{}.pt")
    n_poly_degrees = 3
    for i in range(n_poly_degrees):
        wish_model.save(save_path.format(i + 1))
    # test reading them
    models = get_wish_models(save_path, n_poly_degrees)
    assert len(models) == n_poly_degrees
    for name in models:
        wish, flow, conf = models[name]
        assert isinstance(wish, Wish)
        assert flow is None
        assert isinstance(conf, WishConfigs)


def test_get_caloclouds_models(tmpdir):
    # Need to get the right paths
    cfg = config_creator.make("caloclouds_3")
    test_cm_model_path = "test/example_cm_model.pt"
    # fake the flow model
    test_model_path = str(tmpdir) + "/example_flow_model.pt"
    write_fake_flow_model(cfg, test_model_path)
    models = get_caloclouds_models(test_cm_model_path, test_model_path)
    assert len(models) == 1
