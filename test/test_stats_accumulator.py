""" Module to test the stats_accumulator module. """
import numpy as np
import numpy.testing as npt
import os
from unittest.mock import patch

from pointcloud.utils import stats_accumulator
from pointcloud.config_varients import default

from helpers.mock_read_write import mock_read_raw_regaxes, mock_get_n_events
from helpers.sample_accumulator import make as make_test_accumulator


def test_StatsAccumulator(tmpdir):
    save_dir = tmpdir.mkdir("test_StatsAccumulator")
    save_name = os.path.join(save_dir, "test_stats_accumulator.h5")

    acc = stats_accumulator.StatsAccumulator()
    assert np.all(acc.counts_hist == 0)
    assert np.all(acc.energy_hist == 0)
    assert np.sum(acc.total_events) == 0

    acc.save(save_name)
    assert os.path.exists(save_name)
    acc = stats_accumulator.StatsAccumulator.load(save_name)

    assert np.all(acc.counts_hist == 0)
    assert np.all(acc.energy_hist == 0)
    assert np.sum(acc.total_events) == 0

    # put one event in with one centeral point
    center = np.array([0.0, 0.0, 0.0, 1.0])
    acc.add([0], [50], np.array([[center]]))

    assert np.sum(acc.counts_hist) == 1
    assert np.sum(acc.energy_hist) == 1
    assert np.sum(acc.counts_hist > 0) == 1
    assert np.sum((acc.counts_hist * acc.energy_hist) > 0) == 1
    assert np.sum(acc.total_events) == 1

    # put another two events in with two points, both at the center
    acc.add([1, 2], [50, 50], np.array([[center * 2, center], [center, center]]))

    assert np.sum(acc.counts_hist) == 5
    assert np.sum(acc.energy_hist) == 6
    assert np.sum(acc.counts_hist > 0) == 1
    assert np.sum((acc.counts_hist * acc.energy_hist) > 0) == 1
    assert np.sum(acc.total_events) == 3

    # now put in one more event with two points,
    # one in the same place as before, one in another incident bin
    acc.add([3, 4], [50, 80], np.array([[center], [center]]))

    assert np.sum(acc.counts_hist) == 7
    assert np.sum(acc.energy_hist) == 8
    assert np.sum(acc.counts_hist > 0) == 2
    assert np.sum((acc.counts_hist * acc.energy_hist) > 0) == 2
    assert np.sum(acc.total_events) == 5

    # one more event, in the original incident bin
    # but with a different z value
    acc.add([5], [50], np.array([[[0.0, 0.0, 1.0, 1.0]]]))

    assert np.sum(acc.counts_hist) == 8
    assert np.sum(acc.energy_hist) == 9
    assert np.sum(acc.counts_hist > 0) == 3
    assert np.sum((acc.counts_hist * acc.energy_hist) > 0) == 3
    assert np.sum(acc.total_events) == 6

    acc.save(save_name)
    assert os.path.exists(save_name)
    acc = stats_accumulator.StatsAccumulator.load(save_name)

    assert np.sum(acc.counts_hist) == 8
    assert np.sum(acc.energy_hist) == 9
    assert np.sum(acc.counts_hist > 0) == 3
    assert np.sum((acc.counts_hist * acc.energy_hist) > 0) == 3
    assert np.sum(acc.total_events) == 6


def test_AlignedStatsAccumulator(tmpdir):
    save_dir = tmpdir.mkdir("test_AlignedStatsAccumulator")
    save_name = os.path.join(save_dir, "test_stats_accumulator.h5")

    acc = stats_accumulator.AlignedStatsAccumulator("mean")
    assert np.all(acc.counts_hist == 0)
    assert np.all(acc.energy_hist == 0)
    assert np.all(acc.layer_offset_hist == 0)
    assert np.sum(acc.total_events) == 0

    acc.save(save_name)
    assert os.path.exists(save_name)
    acc = stats_accumulator.AlignedStatsAccumulator.load(save_name, "mean")

    assert np.all(acc.counts_hist == 0)
    assert np.all(acc.energy_hist == 0)
    assert np.all(acc.layer_offset_hist == 0)
    assert np.sum(acc.total_events) == 0

    # put one event in with one centeral point
    center = np.array([0.0, 0.0, 0.0, 1.0])
    acc.add([0], [50], np.array([[center]]))

    assert np.sum(acc.counts_hist) == 1
    assert np.sum(acc.energy_hist) == 1
    assert np.sum(acc.counts_hist > 0) == 1
    assert np.sum(acc.layer_offset_hist) == 1
    assert np.sum(acc.layer_offset_hist > 0) == 1
    assert np.sum((acc.counts_hist * acc.energy_hist) > 0) == 1
    assert np.sum(acc.total_events) == 1

    # put another two events in with two points, both at the center
    acc.add([1, 2], [50, 50], np.array([[center * 2, center], [center, center]]))

    assert np.sum(acc.counts_hist) == 5
    assert np.sum(acc.energy_hist) == 6
    assert np.sum(acc.counts_hist > 0) == 1
    assert np.sum(acc.layer_offset_hist) == 3
    assert np.sum(acc.layer_offset_hist > 0) == 1
    assert np.sum((acc.counts_hist * acc.energy_hist) > 0) == 1
    assert np.sum(acc.total_events) == 3

    # now put in one more event with two points,
    # one in the same place as before, one in another incident bin
    acc.add([3, 4], [50, 80], np.array([[center], [center]]))

    assert np.sum(acc.counts_hist) == 7
    assert np.sum(acc.energy_hist) == 8
    assert np.sum(acc.counts_hist > 0) == 2
    assert np.sum(acc.layer_offset_hist) == 5
    assert np.sum(acc.layer_offset_hist > 0) == 2
    assert np.sum((acc.counts_hist * acc.energy_hist) > 0) == 2
    assert np.sum(acc.total_events) == 5

    # one more event, in the original incident bin
    # but with a different z value
    acc.add([5], [50], np.array([[[0.0, 0.0, 0.5, 1.0]]]))

    assert np.sum(acc.counts_hist) == 8
    assert np.sum(acc.energy_hist) == 9
    assert np.sum(acc.counts_hist > 0) == 2  # shift realigns it back over other events
    assert np.sum(acc.layer_offset_hist) == 6
    assert np.sum(acc.layer_offset_hist > 0) == 3
    assert np.sum((acc.counts_hist * acc.energy_hist) > 0) == 2
    assert np.sum(acc.total_events) == 6

    acc.save(save_name)
    assert os.path.exists(save_name)
    acc = stats_accumulator.AlignedStatsAccumulator.load(save_name, "mean")

    assert np.sum(acc.counts_hist) == 8
    assert np.sum(acc.energy_hist) == 9
    assert np.sum(acc.counts_hist > 0) == 2
    assert np.sum(acc.layer_offset_hist) == 6
    assert np.sum(acc.layer_offset_hist > 0) == 3
    assert np.sum((acc.counts_hist * acc.energy_hist) > 0) == 2
    assert np.sum(acc.total_events) == 6


def test_save_location(tmpdir):
    config = default.Configs()
    config.logdir = tmpdir.mkdir("test_save_location")
    save_name_0 = stats_accumulator.save_location(config, 0, 1)
    save_dir_0 = os.path.dirname(save_name_0)
    assert os.path.exists(save_dir_0)

    save_name_1 = stats_accumulator.save_location(config, 2, 0)
    save_dir_1 = os.path.dirname(save_name_1)
    assert os.path.exists(save_dir_1)

    save_name_2 = stats_accumulator.save_location(config, 2, 1)
    assert save_name_2 != save_name_1
    save_dir_2 = os.path.dirname(save_name_2)
    assert save_dir_2 == save_dir_1
    assert os.path.exists(save_dir_2)


# both get_n_events and read_raw_regaxis are tested elsewhere
# here we will mock the calls to them
@patch("pointcloud.utils.stats_accumulator.get_n_events", new=mock_get_n_events)
@patch("pointcloud.utils.stats_accumulator.read_raw_regaxes", new=mock_read_raw_regaxes)
def test_read_section_to(tmpdir):
    config = default.Configs()
    out_path = os.path.join(
        tmpdir.mkdir("test_save_location"), "test_read_section_to.h5"
    )
    stats_accumulator.read_section_to(config, out_path, 1, 0)
    acc = stats_accumulator.StatsAccumulator.load(out_path)
    assert np.sum(acc.counts_hist) > 1
    assert np.sum(acc.total_events) == 3


# both get_n_events and read_raw_regaxis are tested elsewhere
# here we will mock the calls to them
@patch("pointcloud.utils.stats_accumulator.get_n_events", new=mock_get_n_events)
@patch("pointcloud.utils.stats_accumulator.read_raw_regaxes", new=mock_read_raw_regaxes)
def test_read_section(tmpdir):
    config = default.Configs()
    config.logdir = tmpdir.mkdir("test_save_location")
    stats_accumulator.read_section(1, 0, config)
    out_path = stats_accumulator.save_location(config, 1, 0)
    acc = stats_accumulator.StatsAccumulator.load(out_path)
    assert np.sum(acc.counts_hist) > 1
    assert np.sum(acc.total_events) == 3


def test_HighLevelStats():
    acc = make_test_accumulator()

    hls0 = stats_accumulator.HighLevelStats(acc, 0)
    hls1 = stats_accumulator.HighLevelStats(acc, 1)

    for layer_n in range(acc.n_layers):
        n_pts = hls0.get_n_pts_vs_incident_energy(layer_n)
        assert np.all(n_pts == 4)
        n_pts = hls1.get_n_pts_vs_incident_energy(layer_n)
        assert np.all(n_pts == 4)

    layer_n = 0
    n_pts_coeffs = hls0.n_pts(layer_n)
    npt.assert_allclose(n_pts_coeffs, [4])

    n_pts_coeffs = hls1.n_pts(layer_n)
    npt.assert_allclose(n_pts_coeffs, [0.0, 4.0], atol=1e-6)

    std_n_pts = hls0.stddev_n_pts(layer_n)
    npt.assert_allclose(std_n_pts, [0])

    std_n_pts = hls1.stddev_n_pts(layer_n)
    npt.assert_allclose(std_n_pts, [0, 0])

    mean_pt_energy = hls0.event_mean_point_energy(layer_n)
    npt.assert_allclose(mean_pt_energy, [1.0])

    mean_pt_energy = hls1.event_mean_point_energy(layer_n)
    npt.assert_allclose(mean_pt_energy, [0.0, 1.0], atol=1e-6)

    std_mean_pt_energy = hls0.stddev_event_mean_point_energy(layer_n)
    npt.assert_allclose(std_mean_pt_energy, [0.0])

    std_mean_pt_energy = hls1.stddev_event_mean_point_energy(layer_n)
    npt.assert_allclose(std_mean_pt_energy, [0.0, 0.0])

    std_energy_in_evt = hls0.stddev_point_energy_in_evt(layer_n)
    npt.assert_allclose(std_energy_in_evt, [0.0])

    std_energy_in_evt = hls1.stddev_point_energy_in_evt(layer_n)
    npt.assert_allclose(std_energy_in_evt, [0.0, 0.0])


def test_RadialView():
    acc = make_test_accumulator(True)
    radial = stats_accumulator.RadialView(acc, 0, 0)

    def linear(x, a, b):
        return a * x + b

    bounds = [[-np.inf, 0.0], [np.inf, np.inf]]
    p0 = [-0.5, 1.0]

    expected = [-1, 2.0]

    for layer_n in [0, 1, len(acc.layer_bottom) - 1]:
        popt, pcov = radial.fit_to_energy(
            40.0, layer_n, linear, p0, bounds, ignore_norm=False, quiet=True
        )
        npt.assert_allclose(popt, expected, atol=0.1, rtol=0.01)
