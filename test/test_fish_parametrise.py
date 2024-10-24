"""
Unit tests for the fish_parametrise module.
"""

import os
import numpy as np
import numpy.testing as npt
import torch
import pytest

from pointcloud.models import fish_parametrise
from pointcloud.utils import stats_accumulator

from helpers import sample_accumulator


def test_incident_bin_centers():
    acc = sample_accumulator.make()
    incident_bin_centers = fish_parametrise.incident_bin_centers(acc)
    npt.assert_allclose(incident_bin_centers, np.linspace(12.5, 82.5, 15))


def test_distance_bin_centers():
    acc = sample_accumulator.make(
        start_acc=stats_accumulator.AlignedStatsAccumulator("mean")
    )
    distance_bin_centers = fish_parametrise.distance_bin_centers(acc)
    npt.assert_allclose(distance_bin_centers, np.linspace(-0.98, 0.98, 50))


# maths warning is expected, as layer_offset_sq_hist is fake
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_binned_shift():
    layer_bottom = np.linspace(-1, 1, 50)[:-1]
    cell_thickness = layer_bottom[1] - layer_bottom[0]
    acc = sample_accumulator.make(
        start_acc=stats_accumulator.AlignedStatsAccumulator(
            "mean", layer_bottom=layer_bottom, cell_thickness=cell_thickness
        )
    )
    n_incident = acc.total_events.shape[0] - 2
    # the sample accumulator has an event in every layer
    shift_mean, shift_std = fish_parametrise.binned_shift(acc)
    npt.assert_allclose(shift_mean, np.zeros(n_incident))
    assert np.all(shift_std >= 0)
    assert shift_std.shape == (n_incident,)

    # mess with the accumulator to create a more intresting shift distribution
    acc.layer_offset_bins = np.array([-0.5, 0.5, 1.5, 2.5])
    acc.layer_offset_hist = np.array([[0, 0, 0], [0, 2, 1], [0, 0, 0]])
    acc.layer_offset_sq_hist = np.array([[0, 0, 0], [0, 1, 2], [0, 0, 0]])
    shift_mean, shift_std = fish_parametrise.binned_shift(acc)
    npt.assert_allclose(shift_mean, np.array([4 / 3]))


def test_fit_shift():
    linear = np.arange(10)
    zero = np.zeros(10)
    p_mean, p_std = fish_parametrise.fit_shift(linear, linear, linear)
    npt.assert_allclose(p_mean, np.array([0, 0, 1, 0]), atol=0.01)
    npt.assert_allclose(p_std, np.array([0, 0, 1, 0]), atol=0.01)
    p_mean, p_std = fish_parametrise.fit_shift(linear, -linear, zero)
    npt.assert_allclose(p_mean, np.array([0, 0, -1, 0]), atol=0.01)
    npt.assert_allclose(p_std, np.array([0, 0, 0, 0]), atol=0.01)


def test_binned_mean_nHits():
    acc = sample_accumulator.make(
        start_acc=stats_accumulator.AlignedStatsAccumulator("mean")
    )
    n_hits_mean, incident_weights = fish_parametrise.binned_mean_nHits(acc)
    n_incident, n_distance = acc.total_events[1:-1].shape
    assert n_hits_mean.shape == (n_incident, n_distance)
    assert incident_weights.shape == (n_incident,)
    assert set(n_hits_mean.astype(int).flatten()) == {4}

    # now modify the accumulator to have a known mean nHits
    acc.total_events = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]])
    # axis order (incident, distance, x, z)
    acc.counts_hist = np.zeros((3, 4, 5, 5))
    n_hits_mean, incident_weights = fish_parametrise.binned_mean_nHits(acc)
    npt.assert_allclose(n_hits_mean, np.zeros((1, 4)))
    npt.assert_allclose(incident_weights, np.array([4]))

    # one hit in every bin
    acc.counts_hist = np.ones((3, 4, 5, 5))
    n_hits_mean, incident_weights = fish_parametrise.binned_mean_nHits(acc)
    npt.assert_allclose(n_hits_mean, np.full((1, 4), 5 * 5))
    npt.assert_allclose(incident_weights, np.array([4]))


def test_fit_mean_nHits():
    n_hits_mean = np.zeros((5, 5))
    incident_weights = np.ones(5)
    incident_centers = np.arange(5)
    distance_centers = np.arange(5)

    n_hits_mean[:, 1:-1] = 1
    n_hits_mean[:, 2] = 2

    mu, varience, height = fish_parametrise.fit_mean_nHits(
        n_hits_mean, incident_weights, incident_centers, distance_centers
    )
    npt.assert_allclose(mu, 2)
    npt.assert_allclose(varience, 0.5)
    npt.assert_allclose(height, [0, 2.26], atol=0.01)

    n_hits_mean *= np.arange(5)[:, None] + 1
    mu, varience, height = fish_parametrise.fit_mean_nHits(
        n_hits_mean, incident_weights, incident_centers, distance_centers
    )
    npt.assert_allclose(mu, 2)
    npt.assert_allclose(varience, 0.5)
    npt.assert_allclose(height, [2.26, 2.26], atol=0.01)

    n_hits_mean = np.zeros((5, 5))
    n_hits_mean[:, 1:-1] = 1
    n_hits_mean[:, 2] = 2
    incident_weights = np.arange(5)

    mu, varience, height = fish_parametrise.fit_mean_nHits(
        n_hits_mean, incident_weights, incident_centers, distance_centers
    )
    npt.assert_allclose(mu, 2)
    npt.assert_allclose(varience, 0.5)
    npt.assert_allclose(height, [0, 2.26], atol=0.01)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_binned_cv_nHits():
    acc = sample_accumulator.make(
        start_acc=stats_accumulator.AlignedStatsAccumulator("mean")
    )
    cv_nHits = fish_parametrise.binned_cv_nHits(acc)
    n_incident, n_distance = acc.total_events[1:-1].shape
    npt.assert_allclose(cv_nHits, np.zeros(n_distance))

    # now modify the accumulator to have a known cv nHits
    # two events at each shift
    acc.total_events = np.array([[0, 0, 0, 0], [2, 2, 2, 2], [0, 0, 0, 0]])
    # counts are incident, distance, x, y
    acc.counts_hist = np.zeros((3, 4, 5, 5))
    # counts sq are incident, distance
    acc.evt_counts_sq_hist = np.zeros((3, 4))
    cv_nHits = fish_parametrise.binned_cv_nHits(acc)
    # no hist means 0 cv
    npt.assert_allclose(cv_nHits, np.zeros(4))

    # now for the incident bin 1, add one hit
    acc.counts_hist[1, 1, 1, 1] = 1
    acc.evt_counts_sq_hist[1, 1] = 1
    # mean hits for all shift bins is [0, 1/2, 0, 0]
    # the mean_sq is [0, 1/4, 0, 0]
    # the std is sqrt(1/4) = 0.5
    # the cv is 0.5/0.5 = 1
    cv_nHits = fish_parametrise.binned_cv_nHits(acc)
    npt.assert_allclose(cv_nHits, [0, 1.0, 0.0, 0.0], atol=0.01)


def test_fit_cv_nHits():
    distance = np.arange(5)
    cv_nHits = np.zeros(5)
    cv_fit = fish_parametrise.fit_cv_nHits(distance, cv_nHits)
    npt.assert_allclose(cv_fit, np.zeros(13))

    cv_nHits = np.arange(5)
    cv_fit = fish_parametrise.fit_cv_nHits(distance, cv_nHits)
    expected = np.zeros(13)
    expected[-2] = 1
    npt.assert_allclose(cv_fit, expected, atol=0.1)


def test_binned_mean_energy():
    acc = sample_accumulator.make(
        start_acc=stats_accumulator.AlignedStatsAccumulator("mean")
    )
    mean_energy = fish_parametrise.binned_mean_energy(acc)
    n_incident, n_distance = acc.total_events[1:-1].shape
    npt.assert_allclose(mean_energy, np.full((n_incident, n_distance), 1.0))

    acc.evt_mean_E_hist[:] = 0
    mean_energy = fish_parametrise.binned_mean_energy(acc)
    npt.assert_allclose(mean_energy, np.zeros((n_incident, n_distance)))


def test_fit_mean_energy_vs_distance():
    expected_mu = [0, 1, 2, 2]
    expected_beta = [1, 1, 2, 1]
    expected_height = [1, 1, 2, 2]
    expected_lift = [0, 0, 1, 0]

    n_distance = 100
    n_incident = 4
    distance_centers = np.linspace(-0.98, 0.98, n_distance)
    mean_energy = np.zeros((n_incident, n_distance))

    for i in range(n_incident):
        distribution = torch.distributions.Gumbel(expected_mu[i], expected_beta[i])
        pdf = distribution.log_prob(torch.tensor(distance_centers)).exp().numpy()
        # we arent doing the 1/beta gumbel normalisation
        mean_energy[i] = pdf * expected_beta[i] * expected_height[i] + expected_lift[i]
    (
        found_mu,
        found_beta,
        found_height,
        found_lift,
    ) = fish_parametrise.fit_mean_energy_vs_distance(mean_energy, distance_centers)
    npt.assert_allclose(found_mu, expected_mu, atol=0.1)
    npt.assert_allclose(found_beta, expected_beta, atol=0.1)
    npt.assert_allclose(found_height, expected_height, atol=0.1)
    npt.assert_allclose(found_lift, expected_lift, atol=0.1)


def test_fit_mean_energy():
    incident_centers = np.arange(5)
    # mu and beta get a quadratic fit, height and lift get a linear fit
    found_mu = np.zeros(5)
    found_beta = np.ones(5)
    found_height = np.ones(5)
    found_lift = np.zeros(5)
    mu, beta, height, lift = fish_parametrise.fit_mean_energy(
        incident_centers, found_mu, found_beta, found_height, found_lift
    )
    npt.assert_allclose(mu, [0, 0], atol=0.1)
    npt.assert_allclose(beta, [0, 1], atol=0.1)
    npt.assert_allclose(height, [0, 1], atol=0.1)
    npt.assert_allclose(lift, [0, 0], atol=0.1)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_binned_cv_energy():
    acc = sample_accumulator.make(
        start_acc=stats_accumulator.AlignedStatsAccumulator("mean")
    )
    cv_energy = fish_parametrise.binned_cv_energy(acc)
    n_incident, n_distance = acc.total_events[1:-1].shape
    npt.assert_allclose(cv_energy, np.zeros(n_distance))

    # now modify the accumulator to have a known cv energy
    acc.total_events = np.array([[0, 0, 0, 0], [2, 2, 2, 2], [0, 0, 0, 0]])
    acc.evt_mean_E_hist = np.zeros((3, 4))
    acc.evt_mean_E_sq_hist = np.zeros((3, 4))
    cv_energy = fish_parametrise.binned_cv_energy(acc)
    npt.assert_allclose(cv_energy, np.zeros(4))

    acc.evt_mean_E_hist[1, 1] = 1
    acc.evt_mean_E_sq_hist[1, 1] = 1
    cv_energy = fish_parametrise.binned_cv_energy(acc)
    npt.assert_allclose(cv_energy, [0, 1.0, 0.0, 0.0], atol=0.01)


def test_fit_cv_energy():
    distance = np.arange(5)
    cv_energy = np.zeros(5)
    cv_fit = fish_parametrise.fit_cv_energy(distance, cv_energy)
    npt.assert_allclose(cv_fit, np.zeros(13))

    cv_energy = np.arange(5)
    cv_fit = fish_parametrise.fit_cv_energy(distance, cv_energy)
    expected = np.zeros(13)
    expected[-2] = 1
    npt.assert_allclose(cv_fit, expected, atol=0.1)


def test_binned_stdEWithin_vs_incident():
    acc = sample_accumulator.make(
        start_acc=stats_accumulator.AlignedStatsAccumulator("mean")
    )
    stdEWithin = fish_parametrise.binned_stdEWithin_vs_incident(acc)
    n_incident, n_distance = acc.total_events[1:-1].shape
    npt.assert_allclose(stdEWithin, np.zeros((n_incident,)))

    # now modify the accumulator to have a known stdEWithin
    acc.total_events = np.array([[0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0]])
    acc.pnt_mean_E_sq_hist = np.zeros((3, 4))
    acc.evt_mean_E_sq_hist = np.zeros((3, 4))
    stdEWithin = fish_parametrise.binned_stdEWithin_vs_incident(acc)
    npt.assert_allclose(stdEWithin, np.zeros(1))

    # both events have one point with 2 and one with 3
    acc.pnt_mean_E_sq_hist[1, 1] = 6.5 + 6.5
    acc.evt_mean_E_sq_hist[1, 1] = 6.25 + 6.25
    stdEWithin = fish_parametrise.binned_stdEWithin_vs_incident(acc)
    npt.assert_allclose(stdEWithin, [0.5], atol=0.01)


def test_fit_stdEWithin_vs_incident():
    incident_centers = np.arange(5)
    stdEWithin = np.zeros(5)
    incident_fit = fish_parametrise.fit_stdEWithin_vs_incident(
        incident_centers, stdEWithin
    )
    npt.assert_allclose(incident_fit, np.zeros(2))

    stdEWithin = np.arange(5)
    incident_fit = fish_parametrise.fit_stdEWithin_vs_incident(
        incident_centers, stdEWithin
    )
    npt.assert_allclose(incident_fit, [1, 0], atol=0.1)


def test_rescaled_stdEWithin_vs_distance():
    acc = sample_accumulator.make(
        start_acc=stats_accumulator.AlignedStatsAccumulator("mean")
    )
    incident_fit = np.array([0, 2])
    n_incident, n_distance = acc.total_events[1:-1].shape
    incidents = np.arange(n_incident)
    rescaled = fish_parametrise.rescaled_stdEWithin_vs_distance(
        acc, incident_fit, incidents
    )
    npt.assert_allclose(rescaled, np.zeros(n_distance))

    # now modify the accumulator to have a known stdEWithin
    acc.total_events = np.array([[0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0]])
    acc.pnt_mean_E_sq_hist = np.zeros((3, 4))
    acc.evt_mean_E_sq_hist = np.zeros((3, 4))
    incidents = np.array([1])
    rescaled = fish_parametrise.rescaled_stdEWithin_vs_distance(
        acc, incident_fit, incidents
    )
    npt.assert_allclose(rescaled, np.zeros(4))

    # both events have one point with 2 and one with 3
    acc.pnt_mean_E_sq_hist[1, 1] = 6.5 + 6.5
    acc.evt_mean_E_sq_hist[1, 1] = 6.25 + 6.25
    rescaled = fish_parametrise.rescaled_stdEWithin_vs_distance(
        acc, incident_fit, incidents
    )
    expected = np.zeros(4)
    expected[1] = 0.25
    npt.assert_allclose(rescaled, expected, atol=0.01)


def test_fit_stdEWithin_vs_distance():
    expected_mu = [0, 1, 2, 2]
    expected_beta = [1, 1, 2, 1]
    expected_height = [1, 1, 2, 2]
    expected_lift = [0, 0, 0.1, 0]

    n_distance = 100
    distance_centers = np.linspace(-0.98, 0.98, n_distance)

    for i in range(len(expected_mu)):
        distribution = torch.distributions.Gumbel(expected_mu[i], expected_beta[i])
        pdf = distribution.log_prob(torch.tensor(distance_centers)).exp().numpy()
        # we arent doing the 1/beta gumbel normalisation
        stdEwithin = pdf * expected_beta[i] * expected_height[i] + expected_lift[i]
        (
            found_mu,
            found_beta,
            found_height,
            found_lift,
        ) = fish_parametrise.fit_stdEWithin_vs_distance(distance_centers, stdEwithin)
        npt.assert_allclose(found_mu, expected_mu[i], atol=0.1)
        npt.assert_allclose(found_beta, expected_beta[i], atol=0.1)
        npt.assert_allclose(found_height, expected_height[i], atol=0.1)
        npt.assert_allclose(found_lift, expected_lift[i], atol=0.1)


def test_radial_bin_centers():
    # really only check the form of this, as we are unlikely to use these
    # radial functions in anger
    acc = sample_accumulator.make(
        start_acc=stats_accumulator.AlignedStatsAccumulator("mean")
    )
    radial_bin_centers = fish_parametrise.radial_bin_centers(acc)
    _, _, n_x_bins, n_z_bins = acc.counts_hist.shape
    n_radial_centers = n_x_bins * n_z_bins
    assert radial_bin_centers.shape == (n_radial_centers,)
    assert np.all(radial_bin_centers >= 0)


def test_binned_radial_probs():
    # again, just a form check
    acc = sample_accumulator.make(
        start_acc=stats_accumulator.AlignedStatsAccumulator("mean")
    )
    n_incident, n_distance, n_x_bins, n_z_bins = acc.counts_hist.shape
    n_radial_centers = n_x_bins * n_z_bins
    radial_centers = np.arange(n_radial_centers)
    radial_probs = fish_parametrise.binned_radial_probs(acc, radial_centers)
    assert radial_probs.shape == (n_distance, n_radial_centers)
    assert np.all(radial_probs >= 0)


def test_fit_radial_probs():
    distance_centers = np.arange(5)
    radial_centers = np.arange(5)
    radial_probs = np.zeros((5, 5))
    radial_probs[:, 0] = 10
    radial_probs[:, 1] = 5
    core_fit, to_tail_fit, p_core_fit = fish_parametrise.fit_radial_probs(
        distance_centers, radial_centers, radial_probs, n_attempts=1
    )
    assert core_fit.shape == (3,)
    assert to_tail_fit.shape == (3,)
    assert p_core_fit.shape == (3,)


parametrization_float_attrs = [
    "nhm_vs_dist_mean",
    "nhm_vs_dist_var",
    "sewe_vs_dist_mu",
    "sewe_vs_dist_beta",
    "sewe_vs_dist_height",
    "sewe_vs_dist_lift",
]

parametrization_tensor_attrs = {
    "shift_mean": (4,),
    "shift_std": (4,),
    "cv_energy": (13,),
    "cv_n_hits": (13,),
    "nhm_vs_incident_height": (2,),
    "me_mu": (2,),
    "me_beta": (2,),
    "me_height": (2,),
    "me_lift": (2,),
    "sewe_vs_incident": (2,),
    "radial_vs_dist_core": (3,),
    "radial_vs_dist_to_tail": (3,),
    "radial_vs_dist_p_core": (3,),
}


def check_parametrization_attrs(param):
    for attr in parametrization_float_attrs:
        assert hasattr(param, attr)
        try:
            float(getattr(param, attr))
        except ValueError:
            raise AssertionError(f"Attribute {attr} is not float like")
    for attr, shape in parametrization_tensor_attrs.items():
        assert hasattr(param, attr)
        assert isinstance(getattr(param, attr), torch.Tensor)
        assert getattr(param, attr).shape == shape


def test_Parametrisation(tmpdir):
    # all the maths functions are tested above, so we just need to check the
    # object has the right attributes, and keeps them the same after saving and loading.
    acc = sample_accumulator.make(
        start_acc=stats_accumulator.AlignedStatsAccumulator("mean")
    )

    try:
        # can't work with events that all have identical point energy
        param = fish_parametrise.Parametrisation(acc)
        raise Exception("Should have raised a NotImplementedError")
    except NotImplementedError:
        pass

    # need to add some random points
    sample_accumulator.add_random(acc)
    param = fish_parametrise.Parametrisation(acc, n_attempts=1)

    check_parametrization_attrs(param)

    save_path = str(tmpdir / "param.npz")
    param.save(save_path)
    loaded_param = fish_parametrise.Parametrisation.load(save_path)

    check_parametrization_attrs(loaded_param)
    for attr in parametrization_float_attrs:
        assert getattr(param, attr) == getattr(loaded_param, attr)
    for attr in parametrization_tensor_attrs:
        npt.assert_allclose(getattr(param, attr), getattr(loaded_param, attr))

    # should also be possible to load from a fish model
    test_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(test_dir, "example_fish_model.npz")
    loaded_param = fish_parametrise.Parametrisation.load(model_path)
    check_parametrization_attrs(loaded_param)
