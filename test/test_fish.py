""" Unit tests for fish.py """

import os
import numpy as np
import torch
import numpy.testing as npt

from pointcloud.models import fish_parametrise, fish
from pointcloud.utils.maths import gumbel_params, gumbel
from pointcloud.utils import stats_accumulator

from helpers import sample_accumulator


def get_param():
    test_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(test_dir, "example_fish_model.npz")
    param = fish_parametrise.Parametrisation.load(model_path)
    return param


def test_calc_shifts():
    # try with the example fish model
    param = get_param()
    incidents = np.arange(10) * 10
    shift_gumbel_mu, shift_gumbel_beta = fish.calc_shifts(param, incidents)
    assert shift_gumbel_mu.shape == (10,)
    assert shift_gumbel_beta.shape == (10,)
    assert np.all(shift_gumbel_beta.numpy() >= 0)

    # now set the mean and std to known values, and check the output
    param.shift_mean = torch.tensor([1.0, 0.0])
    param.shift_std = torch.tensor([0.1, 1.0])
    incidents = torch.tensor([0.0, 1.0])
    expected_means = torch.tensor([0.0, 1.0])
    expected_stds = torch.tensor([1.0, 1.1])
    # tested elsewhere
    expected_mu, expected_beta = gumbel_params(expected_means, expected_stds)
    shift_gumbel_mu, shift_gumbel_beta = fish.calc_shifts(param, incidents)
    npt.assert_allclose(shift_gumbel_mu, expected_mu, atol=0.1)
    npt.assert_allclose(shift_gumbel_beta, expected_beta, atol=0.1)


def test_calc_dist_nHits():
    param = get_param()
    incidents = torch.arange(10) * 10
    distances = torch.linspace(-1, 1, 100)
    mean_n_hits, std_n_hits = fish.calc_dist_nHits(param, incidents, distances)
    assert mean_n_hits.shape == (10, 100)
    assert std_n_hits.shape == (10, 100)
    assert np.all(std_n_hits.numpy() >= 0)

    # now set the nhm params to known values, and check the output
    param.nhm_vs_incident_height = torch.tensor([1.0, 0.0])
    param.nhm_vs_dist_mean = -1
    param.nhm_vs_dist_var = 2.0
    param.cv_n_hits = torch.tensor([0.1, 10.0])

    gaussian = np.exp(
        -0.5 * (distances - param.nhm_vs_dist_mean) ** 2 / param.nhm_vs_dist_var
    )
    incident_scales = np.polyval(param.nhm_vs_incident_height, incidents.numpy())
    expected_means = gaussian.numpy() * incident_scales[:, None]
    expected_stds = np.polyval(param.cv_n_hits, distances) * expected_means
    mean_n_hits, std_n_hits = fish.calc_dist_nHits(param, incidents, distances)
    npt.assert_allclose(mean_n_hits, expected_means, rtol=1e-5)
    npt.assert_allclose(std_n_hits, expected_stds, rtol=1e-5)


def test_calc_dist_energy():
    param = get_param()
    incidents = torch.arange(10) * 10
    distances = torch.linspace(-1, 1, 100)
    mean_energy, std_energy = fish.calc_dist_energy(param, incidents, distances)
    assert mean_energy.shape == (10, 100)
    assert std_energy.shape == (10, 100)
    assert np.all(std_energy.numpy() >= 0)

    # now set the me params to known values, and check the output
    param.me_mu = torch.tensor([1.0, 0.0, 1.0])
    param.me_beta = torch.tensor([0.1, 0.0, 0.1])
    param.me_height = torch.tensor([1.0, 1.0])
    param.me_lift = torch.tensor([0.1, 0.0])
    param.cv_energy = torch.tensor([0.1, 10.0])

    numpy_incidents = incidents.numpy()[:, None]
    mu = np.polyval(param.me_mu, numpy_incidents)
    beta = np.polyval(param.me_beta, numpy_incidents)
    height = np.polyval(param.me_height, numpy_incidents)
    lift = np.polyval(param.me_lift, numpy_incidents)

    expected_means = gumbel(distances.numpy(), mu, beta, height, lift)
    expected_stds = np.polyval(param.cv_energy, distances) * expected_means
    mean_energy, std_energy = fish.calc_dist_energy(param, incidents, distances)
    npt.assert_allclose(mean_energy, expected_means, rtol=1e-5, atol=1e-3)
    npt.assert_allclose(std_energy, expected_stds, rtol=1e-5, atol=1e-3)


def test_calc_stdEWithin():
    param = get_param()
    incidents = torch.arange(10) * 10
    distances = torch.linspace(-1, 1, 100)
    std_e_within = fish.calc_stdEWithin(param, incidents, distances)
    assert std_e_within.shape == (10, 100)
    assert np.all(std_e_within.numpy() >= 0)

    # now set the sewe params to known values, and check the output
    param.sewe_vs_incident = torch.tensor([1.0, 0.0])
    param.sewe_vs_dist_mu = 3.1
    param.sewe_vs_dist_beta = 0.1
    param.sewe_vs_dist_height = 1.5
    param.sewe_vs_dist_lift = 0.1

    against_distance = gumbel(
        distances.numpy(),
        param.sewe_vs_dist_mu,
        param.sewe_vs_dist_beta,
        param.sewe_vs_dist_height,
        param.sewe_vs_dist_lift,
    )
    incident_scales = np.polyval(param.sewe_vs_incident, incidents.numpy())
    expected_stds = against_distance * incident_scales[:, None]
    std_e_within = fish.calc_stdEWithin(param, incidents, distances)
    npt.assert_allclose(std_e_within, expected_stds, rtol=1e-5)


def test_calc_radial():
    # since we won't actually use the radial distribution, we can just test that it runs
    param = get_param()
    distances = torch.linspace(-1, 1, 100)
    core, to_tail, prob = fish.calc_radial(param, distances)
    assert core.shape == (100,)
    assert to_tail.shape == (100,)
    assert prob.shape == (100,)
    assert np.all(core.numpy() >= 0)
    assert np.all(to_tail.numpy() >= 0)
    assert np.all(prob.numpy() >= 0)
    assert np.all(prob.numpy() <= 1)


def check_fish_params(model):
    assert isinstance(model.parametrisation, fish_parametrise.Parametrisation)
    assert isinstance(model.distances, torch.Tensor)
    assert isinstance(model.layer_depths, torch.Tensor)
    assert len(model.distances) >= len(model.layer_depths)


def test_Fish(tmpdir):
    # test init
    acc = sample_accumulator.make(
        start_acc=stats_accumulator.AlignedStatsAccumulator("mean")
    )
    sample_accumulator.add_random(acc)
    param = fish_parametrise.Parametrisation(acc, 1)
    model = fish.Fish(param)
    n_depths = len(model.layer_depths)
    check_fish_params(model)

    # test save
    save_path = str(tmpdir.join("fish_model.npz"))
    model.save(save_path)
    assert os.path.exists(save_path)

    # test load
    model2 = fish.Fish.load(save_path)
    check_fish_params(model2)

    # test calc_stats
    n_incidents = 10
    incidents = torch.arange(n_incidents) * 10
    stats = model.calc_stats(incidents, True, True)
    stats2 = model2.calc_stats(incidents, True, True)
    expected_keys = [
        "shift_gumbel_mu",
        "shift_gumbel_beta",
        "n_hits_weibull_scale",
        "n_hits_weibull_concen",
        "energy_lognorm_loc",
        "energy_lognorm_scale",
        "stdEWithin_lognorm_loc",
        "stdEWithin_lognorm_scale",
        "radial_core",
        "radial_to_tail",
        "radial_prob",
    ]
    n_distances = len(model.distances)
    for key in expected_keys:
        assert key in stats
        assert key in stats2
        value = stats[key]
        if "shift" in key:
            assert value.shape == (n_incidents,)
        elif "radial" in key:
            assert value.shape == (n_distances,)
        else:
            assert value.shape == (n_incidents, n_distances)
        assert torch.allclose(stats[key], stats2[key])

    # test sample_axial_components
    components = model.sample_axial_components(stats)
    assert "shift" in components
    assert components["shift"].shape == (n_incidents,)
    assert "n_hits" in components
    assert components["n_hits"].shape == (n_incidents, n_distances)
    assert np.all(components["n_hits"].numpy() >= 0)
    assert "mean_energy" in components
    assert components["mean_energy"].shape == (n_incidents, n_distances)
    assert np.all(components["mean_energy"].numpy() >= 0)

    # test clip_n_hits
    n_hits = torch.full((4, 5), 20)
    # there are 100 hits in each event, so this should not change
    reduced_n_hits, energy_rescale = model.clip_n_hits(n_hits, 100)
    npt.assert_allclose(reduced_n_hits, n_hits)
    npt.assert_allclose(energy_rescale, np.ones((4, 5)))

    # now lets move it slightly down
    reduced_n_hits, energy_rescale = model.clip_n_hits(n_hits, 99)
    npt.assert_allclose(reduced_n_hits, n_hits - 1)
    assert np.all(energy_rescale.numpy() >= 1)

    # move down by 50 percent
    reduced_n_hits, energy_rescale = model.clip_n_hits(n_hits, 50)
    npt.assert_allclose(reduced_n_hits, n_hits / 2)
    npt.assert_allclose(energy_rescale, 2)

    # test combine_components
    event = model.combine_components(stats, components, 100)
    assert event.shape == (n_incidents, 100, 4)
    # energy should always be positive
    assert np.all(event[..., 3].numpy() >= 0)

    # test sample
    event = model.sample(incidents[:, None], 100)
    assert event.shape == (n_incidents, 100, 4)
    # energy should always be positive
    assert np.all(event[..., 3].numpy() >= 0)


# TODO?
# def test_load_fish_from_accumulator():
#    pass
#
#
# def test_accumulate_and_load_fish():
#    pass
