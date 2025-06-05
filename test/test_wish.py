""" Module to test the wish module """
import numpy as np
import numpy.testing as npt
import torch

from pointcloud.models import wish
from pointcloud.utils.stats_accumulator import HighLevelStats

from helpers import sample_accumulator, config_creator


def test_construct_symmetric_2by2():
    # there is a trivial case where the eigenvectors
    # unit vectors
    angle = torch.tensor([0.0])
    matrix = wish.construct_symmetric_2by2(angle, torch.Tensor([1.0, 2.0]))
    npt.assert_almost_equal(matrix, np.array([[1.0, 0.0], [0.0, 2.0]]))
    # or we can rotate by pi/2, and we should get the same
    angle = torch.tensor([np.pi / 2])
    matrix = wish.construct_symmetric_2by2(angle, torch.Tensor([1.0, 2.0]))
    npt.assert_almost_equal(matrix, np.array([[2.0, 0.0], [0.0, 1.0]]))

    # for any value, we should be able to extract the eigenvalues
    eigenvector_angle = 1.34
    angle = torch.tensor([eigenvector_angle])
    eigenvector = np.array([np.cos(eigenvector_angle), np.sin(eigenvector_angle)])
    eigenvalue_1 = 5.0
    eigenvalue_angle_2 = eigenvector_angle + np.pi / 2
    eigenvector_2 = np.array([np.cos(eigenvalue_angle_2), np.sin(eigenvalue_angle_2)])
    eigenvalue_2 = 3.0
    matrix = wish.construct_symmetric_2by2(
        angle, torch.Tensor([eigenvalue_1, eigenvalue_2])
    )
    npt.assert_almost_equal(matrix @ eigenvector, eigenvalue_1 * eigenvector, decimal=3)
    npt.assert_almost_equal(
        matrix @ eigenvector_2, eigenvalue_2 * eigenvector_2, decimal=3
    )


def test_LinearFitSymmetricPositive2by2():
    # if we give everything a 0 gradient we should get a no change from changing
    # the variable
    fit = wish.LinearFitSymmetricPositive2by2(
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        torch.tensor([0.0, 0.0]),
        torch.tensor([1.0, 1.0]),
    )
    found_1 = fit.fit(0.0).detach().numpy()
    found_2 = fit.fit(1.0).detach().numpy()
    npt.assert_almost_equal(found_1, found_2)
    # if both the eigenvalues get the same gradient it should scale the whole matrix
    fit = wish.LinearFitSymmetricPositive2by2(
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        torch.tensor([1.0, 1.0]),
        torch.tensor([0.0, 0.0]),
    )
    found_1 = fit.fit(1.0).detach().numpy()
    found_2 = fit.fit(3.0).detach().numpy()
    npt.assert_almost_equal(found_1 * 3, found_2, decimal=3)


def test_PolyFit1DGauss():
    int_zero = torch.tensor(0)
    int_one = torch.tensor(1)
    # with all values being 0 we get an edge case
    mean_grad = torch.tensor(0.0)
    mean_intercept = torch.tensor(0.0)
    std_grad = torch.tensor(0.0)
    std_intercept = torch.tensor(0.0)
    fit = wish.PolyFit1DGauss([mean_grad, mean_intercept], [std_grad, std_intercept])
    found_std = fit.get_standarddev(0.0)
    npt.assert_almost_equal(found_std, 0.0)
    found_mean = fit.get_mean(0.0)
    npt.assert_almost_equal(found_mean, 0.0)
    # but it's not possible to sample this edge case

    # if the gradient is 0, the rate should be constant
    mean_intercept = torch.tensor(5.0)
    std_intercept = torch.tensor(1.0)
    fit = wish.PolyFit1DGauss([mean_grad, mean_intercept], [std_grad, std_intercept])
    found_std = fit.get_standarddev(0.0)
    npt.assert_almost_equal(found_std, 1.0)
    found_std = fit.get_standarddev(1.0)
    npt.assert_almost_equal(found_std, 1.0)
    found_mean = fit.get_mean(0.0)
    npt.assert_almost_equal(found_mean, 5.0)
    found_mean = fit.get_mean(1.0)
    npt.assert_almost_equal(found_mean, 5.0)
    found_prob_0 = fit.calculate_log_probability(0.0, int_zero)
    found_prob_1 = fit.calculate_log_probability(0.0, int_one)
    assert found_prob_0 < found_prob_1
    found_sample = fit.draw_sample(0.0, 3)
    assert found_sample.shape == (3,)

    # try a standard linear fit
    mean_grad = torch.tensor(1.0)
    mean_intercept = torch.tensor(0.0)
    std_grad = torch.tensor(1.0)
    std_intercept = torch.tensor(0.0)
    fit = wish.PolyFit1DGauss(
        [mean_grad, mean_intercept],
        [std_grad, std_intercept],
    )
    found_std = fit.get_standarddev(1.0)
    npt.assert_almost_equal(found_std, 1.0)
    found_std = fit.get_standarddev(3.0)
    npt.assert_almost_equal(found_std, 3.0)
    found_mean = fit.get_mean(1.0)
    npt.assert_almost_equal(found_mean, 1.0)
    found_mean = fit.get_mean(3.0)
    npt.assert_almost_equal(found_mean, 3.0)
    found_prob_0 = fit.calculate_log_probability(5.0, int_zero)
    found_prob_1 = fit.calculate_log_probability(5.0, int_one)
    assert found_prob_0 < found_prob_1
    found_sample = fit.draw_sample(1.0, 3)
    assert found_sample.shape == (3,)

    # try one quadratic fit
    fit = wish.PolyFit1DGauss(
        torch.tensor([0.0, 1.0, -2.0]),
        torch.tensor([1.0, 1.0, 0.0]),
    )
    found_mean = fit.get_mean(0.0)
    npt.assert_almost_equal(found_mean, -2)
    found_mean = fit.get_mean(1.0)
    npt.assert_almost_equal(found_mean, -1)
    found_mean = fit.get_mean(2.0)
    npt.assert_almost_equal(found_mean, 0)
    found_std = fit.get_standarddev(0.0)
    npt.assert_almost_equal(found_std, 0)
    found_std = fit.get_standarddev(1.0)
    npt.assert_almost_equal(found_std, 2)
    found_std = fit.get_standarddev(2.0)
    npt.assert_almost_equal(found_std, 6)

    found_sample = fit.draw_sample(0.0, 100)
    assert found_sample.shape == (100,)


def test_LinearFit2DGauss():
    class DummyCovarienceMatrix:
        value_table = {
            0.0: torch.Tensor([[1.0, 0.0], [0.0, 1.0]]),
            1.0: torch.Tensor([[2.0, 0.0], [0.0, 2.0]]),
        }

        def fit(self, x):
            return self.value_table[x]

    dummy = DummyCovarienceMatrix()
    fit = wish.LinearFit2DGauss(dummy)
    log_prob_1 = fit.calculate_log_probability(0.0, torch.tensor([0.0, 0.0]))
    log_prob_2 = fit.calculate_log_probability(0.0, torch.tensor([1.0, 0.0]))
    log_prob_3 = fit.calculate_log_probability(0.0, torch.tensor([0.0, 1.0]))
    assert log_prob_1 > log_prob_2
    assert log_prob_1 > log_prob_3
    npt.assert_almost_equal(log_prob_2, log_prob_3)
    log_prob_4 = fit.calculate_log_probability(1.0, torch.tensor([0.0, 0.0]))
    log_prob_5 = fit.calculate_log_probability(1.0, torch.tensor([1.0, 0.0]))
    log_prob_6 = fit.calculate_log_probability(1.0, torch.tensor([0.0, 1.0]))
    assert log_prob_4 > log_prob_5
    assert log_prob_4 > log_prob_6
    npt.assert_almost_equal(log_prob_5, log_prob_6)
    assert log_prob_1 > log_prob_4
    assert log_prob_2 > log_prob_5
    assert log_prob_3 > log_prob_6

    # try sampling
    sample = fit.draw_sample(0.0, (100,))
    assert sample.shape == (100, 2)


def test_PolyFit1DLogNormal():
    fit = wish.PolyFit1DLogNormal(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]))
    found_mean = fit.get_mean(0.0)
    npt.assert_almost_equal(found_mean, 1.0)
    found_std = fit.get_standarddev(0.0)
    npt.assert_almost_equal(found_std, 1.0)
    found_mean = fit.get_mean(1.0)
    npt.assert_almost_equal(found_mean, 1.0)
    found_std = fit.get_standarddev(1.0)
    npt.assert_almost_equal(found_std, 1.0)
    found_prob_0 = fit.calculate_log_probability(0.0, torch.tensor(0.0))
    assert found_prob_0 < -1000  # we know the position is clipped, and the prob is 0
    found_prob_1 = fit.calculate_log_probability(1.0, torch.tensor(1.0))

    mean = 1.0
    std = 1.0
    mean_unlogged = np.log(mean / np.sqrt(1 + std**2 / mean**2))
    var_unlogged = np.log(1 + std**2 / mean**2)
    expected = np.exp(
        -((np.log(1) - mean_unlogged) ** 2) / (2 * var_unlogged)
    ) / np.sqrt(2 * np.pi * var_unlogged)
    # this appears to have lots of numerical instability
    assert np.abs(found_prob_1 - np.log(expected)) < 0.1

    found_prob_2 = fit.calculate_log_probability(-1.0, torch.tensor(1.0))
    assert np.abs(found_prob_2 - np.log(expected)) < 0.1

    found_sample = fit.draw_sample(0.0, 3)
    assert found_sample.shape == (3,)

    # try a standard linear fit
    fit = wish.PolyFit1DLogNormal(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]))
    found_mean = fit.get_mean(1.0)
    npt.assert_almost_equal(found_mean, 1.0)
    found_std = fit.get_standarddev(1.0)
    npt.assert_almost_equal(found_std, 1.0)
    found_mean = fit.get_mean(2.0)
    npt.assert_almost_equal(found_mean, 2.0)
    found_std = fit.get_standarddev(2.0)
    npt.assert_almost_equal(found_std, 2.0)

    # generate 10k values and check the mean and std are close
    values = fit.draw_sample(1.0, 100_000)
    found_mean = values.mean()
    found_std = values.std()
    npt.assert_almost_equal(found_mean, 1.0, decimal=1)
    npt.assert_almost_equal(found_std, 1.0, decimal=1)


def test_PolyFit2DGFlash():
    fit = wish.PolyFit2DGFlash(
        torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([1.0])
    )
    Rc, Rt_extend, p = fit.get_distribution_args(0.0)
    npt.assert_almost_equal(Rc, 1.0, decimal=4)
    npt.assert_almost_equal(Rt_extend, 0.0, decimal=4)
    npt.assert_almost_equal(p, 1.0, decimal=4)
    Rc, Rt_extend, p = fit.get_distribution_args(1.0)
    npt.assert_almost_equal(Rc, 1.0, decimal=4)
    npt.assert_almost_equal(Rt_extend, 0.0, decimal=4)
    npt.assert_almost_equal(p, 1.0, decimal=4)

    prob_0 = fit.calculate_probability(0.0, torch.tensor([0.0, 0.0]))
    npt.assert_almost_equal(prob_0, 0)

    # Rt = 1, Rc = 1, p = 1
    # p(r) = p*2*r*Rc**2/(Rc**2 + r**2)**2 + (1-p)*r*2*Rt**2/(Rt**2 + r**2)**2
    # p(r) = 2*r/(r**2 + 1)**2
    # jacobien = 1/(2*pi)
    prob_1 = fit.calculate_probability(0.0, torch.tensor([1.0, 0.0]))
    npt.assert_almost_equal(prob_1, 2 / (4 * 2 * np.pi))

    prob_2 = fit.calculate_probability(0.0, torch.tensor([0.0, 1.0]))
    npt.assert_almost_equal(prob_2, 2 / (4 * 2 * np.pi))

    # Rc = 1., Rt = 1., p = 1.
    # r = sqrt(2)
    # p(r) = 2*sqrt(2)/(1 + 2)**2
    # jac = 1/(2*pi*sqrt(2))

    prob_3 = fit.calculate_probability(1.0, torch.tensor([1.0, 1.0]))
    npt.assert_almost_equal(prob_3, 1 / (9 * np.pi))

    sample = fit.draw_sample(0.0, (100,))
    assert sample.shape == (100, 2)

    fit = wish.PolyFit2DGFlash(
        torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 1.0]), torch.tensor([2.0, 0.0])
    )
    Rc, Rt_extend, p = fit.get_distribution_args(0.0)
    npt.assert_almost_equal(Rc, 0.0, decimal=4)
    npt.assert_almost_equal(Rt_extend, 1.0, decimal=4)
    npt.assert_almost_equal(p, 0.0, decimal=4)

    Rc, Rt_extend, p = fit.get_distribution_args(1.0)
    npt.assert_almost_equal(Rc, 1.0, decimal=4)
    npt.assert_almost_equal(Rt_extend, 0.0, decimal=4)
    npt.assert_almost_equal(p, 1.0, decimal=4)

    Rc, Rt_extend, p = fit.get_distribution_args(2.0)
    npt.assert_almost_equal(Rc, 2.0, decimal=4)
    npt.assert_almost_equal(Rt_extend, 0.0, decimal=4)
    npt.assert_almost_equal(p, 1.0, decimal=4)

    Rc, Rt_extend, p = fit.get_distribution_args(-1.0)
    npt.assert_almost_equal(Rc, 0.0, decimal=4)
    npt.assert_almost_equal(Rt_extend, 2.0, decimal=4)
    npt.assert_almost_equal(p, 0.0, decimal=4)

    prob_0 = fit.calculate_probability(0.0, torch.tensor([0.0, 0.0]))
    npt.assert_almost_equal(prob_0, 0, decimal=4)

    # Rc = 0., Rt = 1., p = 0.
    # p(r) = 2*r/(r**2 + 1)**2
    # jacobien = 1/(2*pi)

    prob_1 = fit.calculate_probability(0.0, torch.tensor([1.0, 0.0]))
    npt.assert_almost_equal(prob_1, 1 / (4 * np.pi), decimal=4)

    # Rc = 1., Rt = 0., p = 1.
    # p(r) = 2*r/(r**2 + 1)**2
    # jacobien = 1/(2*pi)
    prob_2 = fit.calculate_probability(1.0, torch.tensor([0.0, 1.0]))
    npt.assert_almost_equal(prob_2, 1 / (4 * np.pi), decimal=4)

    sample = fit.draw_sample(0.0, (100,))
    assert sample.shape == (100, 2)
    sample = fit.draw_sample(1.0, (100,))
    assert sample.shape == (100, 2)


def test_padding_mask():
    events = torch.ones(4, 10)
    mask = wish.padding_mask(events)
    assert all(mask)
    events = torch.zeros(4, 10)
    mask = wish.padding_mask(events)
    assert not any(mask)
    events[0, 3] = 1
    mask = wish.padding_mask(events)
    assert mask[0]
    assert not any(mask[1:])


def test_LayersBackBone():
    backbone = wish.LayersBackBone(3, 2)
    assert backbone.n_layers == 3
    assert backbone.poly_degree == 2
    assert len(backbone.n_pts_fits) == 3
    assert len(backbone.energy_fits) == 3
    # the backbone parameters should have 4 parameters per layer
    assert len(backbone.all_params) == 3 * 4

    # try calling reset_params and check the chosen params are changed
    # but the others stay static
    prior_energy_means = [
        np.copy(e.mean_coeffs.detach().numpy()) for e in backbone.energy_fits
    ]
    prior_energy_stds = [
        np.copy(e.standarddev_coeffs.detach().numpy()) for e in backbone.energy_fits
    ]
    all_prior_params = [np.copy(p.detach().numpy()) for p in backbone.all_params]

    backbone.reset_params(energy_mean_coeffs={0: np.zeros(3), 2: np.ones(3)})

    new_energy_means = [
        np.copy(e.mean_coeffs.detach().numpy()) for e in backbone.energy_fits
    ]
    new_energy_stds = [
        np.copy(e.standarddev_coeffs.detach().numpy()) for e in backbone.energy_fits
    ]
    new_all_params = [np.copy(p.detach().numpy()) for p in backbone.all_params]

    npt.assert_allclose(prior_energy_means[1], new_energy_means[1])
    npt.assert_allclose(new_energy_means[0], np.zeros(3))
    change = np.abs(new_energy_means[2] - prior_energy_means[2])
    assert np.all(change > 1e-6)
    npt.assert_allclose(new_energy_means[2], np.ones(3))
    change = np.abs(new_energy_means[2] - prior_energy_means[2])
    assert np.all(change > 1e-6)

    npt.assert_allclose(prior_energy_stds[0], new_energy_stds[0])
    npt.assert_allclose(prior_energy_stds[1], new_energy_stds[1])
    npt.assert_allclose(prior_energy_stds[2], new_energy_stds[2])

    changed = np.abs(np.array(new_all_params) - np.array(all_prior_params)) > 1e-6
    assert np.sum(changed) == 6

    coeffs = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [-1, 0, 0], [-1, 0, 1], [0, 0, -1]]
    for energy_mean_coeffs in coeffs:
        for energy_std_coeffs in coeffs:
            for hits_mean_coeffs in coeffs:
                for hits_std_coeffs in coeffs:
                    backbone.reset_params(
                        energy_mean_coeffs={
                            i: np.array(energy_mean_coeffs) for i in range(3)
                        },
                        energy_standarddev_coeffs={
                            i: np.array(energy_std_coeffs) for i in range(3)
                        },
                        n_pts_coeffs={i: np.array(hits_mean_coeffs) for i in range(3)},
                        n_pts_standarddev_coeffs={
                            i: np.array(hits_std_coeffs) for i in range(3)
                        },
                    )
                    # check we can sample
                    layers_samples = backbone.draw_sample(1.0)
                    assert len(layers_samples) == 3
                    finite_list = []
                    for i, sample in enumerate(layers_samples):
                        assert len(sample) == 2, f"sample {i} is {sample}"
                        if torch.isfinite(sample[0]).all():
                            assert sample[0] >= 0, f"n_points {i} is {sample[0]}"
                        if torch.isfinite(sample[1]).all():
                            assert sample[1] >= 0, f"energy {i} is {sample[1]}"
                            finite_list.append(sample)

                    # check if the sample pulled is real
                    if finite_list:
                        # check we can get a likelihood
                        layers_likelihood = backbone.log_likelihood(1.0, finite_list)
                        # don't even check it's a number.... these are weird edge cases
                        assert isinstance(layers_likelihood, torch.Tensor)


def test_WishLayer():
    layer = wish.WishLayer(0, 2)
    assert layer.layer_n == 0
    assert layer.poly_degree == 2

    # try calling reset_params and check the chosen params are changed
    # but the others stay static
    prior_Rt = np.copy(layer.displacement_fit.Rt_coeffs.detach().numpy())
    prior_Rc = np.copy(layer.displacement_fit.Rc_coeffs.detach().numpy())
    prior_p = np.copy(layer.displacement_fit.p_coeffs.detach().numpy())
    prior_energy_stds = np.copy(layer.energy_fit.standarddev_coeffs.detach().numpy())
    all_prior_params = [np.copy(p.detach().numpy()) for p in layer.all_params]

    layer.reset_params(Rt_coeffs=np.ones(3), energy_standarddev_coeffs=np.ones(3))

    new_Rt = np.copy(layer.displacement_fit.Rt_coeffs.detach().numpy())
    new_Rc = np.copy(layer.displacement_fit.Rc_coeffs.detach().numpy())
    new_p = np.copy(layer.displacement_fit.p_coeffs.detach().numpy())
    new_energy_stds = np.copy(layer.energy_fit.standarddev_coeffs.detach().numpy())
    new_all_params = [np.copy(p.detach().numpy()) for p in layer.all_params]

    npt.assert_allclose(new_Rt, np.ones(3))
    change = np.abs(new_Rt - prior_Rt)
    assert np.all(change > 1e-6)
    npt.assert_allclose(new_Rc, prior_Rc)
    npt.assert_allclose(new_p, prior_p)
    npt.assert_allclose(new_energy_stds, np.ones(3))
    change = np.abs(new_energy_stds - prior_energy_stds)
    assert np.all(change > 1e-6)
    changed = np.abs(np.array(new_all_params) - np.array(all_prior_params)) > 1e-6
    assert np.sum(changed) == 6

    # check we can sample
    n_pts = 10
    displacements, energies = layer.draw_sample(1.0, torch.Size([n_pts]))
    assert displacements.shape == (n_pts, 2)
    assert energies.shape == (n_pts,)
    assert all(energies >= 0)


def test_Wish(tmpdir):
    # test each of the functions is callable in princple.
    # all of the internal working are tested in the other tests
    config = config_creator.make("wish", my_tmpdir=tmpdir)

    wish_model = wish.Wish(config)

    # check the created attributes
    assert wish_model.n_layers == 30
    assert wish_model.poly_degree == config.poly_degree
    assert isinstance(wish_model.backbone, wish.LayersBackBone)
    assert len(wish_model.layers) == 30
    for layer in wish_model.layers:
        assert isinstance(layer, wish.WishLayer)

    acc = sample_accumulator.make(add_varients=True)
    hls = HighLevelStats(acc, wish_model.poly_degree)
    wish_model.set_from_stats(hls)

    fake_batch = {
        "event": torch.ones(1, 10, 100, 4),
        "energy": torch.ones(1, 100),
        "points": torch.ones(1, 100),
    }
    loss = wish_model.get_loss(fake_batch)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad

    log_likelihood = wish_model.log_likelihood(fake_batch)
    assert isinstance(log_likelihood, torch.Tensor)

    event_log_likelihood = wish_model.event_log_likelihood(
        torch.ones(1000, 4), torch.ones(1)
    )
    assert isinstance(event_log_likelihood, torch.Tensor)

    layers = wish_model.draw_all_layers(1.0)
    assert isinstance(layers, list)
    for displacement, energy in layers:
        assert isinstance(displacement, torch.Tensor)
        assert displacement.shape[1] == 2
        assert isinstance(energy, torch.Tensor)
        assert len(energy.shape) == 1
        assert torch.all(energy >= 0)

    all_x, all_y, all_z, all_e = wish_model.inference(50.0)
    assert len(all_x) == len(all_y) == len(all_z) == len(all_e)
    assert np.all(np.array(all_e) >= 0)

    incidents = np.arange(1, 10).reshape(-1, 1)
    sample = wish_model.sample(incidents, 100)
    assert sample.shape == (9, 100, 4)
    assert np.all(sample[:, :, 3] >= 0)

    # now chek the method that returns things as a torch tensor.
    found = wish_model.forward(fake_batch)
    for f in found:
        assert isinstance(f, torch.Tensor)

    # Finally, save and load, checking that the parameters are
    # preserved
    backbone_params = [p.detach().numpy() for p in wish_model.backbone.all_params]
    layer_params = [
        p.detach().numpy() for layer in wish_model.layers for p in layer.all_params
    ]
    save_location = str(tmpdir / "wish_model.pth")

    wish_model.save(save_location)
    loaded_model = wish.Wish.load(save_location)

    new_backbone_params = [p.detach().numpy() for p in loaded_model.backbone.all_params]
    for (
        old_p,
        new_p,
    ) in zip(backbone_params, new_backbone_params):
        npt.assert_allclose(old_p, new_p)
    new_layer_params = [
        p.detach().numpy() for layer in loaded_model.layers for p in layer.all_params
    ]
    for (
        old_l,
        new_l,
    ) in zip(layer_params, new_layer_params):
        npt.assert_allclose(old_l, new_l)
