""" Module to test the maths module """
import numpy as np
import numpy.testing as npt
import torch

from pointcloud.utils import maths


def test_torch_polyval():
    coefficients = torch.tensor([1.0])
    x = torch.linspace(-1, 1, 100)
    y = maths.torch_polyval(coefficients, x)
    npt.assert_allclose(y, 1, atol=1e-6)
    coefficients = torch.tensor([0.0, 1.0])
    y = maths.torch_polyval(coefficients, x)
    npt.assert_allclose(y, 1, atol=1e-6)
    coefficients = torch.tensor([0.0, 1.0, 1.0])
    y = maths.torch_polyval(coefficients, x)
    npt.assert_allclose(y, x + 1, atol=1e-6)
    coefficients = torch.tensor([1.0, 1.0, 1.0])
    y = maths.torch_polyval(coefficients, x)
    npt.assert_allclose(y, x**2 + x + 1, atol=1e-6)
    coefficients = torch.tensor([-1.0, 0.0, 1.0])
    y = maths.torch_polyval(coefficients, x)
    npt.assert_allclose(y, -(x**2) + 1, atol=1e-6)


def test_gaussian():
    xs = np.linspace(-10, 10, 100)
    ys = maths.gaussian(xs, 0, 1, 1)
    assert np.argmax(ys) == 49
    npt.assert_allclose(ys[49], 1, atol=1e-2)
    npt.assert_allclose(ys[:49], ys[51:][::-1], atol=1e-2)

    ys = maths.gaussian(xs, 0, 1, 2)
    assert np.argmax(ys) == 49
    npt.assert_allclose(ys[49], 2, atol=1e-1)

    ys = maths.gaussian(xs, 10, 1, 0)
    npt.assert_allclose(ys, 0, atol=1e-6)

    ys = maths.gaussian(xs, -10, 1, 1)
    assert np.argmax(ys) == 0
    npt.assert_allclose(ys[0], 1, atol=1e-1)

    ys_narrow = maths.gaussian(xs, 0, 0.1, 1)
    ys_wide = maths.gaussian(xs, 0, 10, 1)
    assert np.all(ys_narrow[51:] < ys_wide[51:])
    assert np.all(ys_narrow[:48] < ys_wide[:48])


def test_gumbel():
    xs = np.linspace(0, 10, 100)
    ys = maths.gumbel(xs, 1, 1, 1, 0.0)
    assert np.argmax(ys) == 10
    npt.assert_allclose(ys[10], np.exp(-1), atol=1e-2)

    ys = maths.gumbel(xs, 1, 1, 2, 0.0)
    assert np.argmax(ys) == 10
    npt.assert_allclose(ys[10], 2 * np.exp(-1), atol=1e-2)

    ys = maths.gumbel(xs, 1, 1, 0, 0.0)
    npt.assert_allclose(ys, 0, atol=1e-6)

    ys = maths.gumbel(xs, 5, 1, 1, 0.0)
    assert np.argmax(ys) == 50
    npt.assert_allclose(ys[50], np.exp(-1), atol=1e-2)

    ys_lifted = maths.gumbel(xs, 5, 1, 1, 1.0)
    npt.assert_allclose(ys_lifted, ys + 1, atol=1e-6)

    ys_narrow = maths.gumbel(xs, 5, 0.1, 1, 0.0)
    ys_wide = maths.gumbel(xs, 5, 10, 1, 0.0)
    assert np.all(ys_narrow[51:] < ys_wide[51:])
    assert np.all(ys_narrow[:49] < ys_wide[:49])


def test_gumbel_params():
    mu, beta = maths.gumbel_params(torch.tensor(1), torch.tensor(1))
    distribution = torch.distributions.gumbel.Gumbel(mu, beta)
    npt.assert_allclose(distribution.mean, 1, atol=1e-6)
    npt.assert_allclose(distribution.stddev, 1, atol=1e-6)

    mu, beta = maths.gumbel_params(torch.tensor(1), torch.tensor(2))
    distribution = torch.distributions.gumbel.Gumbel(mu, beta)
    npt.assert_allclose(distribution.mean, 1, atol=1e-6)
    npt.assert_allclose(distribution.stddev, np.sqrt(2), atol=1e-6)

    mu, beta = maths.gumbel_params(torch.tensor(5), torch.tensor(1))
    distribution = torch.distributions.gumbel.Gumbel(mu, beta)
    npt.assert_allclose(distribution.mean, 5, atol=1e-6)
    npt.assert_allclose(distribution.stddev, 1, atol=1e-6)


def test_weibull_params():
    scale, concentration = maths.weibull_params(torch.tensor(1), torch.tensor(1))
    distribution = torch.distributions.weibull.Weibull(scale, concentration)
    npt.assert_allclose(distribution.mean, 1, atol=1e-6)
    npt.assert_allclose(distribution.stddev, 1, atol=1e-6)

    # without tuning, the standard devation is often rather off
    scale, concentration = maths.weibull_params(torch.tensor(1), torch.tensor(2))
    distribution = torch.distributions.weibull.Weibull(scale, concentration)
    npt.assert_allclose(distribution.mean, 1, atol=1e-6)
    npt.assert_allclose(distribution.stddev, 2, atol=1)

    # tuning doesn't actually improve things that much
    scale, concentration = maths.weibull_params(
        torch.tensor(1), torch.tensor(2.5), tune=True
    )
    distribution = torch.distributions.weibull.Weibull(scale, concentration)
    npt.assert_allclose(distribution.mean, 1, atol=0.2)
    npt.assert_allclose(distribution.stddev, 2.5, atol=0.2)

    scale, concentration = maths.weibull_params(
        torch.tensor(2), torch.tensor(3), tune=True
    )
    distribution = torch.distributions.weibull.Weibull(scale, concentration)
    npt.assert_allclose(distribution.mean, 2, atol=0.2)
    npt.assert_allclose(distribution.stddev, 3, atol=0.2)


def test_logNorm_params():
    location, scale = maths.logNorm_params(torch.tensor(1), torch.tensor(1))
    distribution = torch.distributions.log_normal.LogNormal(location, scale)
    npt.assert_allclose(distribution.mean, 1, atol=1e-6)
    npt.assert_allclose(distribution.stddev, 1, atol=1e-6)

    location, scale = maths.logNorm_params(torch.tensor(1), torch.tensor(2))
    distribution = torch.distributions.log_normal.LogNormal(location, scale)
    npt.assert_allclose(distribution.mean, 1, atol=1e-6)
    npt.assert_allclose(distribution.stddev, 2, atol=1e-6)

    location, scale = maths.logNorm_params(torch.tensor(5), torch.tensor(1))
    distribution = torch.distributions.log_normal.LogNormal(location, scale)
    npt.assert_allclose(distribution.mean, 5, atol=1e-6)
    npt.assert_allclose(distribution.stddev, 1, atol=1e-6)
