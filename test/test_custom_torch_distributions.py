""" Module to test the anomalies module. """
import numpy as np
import numpy.testing as npt
import torch

from pointcloud.models.custom_torch_distributions import (
    GFlashRadial,
    PartGFlashRadial,
    RadialPlane,
)


def test_PartGFlashRadial():
    radial_part1 = PartGFlashRadial(torch.tensor(0.5))
    assert radial_part1.R == 0.5
    radial_part2 = PartGFlashRadial(torch.tensor(0.7))
    assert radial_part1.mean < radial_part2.mean

    sample = radial_part1.sample((10, 10)).detach().numpy()
    assert np.all(sample >= 0)

    values = torch.linspace(1.0, 0.5, 100)
    probs = radial_part1.prob(values).detach().numpy()
    assert np.all(probs >= 0)
    ordered = np.argsort(probs)
    assert np.all(ordered == np.arange(100))

    values = torch.linspace(0, 1, 100)
    cdf = radial_part1.cdf(values).detach().numpy()
    assert np.all(cdf >= 0)
    ordered = np.argsort(cdf)
    assert np.all(ordered == np.arange(100))


def test_GFlashRadial():
    radial_part1 = PartGFlashRadial(torch.tensor(0.5))
    radial_part2 = PartGFlashRadial(torch.tensor(0.7))

    radial1 = GFlashRadial(torch.tensor(0.5), torch.tensor(0.2), torch.tensor(0.5))
    assert radial1.Rc == 0.5
    assert radial1.Rt == 0.7
    assert radial1.mean == 0.5 * (radial_part1.mean + radial_part2.mean)

    sample = radial1.sample((10, 10)).detach().numpy()
    assert np.all(sample >= 0)

    values = torch.linspace(7, 0.7, 100)
    probs = radial1.prob(values).detach().numpy()
    assert np.all(probs >= 0)
    ordered = np.argsort(probs)
    assert np.all(ordered == np.arange(100))

    radial2 = GFlashRadial(torch.tensor(0.5), torch.tensor(0.2), torch.tensor(1.0))
    npt.assert_allclose(radial2.mean, radial_part1.mean, atol=1e-6)
    npt.assert_allclose(radial2.prob(values), radial_part1.prob(values), atol=1e-6)
    sample = radial2.sample((10, 10)).detach().numpy()
    assert np.all(sample >= 0)

    radial3 = GFlashRadial(torch.tensor(0.5), torch.tensor(0.2), torch.tensor(0.0))
    npt.assert_allclose(radial3.mean, radial_part2.mean, atol=1e-6)
    npt.assert_allclose(radial3.prob(values), radial_part2.prob(values), atol=1e-6)
    sample = radial3.sample((10, 10)).detach().numpy()
    assert np.all(sample >= 0)


def test_RadialPlane():
    radial_part1 = PartGFlashRadial(torch.tensor(0.5))
    radial_plane = RadialPlane(radial_part1)
    assert torch.all(radial_plane.mean == 0)
    sample = radial_plane.sample((10, 10)).detach().numpy()
    assert sample.shape == (10, 10, 2)

    values = torch.rand(100, 2)
    probs = radial_plane.prob(values).detach().numpy()
    assert np.all(probs >= 0)
