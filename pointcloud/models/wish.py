""" A really simple model to verify properties of the dataset. """

import torch
from torch.nn import Module, Parameter
from torch.distributions import (
    Normal,
    LogNormal,
    Weibull,
    MultivariateNormal,
    transformed_distribution,
    transforms,
)
import numpy as np

from ..configs import Configs
from ..utils.metadata import Metadata
from ..utils.detector_map import split_to_layers
from ..utils import stats_accumulator

from .custom_torch_distributions import GFlashRadial, RadialPlane


def torch_polyval(coefficients, x):
    """
    Same as numpy's polyval, only if either the coefficients or x are torch tensors,
    it preserves their tensor behaviour and gradient.

    Parameters
    ----------
    coefficients : torch.Tensor (n_coeffs,)
        The polynomial coefficients, from the highest order term to the constant term.
    x : torch.Tensor
        The independent variable.
    Returns
    -------
    y : torch.Tensor
        The value of the polynomial at x.
    """
    n_coeffs = len(coefficients)
    y = sum(
        coeff * x ** (n_coeffs - 1 - exponent)
        for exponent, coeff in enumerate(coefficients)
    )
    return y


def construct_symmetric_2by2(first_eigenvector_angle, eigenvalues):
    """
    By constructing a symmetric 2x2 matrix from eigenvector and eigenvalue information
    we can control it's properties more easily.

    Parameters
    ----------
    first_eigenvector_angle : torch.Tensor
        The angle of the first eigenvector in radians.
    eigenvalues : torch.Tensor length 2
        The two eigenvalues.

    Returns
    -------
    symmetric_matrix : 2-D torch.Tensor, of shape (2, 2)
        Matrix constructed from the eigenvector and eigenvalue information.

    """
    second_eigenvector_angle = first_eigenvector_angle + np.pi * 0.5
    eigenvector_matrix = torch.stack(
        [
            torch.cat(
                [
                    torch.cos(first_eigenvector_angle),
                    torch.cos(second_eigenvector_angle),
                ]
            ),
            torch.cat(
                [
                    torch.sin(first_eigenvector_angle),
                    torch.sin(second_eigenvector_angle),
                ]
            ),
        ]
    )
    eigenvalue_matrix = torch.diag(eigenvalues)
    symmetric_matrix = eigenvector_matrix @ eigenvalue_matrix @ eigenvector_matrix.T
    # enforce the symmetric property even with floating point errors
    symmetric_matrix = 0.5 * (symmetric_matrix + symmetric_matrix.T)
    return symmetric_matrix


class LinearFitSymmetricPositive2by2:
    def __init__(
        self,
        eigenvector_angle_grad,
        eigenvector_angle_intercept,
        eigenvalues_grad,
        eigenvalues_intercept,
    ):
        """
        A 2x2 symmetric positive semi-definite matrix where the eigenvectors and eigenvalues
        are linear functions of a single independent variable.

        Parameters
        ----------
        eigenvector_angle_grad : torch.Tensor
            The gradient of the angle of the first eigenvector in radians.
        eigenvector_angle_intercept : torch.Tensor
            The intercept of the angle of the first eigenvector in radians.
        eigenvalues_grad : torch.Tensor, of shape (2,)
            The gradient of the eigenvalues.
        eigenvalues_intercept : torch.Tensor, of shape (2,)
            The intercept of the eigenvalues.
        """
        self.eigenvector_angle_grad = eigenvector_angle_grad
        self.eigenvector_angle_intercept = eigenvector_angle_intercept
        self.eigenvalues_grad = eigenvalues_grad
        self.eigenvalues_intercept = eigenvalues_intercept

    def fit(self, variable):
        """
        Construct the matrix given a value of the independent variable.

        Parameters
        ----------
        variable : float
            The value of the independent variable.
        Returns
        -------
        matrix : 2-D torch.Tensor, of shape (2, 2)
            The 2x2 symmetric positive semi-definite matrix.
        """
        eigenvector_angle = (
            self.eigenvector_angle_grad * variable + self.eigenvector_angle_intercept
        )
        # Force the eigenvalues to be positive
        eigenvalues = (
            self.eigenvalues_grad * variable + self.eigenvalues_intercept
        ).abs() + 0.00001
        return construct_symmetric_2by2(eigenvector_angle, eigenvalues)


class AbsPolyFit1DDistribution:
    def __init__(
        self,
        distribution,
        mean_coeffs,
        standarddev_coeffs,
        only_positive_domain=False,
    ):
        """
        A 1 dimensional distribution where mean and standarddev
        is linear functions of a single independent variable.

        Parameters
        ----------
        distribution : subclass of torch.distributions.Distribution
            The distribution to fit.
        mean_coeffs : torch.Tensor
            The polynomial coefficients of the mean with
            respect to the independent variable.
        standarddev_coeffs : torch.Tensor
            The polynomial coefficients of the standard devation with
            respect to the independent variable.
        only_positive_domain : bool, optional
            Force the domain to be positive. This modifies the distribution.
            Default is False.
        """
        self._distribution = distribution
        self.mean_coeffs = mean_coeffs
        self.standarddev_coeffs = standarddev_coeffs
        self.only_positive_domain = only_positive_domain

    def get_distribution_args(self, variable):
        """
        The construction arguments for the distribution chosen for the fit.
        The defaults returned here are (mean, standarddev).
        Fits can override this method to provide different arguments.

        Parameters
        ----------
        variable : float
            The value of the independent variable.
        Returns
        -------
        mean : torch.Tensor
            The mean of the distribution.
        standarddev : torch.Tensor
            The standard devation of the distribution.
        """
        mean = self.get_mean(variable)
        standarddev = self.get_standarddev(variable)
        return mean, standarddev

    def get_standarddev(self, variable):
        """
        Standard devation matrix of the gaussian distribution given a value of the
        independent variable.

        Parameters
        ----------
        variable : float
            The value of the independent variable.
        Returns
        -------
        standarddev : torch.Tensor
            Standard devation matrix of the gaussian distribution.
        """
        eps = torch.tensor(1e-10)
        standarddev = torch_polyval(self.standarddev_coeffs, variable)
        return standarddev.abs() + eps

    def get_mean(self, variable):
        """
        Mean of the gaussian distribution given a value of the
        independent variable.

        Parameters
        ----------
        variable : float
            The value of the independent variable.
        Returns
        -------
        mean : torch.Tensor
            Mean of the gaussian distribution.
        """
        return torch_polyval(self.mean_coeffs, variable)

    def calculate_log_probability(self, variable, position):
        """
        Probability density function of the distribution given a value of the
        independent variable and a position in the N dimensional space.

        Parameters
        ----------
        variable : float
            The value of the independent variable.
        position : 1-D array_like, of length n_pts
            The position in the 1 dimensional space (of one or more points).
        Returns
        -------
        probability : torch.Tensor or 1-D torch.Tensor, of length n_pts
            Probability of the position(s) in the 2 dimensional space.
        """
        distribution = self._distribution(*self.get_distribution_args(variable))
        return distribution.log_prob(position)

    def draw_sample(self, variable, n_samples=1):
        """
        Draw a sample from the gaussian given a value of the
        independent variable.

        Parameters
        ----------
        variable : float
            The value of the independent variable.
        n_samples : int, or iterable of int, optional
            Desired number of samples. Default is 1.
        Returns
        -------
        sample : 1-D array_like, of length (n_samples, N)
        """
        # TODO, warning, if self.only_positive_domain is True, the mean is not
        # the mean of the distributions, it's just a parameter....
        if not hasattr(n_samples, "__iter__"):
            n_samples = [n_samples]
        distribution = self._distribution(*self.get_distribution_args(variable))
        if self.only_positive_domain:
            distribution = transformed_distribution.TransformedDistribution(
                distribution, transforms.AbsTransform()
            )
        sample = distribution.sample(n_samples)
        return sample.flatten()


class PolyFit1DGauss(AbsPolyFit1DDistribution):
    def __init__(
        self,
        mean_coeffs,
        standarddev_coeffs,
        only_positive_domain=False,
    ):
        """
        A 1 dimensional gaussian distribution where mean and standarddev
        is linear functions of a single independent variable.

        Parameters
        ----------
        mean_coeffs : torch.Tensor
            The polynomial coefficients of the mean with
            respect to the independent variable.
        standarddev_coeffs : torch.Tensor
            The polynomial coefficients of the standard devation with
            respect to the independent variable.
        only_positive_domain : bool, optional
            Force the domain to be positive. This modifies the distribution.
            Default is False.
        """
        super().__init__(
            Normal,
            mean_coeffs,
            standarddev_coeffs,
            only_positive_domain,
        )


class PolyFit1DLogNormal(AbsPolyFit1DDistribution):
    def __init__(
        self,
        mean_coeffs,
        standarddev_coeffs,
    ):
        """
        A 1 dimensional gaussian distribution where mean and standarddev
        is linear functions of a single independent variable.

        Parameters
        ----------
        mean_coeffs : torch.Tensor
            The polynomial coefficients of the mean with
            respect to the independent variable.
        standarddev_coeffs : torch.Tensor
            The polynomial coefficients of the standard devation with
            respect to the independent variable.
        """
        super().__init__(
            LogNormal,
            mean_coeffs,
            standarddev_coeffs,
        )

    def get_mean(self, variable):
        """
        Mean of the LogNormal distribution given a value of the
        independent variable.
        For a LogNormal distribution, the mean can only be positive.

        Parameters
        ----------
        variable : float
            The value of the independent variable.

        Returns
        -------
        mean : torch.Tensor
            The mean of the LogNormal distribution.
        """
        mean = super().get_mean(variable)
        return mean.abs()

    def calculate_log_probability(self, variable, position):
        """
        Probability density function of the log normal distribution given a value of the
        independent variable and a position in the 1 dimensional space.
        As the log normal distribution goes to 0 at 0, this case
        is clipped to a small value.

        Parameters
        ----------
        variable : float
            The value of the independent variable.
        position : 1-D array_like, of length n_pts
            The position in the 1 dimensional space (of one or more points).
        Returns
        -------
        probability : torch.Tensor or 1-D torch.Tensor, of length n_pts
            Probability of the position(s) in the 2 dimensional space.
        """
        distribution = self._distribution(*self.get_distribution_args(variable))
        position = torch.clip(position, torch.finfo(position.dtype).tiny, None)
        return distribution.log_prob(position)


class PolyFit1DWeibull(AbsPolyFit1DDistribution):
    def __init__(
        self,
        mean_coeffs,
        standarddev_coeffs,
    ):
        """
        A 1 dimensional Weibull distribution where mean and standarddev
        is linear functions of a single independent variable.

        Parameters
        ----------
        mean_coeffs : torch.Tensor
            The polynomial coefficients of the mean with
            respect to the independent variable.
        standarddev_coeffs : torch.Tensor
            The polynomial coefficients of the standard devation with
            respect to the independent variable.
        """
        super().__init__(
            Weibull,
            mean_coeffs,
            standarddev_coeffs,
        )
        self._default_dist = Weibull(torch.Tensor([1.0]), torch.Tensor([1.0]))

    def draw_sample(self, variable, n_samples=1):
        """
        Draw a sample from the gaussian given a value of the
        independent variable.

        Parameters
        ----------
        variable : float
            The value of the independent variable.
        n_samples : int, or iterable of int, optional
            Desired number of samples. Default is 1.
        Returns
        -------
        sample : 1-D array_like, of length (n_samples, N)
        """
        try:
            # do a regular sampling
            sample = super().draw_sample(variable, n_samples)
        except ValueError:
            # something weird, just return zeros
            # weibull distribution has bad limiting cases at 0
            if not hasattr(n_samples, "__iter__"):
                n_samples = [n_samples]
            sample = self._default_dist.sample(n_samples)
            sample[:] *= 0

        return sample.flatten()

    def get_distribution_args(self, variable):
        """
        The construction arguments for the Weibull distribution.
        These are (concentration, scale).

        Parameters
        ----------
        variable : float
            The value of the independent variable.
        Returns
        -------
        scale : torch.Tensor
            The scale of the distribution.
        concentration : torch.Tensor
            The concentration of the distribution.
        """
        finfo = torch.finfo(self.mean_coeffs.dtype)
        small_value = finfo.tiny

        # make sure we start with reasonable values
        mean = torch.clip(self.get_mean(variable), small_value, None)
        standarddev = torch.clip(self.get_standarddev(variable), small_value, None)

        # do the standard calculation
        concentration = (mean / standarddev) ** (1.086)
        denominator = (1 + 1 / concentration).lgamma().exp()
        scale = torch.clip(mean / denominator, small_value, None)

        return scale, concentration

    def calculate_log_probability(self, variable, position):
        """
        Probability density function of the distribution given a value of the
        independent variable and a position in the 1 dimensional space.

        Parameters
        ----------
        variable : float
            The value of the independent variable.
        position : 1-D array_like, of length n_pts
            The position in the 1 dimensional space (of one or more points).
        Returns
        -------
        probability : torch.Tensor or 1-D torch.Tensor, of length n_pts
            Probability of the position(s) in the 2 dimensional space.
        """
        eps = torch.tensor(1e-10)
        distribution = self._distribution(*self.get_distribution_args(variable))
        position = torch.clip(position, eps, None)
        return distribution.log_prob(position)


class LinearFit2DGauss:
    def __init__(
        self,
        covarience_matrix_fit,
    ):
        """
        An N dimensional gaussian distribution where the mean and covarience
        are linear functions of a single independent variable.

        Parameters
        ----------
        covarience_matrix_fit : LinearFitSymmetricPositive2by2
            A linear fir object for the covarience matrix.
        """
        self.covarience_matrix_fit = covarience_matrix_fit
        self._mean = torch.nn.Parameter(torch.zeros(2), requires_grad=False)

    def calculate_log_probability(self, variable, position):
        """
        Probability density function of the gaussian distribution given a value of the
        independent variable and a position in the N dimensional space.

        Parameters
        ----------
        variable : float
            The value of the independent variable.
        position : 1-D array_like, of length N, or 2-D array_like, of shape (n_pts, 2)
            The position in the N dimensional space (of one or more points.
        Returns
        -------
        probability : torch.Tensor or 1-D torch.Tensor, of length n_pts
            Probability of the position(s) in the 2 dimensional space.
        """
        covarience = self.covarience_matrix_fit.fit(variable)
        distribution = MultivariateNormal(self._mean, covarience)
        return distribution.log_prob(position)

    def draw_sample(self, variable, n_samples=(1,)):
        """
        Draw a sample from the gaussian distribution given a value of the
        independent variable.

        Parameters
        ----------
        variable : float
            The value of the independent variable.
        n_samples : iterable of ints, optional
            Desired number of samples. Default is (1,).
        Returns
        -------
        sample : 1-D array_like, of shape (<n_samples>, N)
        """
        covarience = self.covarience_matrix_fit.fit(variable)
        distribution = MultivariateNormal(self._mean, covarience)
        return distribution.sample(n_samples)


class PolyFit2DGFlash:
    def __init__(
        self,
        Rc_coeffs,
        Rt_coeffs,
        p_coeffs,
    ):
        """
        A 2 dimensional distribution modeling the distribution
        in a layer using the radial distribution of the GFlash model.
        All parameters are linear functions of a single independent variable.

        In the GFlash model, the radial distribution of hits is modeled as
        Rt = Rc + Rt_extend
        p(r) = p*r*Rc**2/(Rc**2 + r**2)**2 + (1-p)*r*Rt**2/(Rt**2 + r**2)**2

        Parameters
        ----------
        Rc_coeffs : torch.Tensor
            The coefficients for the fit of the Rc parameter for the GFlash model.
        Rt_coeffs : torch.Tensor
            The coefficients for the fit of the Rt extention parameter
            for the GFlash model.
        p_coeffs : torch.Tensor
            The coefficients for the fit of the p parameter for the GFlash model.
        """
        self.Rc_coeffs = Rc_coeffs
        self.Rt_coeffs = Rt_coeffs
        self.p_coeffs = p_coeffs

    def distribution(self, *args):
        """
        Construct the GFlashRadial distribution given the parameters.

        Parameters
        ----------
        args : tuple of torch.Tensor
            The parameters for the GFlash model.
        Returns
        -------
        distribution : GFlashRadial
            The GFlashRadial distribution.
        """
        return RadialPlane(GFlashRadial(*args))

    def get_distribution_args(self, variable):
        """
        Given a value of the independent variable, return the parameters
        of the GFlash model.

        Parameters
        ----------
        variable : float
            The value of the independent variable.
        Returns
        -------
        Rc : torch.Tensor
            The Rc parameter for the GFlash model.
        Rt_extend : torch.Tensor
            The Rt extention parameter for the GFlash model.
        p : torch.Tensor
            The p parameter for the GFlash model.
        """
        eps = torch.tensor(0.00001)
        Rc = torch_polyval(self.Rc_coeffs, variable)
        Rc = torch.clamp(Rc, min=eps)
        Rt_extend = torch_polyval(self.Rt_coeffs, variable)
        Rt_extend = torch.clamp(Rt_extend, min=eps)
        p = torch_polyval(self.p_coeffs, variable)
        p = torch.clamp(p, min=eps, max=1 - eps)
        return Rc, Rt_extend, p

    def calculate_probability(self, variable, position):
        """
        Value of probability density function of the distribution
        given a value of the independent variable and a position in
        the 2 dimensional space.

        Parameters
        ----------
        variable : float
            The value of the independent variable.
        position : 1-D array_like, of length N, or 2-D array_like, of shape (n_pts, 2)
            The position in the N dimensional space (of one or more points.
        Returns
        -------
        probability : torch.Tensor or 1-D torch.Tensor, of length n_pts
            Probability of the position(s) in the 2 dimensional space.
        """
        distribution = self.distribution(*self.get_distribution_args(variable))
        return distribution.prob(position)

    def calculate_log_probability(self, variable, position):
        """
        Logarithm of probability density function of the distribution
        given a value of the independent variable and a position in
        the 2 dimensional space.

        Parameters
        ----------
        variable : float
            The value of the independent variable.
        position : 1-D array_like, of length N, or 2-D array_like, of shape (n_pts, 2)
            The position in the N dimensional space (of one or more points.
        Returns
        -------
        log_probability : torch.Tensor or 1-D torch.Tensor, of length n_pts
            Probability of the position(s) in the 2 dimensional space.
        """
        distribution = GFlashRadial(*self.get_distribution_args(variable))
        radii = torch.norm(position, dim=-1)
        return distribution.log_prob(radii)

    def draw_sample(self, variable, n_samples=(1,)):
        """
        Draw a sample from the distribution given a value of the
        independent variable.

        Parameters
        ----------
        variable : float
            The value of the independent variable.
        n_samples : iterable of ints, optional
            Desired number of samples. Default is (1,).
        Returns
        -------
        sample : 1-D array_like, of shape (<n_samples>, 2)
        """
        distribution = self.distribution(*self.get_distribution_args(variable))
        drawn = distribution.sample(n_samples)
        return drawn


def padding_mask(event):
    """
    Events are padded with 0 particles to make them all the same length.
    This function creates a mask to remove the padding.

    Parameters
    ----------
    event : 2-D array_like, of shape (padding + n_pts, 4)
        x, y, z and energy of the points in the event.

    Returns
    -------
    mask : 1-D array_like, of length (padding + n_pts)
    """
    return event[:, 3] > 0


class LayersBackBone:
    MAX_PTS_PER_LAYER = 1000

    def __init__(self, n_layers, poly_degree):
        """
        Component for inputs to each layer in the detector.
        Functionally equivalent to ShowerFlow.
        In theory, could correlate the inputs to each layer, but
        currently just creates n_layers of the same thing.

        Parameters
        ----------
        n_layers : int
            The number of layers in the detector.
        poly_degree : int
            The degree of the polynomial fits for variation with
            incident energy.
        """
        self.n_layers = n_layers
        self.poly_degree = poly_degree
        self._setup_fits()
        self.all_params = self._collect_all_params()
        self.displayed_layers = range(n_layers)

    def _setup_fits(self):
        """
        Set up the fits with trainable parameters.
        """

        self.n_pts_fits = []
        for i in range(self.n_layers):
            mean_coeffs = Parameter(
                torch.randn(1 + self.poly_degree), requires_grad=True
            )
            standarddev_coeffs = Parameter(
                torch.randn(1 + self.poly_degree), requires_grad=True
            )
            # npts_fit = LinearFit1DBinomial(
            npts_fit = PolyFit1DWeibull(mean_coeffs, standarddev_coeffs)
            self.n_pts_fits.append(npts_fit)

        self.energy_fits = []
        for i in range(self.n_layers):
            mean_coeffs = Parameter(
                torch.randn(1 + self.poly_degree), requires_grad=True
            )
            standarddev_coeffs = Parameter(
                torch.randn(1 + self.poly_degree), requires_grad=True
            )
            # energy_fit = PolyFit1DGauss(
            #    mean_coeffs,
            #    standarddev_coeffs,
            #    only_positive_domain=True,
            # )
            energy_fit = PolyFit1DLogNormal(mean_coeffs, standarddev_coeffs)
            self.energy_fits.append(energy_fit)

    def _collect_all_params(self):
        """
        Go through the fits, adding their params to a list.
        """
        all_params = []
        for fit in self.n_pts_fits + self.energy_fits:
            all_params.append(fit.mean_coeffs)
            all_params.append(fit.standarddev_coeffs)
        return all_params

    def reset_params(
        self,
        n_pts_coeffs=None,
        n_pts_standarddev_coeffs=None,
        energy_mean_coeffs=None,
        energy_standarddev_coeffs=None,
    ):
        """
        Set the parameters of the fits to specific values.

        Parameters
        ----------
        n_pts_coeffs : dict of arrays, optional
            The coefficients of the fits for the number of points in each layer.
            Keys are the layer numbers.
        n_pts_standarddev_coeffs : dict of arrays, optional
            The coefficients of the fits for the standard devation
            of the number of points in each layer.
            Keys are the layer numbers.
        energy_mean_coeffs : dict of arrays, optional
            The coefficients of the fits for the mean energy of points in each layer.
            Keys are the layer numbers.
        energy_standarddev_coeffs : dict of arrays, optional
            The coefficients of the fits for the standard devation of the energy of
            points in each layer.
            Keys are the layer numbers.
        """
        # start by making empty dicts if they are not provided
        # this can't be the default value of the function as it
        # would be shared between calls
        if n_pts_coeffs is None:
            n_pts_coeffs = {}
        if n_pts_standarddev_coeffs is None:
            n_pts_standarddev_coeffs = {}
        if energy_mean_coeffs is None:
            energy_mean_coeffs = {}
        if energy_standarddev_coeffs is None:
            energy_standarddev_coeffs = {}

        with torch.no_grad():
            for i in range(self.n_layers):
                # number of points in each layer
                if i in n_pts_coeffs:
                    if np.any(np.isnan(n_pts_coeffs[i])):
                        raise ValueError(f"n_pts_coeffs[{i}] has nan")
                    self.n_pts_fits[i].mean_coeffs[:] = torch.from_numpy(
                        n_pts_coeffs[i]
                    )
                    del n_pts_coeffs[i]
                if i in n_pts_standarddev_coeffs:
                    if np.any(np.isnan(n_pts_standarddev_coeffs[i])):
                        raise ValueError(f"n_pts_standarddev_coeffs[{i}] has nan")
                    self.n_pts_fits[i].standarddev_coeffs[:] = torch.from_numpy(
                        n_pts_standarddev_coeffs[i]
                    )
                    del n_pts_standarddev_coeffs[i]
                if i in energy_mean_coeffs:
                    if np.any(np.isnan(energy_mean_coeffs[i])):
                        raise ValueError(f"energy_mean_coeffs[{i}] has nan")
                    self.energy_fits[i].mean_coeffs[:] = torch.from_numpy(
                        energy_mean_coeffs[i]
                    )
                    del energy_mean_coeffs[i]
                if i in energy_standarddev_coeffs:
                    if np.any(np.isnan(energy_standarddev_coeffs[i])):
                        raise ValueError(f"energy_standarddev_coeffs[{i}] has nan")
                    self.energy_fits[i].standarddev_coeffs[:] = torch.from_numpy(
                        energy_standarddev_coeffs[i]
                    )
                    del energy_standarddev_coeffs[i]

        assert not n_pts_coeffs, f"n_pts_coeffs contains unused items: {n_pts_coeffs}"
        assert (
            not n_pts_standarddev_coeffs
        ), f"n_pts_standarddev_coeffs contains unused items: {n_pts_standarddev_coeffs}"
        assert (
            not energy_mean_coeffs
        ), f"energy_mean_coeffs contains unused items: {energy_mean_coeffs}"
        assert (
            not energy_standarddev_coeffs
        ), f"energy_standarddev_coeffs contains unused items: {energy_standarddev_coeffs}"
        # self.all_params = self._collect_all_params()

    def __str__(self):
        text = f"LayersBackBone({self.n_layers})\n"
        return text

    def draw_sample(self, incident_energy):
        """
        Draw a sample of points for each layer in the detector.

        Parameters
        ----------
        incident_energy : float
            The energy of the particle incident on the detector.
        Returns
        -------
        samples : list
            List of samples for each layer in the detector.
        """
        samples = []
        for n_layer in range(self.n_layers):
            n_pts = self.n_pts_fits[n_layer].draw_sample(incident_energy)
            energy = self.energy_fits[n_layer].draw_sample(
                incident_energy  # , n_pts.numpy().astype(int)
            )
            samples.append((n_pts, energy))
        return samples

    def log_likelihood(self, incident_energy, samples):
        """
        Given an observed event backbone, calculate the log likelihood of the model parameters.

        Parameters
        ----------
        incident_energy : float
            The energy of the particle incident on the detector.
        samples : list of (int, float)
            List of number of points, and total energy for each layer in the detector.
        Returns
        -------
        log_likelihood : torch.Tensor
        """
        log_likelihood = 0.0
        for layer_n, (n_pts, energy) in enumerate(samples):
            log_likelihood += (
                self.n_pts_fits[layer_n]
                .calculate_log_probability(incident_energy, n_pts)
                .sum()
            )
            if n_pts:
                mean_energy = energy.sum() / n_pts
                log_likelihood += torch.sum(
                    self.energy_fits[layer_n].calculate_log_probability(
                        incident_energy, mean_energy
                    )
                )
        return log_likelihood


class WishLayer:
    def __init__(self, layer_n, poly_degree, fit_attempts=100):
        """
        One layer of the ditector in the wish model.
        Composed of a set of 4 distributions whose parameters are linear
        fits in the energy of the particle.

        Parameters
        ----------
        layer_n : int
            The layer number.
        """
        self.layer_n = layer_n
        self.poly_degree = poly_degree
        self.fit_attempts = fit_attempts
        self._setup_fits()

    def _setup_fits(self):
        """
        Set up the fits with trainable parameters.
        """
        self.all_params = []

        Rt_coeffs = Parameter(torch.randn(1 + self.poly_degree), requires_grad=True)
        Rc_coeffs = Parameter(torch.randn(1 + self.poly_degree), requires_grad=True)
        p_coeffs = Parameter(torch.randn(1 + self.poly_degree), requires_grad=True)
        self.all_params += [
            Rt_coeffs,
            Rc_coeffs,
            p_coeffs,
        ]

        self.displacement_fit = PolyFit2DGFlash(
            Rc_coeffs,
            Rt_coeffs,
            p_coeffs,
        )

        # the energy has a strict mean of 1.
        # it get's rescaled by the Backbones energy fit
        mean_coeffs = Parameter(torch.zeros(1 + self.poly_degree), requires_grad=True)
        with torch.no_grad():
            mean_coeffs[-1] += 1.0
        standarddev_coeffs = Parameter(
            torch.randn(1 + self.poly_degree), requires_grad=True
        )
        self.all_params += [standarddev_coeffs]
        self.energy_fit = PolyFit1DLogNormal(mean_coeffs, standarddev_coeffs)

    def draw_sample(self, incident_energy, n_pts):
        """
        Given the incident energy, draw a sample of points for the layer.

        Parameters
        ----------
        incident_energy : float
            The energy of the particle incident on the layer.
        n_pts : torch.Size
            The number of points to draw.
        Returns
        -------
        sample_displacement : 2-D array_like, of shape (n_pts, 2)
            Position of the poitns in the layer relative to the shower axis.
            Format (x, y).
        sample_energy : 1-D array_like, of length n_pts
            Energy of the points in the layer.
            Unnormalised.
        """
        displacement = self.displacement_fit.draw_sample(incident_energy, n_pts)
        energies = self.energy_fit.draw_sample(incident_energy, n_pts)
        return displacement, energies

    def log_likelihood(self, incident_energy, displacement, energies):
        """
        Log likelihood of the model parameters given the incident energy,
        and a sample of displacements and energies.

        Parameters
        ----------
        incident_energy : float
            The energy of the particle incident on the layer.
        displacement : 2-D array_like, of shape (n_pts, 2)
            Position of the points in the layer relative to the shower axis.
            Format (x, y).
        energies : 1-D array_like, of length n_pts
            Energy of the points in the layer.
        Returns
        -------
        probability : float
        """
        probability_displacement = self.displacement_fit.calculate_log_probability(
            incident_energy, displacement
        )
        # the mean value of the energies is set in the backbone
        # so we need to normalise the energies to have a mean of 1
        mean_energy = energies.mean()
        if mean_energy > 0:
            energies = energies / mean_energy
        probability_energies = self.energy_fit.calculate_log_probability(
            incident_energy, energies
        )
        return probability_displacement.sum() + probability_energies.sum()

    def reset_with_radialView(self, radialView):
        """
        Fit the parameters of the lateral hits distribution using a RadialView.
        This involves fitting at each energy bin, then getting the linear fit
        through those values.

        Parameters
        ----------
        radialView : stats_accumulator.RadialView
            A radial view of accumulated statistics that has a fit_to_hits
            method.
        """
        incedent_energies = radialView.incident_energy_bin_centers
        rt_values = np.empty_like(incedent_energies)
        rc_values = np.empty_like(incedent_energies)
        p_values = np.empty_like(incedent_energies)

        eps = 0.00001  # because if the radius is 0, the model has a singularity
        p0 = [0.1, 0.0005, 0.005]
        bounds = [[0.0, eps, eps], [1.0, np.inf, np.inf]]

        def to_fit(radial_points, p, Rt_extend, Rc):
            # model = GFlashRadial(Rc, Rt_extend, p)
            model = GFlashRadial(Rc, Rt_extend, p)
            radial_points = torch.tensor(radial_points, dtype=torch.float32)
            found = model.prob(radial_points).detach().numpy()
            return found

        for i, energy in enumerate(incedent_energies):
            print(".", end="")
            try:
                popt, _ = radialView.fit_to_hits(
                    i,
                    self.layer_n,
                    to_fit,
                    p0=p0,
                    bounds=bounds,
                    ignore_norm=True,  # Todo: check if this is correct
                    n_attempts=self.fit_attempts,
                    quiet=True,
                )
            except ValueError:
                print(f"Failed to fit layer {self.layer_n} at energy {energy}")
                print("Will use the next/previous value if possible")
                popt = [-1, -1, -1]
            p_values[i] = popt[0]
            rt_values[i] = popt[1]
            rc_values[i] = popt[2]
        print(" ")

        # check for fits that failed, and borrow from the next/previous value
        for i, p in enumerate(p_values):
            if p > 0:
                continue
            forward = next(
                (j for j in range(i, len(p_values)) if p_values[j] > 0), None
            )
            backward = next((j for j in range(i, -1, -1) if p_values[j] > 0), None)
            if forward is not None and backward is not None:
                pick = forward if forward - i < i - backward else backward
            elif forward is not None:
                pick = forward
            elif backward is not None:
                pick = backward
            else:
                raise ValueError("Model fit failed for all points")
            p_values[i] = p_values[pick]
            rt_values[i] = rt_values[pick]
            rc_values[i] = rc_values[pick]

        # now do fits with the energy for the parameters
        Rt_coeffs = np.polyfit(incedent_energies, rt_values, self.poly_degree)
        Rc_coeffs = np.polyfit(incedent_energies, rc_values, self.poly_degree)
        p_coeffs = np.polyfit(incedent_energies, p_values, self.poly_degree)
        self.reset_params(
            Rt_coeffs=Rt_coeffs,
            Rc_coeffs=Rc_coeffs,
            p_coeffs=p_coeffs,
        )

    def reset_params(
        self,
        energy_standarddev_coeffs=None,
        Rt_coeffs=None,
        Rc_coeffs=None,
        p_coeffs=None,
    ):
        """
        Set the parameters of the fits to specific values.

        Parameters
        ----------
        energy_standarddev_coeffs : torch.Tensor, optional
            The coefficients for the fit of the standard devation of
            the energy of points in the layer.
        Rt_coeffs : torch.Tensor, optional
            The coefficients for the fit of the Rt extention parameter for the model.
        Rc_coeffs : torch.Tensor, optional
            The coefficients for the fit of the Rc parameter for the model.
        p_coeffs : torch.Tensor, optional
            The coefficients for the fit of the p parameter for the model.
        """
        with torch.no_grad():
            if energy_standarddev_coeffs is not None:
                if np.any(np.isnan(energy_standarddev_coeffs)):
                    raise ValueError("energy_standarddev_coeffs has nan")
                self.energy_fit.standarddev_coeffs[:] = torch.from_numpy(
                    energy_standarddev_coeffs
                )
            if Rt_coeffs is not None:
                if np.any(np.isnan(Rt_coeffs)):
                    raise ValueError("Rt_coeffs has nan")
                self.displacement_fit.Rt_coeffs[:] = torch.from_numpy(Rt_coeffs)
            if Rc_coeffs is not None:
                if np.any(np.isnan(Rc_coeffs)):
                    raise ValueError("Rc_coeffs has nan")
                self.displacement_fit.Rc_coeffs[:] = torch.from_numpy(Rc_coeffs)
            if p_coeffs is not None:
                if np.any(np.isnan(p_coeffs)):
                    raise ValueError("p_coeffs has nan")
                self.displacement_fit.p_coeffs[:] = torch.from_numpy(p_coeffs)


class Wish(Module):
    def __init__(self, config):
        """
        A wish model.

        Parameters
        ----------
        config : configs.Configs
            Configuration for the run, used to obtain metadata about
            the detector.

        Attributes
        ----------
        n_layers : int
            The number of layers in the detector.
        poly_degree : int
            The degree of all polynomial fits used in the model.
        max_n_pts_per_layer : float
            The maximum number of points in a single layer.
        backbone : LayersBackBone
            The backbone of the model, containing the fits for the inputs to each layer.
        normed_layer_bottom : 1-D array_like, of length n_layers
            The normalised position of the bottom of each layer.
        cell_thickness : float
            Normalised thickness of a cell in the detector.
        layer_params : list of torch.Tensor
            The parameters of the fits for each layer.
        layers : list of WishLayer
            The layers of the detector.
        config : configs.Configs
            Configuration for the run, used to obtain metadata about
            the detector.

        """
        super().__init__()
        metadata = Metadata(config)
        self.n_layers = len(metadata.layer_bottom_pos)
        self.poly_degree = config.poly_degree
        self.max_n_pts_per_layer = config.max_points / 2.0
        self.backbone = LayersBackBone(self.n_layers, self.poly_degree)
        for i, param in enumerate(self.backbone.all_params):
            self.register_parameter(f"backbone_param{i}", param)
        self.normed_layer_bottom = np.linspace(-1, 1, self.n_layers + 1)[
            : self.n_layers
        ]
        self.cell_thickness = 0.5 * (
            self.normed_layer_bottom[1] - self.normed_layer_bottom[0]
        )
        self.layer_params = []
        self.layers = []
        self.config = config
        for i in range(self.n_layers):
            layer = WishLayer(i, self.poly_degree, config.fit_attempts)
            layer_params = layer.all_params
            for p, param in enumerate(layer_params):
                self.register_parameter(f"layer_{i}_param_{p}", param)
            self.layer_params += layer_params
            self.layers.append(layer)

    def set_from_stats(self, high_level_stats):
        """
        Set the parameters of the fits to specific values,
        determined by an object that can give high level stats of the
        data.

        Parameters
        ----------
        high_level_stats : utils.stats_accumulator.HighLevelStats
            An object that can give high level stats of the data.
        """
        # start by asking for params for the backbone
        n_pts_coeffs = {}
        n_pts_standarddev_coeffs = {}
        energy_coeffs = {}
        energy_standarddev_coeffs = {}
        for i in range(self.n_layers):
            n_pts_coeffs[i] = high_level_stats.n_pts(i)
            n_pts_standarddev_coeffs[i] = high_level_stats.stddev_n_pts(i)
            energy_coeffs[i] = high_level_stats.event_mean_point_energy(i)
            energy_standarddev_coeffs[
                i
            ] = high_level_stats.stddev_event_mean_point_energy(i)
        self.backbone.reset_params(
            n_pts_coeffs,
            n_pts_standarddev_coeffs,
            energy_coeffs,
            energy_standarddev_coeffs,
        )
        # now for each layer, ask for the params
        radial_view = stats_accumulator.RadialView(high_level_stats.accumulator)
        for i, layer in enumerate(self.layers):
            energy_standarddev_coeffs = high_level_stats.stddev_point_energy_in_evt(i)
            layer.reset_params(
                energy_standarddev_coeffs=energy_standarddev_coeffs,
            )
            print(f"Params for layer {i}", end=" ")
            layer.reset_with_radialView(radial_view)

    def get_loss(self, batch, writer=None, it=None):
        """
        Get the loss for a batch of events.

        Parameters
        ----------
        batch : dict
            A batch of events.
            Has keys "event" and "energy",
            where "event" is a 3-D tensor of shape (1, n_events, n_points, 4),
            the first "1" is optional, but in theory could be a batch dimension.
            The "4" is x, y, z, energy.
            "energy" is a 1-D tensor of shape (1, n_events),
            the energy of the incident particle.
            Again, the "1" is optional.
        writer : None
            Not used.
        it : None
            Not used.

        Returns
        -------
        loss : torch.Tensor, (1,)
            The combined loss for the batch.

        """
        return -self.log_likelihood(batch)

    def log_likelihood(self, batch):
        """
        Calculate the log likelihood of the model parameters given the observed events.

        Parameters
        ----------
        batch : dict
            A batch of events.
            Has keys "event" and "energy",
            where "event" is a 3-D tensor of shape (1, n_events, n_points, 4),
            the first "1" is optional, but in theory could be a batch dimension.
            The "4" is x, y, z, energy.
            "energy" is a 1-D tensor of shape (1, n_events),
            the energy of the incident particle.
            Again, the "1" is optional.

        Returns
        -------
        log_likelihood : torch.Tensor, (1,)
            The sum of the log likelihood of the model parameters
            given the observed events in the batch.
        """
        log_likelihood = 0.0
        events = batch["event"]
        if len(events.shape) > 3:
            assert events.shape[0] == 1
            events = events[0]
        incident = batch["energy"].flatten()
        for incident_energy, event in zip(incident, events):
            event = event[padding_mask(event)]
            log_likelihood += self.event_log_likelihood(event, incident_energy)
        return log_likelihood

    def event_log_likelihood(self, event, incident_energy):
        """
        Calculate the log likelihood of the model parameters for a single event.

        Parameters
        ----------
        event : 2-D array_like, of shape (n_points, 4)
            The x, y, z, energy of the points in the event.
        incident_energy : float or torch.Tensor
            The energy of the particle incident on the detector.

        Returns
        -------
        log_likelihood : torch.Tensor, (1,)
            The log likelihood of the model parameters given the observed event.
        """
        # for each layer, calculate the log likelihood of the parameters
        log_likelihood = 0.0
        backbone_observed = []
        for layer_n, points_in_layer in enumerate(
            split_to_layers(event, self.normed_layer_bottom, self.cell_thickness)
        ):
            displacement = points_in_layer[:, [0, 2]]
            energies = points_in_layer[:, 3]
            n_points = torch.tensor((energies.shape[0]))
            backbone_observed.append((n_points, energies))
            if n_points == 0:
                continue
            log_likelihood += self.layers[layer_n].log_likelihood(
                incident_energy, displacement, energies
            )
        backbone_likelihood = self.backbone.log_likelihood(
            incident_energy, backbone_observed
        )
        log_likelihood += backbone_likelihood
        return log_likelihood

    def draw_all_layers(self, incident_energy):
        """
        Draw a sample of points from each layer in the detector.

        Parameters
        ----------
        incident_energy : torch.Tensor
            The energy of the particle incident on the detector.

        Returns
        -------
        layers : list of (torch.Tensor, torch.Tensor)
            The displacement and energy of the points in each layer.
            The first element is the displacement, the second is the energy.
            The displacement is a 2-D tensor of shape (n_points, 2),
            the energy is a 1-D tensor of shape (n_points,).
        """
        layers = []
        for layer_n, (n_pts, energy) in enumerate(
            self.backbone.draw_sample(incident_energy)
        ):
            torch.clamp(n_pts, 0, self.max_n_pts_per_layer, out=n_pts)
            layer = self.layers[layer_n]
            displacement, energies = layer.draw_sample(
                incident_energy, n_pts.numpy().round().astype(int)
            )
            rescaled_energy = energies * energy
            layers.append((displacement, rescaled_energy))
        return layers

    def inference(self, incident_energy):
        """
        Draw a sample of points from the detector.
        No gradients are calculated, for inference only.

        Parameters
        ----------
        incident_energy : float like
            The energy of the particle incident on the detector.

        Returns
        -------
        x : list of float
            The x position of the points without padding.
        y : list of float
            The y position of the points without padding.
        z : list of float
            The z position of the points without padding.
        e : list of float
            The energy of the points without padding.
        """
        with torch.no_grad():
            all_x, all_y, all_z, all_e = [], [], [], []
            for y, (displacement, energy) in enumerate(
                self.draw_all_layers(incident_energy)
            ):
                all_x += displacement[:, 0].tolist()
                all_y += np.full(len(displacement), y).tolist()
                all_z += displacement[:, 1].tolist()
                all_e += energy.tolist()
        return all_x, all_y, all_z, all_e

    def sample(self, conditioning, max_hits):
        """
        Draw multiple samples of points from the detector.
        No gradients are calculated, for inference only.

        Parameters
        ----------
        conditioning : array of values, (n_samples, n_input_features)
            A set of conditioning values for each sample to be drawn.
            The columns will be [n_hits, incident_energy], but n_hits
            is not used.
        max_hits : int
            The maximum number of hits to draw for each sample.

        Returns
        -------
        samples : array of values, (n_samples, max_hits, 4)
            The x, y, z and energy of the points in each sample.
        """
        samples = np.zeros((conditioning.shape[0], max_hits, 4))
        for i, incident_energy in enumerate(conditioning[:, -1]):
            x, y, z, e = self.inference(incident_energy)
            n_hits = min(len(e), max_hits)
            samples[i, :n_hits, 0] = x[:n_hits]
            samples[i, :n_hits, 1] = y[:n_hits]
            samples[i, :n_hits, 2] = z[:n_hits]
            samples[i, :n_hits, 3] = e[:n_hits]
        return samples

    def forward(self, batch):
        coords = []
        for incident in batch["energy"].flatten():
            all_x, all_y, all_z, all_e = [], [], [], []
            for y, (displacement, energy) in enumerate(self.draw_all_layers(incident)):
                all_x.append(displacement[:, 0])
                all_y.append(torch.full((len(displacement),), y))
                all_z.append(displacement[:, 1])
                all_e.append(energy)
            all_x = torch.cat(all_x)
            all_y = torch.cat(all_y)
            all_z = torch.cat(all_z)
            all_e = torch.cat(all_e)
            coords.append(torch.stack([all_x, all_y, all_z, all_e], dim=1))
        return coords

    def save(self, path):
        """
        Save the model to a file.

        Parameters
        ----------
        path : str
            The path to save the model to.
        """
        to_save = {"state_dict": self.state_dict(), "config": self.config}
        torch.save(to_save, path)

    @classmethod
    def load(cls, path):
        """
        Load a model from a file.

        Parameters
        ----------
        path : str
            The path to load the model from.
        """
        loaded = torch.load(path)
        model = cls(loaded["config"])
        model.load_state_dict(loaded["state_dict"])
        return model


def load_wish_from_accumulator(
    accumulator="../point-cloud-diffusion-logs/wish/dataset_accumulators/10-90GeV_x36_grid_regular_524k_float32/from_10.h5",
    config=Configs(),
):
    """
    Load the wish model and set it's values from the statistics gathered by an accumulator.

    Parameters
    ----------
    accumulator: str or stats_accumulator.StatsAccumulator
        The statistics to set the model from
        If a string, it is the path to the file containing the statistics
    config: configs.Configs
        The configuration object
        Optional, the default is the default configuration

    Returns
    -------
    wish.Wish
        The wish model with its values set from the statistics
    """
    if isinstance(accumulator, str):
        accumulator = stats_accumulator.StatsAccumulator.load(accumulator)
    hls = stats_accumulator.HighLevelStats(accumulator, config.poly_degree)

    loaded = Wish(config)
    loaded.set_from_stats(hls)
    return loaded


def accumulate_and_load_wish(configs=Configs()):
    """
    This function runs an accumulator over the whole dataset,
    then loads the wish model with the statistics from the accumulator.
    It returns both the model and the accumulator.

    Parameters
    ----------
    configs: configs.Configs
        The configuration object
        Optional, the default is the default configuration

    Returns
    -------
    wish.Wish
        The wish model with its values set from the statistics
    stats_accumulator.StatsAccumulator
        The accumulator object
    """
    print("Accumulating stats")
    acc = stats_accumulator.read_section_to(configs, False, 1, 0)
    print("Loading model")
    model = load_wish_from_accumulator(acc, configs)
    return model, acc
