"""
Arithmatic
"""
import warnings
import numpy as np
import torch
from scipy.optimize import minimize

_TORCH_TINY = torch.tensor(torch.finfo(torch.float32).tiny)
_TORCH_MAX = torch.tensor(torch.finfo(torch.float32).max)


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


def gaussian(x, mean, varience, height):
    """
    Gaussian function, with height scaling.

    Works with numpy arrays or torch tensors.

    Parameters
    ----------
    x : array_like
        The input values.
    mean : float
        The mean of the Gaussian.
    varience : float
        The standard deviation squared of the Gaussian.
    height : float
        Scale factor for the height of the Gaussian.

    Returns
    -------
    array_like
        The Gaussian function evaluated at x.
    """
    return height * np.exp(-0.5 * ((x - mean) ** 2 / varience))


def gumbel(x, mu, beta, height, lift):
    """
    Gumbel function, with height scaling and lift.

    Works with numpy arrays or torch tensors.

    Parameters
    ----------
    x : array_like
        The input values.
    mu : float
        The location of the peak of the Gumbel.
    beta : float
        The steepness of the Gumbel.
    height : float
        Scale factor for the height of the Gumbel.
    lift : float
        The minimum value of the Gumbel.

    Returns
    -------
    array_like
        The Gumbel function evaluated at x.
    """
    # this can create overflow errors and other numerical issues
    # need safeguards
    z = (x - mu) / beta
    with np.errstate(over="raise"):
        try:
            # this is the correct formula
            distribution = lift + height * np.exp(-z - np.exp(-z))
        except FloatingPointError:
            warnings.warn("Warning: Numerical issues with Gumbel function.")
            # assume that the -z - exp(-z) term is very negative
            shaped = (x - x) + 1
            distribution = lift * shaped
    # TODO maybe remove
    # max_unnormed_gumbel = 0.37  # unnormalised gumbel is between 0 and 0.3679...
    # max_value = lift + height * max_unnormed_gumbel
    # if (distribution > max_value).any():
    #    # if we exceeded the maximum value, assume the results are bad
    #    distribution[distribution > max_value] = lift
    return distribution


def gumbel_params(mean, varience):
    """
    Calculates mu and beta from a mean and standard deviation.

    Parameters
    ----------
    mean : float or array_like of float
        The mean of the Gumbel.
    varience : float or array_like of float
        The standard deviation squared of the Gumbel.

    Returns
    -------
    mu : float or array_like of float
        The location of the peak of the Gumbel.
    beta : float or array_like of float
        The steepness of the Gumbel.
    """
    beta = np.sqrt(varience * 6) / np.pi
    mu = mean - np.euler_gamma * beta
    return mu, beta


def weibull_params(mean, standarddev, tune=False):
    """
    Calculates approximate concentration and scale from the mean
    and standard devation.

    Parameters
    ----------
    mean : float or array_like of float
        The mean of the Gumbel.
    standarddev : float or array_like of float
        The standard deviation of the Gumbel.

    Returns
    -------
    scale : float or array_like of float
        The scale of the distribution.
    concentration : float or array_like of float
        The concentration of the distribution.
    """
    # this is an approximation, but it is good enough for many purposes
    concentration = (mean / standarddev) ** (1.086)
    concentration = torch.clip(concentration, _TORCH_TINY, _TORCH_MAX)
    denominator = (1 + 1 / concentration).lgamma().exp()
    scale = torch.clip(mean / denominator, _TORCH_TINY, None)
    if tune:  # TODO, should do this with torch.
        # use an optimiser to improve the calculation
        # exspensive, but effective
        def to_minimise(args):
            penalty = 0
            for i in range(2):
                if args[i] < 0:
                    penalty += args[i] ** 2
                    args[i] = _TORCH_TINY
            distribution = torch.distributions.Weibull(*args)
            mean_diff = (distribution.mean - mean) ** 2
            std_diff = (distribution.stddev - standarddev) ** 2
            return mean_diff + std_diff + penalty

        x0 = np.array([scale, concentration])
        initial_cost = to_minimise(x0)
        result = minimize(to_minimise, x0)
        scale, concentration = result.x
        optimised_cost = to_minimise([scale, concentration])
        print(optimised_cost, initial_cost)
        if optimised_cost > initial_cost:
            warnings.warn("Warning: Tuning did not improve the result.")
            scale, concentration = x0
    return scale, concentration


def logNorm_params(mean, standarddev):
    """
    Calculates the location and scale from the mean
    and standard devation.
    Uses operations that require torch Tensors.

    Parameters
    ----------
    mean : torch Tensor
        The mean of the Gumbel.
    standarddev : torch Tensor
        The standard deviation of the log normal.

    Returns
    -------
    location : torch Tensor
        The location of the distribution.
    scale : torch Tensor
        The scale of the distribution.
    """
    # TODO, this seems to throw so really odd results...
    ratio = (standarddev / mean).nan_to_num()
    scale2 = torch.clip(torch.log(1 + ratio**2), _TORCH_TINY, None)
    scale = torch.sqrt(scale2)
    location = torch.log(mean) - 0.5 * scale2
    # can't just set nan to 0, as nan appears where location should be negative
    minimal_location = location[~torch.isnan(location)].min().item()
    clean_location = location.nan_to_num(nan=minimal_location)
    max_val = torch.clip(clean_location.abs(), _TORCH_TINY, 1.0) * _TORCH_MAX
    clean_scale = torch.clip(scale.nan_to_num(), _TORCH_TINY, max_val.max())
    return clean_location, clean_scale
