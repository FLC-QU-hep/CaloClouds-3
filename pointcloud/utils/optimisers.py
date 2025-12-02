"""
Custom optimisers
"""
import numpy as np

from scipy.optimize import curve_fit as _curve_fit
from scipy.optimize._minimize import Bounds
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._lsq.least_squares import prepare_bounds
from scipy.linalg import solve_triangular, cholesky, LinAlgError


def _wrap_func(func, xdata, ydata, transform):
    """
    Stolen directly from scipy.optimize._minipack_py, which is hidden from the user
    """
    if transform is None:

        def func_wrapped(params):
            return func(xdata, *params) - ydata

    elif transform.size == 1 or transform.ndim == 1:

        def func_wrapped(params):
            return transform * (func(xdata, *params) - ydata)

    else:
        # Chisq = (y - yd)^T C^{-1} (y-yd)
        # transform = L such that C = L L^T
        # C^{-1} = L^{-T} L^{-1}
        # Chisq = (y - yd)^T L^{-T} L^{-1} (y-yd)
        # Define (y-yd)' = L^{-1} (y-yd)
        # by solving
        # L (y-yd)' = (y-yd)
        # and minimize (y-yd)'^T (y-yd)'
        def func_wrapped(params):
            return solve_triangular(transform, func(xdata, *params) - ydata, lower=True)

    return func_wrapped


def _initialize_feasible(lb, ub):
    """
    Stolen directly from scipy.optimize._minipack_py, which is hidden from the user
    """
    p0 = np.ones_like(lb)
    lb_finite = np.isfinite(lb)
    ub_finite = np.isfinite(ub)

    mask = lb_finite & ub_finite
    p0[mask] = 0.5 * (lb[mask] + ub[mask])

    mask = lb_finite & ~ub_finite
    p0[mask] = lb[mask] + 1

    mask = ~lb_finite & ub_finite
    p0[mask] = ub[mask] - 1

    return p0


def get_p0_n_bounds(f, p0=None, bounds=(-np.inf, np.inf)):
    """
    Given the users input for the function, and optionally the initial parameters,
    and the bounds, determine the initial parameters and the number of parameters
    and bounds. Same as done in the scipy curve_fit function.

    Parameters
    ----------
    f : callable
        The model function, f(x, ...). It must take the independent
        variable as the first argument and the parameters to fit as
        separate remaining arguments.
    p0 : array_like, optional
        Initial guess for the parameters (length N). If None, then the
        initial values will all be 1 (if the number of parameters for the
        function can be determined using introspection, otherwise a
        ValueError is raised).
        In the case that finite bounds are provided, the initial guess will
        be chosen to be within the bounds, centered if both bounds are finite.
    bounds : 2-tuple of array_like or `Bounds`, optional
        Lower and upper bounds on parameters. Defaults to no bounds.
        There are two ways to specify the bounds:

            - Instance of `Bounds` class.

            - 2-tuple of array_like: Each element of the tuple must be either
              an array with the length equal to the number of parameters, or a
              scalar (in which case the bound is taken to be the same for all
              parameters). Use ``np.inf`` with an appropriate sign to disable
              bounds on all or some parameters.

    Returns
    -------
    p0 : array
        Initial guess for the parameters.
    n : int
        Number of parameters in the model.
    lower_bounds : array
        Lower bounds on the parameters.
    upper_bounds : array
        Upper bounds on the parameters.
    """
    if p0 is None:
        # determine number of parameters by inspecting the function
        sig = _getfullargspec(f)
        args = sig.args
        if len(args) < 2:
            raise ValueError("Unable to determine number of fit parameters.")
        n = len(args) - 1
    else:
        p0 = np.atleast_1d(p0)
        n = p0.size

    if isinstance(bounds, Bounds):
        lower_bounds = bounds.lb
        upper_bounds = bounds.ub
    else:
        lower_bounds, upper_bounds = prepare_bounds(bounds, n)

    if p0 is None:
        p0 = _initialize_feasible(lower_bounds, upper_bounds)

    return p0, n, lower_bounds, upper_bounds


def evaluator_factory(f, xdata, ydata, sigma=None):
    """
    Make a function that can score chosen parameters for the fit.
    The lower the score, the better the fit.

    Parameters
    ----------
    f : callable
        The model function, f(x, ...). It must take the independent
        variable as the first argument and the parameters to fit as
        separate remaining arguments.
    xdata : array_like
        The independent variable where the data is measured.
        Should usually be an M-length sequence or an (k,M)-shaped array for
        functions with k predictors, and each element should be float
        convertible if it is an array like object.
    ydata : array_like
        The dependent data, a length M array - nominally ``f(xdata, ...)``.
    sigma : None or scalar or M-length sequence or MxM array, optional
        Determines the uncertainty in `ydata`. If we define residuals as
        ``r = ydata - f(xdata, *popt)``, then the interpretation of `sigma`
        depends on its number of dimensions:

            - A scalar or 1-D `sigma` should contain values of standard deviations of
              errors in `ydata`. In this case, the optimized function is
              ``chisq = sum((r / sigma) ** 2)``.

            - A 2-D `sigma` should contain the covariance matrix of
              errors in `ydata`. In this case, the optimized function is
              ``chisq = r.T @ inv(sigma) @ r``.


        None (default) is equivalent of 1-D `sigma` filled with ones.

    Returns
    -------
    score_func : callable, ``score_func(params)``
        A function that calculates the score of the chosen parameters.
    """
    # Determine type of sigma
    if sigma is not None:
        sigma = np.asarray(sigma)

        # if 1-D or a scalar, sigma are errors, define transform = 1/sigma
        if sigma.size == 1 or sigma.shape == (ydata.size,):
            transform = 1.0 / sigma
        # if 2-D, sigma is the covariance matrix,
        # define transform = L such that L L^T = C
        elif sigma.shape == (ydata.size, ydata.size):
            try:
                # scipy.linalg.cholesky requires lower=True to return L L^T = A
                transform = cholesky(sigma, lower=True)
            except LinAlgError as e:
                raise ValueError("`sigma` must be positive definite.") from e
        else:
            raise ValueError("`sigma` has incorrect shape.")
    else:
        transform = None

    def score_func(params):
        func = _wrap_func(f, xdata, ydata, transform)
        return (func(params) ** 2).sum()

    return score_func


def chose_trials(n_attempts, p0, lower_bounds, upper_bounds):
    """
    Choose the parameters for the trials.
    One attempt is always made with the initial parameters,
    then for the remaining attempts, for parameters with finite bounds,
    random values are chosen within the bounds, for parameters with infinite bounds,
    random values are chosen as a normal distribution with
    mean for the initial parameters and standard devation equal to the mean.

    Parameters
    ----------
    n_attempts : int
        Number of attempts to make including the initial guess.
    p0 : array
        Initial guess for the parameters.
    lower_bounds : array
        Lower bounds on the parameters.
    upper_bounds : array
        Upper bounds on the parameters.

    Returns
    -------
    trials : array, shape (n_attempts, p0.size)
        The parameters for the trials.
    """
    trials = np.empty((n_attempts, p0.size))
    trials[0] = p0
    finite_bounds = np.isfinite(lower_bounds) & np.isfinite(upper_bounds)
    min_varience = 0.01
    for i in np.where(finite_bounds)[0]:
        trials[1:, i] = np.random.uniform(
            lower_bounds[i], upper_bounds[i], n_attempts - 1
        )
    for i in np.where(~finite_bounds)[0]:
        var = max(np.abs(p0[i]), min_varience)
        random = np.random.normal(p0[i], var, n_attempts - 1)
        # might still have one finite bound
        trials[1:, i] = np.clip(random, lower_bounds[i], upper_bounds[i])
    return trials


def curve_fit(
    f,
    xdata,
    ydata,
    n_attempts=10,
    p0=None,
    sigma=None,
    absolute_sigma=False,
    check_finite=None,
    bounds=(-np.inf, np.inf),
    method=None,
    jac=None,
    *,
    full_output=False,
    nan_policy=None,
    quiet=False,
    **kwargs,
):
    """
    Same idea but more robust than the scipy curve_fit.
    It will make multiple attempts within the bounds to find a fit.
    One attempt is always made with the initial parameters,
    then for the remaining attempts, for parameters with finite bounds,
    random values are chosen within the bounds, for parameters with infinite bounds,
    random values are chosen as a normal distribution with
    mean for the initial parameters and standard devation equal to the mean.

    Parameters
    ----------
    f : callable
        The model function, f(x, ...). It must take the independent
        variable as the first argument and the parameters to fit as
        separate remaining arguments.
    xdata : array_like
        The independent variable where the data is measured.
        Should usually be an M-length sequence or an (k,M)-shaped array for
        functions with k predictors, and each element should be float
        convertible if it is an array like object.
    ydata : array_like
        The dependent data, a length M array - nominally ``f(xdata, ...)``.
    n_attempts : int, optional
        Number of attempts to make including the initial guess. Default is 10.
    p0 : array_like, optional
        Initial guess for the parameters (length N). If None, then the
        initial values will all be 1 (if the number of parameters for the
        function can be determined using introspection, otherwise a
        ValueError is raised).
    sigma : None or scalar or M-length sequence or MxM array, optional
        Determines the uncertainty in `ydata`. If we define residuals as
        ``r = ydata - f(xdata, *popt)``, then the interpretation of `sigma`
        depends on its number of dimensions:

            - A scalar or 1-D `sigma` should contain values of standard deviations of
              errors in `ydata`. In this case, the optimized function is
              ``chisq = sum((r / sigma) ** 2)``.

            - A 2-D `sigma` should contain the covariance matrix of
              errors in `ydata`. In this case, the optimized function is
              ``chisq = r.T @ inv(sigma) @ r``.


        None (default) is equivalent of 1-D `sigma` filled with ones.
    absolute_sigma : bool, optional
        If True, `sigma` is used in an absolute sense and the estimated parameter
        covariance `pcov` reflects these absolute values.

        If False (default), only the relative magnitudes of the `sigma` values matter.
        The returned parameter covariance matrix `pcov` is based on scaling
        `sigma` by a constant factor. This constant is set by demanding that the
        reduced `chisq` for the optimal parameters `popt` when using the
        *scaled* `sigma` equals unity. In other words, `sigma` is scaled to
        match the sample variance of the residuals after the fit. Default is False.
        Mathematically,
        ``pcov(absolute_sigma=False) = pcov(absolute_sigma=True) * chisq(popt)/(M-N)``
    check_finite : bool, optional
        If True, check that the input arrays do not contain nans of infs,
        and raise a ValueError if they do. Setting this parameter to
        False may silently produce nonsensical results if the input arrays
        do contain nans. Default is True if `nan_policy` is not specified
        explicitly and False otherwise.
    bounds : 2-tuple of array_like or `Bounds`, optional
        Lower and upper bounds on parameters. Defaults to no bounds.
        There are two ways to specify the bounds:

            - Instance of `Bounds` class.

            - 2-tuple of array_like: Each element of the tuple must be either
              an array with the length equal to the number of parameters, or a
              scalar (in which case the bound is taken to be the same for all
              parameters). Use ``np.inf`` with an appropriate sign to disable
              bounds on all or some parameters.

    method : {'lm', 'trf', 'dogbox'}, optional
        Method to use for optimization. See `least_squares` for more details.
        Default is 'lm' for unconstrained problems and 'trf' if `bounds` are
        provided. The method 'lm' won't work when the number of observations
        is less than the number of variables, use 'trf' or 'dogbox' in this
        case.

    jac : callable, string or None, optional
        Function with signature ``jac(x, ...)`` which computes the Jacobian
        matrix of the model function with respect to parameters as a dense
        array_like structure. It will be scaled according to provided `sigma`.
        If None (default), the Jacobian will be estimated numerically.
        String keywords for 'trf' and 'dogbox' methods can be used to select
        a finite difference scheme, see `least_squares`.

    nan_policy : {'raise', 'omit', None}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is None):

          * 'raise': throws an error
          * 'omit': performs the calculations ignoring nan values
          * None: no special handling of NaNs is performed
            (except what is done by check_finite); the behavior when NaNs
            are present is implementation-dependent and may change.

        Note that if this value is specified explicitly (not None),
        `check_finite` will be set as False.

    **kwargs
        Keyword arguments passed to `leastsq` for ``method='lm'`` or
        `least_squares` otherwise.

    Returns
    -------
    popt : array
        Optimal values for the parameters so that the sum of the squared
        residuals of ``f(xdata, *popt) - ydata`` is minimized.
    pcov : 2-D array
        The estimated approximate covariance of popt. The diagonals provide
        the variance of the parameter estimate. To compute one standard
        deviation errors on the parameters, use
        ``perr = np.sqrt(np.diag(pcov))``. Note that the relationship between
        `cov` and parameter error estimates is derived based on a linear
        approximation to the model function around the optimum [1].
        When this approximation becomes inaccurate, `cov` may not provide an
        accurate measure of uncertainty.

        How the `sigma` parameter affects the estimated covariance
        depends on `absolute_sigma` argument, as described above.

        If the Jacobian matrix at the solution doesn't have a full rank, then
        'lm' method returns a matrix filled with ``np.inf``, on the other hand
        'trf'  and 'dogbox' methods use Moore-Penrose pseudoinverse to compute
        the covariance matrix. Covariance matrices with large condition numbers
        (e.g. computed with `numpy.linalg.cond`) may indicate that results are
        unreliable.

    Raises
    ------
    ValueError
        if either `ydata` or `xdata` contain NaNs, or if incompatible options
        are used.

    RuntimeError
        if the least-squares minimization fails.

    OptimizeWarning
        if covariance of the parameters can not be estimated.

    """
    # never return full_output
    kwargs["full_output"] = False
    p0, n, lower_bounds, upper_bounds = get_p0_n_bounds(f, p0, bounds)
    score_func = evaluator_factory(f, xdata, ydata, sigma)
    trials = chose_trials(n_attempts, p0, lower_bounds, upper_bounds)
    found = np.empty_like(trials)
    scores = np.empty(n_attempts)
    covariences = np.empty((n_attempts, n, n))
    for i, trial in enumerate(trials):
        try:
            popt, pcov = _curve_fit(
                f,
                xdata,
                ydata,
                p0=trial,
                sigma=sigma,
                absolute_sigma=absolute_sigma,
                check_finite=check_finite,
                bounds=bounds,
                method=method,
                jac=jac,
                nan_policy=nan_policy,
                **kwargs,
            )
        except RuntimeError:
            scores[i] = np.inf
            continue
        found[i] = popt
        scores[i] = score_func(popt)
        covariences[i] = pcov
        if not quiet and (n_attempts < 10 or n_attempts % 10 == 0):
            format_input = ", ".join([f"{p:.6g}" for p in trial])
            format_opt = ", ".join([f"{p:.6g}" for p in popt])
            print(
                f"Optimisation trial {i+1}/{n_attempts} complete\n"
                f"Initial parameters: {format_input}\n"
                f"Optimal parameters: {format_opt}\n"
                f"Score: {scores[i]:.6g}\n"
            )
    best = np.argmin(scores)
    if not quiet:
        print(f"All {n_attempts} optimisation trials complete\n" "Best trial;")
        format_input = ", ".join([f"{p:.6g}" for p in trials[best]])
        format_opt = ", ".join([f"{p:.6g}" for p in found[best]])
        print(
            f"Initial parameters: {format_input}\n"
            f"Optimal parameters: {format_opt}\n"
            f"Score: {scores[i]:.6g}\n"
        )

    return found[best], covariences[best]
