"""
Fish==FastWish

Taking the lessons learned from the wish model, this is a continuous fast varient.
This module get the parameters that form the model for the fish simulation.
"""

import numpy as np
import torch

from ..utils.maths import gaussian, gumbel, gumbel_params
from ..utils.optimisers import curve_fit
from .custom_torch_distributions import make_tailed_exponential


# lots of our fits range in bins of incident energy and distance from center
# so we need to be able to calculate the bin centers
def incident_bin_centers(acc):
    """
    Calculate the incident energy bin centers.

    Parameters
    ----------
    acc : AlignedStatsAccumulator
        Histogramed data about the events.

    Returns
    -------
    incident_centers : numpy array of float (n_incident,)
        The center of the incident energy bins.
    """
    incident_bins = acc.incident_bin_boundaries
    incident_centers = 0.5 * (incident_bins[1:] + incident_bins[:-1])
    return incident_centers


def distance_bin_centers(acc):
    """
    Calculate the distance from center bin centers.

    Parameters
    ----------
    acc : AlignedStatsAccumulator
        Histogramed data about the events.

    Returns
    -------
    distance_centers : numpy array of float (n_distance,)
        The center of the distance from center bins.
    """
    distance_bins = acc.layer_offset_bins
    distance_centers = 0.5 * (distance_bins[1:] + distance_bins[:-1])
    return distance_centers


# Polynomial for overall trend
# Then gumbel at each incident energy
def binned_shift(acc):
    """
    Values for the shift of the distributions.
    These are fitted only to incident energy.

    Parameters
    ----------
    acc : AlignedStatsAccumulator
        Histogramed data about the events.

    Returns
    -------
    shift_mean : numpy array of float (n_incident,)
        The mean of the shift of the distributions v.s. incident energy.
    shift_std : numpy array of float (n_incident,)
        The standard deviation of the shift of the distributions v.s. incident energy.
    """
    shift_bins = acc.layer_offset_bins
    shift_centers = 0.5 * (shift_bins[1:] + shift_bins[:-1])
    shift_hist = acc.layer_offset_hist[1:-1]
    sum_per_incident = np.nansum(shift_hist, axis=1)
    shift_mean = np.nansum(shift_hist * shift_centers, axis=1) / sum_per_incident
    shift_sq_hist = acc.layer_offset_sq_hist[1:-1]
    shift_std = np.sqrt(
        np.nansum(shift_sq_hist * shift_centers, axis=1) / sum_per_incident
        - shift_mean**2
    )
    return shift_mean, shift_std


def fit_shift(incident_energy, shift_mean, shift_std, poly_order=3):
    """
    Fit the shift with a polynomial.
    These values define a gumbel from which the shift can be drawn.

    Parameters
    ----------
    incident_energy : numpy array of float (n_incident,)
        Incident energy values at the center of the bins.
    shift_mean : numpy array of float (n_incident,)
        The mean of the shift of the distributions v.s. incident energy.
    shift_std : numpy array of float (n_incident,)
        The standard deviation of the shift of the distributions v.s. incident energy.
    poly_order : int
        The order of the polynomial to fit.

    Returns
    -------
    p_mean : numpy array of float (poly_order+1,)
        The coefficients of the polynomial fit to the mean of the shift.
    p_std : numpy array of float (poly_order+1,)
        The coefficients of the polynomial fit to the standard deviation of the shift.
    """
    p_mean = np.polyfit(incident_energy, shift_mean, poly_order)
    p_std = np.polyfit(incident_energy, shift_std, poly_order)
    return p_mean, p_std


# Linear in mean n hits vs incident energy
# Gaussian in (recsaled for incident energy) mean n hits vs distance from center
def binned_mean_nHits(acc):
    """
    Bin the number of hits against the incident energy and distance from center.

    Parameters
    ----------
    acc : AlignedStatsAccumulator
        Histogramed data about the events.

    Returns
    -------
    n_hits_mean : numpy array of float (n_incident, distance)
        The mean number of hits for each incident energy and distance from the center.
    incident_weights : numpy array of float (n_incident,)
        Total number of events for each incident energy.

    """
    n_hits_per_bin = np.nansum(acc.counts_hist[1:-1], axis=(2, 3))
    events_per_bin = acc.total_events[1:-1]
    with np.errstate(divide="ignore"):
        n_hits_mean = n_hits_per_bin / events_per_bin
    n_hits_mean[events_per_bin == 0] = 0.0
    incident_weights = np.sum(events_per_bin, axis=1)
    return n_hits_mean, incident_weights


def fit_mean_nHits(
    n_hits_mean, incident_weights, incident_centers, distance_centers, poly_order=1
):
    """
    Fit the mean number of hits vs distance from center with a gaussian,
    where the height of the gaussian is a polynomial (by default linear)
    function of the incident energy.

    Parameters
    ----------
    n_hits_mean : numpy array of float (n_incident, n_distance)
        The mean number of hits for each bin.
    incident_weights : numpy array of float (n_incident,)
        Total number of events for each incident energy.
    incident_centers : numpy array of float (n_incident,)
        The center of the incident energy bins.
    distance_centers : numpy array of float (n_distance,)
        The center of the distance from center bins.
    poly_order : int
        The order of the polynomial to fit against incident energy.

    Returns
    -------
    mu : float
        The mean of the gaussian fit along the distance from the center.
    varience : float
        The varience of the gaussian fit along the distance from the center.
    height : numpy array of float (poly_order+1,)
        The height of the gaussian fit, with scaling for incident energy
    """
    # the weighting for each distance bin, rescaled for mean hits and incident energy
    # event frequency
    weight_array = incident_weights[:, np.newaxis] * n_hits_mean
    sum_weights = np.nansum(weight_array)
    mu = np.nansum(distance_centers * weight_array) / sum_weights
    var = np.nansum((distance_centers - mu) ** 2 * weight_array) / sum_weights
    # now get the height change with incident energy
    height_1_prediction = gaussian(distance_centers, mu, var, 1.0)
    ratios = np.nansum(n_hits_mean, axis=1) / np.nansum(height_1_prediction)
    # ratios = np.nanmean(n_hits_mean/height_1_prediction, axis=1)
    height = np.polyfit(incident_centers, ratios, poly_order)
    return mu, var, height


# coefficient of variation is insensitive to incident energy,
# it varies only with the distance from the center
# the variation is complex and it is fitted with a 12th order polynomial
def binned_cv_nHits(acc):
    """
    Fit the coefficient of variation of the number of hits.
    These are fitted to distance from the center.

    Parameters
    ----------
    acc : AlignedStatsAccumulator
        Histogramed data about the events.

    Returns
    -------
    cv_n_hits : numpy array of float (n_distance, )
        The coefficient of variation for each distance from the center.
    """
    events_per_bin = acc.total_events[1:-1]
    mask = events_per_bin == 0
    n_events_per_incident = np.nansum(events_per_bin, axis=0)
    # avoid division by zero
    events_per_bin[mask] = 1
    n_hits_per_event = np.nansum(acc.counts_hist[1:-1], axis=(2, 3)) / events_per_bin
    n_hits_sq_per_event = acc.evt_counts_sq_hist[1:-1] / events_per_bin
    std_n_hits_both = np.sqrt(n_hits_sq_per_event - n_hits_per_event**2)
    std_n_hits_both[mask] = 0.0
    std_n_hits = np.nansum(std_n_hits_both * n_events_per_incident, axis=0) / np.nansum(
        n_events_per_incident
    )
    mean_hits = np.nansum(n_hits_per_event * n_events_per_incident, axis=0) / np.nansum(
        n_events_per_incident
    )
    mean_hits[mean_hits == 0] = 1
    cv_n_hits = std_n_hits / mean_hits
    return cv_n_hits


def fit_cv_nHits(distance, cv_n_hits, poly_order=12):
    """
    Fit the coefficient of variation of the number of hits vs
    distance from center with a polynomial.

    Parameters
    ----------
    distance : numpy array of float (n_distance,)
        The distance from the center of the detector.
    cv_n_hits : numpy array of float (n_distance,)
        The coefficient of variation for each distance from the center.
    poly_order : int
        The order of the polynomial to fit.

    Returns
    -------
    cv_fit : numpy array of float (poly_order+1,)
        The coefficients of the polynomial fit to the coefficient of variation.
    """
    cv_fit = np.polyfit(distance, cv_n_hits, poly_order)
    return cv_fit


# The energy has 2d fits, so bin in both
# the incident energy and the distance from the center
def binned_mean_energy(acc):
    """
    Return the mean energy in bins of incident energy and distance from center.

    Parameters
    ----------
    acc : AlignedStatsAccumulator
        Histogramed data about the events.

    Returns
    -------
    mean_energy : numpy array of float (n_incident, n_distance)
        The mean energy for each incident energy and distance from the center.
    """
    total_energy = acc.evt_mean_E_hist[1:-1]
    events_per_bin = acc.total_events[1:-1]
    zero_mask = events_per_bin == 0
    events_per_bin[zero_mask] = 1
    mean_energy = total_energy / events_per_bin
    mean_energy[zero_mask] = 0.0
    return mean_energy


# The mean energy is fit to a gumbel distribution in the distance direction
# with the parameters of the gumbel varying in the incident energy direction
def fit_mean_energy_vs_distance(mean_energy, distance_centers):
    """
    Fit the mean energy across the distance from the center in
    each bin of incident energy. Uses a gumbel distribution,
    with height (vertical multiplier) and lift (vertical shift).

    Parameters
    ----------
    mean_energy : numpy array of float (n_incident, n_distance)
        The mean energy for each incident energy and distance from the center.
    distance_centers : numpy array of float (n_distance,)
        The center of the distance from center bins.

    Returns
    -------
    found_mu : numpy array of float (n_incident,)
        The mean of the gumbel fit for each incident energy.
    found_beta : numpy array of float (n_incident,)
        The beta of the gumbel fit for each incident energy.
    found_height : numpy array of float (n_incident,)
        Multiplier for the value of the gumbel fit.
    found_lift : numpy array of float (n_incident,)
        Addition (post multiplication by hight)
        to the height of the gumbel fit.
    """

    n_incident, n_distance = mean_energy.shape
    found_mu = np.zeros(n_incident)
    found_beta = np.zeros(n_incident)
    found_height = np.zeros(n_incident)
    found_lift = np.zeros(n_incident)

    p0 = [0.0, 1.0, 1.0, 0.0]
    bounds = [[-10.0, 0.0, 0.0, -10.0], [10.0, np.inf, 10.0, 10.0]]
    # we only fit the center of the distribution
    mask = (distance_centers > -1.0) & (distance_centers < 1.0)
    for i in range(n_incident):
        print(".", end="")
        popt, _ = curve_fit(
            gumbel,
            distance_centers[mask],
            mean_energy[i, mask],
            p0=p0,
            bounds=bounds,
            n_attempts=5,
            quiet=True,
        )
        found_mu[i] = popt[0]
        found_beta[i] = popt[1]
        found_height[i] = popt[2]
        found_lift[i] = popt[3]
    return found_mu, found_beta, found_height, found_lift


def fit_mean_energy(incident_centers, found_mu, found_beta, found_height, found_lift):
    """
    Using the gumble fits to the mean energy vs distance from the center
    in each incident energy bin, fit the parameters of the gumbel
    to simple polynomials in the incident energy.

    Parameters
    ----------
    incident_centers : numpy array of float (n_incident,)
        The center of the incident energy bins.
    found_mu : numpy array of float (n_incident,)
        The mean of the gumbel fit for each incident energy.
    found_beta : numpy array of float (n_incident,)
        The beta of the gumbel fit for each incident energy.
    found_height : numpy array of float (n_incident,)
        Multiplier for the value of the gumbel fit.
    found_lift : numpy array of float (n_incident,)
        Addition (post multiplication by hight)
        to the height of the gumbel fit.

    Returns
    -------
    mu : numpy array of float (2,)
        The mean of the gumbel fit for each incident energy.
    beta : numpy array of float (2,)
        The beta of the gumbel fit for each incident energy.
    height : numpy array of float (2,)
        Multiplier for the value of the gumbel fit.
    lift: numpy array of float (2,)
        Addition (post multiplication by hight)
        to the height of the gumbel fit.
    """
    # mu = np.polyfit(incident_centers, found_mu, 2)
    # beta = np.polyfit(incident_centers, found_beta, 2)
    # height = np.polyfit(incident_centers, found_height, 1)
    # lift = np.polyfit(incident_centers, found_lift, 1)
    mu = np.polyfit(incident_centers, found_mu, 1)
    beta = np.polyfit(incident_centers, found_beta, 1)
    height = np.polyfit(incident_centers, found_height, 1)
    lift = np.polyfit(incident_centers, found_lift, 1)
    return mu, beta, height, lift


# the cv of the energy fit is very like the cv of the number of points
def binned_cv_energy(acc):
    """
    Fit the coefficient of variation of the energy.
    These are fitted to distance from the center.

    Parameters
    ----------
    acc : AlignedStatsAccumulator
        Histogramed data about the events.

    Returns
    -------
    cv_energy : numpy array of float (n_distance, )
        The coefficient of variation for each distance from the center.
    """
    events_per_bin = acc.total_events[1:-1]
    mask = events_per_bin == 0
    n_events_per_incident = np.nansum(events_per_bin, axis=0)
    # avoid division by zero
    events_per_bin[mask] = 1

    energy_per_event = acc.evt_mean_E_hist[1:-1] / events_per_bin
    energy_sq_per_event = acc.evt_mean_E_sq_hist[1:-1] / events_per_bin
    std_energy_both = np.sqrt(energy_sq_per_event - energy_per_event**2)
    std_energy_both[mask] = 0.0
    std_energy = np.nansum(std_energy_both * n_events_per_incident, axis=0) / np.nansum(
        n_events_per_incident
    )
    mean_energy = np.nansum(
        energy_per_event * n_events_per_incident, axis=0
    ) / np.nansum(n_events_per_incident)
    # avoid division by zero
    mean_energy[mean_energy == 0] = 1
    cv_energy = std_energy / mean_energy

    return cv_energy


def fit_cv_energy(distance, cv_energy, poly_order=12):
    """
    Fit the coefficient of variation of the energy vs
    distance from center with a polynomial.

    Parameters
    ----------
    distance : numpy array of float (n_distance,)
        The distance from the center of the detector.
    cv_energy : numpy array of float (n_distance,)
        The coefficient of variation for each distance from the center.
    poly_order : int
        The order of the polynomial to fit.

    Returns
    -------
    cv_fit : numpy array of float (poly_order+1,)
        The coefficients of the polynomial fit to the coefficient of variation.
    """
    cv_fit = np.polyfit(distance, cv_energy, poly_order)
    return cv_fit


# Finally, the standard devation of the energy in an event is
# fit rather like the number of hits
# a linear fit on the incident energy, then gumbel on the distance
# Linear in mean n hits vs incident energy
def binned_stdEWithin_vs_incident(acc):
    """
    The standard devation of points inside events is binned
    per incident energy.

    Parameters
    ----------
    acc : AlignedStatsAccumulator
        Histogramed data about the events.

    Returns
    -------
    std_e_within : numpy array of float (n_incident, )
        The mean standard devation of points within an event
        For each incident energy bin.

    """
    events_per_incident = np.nansum(acc.total_events[1:-1], axis=1)
    zero_mask = events_per_incident == 0
    events_per_incident[zero_mask] = 1
    pnt_mean_E_sq = (
        np.nansum(acc.pnt_mean_E_sq_hist[1:-1], axis=1) / events_per_incident
    )
    mean_pnt_E_sq = (
        np.nansum(acc.evt_mean_E_sq_hist[1:-1], axis=1) / events_per_incident
    )
    std_e_within = np.sqrt(pnt_mean_E_sq - mean_pnt_E_sq)
    std_e_within[zero_mask] = 0.0
    return std_e_within


def fit_stdEWithin_vs_incident(incident_energy, std_e_within, poly_order=1):
    """
    Fit the standard deviation of points within events vs
    incident energy with a polynomial.  This is half the fit used for drawing
    the standard deviation of points within events.

    Parameters
    ----------
    incident_energy : numpy array of float (n_incident,)
        Incident energy values at the center of the bins.
    std_e_within : numpy array of float (n_incident,)
        The mean number of hits for each incident energy.
    poly_order : int
        The order of the polynomial to fit.

    Returns
    -------
    incident_fit : numpy array of float (poly_order+1,)
        The coefficients of the polynomial fit to
        the standard deviation of points within events.
    """
    incident_fit = np.polyfit(incident_energy, std_e_within, poly_order)
    return incident_fit


#  in (recsaled for incident energy) mean n hits vs distance from center
def rescaled_stdEWithin_vs_distance(acc, incident_fit, incident_centers):
    """
    After rescaling the standard deviation of points within events
    acording to the incident_fit, fit the standard deviation of points
    within events vs distance from center.

    Parameters
    ----------
    acc : AlignedStatsAccumulator
        Histogramed data about the events.
    incident_fit : numpy array of float (poly_order+1,)
        The coefficients for the fit between incident energy and
        standard deviation of points within events.
    incident_centers : numpy array of float (n_incident,)
        The center of the incident energy bins.

    Returns
    -------
    stdEWithin : numpy array of float (n_distance, )
        The standard deviation of points within events
        for each distance from the center.
    """
    std_per_incident = np.polyval(incident_fit, incident_centers)
    events_per_bin = acc.total_events[1:-1]
    sum_events_per_bin = np.nansum(events_per_bin, axis=0)
    sum_events_per_bin[sum_events_per_bin == 0] = 1
    zero_mask = events_per_bin == 0
    events_per_bin[zero_mask] = 1
    pnt_mean_E_sq = acc.pnt_mean_E_sq_hist[1:-1] / events_per_bin
    mean_pnt_E_sq = acc.evt_mean_E_sq_hist[1:-1] / events_per_bin
    std_e_within = np.sqrt(pnt_mean_E_sq - mean_pnt_E_sq)
    rescaled = std_e_within / std_per_incident[:, np.newaxis]
    rescaled[zero_mask] = 0.0
    stdEWithin = np.nansum(rescaled * events_per_bin, axis=0) / sum_events_per_bin
    return stdEWithin


def fit_stdEWithin_vs_distance(distance, incident_rescaled):
    """
    Give the parameters of a gumbel to fit the standard deviation
    of energy within an event vs distance from center.

    Parameters
    ----------
    distance : numpy array of float (n_distance,)
        The distance from the center of the detector.
    incident_rescaled : numpy array of float (n_distance,)
        The rescaled standard devation of energy within an event
        in each distrance bin.

    Returns
    -------
    mu : float
        The mean of the gumbel fit.
    beta : float
        The varience of the gumbel fit.
    height : float
        The height of the gumbel fit.
    lift: float
        The lift of the gumbel fit.
    """
    mask = (distance > -1.0) & (distance < 1.0) & (incident_rescaled > 0)
    if not np.any(mask):
        if np.all(incident_rescaled <= 0):
            raise ValueError("All values of incident_rescaled are zero or less")
        elif np.all(distance < -1.0):
            raise ValueError("All values of distance are less than -1")
        else:
            raise ValueError("All values of distance are greater than 1")
    masked_distance = distance[mask]
    masked_incident_rescaled = incident_rescaled[mask]

    # choose the initial guess based on the mean and variance of the data
    incidence_sum = np.nansum(masked_incident_rescaled)
    dist_mean = np.nansum(masked_distance * masked_incident_rescaled) / incidence_sum
    non_zero_weights = np.nansum(masked_incident_rescaled > 0)
    dist_var = (non_zero_weights / (non_zero_weights - 1)) * np.nansum(
        (masked_distance - dist_mean) ** 2 * masked_incident_rescaled
    )
    gumbel_mu, gumbel_beta = gumbel_params(dist_mean, dist_var)

    p0 = [gumbel_mu, gumbel_beta, 1.0, 0.0]
    bounds = [[-10.0, 0.0, 0.0, -5.0], [10.0, 10.0, 100.0, 0.3]]
    params, _ = curve_fit(
        gumbel,
        masked_distance,
        masked_incident_rescaled,
        p0=p0,
        bounds=bounds,
        n_attempts=5,
        quiet=True,
    )
    return params


# Also, there is a radial distribution of the number of hits
# this is a pair of exponentials, one for the core and one for the tail
# The parameters of the exponentials vary only with distance
def radial_bin_centers(acc):
    """
    Calculate the radial bin centers, shifting so counts center across all
    incident energies and distances is at the radial center.

    Parameters
    ----------
    acc : AlignedStatsAccumulator
        Histogramed data about the events.

    Returns
    -------
    radial_centers : numpy array of float (n_radial,)
        The center of the radial bins.
        May not be evenly spaced.
    """
    x_bin_centers, y_bin_centers, y_bin_centers = acc._get_bin_centers()
    counts = acc.counts_hist.sum(axis=(0, 1))
    counts_x_marginal = counts.sum(axis=1)
    x_center = np.nansum(x_bin_centers * counts_x_marginal) / counts_x_marginal.sum()
    counts_y_marginal = counts.sum(axis=0)
    y_center = np.nansum(y_bin_centers * counts_y_marginal) / counts_y_marginal.sum()
    corrected_x = x_bin_centers - x_center
    corrected_y = y_bin_centers - y_center
    radial_bins = np.sqrt(corrected_x[:, np.newaxis] ** 2 + corrected_y**2)
    return radial_bins.flatten()


def binned_radial_probs(acc, radial_centers):
    """
    The probility of a hit at the radius of the bin.
    This is not equal to the density of the bin, there is a jacobian
    transformation to account for the growing area of the cylindrical
    shell.

    Parameters
    ----------
    acc : AlignedStatsAccumulator
        Histogramed data about the events.

    Returns
    -------
    radial_probs : numpy array of float (n_distances, n_radial)
        The probability of a hit for each radial distance from the center.
        Split by the distance along the axis of the shower.
    """
    bin_jacobean = np.pi * radial_centers / (acc.lateral_bin_size**2)
    radial_hist = np.nansum(acc.counts_hist, axis=0)
    radial_hist = radial_hist.reshape(radial_hist.shape[0], len(radial_centers))
    unnormed_radial_probs = radial_hist * bin_jacobean
    denominator = np.nansum(unnormed_radial_probs, axis=1)
    zero_mask = denominator == 0
    denominator[zero_mask] = 1
    radial_probs = unnormed_radial_probs / denominator[:, np.newaxis]
    return radial_probs


# this is the one that takes the longest
# consider using torch for the optimisation instead.
# It takes a long time because the fits are quite unstable and need
# a high number of attempts
# 100 attempts are wanted for a confident fit
def fit_radial_probs(
    distance, radial_centers, radial_probs, poly_order=2, n_attempts=10
):
    """
    Fit the radial probability distribution with a tailed exponential distribution
    of the form:

    p(x) = p * (exp(-x/core)/core) + (1-p) * (exp(-x/tail)/tail)
        where
    tail = core + to_tail

    A fit is made at each distance from the center, then the parameters of the fit
    vs distance are fitted with a polynomial.

    Parameters
    ----------
    distance : numpy array of float (n_distance,)
        The distance from the center of the detector.
    radial_centers : numpy array of float (n_radial,)
        The center of the radial bins (all points at which the probability is known).
    radial_probs : numpy array of float (n_distance, n_radial)
        The probability corrisponding to each radial center.
    poly_order : int
        The order of the polynomial to fit for each
        parameter against distance from the center.
    n_attempts : int
        The number of attempts to make at fitting the tailed exponential.
        Use 100 for reliable fits, use 10 for passable fits, use 1 for quick fits.

    Returns
    -------
    core_fit : numpy array (poly_order+1,)
        The coefficients of the polynomial fit to the core of the exponential.
    to_tail_fit : numpy array (poly_order+1,)
        The coefficients of the polynomial fit to the distance from
        the core to the tail.
    p_core_fit : numpy array (poly_order+1,)
        The coefficients of the polynomial fit to the proportion of the core.

    """

    def to_fit(xs, core, to_tail, p_core, normalisation):
        distribution = make_tailed_exponential(core, to_tail, p_core)
        found = np.exp(distribution.log_prob(torch.tensor(xs)))
        normed = found * normalisation
        return normed

    p0 = [0.01, 0.1, 0.5, 0.1]
    bounds = [[1e-10, 1e-10, 1e-5, 0.0], [10.0, 1000.0, 1 - 1e-5, 1.0]]
    # we don't keep the normalisation, it is not useful
    params_per_distance = np.zeros((radial_probs.shape[0], len(p0) - 1))
    for i, probs in enumerate(radial_probs):
        print(".", end="")
        params_here, _ = curve_fit(
            to_fit,
            radial_centers,
            probs,
            p0=p0,
            bounds=bounds,
            n_attempts=n_attempts,
            quiet=True,
        )
        params_per_distance[i] = params_here[:-1]
    core_fit = np.polyfit(distance, params_per_distance[:, 0], poly_order)
    to_tail_fit = np.polyfit(distance, params_per_distance[:, 1], poly_order)
    p_core_fit = np.polyfit(distance, params_per_distance[:, 2], poly_order)
    return core_fit, to_tail_fit, p_core_fit


class Parametrisation:
    """
    Generate, save and load the parameters for the fish model.

    Attributes
    ----------
    acc : AlignedStatsAccumulator
        Histogramed data about the events.
    n_attempts : int
        The number of attempts to make at fitting the tailed exponential.
        Use 100 for reliable fits, use 10 for passable fits, use 1 for quick fits.
    incident_centers : torch tensor of float (n_incident,)
        The center of the incident energy bins. Mostly used internally.
    distance_centers : torch tensor of float (n_distance,)
        The center of the distance from center bins. Mostly used internally.
    shift_mean : torch tensor of float (4,)
        The coefficients of the polynomial fit to the mean of the shift
        where the shift is the displacement along the axis of the shower
        required to align the center of each shower with 0.
    shift_std : torch tensor of float (4,)
        The coefficients of the polynomial fit to the standard deviation of the shift.
    nhm_vs_dist_mean : float
        The mean of the gaussian fit to the number of hits vs distance from the
        aligned (by the shift) center of the shower.
    nhm_vs_dist_var : float
        The varience of the gaussian fit to the number of hits vs distance from the
        aligned (by the shift) center of the shower.
    nhm_vs_incident_height : torch tensor of float (2,)
        The rescaling factor for the gaussian fit to the number of hits,
        varies with incident energy.
    cv_n_hits : torch tensor of float (13,)
        cv == coefficient of variation
        The coefficients of the polynomial fit to the coefficient of variation
        of the number of hits against distance from the aligned (by the shift).
        No variation with incident energy.
    me_mu : torch tensor of float (2,)
        me == mean energy
        The mean of the gumbel fit for the energy vs distance from the
        aligned (by the shift) center of the shower.
    me_beta : torch tensor of float (2,)
        The beta of the gumbel fit for the energy vs distance from the
        aligned (by the shift) center of the shower.
    me_height : torch tensor of float (2,)
        The multiplier for the value of the gumbel fit for the energy vs
        distance from the aligned (by the shift) center of the shower.
    me_lift : torch tensor of float (2,)
        The addition (post multiplication by hight) to the height of the
        gumbel fit for the energy vs distance from the aligned (by the shift)
        center of the shower.
    cv_energy : torch tensor of float (13,)
        The coefficients of the polynomial fit to the coefficient of variation
        of the energy against distance from the aligned (by the shift) center.
        No variation with incident energy.
    sewe_vs_incident : torch tensor of float (2,)
        sewe == standard energy within event
        The coefficients of the linear fit between the standard deviation of energy
        within an event and the incident energy.
    sewe_vs_dist_mu : float
        The mean of the gumbel fit for the standard deviation of energy within an event
        vs distance from the aligned (by the shift) center of the shower.
    sewe_vs_dist_beta : float
        The beta of the gumbel fit for the standard deviation of energy within an event
        vs distance from the aligned (by the shift) center of the shower.
    sewe_vs_dist_height : float
        The multiplier for the value of the gumbel fit for the standard deviation of
        energy within an event vs distance from the aligned (by the shift)
        center of the shower.
    sewe_vs_dist_lift : float
        The addition (post multiplication by hight) to the height of the
        gumbel fit for the standard deviation of energy within an event vs
        distance from the aligned (by the shift) center of the shower.
    radial_vs_dist_core : torch tensor of float (3,)
        The coefficients of the polynomial fit to the core of the exponential
        distribution of the radial distribution of hits.
    radial_vs_dist_to_tail : torch tensor of float (3,)
        The coefficients of the polynomial fit to the distance from the core to the tail
        of the exponential distribution of the radial distribution of hits.
    radial_vs_dist_p_core : torch tensor of float (3,)
        The coefficients of the polynomial fit to the proportion of the core of the
        exponential distribution of the radial distribution of hits.

    """

    def __init__(self, acc=None, n_attempts=20):
        """
        Constructor for the parametrisation.

        Parameters
        ----------
        acc : AlignedStatsAccumulator
            Histogramed data about the events.
        n_attempts : int
            The number of attempts to make at fitting the tailed exponential.
            Use 100 for reliable fits, use 10 for passable fits, use 1 for quick fits.
        """
        self.acc = acc
        self.n_attempts = n_attempts
        if acc is not None:
            print("Generating parameters")
            self.incident_centers = incident_bin_centers(acc)
            self.distance_centers = distance_bin_centers(acc)
            params = self.get_params()
            for key, value in params.items():
                value = torch.tensor(value)
                setattr(self, key, value)
            self._save_attrs = list(params.keys())
            self._save_attrs.extend(
                ["incident_centers", "distance_centers", "n_attempts"]
            )

    def get_params(self):
        """
        Get the parameters for the fish model from the acc.
        """
        params = {}
        shift_means, shift_stds = binned_shift(self.acc)
        print("Fitting shift")
        params["shift_mean"], params["shift_std"] = fit_shift(
            self.incident_centers, shift_means, shift_stds
        )
        n_hits_means, incident_weights = binned_mean_nHits(self.acc)
        # nhm == number of hist mean
        print("Fitting mean nHits ")
        print("Fitting mean nHits")
        (
            params["nhm_vs_dist_mean"],
            params["nhm_vs_dist_var"],
            params["nhm_vs_incident_height"],
        ) = fit_mean_nHits(
            n_hits_means, incident_weights, self.incident_centers, self.distance_centers
        )
        cv_n_hits = binned_cv_nHits(self.acc)
        print("Fitting cv nHits")
        params["cv_n_hits"] = fit_cv_nHits(self.distance_centers, cv_n_hits)
        mean_energy = binned_mean_energy(self.acc)
        print("Fitting mean energy")
        found_mu, found_beta, found_height, found_lift = fit_mean_energy_vs_distance(
            mean_energy, self.distance_centers
        )

        # me == mean energy
        (
            params["me_mu"],
            params["me_beta"],
            params["me_height"],
            params["me_lift"],
        ) = fit_mean_energy(
            self.incident_centers, found_mu, found_beta, found_height, found_lift
        )
        cv_energy = binned_cv_energy(self.acc)
        print("Fitting cv energy")
        params["cv_energy"] = fit_cv_energy(self.distance_centers, cv_energy)
        std_e_within = binned_stdEWithin_vs_incident(self.acc)
        # sewe == standard energy within event
        print("Fitting stdEWithin vs incident")
        params["sewe_vs_incident"] = fit_stdEWithin_vs_incident(
            self.incident_centers, std_e_within
        )
        if np.all(params["sewe_vs_incident"] == 0):
            msg = (
                "All points have same energy, this class isn't build to handle that.\n"
                "If this is intentional you need to write a special case for this"
            )
            raise NotImplementedError(msg)
        rescaled_std_e_within = rescaled_stdEWithin_vs_distance(
            self.acc, params["sewe_vs_incident"], self.incident_centers
        )
        (
            params["sewe_vs_dist_mu"],
            params["sewe_vs_dist_beta"],
            params["sewe_vs_dist_height"],
            params["sewe_vs_dist_lift"],
        ) = fit_stdEWithin_vs_distance(self.distance_centers, rescaled_std_e_within)
        radial_centers = radial_bin_centers(self.acc)
        radial_probs = binned_radial_probs(self.acc, radial_centers)
        print("Fitting radial probs")
        (
            params["radial_vs_dist_core"],
            params["radial_vs_dist_to_tail"],
            params["radial_vs_dist_p_core"],
        ) = fit_radial_probs(
            self.distance_centers,
            radial_centers,
            radial_probs,
            n_attempts=self.n_attempts,
        )
        for key, value in params.items():
            if np.any(np.isnan(value)):
                print(f"NaN in {key}")
                import ipdb

                ipdb.set_trace()
                pass
        return params

    def save(self, filename):
        """
        Save the parameters to a file.

        Parameters
        ----------
        filename : str
            The name of the file to save the parameters to.
        """
        save_dict = {}
        for attr in self._save_attrs:
            try:
                save_dict[attr] = getattr(self, attr).numpy()
            except AttributeError:
                save_dict[attr] = getattr(self, attr)
        np.savez(filename, **save_dict)

    @classmethod
    def load(cls, filename):
        """
        Load the parameters from a file.

        Parameters
        ----------
        filename : str
            The name of the file to load the parameters from.
        """
        loaded = np.load(filename)
        parametrisation = cls.direct_load(loaded)
        return parametrisation

    @classmethod
    def direct_load(cls, numpy_loaded):
        """
        Load the parameters from a numpy loaded file.

        Parameters
        ----------
        numpy_loaded : numpy loaded file
            The numpy loaded file to load the parameters from.
        """
        acc = None
        parametrisation = cls(acc)
        for key in numpy_loaded.keys():
            value = torch.tensor(numpy_loaded[key])
            setattr(parametrisation, key, value)
        return parametrisation
