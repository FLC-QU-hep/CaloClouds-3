"""
Fish==FastWish

Taking the lessons learned from the wish model, this is a continuous fast varient.
This module uses the parameters caluclated in fish_parmaetrise and generates events.
"""

import numpy as np
import torch
from torch.distributions import Gumbel, LogNormal, Weibull

from ..configs import Configs
from ..utils.maths import (
    gumbel,
    gumbel_params,
    weibull_params,
    logNorm_params,
    torch_polyval,
    gaussian,
)
from ..utils import stats_accumulator, metadata
from .custom_torch_distributions import make_tailed_exponential
from .fish_parametrise import Parametrisation, distance_bin_centers


def calc_shifts(parameters, incidents):
    """
    Get the mean and std of the shifts for the incident energies.

    Parameters
    ----------
    parameters : .fish_paramterise.Parametrisation
        The parameters, with attributes shift_mean, an array of (poly_order,)
        floats, and shift_std, an array of (poly_order,) floats.
    incidents : array_like of float (n_events,)
        Incident energy values for each event to be drawn.

    Returns
    -------
    shift_gumbel_mu : array of float (n_events,)
        The mean of the gumbel distribution for the shift.
    shift_gumbel_beta : array of float (n_events,)
        The beta of the gumbel distribution for the shift.
    """
    means = torch_polyval(parameters.shift_mean, incidents)
    stds = torch_polyval(parameters.shift_std, incidents)
    stds = torch.clip(stds, 0.0, None)
    shift_gumbel_mu, shift_gumbel_beta = gumbel_params(means, stds**2)
    return shift_gumbel_mu, shift_gumbel_beta


def calc_dist_nHits(parameters, incidents, distances):
    """
    Get the mean and std number of hits for the given incidents and distances.

    Parameters
    ----------
    parameters : .fish_paramterise.Parametrisation
        The parameters, with attributes
        nhm_vs_dist_mean : float
            The mean of the gaussian fit to the number of hits vs distance from the
            aligned (by the shift) center of the shower.
        nhm_vs_dist_var : float
            The varience of the gaussian fit to the number of hits vs distance from the
            aligned (by the shift) center of the shower.
        nhm_vs_incident_height : numpy array of float (2,)
            nhm == number of hist mean
            The coefficients of the linear fit between the mean number of hits
            and the incident energy.
        cv_n_hits : numpy array of float (13,)
            cv == coefficient of variation
            The coefficients of the polynomial fit to the coefficient of variation
            of the number of hits against distance from the aligned (by the shift).
            No variation with incident energy.
    incidents : array_like of float (n_events,)
        Incident energy values for each event to be drawn.
    distances : array_like of float (n_distances,)
        Distances from distribution center for each event to be drawn.

    Returns
    -------
    mean_n_hits : numpy array of float (n_events, n_distances)
        The mean number of hits expected.
    std_n_hits : numpy array of float (n_events, n_distances)

    """
    # get the heights
    incident_scale = torch_polyval(parameters.nhm_vs_incident_height, incidents)
    mean_n_hits = gaussian(
        distances,
        parameters.nhm_vs_dist_mean,
        parameters.nhm_vs_dist_var,
        incident_scale[:, None],
    )
    mean_n_hits = torch.clip(mean_n_hits, 0.0, None)

    # Now calculate the standard deviation of the number of hits
    cv = torch_polyval(parameters.cv_n_hits, distances)
    cv = torch.clip(cv, 0.0, None)
    std_n_hits = mean_n_hits * cv
    return mean_n_hits, std_n_hits


def calc_dist_energy(parameters, incidents, distances):
    """
    Get the mean and std energy for the given incidents and distances.

    Parameters
    ----------
    parameters : .fish_paramterise.Parametrisation
        The parameters, with attributes
        me_mu : numpy array of float (3,)
            me == mean energy
            The mean of the gumbel fit for the energy vs distance from the
            aligned (by the shift) center of the shower.
        me_beta : numpy array of float (3,)
            The beta of the gumbel fit for the energy vs distance from the
            aligned (by the shift) center of the shower.
        me_height : numpy array of float (2,)
            The multiplier for the value of the gumbel fit for the energy vs
            distance from the aligned (by the shift) center of the shower.
        me_lift : numpy array of float (2,)
            The addition (post multiplication by hight) to the height of the
            gumbel fit for the energy vs distance from the aligned (by the shift)
            center of the shower.
        cv_energy : numpy array of float (13,)
            The coefficients of the polynomial fit to the coefficient of variation
            of the energy against distance from the aligned (by the shift) center.
            No variation with incident energy.
    incidents : numpy array of float (n_incident,)
        Incident energy values for each event to be drawn.
    distances : numpy array of float (n_distances,)
        Distances from distribution center for each event to be drawn.

    Returns
    -------
    mean_energy : numpy array of float (n_incident, n_distances)
        The mean energy expected for each event at each distance.
    std_energy : numpy array of float (n_incident, n_distances)
        The standard deviation of the energy expected for each event
        at each distance.
    """
    incidents = incidents[:, None]
    mu = torch_polyval(parameters.me_mu, incidents)
    mu = mu.nan_to_num()
    beta = torch_polyval(parameters.me_beta, incidents)
    beta = torch.clip(beta.nan_to_num(), 0.0, None)
    height = torch_polyval(parameters.me_height, incidents)
    lift = torch_polyval(parameters.me_lift, incidents)
    mean_energy = gumbel(distances, mu, beta, height, lift)
    mean_energy = torch.clip(mean_energy.nan_to_num(), 0.0, None)
    cv = torch_polyval(parameters.cv_energy, distances)
    cv = torch.clip(cv, 0.0, None)
    std_energy = mean_energy * cv
    return mean_energy, std_energy


def calc_stdEWithin(parameters, incidents, distances):
    """
    Get the standard devation of energy within an event
    for the given incidents and distances.

    Parameters
    ----------
    parameters : .fish_paramterise.Parametrisation
        The parameters, with attributes
        sewe_vs_incident : numpy array of float (2,)
            sewe == standard energy within event
            The coefficients of the linear fit between the standard deviation
            of energy within an event and the incident energy.
        sewe_vs_dist_mu : float
            The mean of the gumbel fit for the standard deviation of energy within
            an event vs distance from the aligned (by the shift) center of the shower.
        sewe_vs_dist_beta : float
            The beta of the gumbel fit for the standard deviation of energy within
            an event vs distance from the aligned (by the shift) center of the shower.
        sewe_vs_dist_height : float
            The multiplier for the value of the gumbel fit for the standard deviation of
            energy within an event vs distance from the aligned (by the shift)
            center of the shower.
        sewe_vs_dist_lift : float
            The addition (post multiplication by hight) to the height of the
            gumbel fit for the standard deviation of energy within an event vs
            distance from the aligned (by the shift) center of the shower.
    incidents : numpy array of float (n_events,)
        Incident energy values for each event to be drawn.
    distances : numpy array of float (n_distances,)
        Distances from distribution center for each event to be drawn.

    Returns
    -------
    std_e_within : numpy array of float (n_events, n_distances)
        The standard deviation of energy within an event expected.

    """
    # start by calculating the gumbel predictions
    gumble = gumbel(
        distances,
        parameters.sewe_vs_dist_mu,
        parameters.sewe_vs_dist_beta,
        parameters.sewe_vs_dist_height,
        parameters.sewe_vs_dist_lift,
    )
    incident_scale = torch_polyval(parameters.sewe_vs_incident, incidents[:, None])
    std_e_within = gumble * incident_scale
    std_e_within = torch.clip(std_e_within, 0.0, None)
    return std_e_within


def calc_radial(parameters, distances):
    """
    Get the parameters for the radial distribution of hits at the specifed distances.
    The parameters are the core, distance to tail and core probability.

    Parameters
    ----------
    parameters : .fish_paramterise.Parametrisation
        The parameters, with attributes
        radial_vs_dist_core : numpy array of float (3,)
            Polynomial coefficients for the core of the radial distribution
            against distance from the aligned (by the shift) center.
        radial_vs_dist_to_tail : numpy array of float (3,)
            Polynomial coefficients for the distance to tail of the radial distribution
            against distance from the aligned (by the shift) center.
        radial_vs_dist_p_core : numpy array of float (3,)
            Polynomial coefficients for the probability of the radial distribution
            against distance from the aligned (by the shift) center.
    distances : numpy array of float (n_distances,)
        Distances from distribution center for each event to be drawn.

    Returns
    -------
    core : numpy array of float (n_distances,)
        The core of the radial distribution.
    to_tail : numpy array of float (n_distances,)
        The distance to the tail of the radial distribution.
    prob : numpy array of float (n_distances,)
        The probability of the radial distribution.
    """
    core = torch_polyval(parameters.radial_vs_dist_core, distances)
    core = torch.clip(core, 0.0, None)
    to_tail = torch_polyval(parameters.radial_vs_dist_to_tail, distances)
    to_tail = torch.clip(to_tail, 0.0, None)
    prob = torch_polyval(parameters.radial_vs_dist_p_core, distances)
    prob = torch.clip(prob, 0.0, 1.0)
    return core, to_tail, prob


class Fish:
    parameters = [
        "shift_gumbel_mu",
        "shift_gumbel_beta",
        "n_hits_weibull_scale",
        "n_hits_weibull_concen",
        "energy_lognorm_loc",
        "energy_lognorm_scale",
        "stdEWithin_lognorm_loc",
        "stdEWithin_lognorm_scale",
    ]

    def __init__(self, parametrisation, acc=None):
        """
        Parameters
        ----------
        parametrisation : dict
            The parameters for the fish model.
        """
        if isinstance(parametrisation, str):
            parametrisation = Parametrisation.load(parametrisation)
        self.parametrisation = parametrisation
        acc = acc or parametrisation.acc
        if acc is not None:
            self.distances = torch.tensor(distance_bin_centers(acc))
            n_distances = len(self.distances)
            first_quarter = int(np.floor(0.25 * n_distances))
            last_quarter = int(np.floor(0.75 * n_distances))
            self.layer_depths = self.distances[first_quarter:last_quarter]

    # TODO should this clip to the detector area?
    # currently it just gives all the points, including ones outside the detector
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
        incident_energies = conditioning[:, -1]
        stats = self.calc_stats(incident_energies, True, True)
        components = self.sample_axial_components(stats)
        samples = self.combine_components(stats, components, max_hits)
        return samples

    all_predictions = ["shift", "n_hits", "mean_energy", "stdEWithin"]

    def calc_stats(
        self, incident_energies, include_stdEWithin=True, include_radial=False
    ):
        """
        Predict a (possibly limited) set of high level parameters for a sample
        at each incident energy.

        Parameters
        ----------
        incident_energies : array of values, (n_incident,)
            The incident energies to predict the stats for.
        include_stdEWithin : bool, optional
            Should we also predict the standard deviation of energy within an event.
            Default True.
        include_radial : bool, optional
            Should we also predict radial parts of the distribution.
            Default False.

        Returns
        -------
        stats : dict of str, (n_incident, n_distance)
            Each of the stats requested.
            Keys always present are; "shift_gumbel_mu", "shift_gumbel_beta"
            "n_hits_weibull_scale", "n_hits_weibull_concen",
            "energy_lognorm_loc", "energy_lognorm_scale",
            If include_stdEWithin is True, also contains
            "stdEWithin_lognorm_loc", "stdEWithin_lognorm_scale"
            If include_radial is True, also contains
            "radial_core", "radial_to_tail", "radial_prob"
        """
        stats = {}
        stats["shift_gumbel_mu"], stats["shift_gumbel_beta"] = calc_shifts(
            self.parametrisation, incident_energies
        )
        mean_n_hits, std_n_hits = calc_dist_nHits(
            self.parametrisation, incident_energies, self.distances
        )
        (
            stats["n_hits_weibull_scale"],
            stats["n_hits_weibull_concen"],
        ) = weibull_params(mean_n_hits, std_n_hits)
        mean_energy, std_energy = calc_dist_energy(
            self.parametrisation, incident_energies, self.distances
        )
        stats["energy_lognorm_loc"], stats["energy_lognorm_scale"] = logNorm_params(
            mean_energy, std_energy
        )
        if include_stdEWithin:
            std_e_within = calc_stdEWithin(
                self.parametrisation, incident_energies, self.distances
            )
            (
                stats["stdEWithin_lognorm_loc"],
                stats["stdEWithin_lognorm_scale"],
            ) = logNorm_params(torch.ones(1), std_e_within)
        if include_radial:
            (
                stats["radial_core"],
                stats["radial_to_tail"],
                stats["radial_prob"],
            ) = calc_radial(self.parametrisation, self.distances)
        return stats

    @classmethod
    def sample_axial_components(cls, stats):
        """
        Sample the components for a set of per event values
        not including per hit values.

        Parameters
        ----------
        stats : dict of str, (n_incident, n_distance)
            Each of the stats needed.
            Keys always present are; "shift_gumbel_mu", "shift_gumbel_beta"
            "n_hits_weibull_scale", "n_hits_weibull_concen",
            "energy_lognorm_loc", "energy_lognorm_scale",

        Returns
        -------
        components : dict
            The component samples.
            Contains "shift", "n_hits" and "mean_energy".

        """
        components = {}
        shift_dist = Gumbel(stats["shift_gumbel_mu"], stats["shift_gumbel_beta"])
        components["shift"] = shift_dist.sample()
        n_hits_dist = Weibull(
            stats["n_hits_weibull_scale"], stats["n_hits_weibull_concen"]
        )
        n_hits = n_hits_dist.sample()
        components["n_hits"] = n_hits
        energy_dist = LogNormal(
            stats["energy_lognorm_loc"], stats["energy_lognorm_scale"]
        )
        mean_energy = energy_dist.sample()
        components["mean_energy"] = mean_energy
        return components

    @classmethod
    def clip_n_hits(cls, n_hits, max_hits):
        """
        For all events, reduce the number of hits to a maximum value,
        and where this has reduced the number of hits, calculate apropriate
        rescaling of the energy of each point to maintain the mean energy.

        Parameters
        ----------
        n_hits : torch Tensor (n_events, n_layers)
            The number of hits in each layer for each event.
        max_hits : int
            The maximum number of hits to allow in a single event.

        Returns
        -------
        reduced_n_hits : torch Tensor (n_events, n_layers)
            The number of hits in each layer for each event, clipped such that
            each event has at most max_hits hits.
        energy_rescale : torch Tensor (n_events, n_layers)
            The rescale factor for the energy of each point in each layer.

        """
        hits_in_event = n_hits.sum(1)
        excess_hits = torch.clip(hits_in_event - max_hits, 0, None)
        percent_reduction = excess_hits / hits_in_event
        # this will hit sparse layers worst
        #reduced_n_hits = torch.floor(n_hits - n_hits * percent_reduction[:, None]).int()
        reduced_n_hits = (n_hits - n_hits * percent_reduction[:, None]).int()
        new_hits_in_event = reduced_n_hits.sum(1)
        remaining_excess_hits = torch.clip(new_hits_in_event - max_hits, 0, None)
        # just take these off the longest layer
        reduced_n_hits[
            torch.arange(reduced_n_hits.shape[0]), reduced_n_hits.argmax(1)
        ] -= remaining_excess_hits
        energy_rescale = n_hits / reduced_n_hits
        return reduced_n_hits, energy_rescale

    def combine_components(self, stats, components, max_hits, quiet=False):
        """
        Given the output of sample_axial_components and the radial distribution,
        create the points of an event.

        Parameters
        ----------
        stats : dict of str, (n_incident, n_distance)
            Each of the stats needed, in particular for the radial distribution
            and the standard devation of energies in an event.
            Will contain
            "radial_core", "radial_to_tail", "radial_prob"
            "stdEWithin_lognorm_loc", "stdEWithin_lognorm_scale"
        components : dict
            The dictionary returned by sample_axial_components.
        max_hits : int
            The length to pad events to, clipping any longer events
            (leaving the mean energy unchanged).
        quiet : bool, optional
            Print progress updates. Default False.

        Returns
        -------
        event : torch Tensor (n_events, max_hits, 4)
            The points of each event, with last axis x,y,z,e,
            padded with 0.
        """
        n_events, n_layers = components["n_hits"].shape
        # angles are completely randomly distributed
        # so we could sample oustide the loop for speed if desired.
        # TODO this could be optimised out of the loops in various places...
        points = torch.zeros(n_events, max_hits, 4)
        radial_dists = [
            make_tailed_exponential(core, to_tail, prob)
            for core, to_tail, prob in zip(
                stats["radial_core"], stats["radial_to_tail"], stats["radial_prob"]
            )
        ]
        reduced_n_hits, energy_rescale = self.clip_n_hits(
            components["n_hits"], max_hits
        )
        for event_n, n_hits_per_layer in enumerate(reduced_n_hits):
            if event_n % 100 == 0 and not quiet:
                print(f"{event_n/n_events:.0%}", end="\r", flush=True)
            cumulative_hits = n_hits_per_layer.cumsum(0)
            shift = components["shift"][event_n]
            for layer_n, n_hits in enumerate(n_hits_per_layer):
                if n_hits == 0:
                    continue
                slice_n = slice(
                    cumulative_hits[layer_n] - n_hits, cumulative_hits[layer_n]
                )
                angles = 2 * torch.rand(n_hits) * torch.pi
                radii = radial_dists[layer_n].sample([n_hits]).flatten()
                xs = radii * torch.cos(angles)
                ys = radii * torch.sin(angles)
                zs = torch.ones(n_hits) * self.distances[layer_n] + shift
                # the stdEWithin varies both with incident energy and distance
                # so it gets generated inside the loop
                energy_varation_dist = LogNormal(
                    stats["stdEWithin_lognorm_loc"][event_n, layer_n],
                    stats["stdEWithin_lognorm_scale"][event_n, layer_n],
                )
                energy_variations = energy_varation_dist.sample([n_hits])
                es = (
                    components["mean_energy"][event_n, layer_n]
                    * energy_variations
                    * energy_rescale[event_n, layer_n]
                )
                points[event_n, slice_n, 0] = xs
                points[event_n, slice_n, 1] = ys
                points[event_n, slice_n, 2] = zs
                points[event_n, slice_n, 3] = es

        if not quiet:
            print("100% - Done")
        return points

    def save(self, path):
        """
        Save the model to a file.

        Parameters
        ----------
        path : str
            The path to save the model to.
        """
        save_dict = {
            attr: getattr(self.parametrisation, attr)
            for attr in self.parametrisation._save_attrs
        }
        save_dict["layer_depths"] = self.layer_depths
        save_dict["distances"] = self.distances
        np.savez(path, **save_dict)

    @classmethod
    def load(cls, path):
        """
        Load a model from a file.

        Parameters
        ----------
        path : str
            The path to load the model from.

        Returns
        -------
        model : Fish
            The loaded model.
        """
        loaded = np.load(path)
        parametrisation = Parametrisation.direct_load(loaded)
        layer_depths = torch.tensor(loaded["layer_depths"])
        distances = torch.tensor(loaded["distances"])
        created = cls(parametrisation)
        created.distances = distances
        created.layer_depths = layer_depths
        return created


def load_fish_from_accumulator(
    accumulator="../point-cloud-diffusion-logs/wish/dataset_accumulators/"
    "p22_th90_ph90_en10-1/p22_th90_ph90_en10-100_seedAll_alignMean.h5",
    config=Configs(),
):
    """
    Load the fish model and set it's values from the statistics gathered by an accumulator.

    Parameters
    ----------
    accumulator: str or stats_accumulator.StatsAccumulator
        The statistics to set the model from
        If a string, it is the path to the file containing the statistics
    config: config.Configs
        The configuration object
        Optional, the default is the default configuration

    Returns
    -------
    fish.Fish
        The fish model with its values set from the statistics
    """
    if isinstance(accumulator, str):
        accumulator = stats_accumulator.AlignedStatsAccumulator.load(accumulator)
    parametrisation = Parametrisation(accumulator)
    model = Fish(parametrisation, acc=accumulator)
    return model


def accumulate_and_load_fish(config=Configs()):
    """
    This function runs an accumulator over the whole dataset,
    then loads the fish model with the statistics from the accumulator.
    It returns both the model and the accumulator.

    Parameters
    ----------
    config: config.Configs
        The configuration object
        Optional, the default is the default configuration

    Returns
    -------
    fish.Fish
        The fish model with its values set from the statistics
    stats_accumulator.StatsAccumulator
        The accumulator object
    """
    print("Accumulating stats")
    acc = stats_accumulator.read_section_to(config, False, 1, 0, "alignMean")
    print("Loading model")
    model = load_fish_from_accumulator(acc, config)
    return model, acc
