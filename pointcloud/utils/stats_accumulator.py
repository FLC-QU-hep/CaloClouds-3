"""
Gather statistical summaries from the data.
Used to generate parameters for the wish model.
"""
import os
from tqdm import tqdm
import h5py
import numpy as np
import warnings
from matplotlib import pyplot as plt

from pointcloud.utils.plotting import heatmap_stack
from pointcloud.utils.optimisers import curve_fit
from pointcloud.utils.detector_map import split_to_layers, floors_ceilings
from pointcloud.data.dataset import dataset_class_from_config
from pointcloud.data.read_write import read_raw_regaxes, get_n_events
from pointcloud.configs import Configs


class StatsAccumulator:
    min_incident_energy = 10
    max_incident_energy = 90

    def __init__(
        self,
        Xmin=-1,
        Xmax=1,
        Ymin=-1,
        Ymax=1,
        layer_bottom=np.linspace(-1, 1, 31)[:-1],
        cell_thickness=None,
        incident_energy_bin_size=5.0,
        lateral_bin_size=None,
    ):
        """
        Accumulate statistical overview of the the events in the dataset.
        Allows events to be added incrementally without overloading the memory.

        Parameters
        ----------
        Xmin: float (optional)
            Minimum x coordinate of the detector
            Optional, default is -1.0
        Xmax: float (optional)
            Maximum x coordinate of the detector
            Optional, default is 1.0
        Ymin: float (optional)
            Minimum y coordinate of the detector
            Optional, default is -1.0
        Ymax: float (optional)
            Maximum y coordinate of the detector
            Optional, default is 1.0
        layer_bottom: array of floats (n_layers,) (optional)
            The y coordinates of the bottom of each detector layer
            Optional, default is 30 evenly spaced layers between -1 and 1
        cell_thickness : float (optional)
            The thickness of each layer
            Optional, default is distance between the first two layer bottoms
        incident_energy_bin_size: float
            Size of the bins for the incident energy
            Optional, default is 5.0
        lateral_bin_size: float
            Size of the bins in the x and z direction
            Optional, default is to make 50 bins in the X direction.

        Attributes
        ----------
        total_events: array (n_incident_bins + 2)
            Total number of events in each incident energy bin
            with under and overflow bins.
            Formula;
                 n_{events}
        counts_hist: array (n_incident_bins + 2, n_layers, n_x_bins, n_z_bins)
            Number of points in each bin summed over events, bins being
            - incident energy with under and overflow bins
            - layers
            - x coordinates
            - z coordinates
            This is the finest possible bins
            Formula;
                 Sum_{events} N(layer, incident_energy, event, x, z)
        counts_sq_hist: array (n_incident_bins + 2, n_layers, n_x_bins, n_z_bins)
            Number of points squared in each bin summed over events, bins being
            - incident energy with under and overflow bins
            - layers
            - x coordinates
            - z coordinates
            This is the finest possible bins
            Formula;
                 Sum_{events} N(layer, incident_energy, event, x, z)^2
        evt_counts_sq_hist: array (n_incident_bins + 2, n_layers)
            Mean nnumber of points squared in each bin summed over events, bins being
            - incident energy with under and overflow bins
            - layers
            So we are binning at the incident energy and layer level.
            Formula;
                Sum_{events} (Sum_{x, z} N(layer, incident_energy, event, x, z))^2
        energy_hist: array (n_incident_bins + 2, n_layers, n_x_bins, n_z_bins)
            Energy of the points in each bin summed over events, bins being
            - incident energy with under and overflow bins
            - layers
            - x coordinates
            - z coordinates
            This is the finest possible bins
            We don't need this for wish calculations,
            just gathering it for visulisation.
            Formula;
                Sum_{events} Sum_{point_i}
                E(layer, incident_energy, event, x, z, point_i)
        energy_sq_hist: array (n_incident_bins + 2, n_layers, n_x_bins, n_z_bins)
            Energy of the points squared in each bin summed over events, bins being
            - incident energy with under and overflow bins
            - layers
            - x coordinates
            - z coordinates
            This is the finest possible bins
            We don't need this for wish calculations,
            just gathering it for visulisation.
            Formula;
                Sum_{events} Sum_{point_i}
                E(layer, incident_energy, event, x, z, point_i)^2
        evt_mean_E_hist: array (n_incident_bins + 2, n_layers)
            Mean energy of the points in each bin summed over events, bins being
            - incident energy with under and overflow bins
            - layers
            So we are binning at the incident energy and layer level.
            Formula;
                Sum_{events} ((Sum_{x, z, point_i}
                              E(layer, incident_energy, event, x, z, point_i))
                              / (Sum_{x, z} N(layer, incident_energy, event, x, z)))
        evt_mean_E_sq_hist: array (n_incident_bins + 2, n_layers)
            Mean energy squared of the points in each bin summed over events, bins being
            - incident energy with under and overflow bins
            - layers
            So we are binning at the incident energy and layer level.
            Formula;
                Sum_{events} ((Sum_{x, z, point_i}
                              E(layer, incident_energy, event, x, z, point_i))
                              / (Sum_{x, z} N(layer, incident_energy, event, x, z)))^2
        pnt_mean_E_sq_hist: array (n_incident_bins + 2, n_layers)
            Mean of the square of the energy of
            each point in each bin summed over events, bins being
            - incident energy with under and overflow bins
            - layers
            So we are binning at the incident energy and layer level.
            Formula;
                Sum_{events} ((Sum_{x, z, point_i}
                              E(layer, incident_energy, event, x, z, point_i)^2)
                              / (Sum_{x, z} N(layer, incident_energy, event, x, z)))
        evt_mean_counts_hist: array (n_incident_bins + 2, n_layers, n_x_bins, n_z_bins)
            Mean number of points per event in each bin summed, bins being
            - incident energy with under and overflow bins
            - layers
            - x coordinates
            - z coordinates
            Formula;
                Sum_{events} (N(layer, incident_energy, event, x, z)/
                              (Sum_{x, z} N(layer, incident_energy, event, x, z))

        """
        self.accumulated_indices = []

        self.Xmin = Xmin
        self.Xmax = Xmax
        self.Ymin = Ymin
        self.Ymax = Ymax
        self.incident_energy_bin_size = incident_energy_bin_size
        if lateral_bin_size is None:
            lateral_bin_size = (Xmax - Xmin) / 50
        self.lateral_bin_size = lateral_bin_size

        self.layer_bottom = layer_bottom
        self.n_layers = len(layer_bottom)
        if cell_thickness is None:
            cell_thickness = layer_bottom[1] - layer_bottom[0]
        self.cell_thickness = cell_thickness

        self.incident_bin_boundaries = np.arange(
            self.min_incident_energy,
            self.max_incident_energy,
            self.incident_energy_bin_size,
        )
        n_incident_bins = len(self.incident_bin_boundaries) - 1
        self.lateral_x_bin_boundaries = np.arange(
            Xmin, Xmax + lateral_bin_size, lateral_bin_size
        )
        n_x_bins = len(self.lateral_x_bin_boundaries) - 1
        self.lateral_y_bin_boundaries = np.arange(
            Ymin, Ymax + lateral_bin_size, lateral_bin_size
        )
        n_y_bins = len(self.lateral_y_bin_boundaries) - 1

        # the things we are accumulating
        self.total_events = np.zeros(n_incident_bins + 2)
        # incident bins get under and overflow
        self.counts_hist = np.zeros(
            (n_incident_bins + 2, self.n_layers, n_x_bins, n_y_bins)
        )
        self.counts_sq_hist = np.zeros(
            (n_incident_bins + 2, self.n_layers, n_x_bins, n_y_bins)
        )
        self.evt_counts_sq_hist = np.zeros((n_incident_bins + 2, self.n_layers))
        self.energy_hist = np.zeros(
            (n_incident_bins + 2, self.n_layers, n_x_bins, n_y_bins)
        )
        self.energy_sq_hist = np.zeros(
            (n_incident_bins + 2, self.n_layers, n_x_bins, n_y_bins)
        )
        self.evt_mean_E_hist = np.zeros((n_incident_bins + 2, self.n_layers))
        self.evt_mean_E_sq_hist = np.zeros((n_incident_bins + 2, self.n_layers))
        self.pnt_mean_E_sq_hist = np.zeros((n_incident_bins + 2, self.n_layers))
        self.evt_mean_counts_hist = np.zeros(
            (n_incident_bins + 2, self.n_layers, n_x_bins, n_y_bins)
        )

    def layer_hists(self, points_in_layer):
        n_pts_hist, _, _ = np.histogram2d(
            points_in_layer[:, 0],
            points_in_layer[:, 1],
            bins=(self.lateral_x_bin_boundaries, self.lateral_y_bin_boundaries),
        )
        energies_hist, _, _ = np.histogram2d(
            points_in_layer[:, 0],
            points_in_layer[:, 1],
            bins=(self.lateral_x_bin_boundaries, self.lateral_y_bin_boundaries),
            weights=points_in_layer[:, 3],
        )
        energies_hist_sq, _, _ = np.histogram2d(
            points_in_layer[:, 0],
            points_in_layer[:, 1],
            bins=(self.lateral_x_bin_boundaries, self.lateral_y_bin_boundaries),
            weights=points_in_layer[:, 3] ** 2,
        )
        return n_pts_hist, energies_hist, energies_hist_sq

    def add_event(self, incident_bin, counts, energies, energies_sq):
        self.counts_hist[incident_bin] += counts
        self.counts_sq_hist[incident_bin] += counts**2
        sum_counts = np.sum(counts, axis=(1, 2))
        total_counts = np.sum(sum_counts)
        if total_counts:
            self.evt_mean_counts_hist[incident_bin] += counts / total_counts
        self.energy_hist[incident_bin] += energies
        self.energy_sq_hist[incident_bin] += energies_sq
        self.evt_counts_sq_hist[incident_bin] += sum_counts**2
        # we don't want zeros here, as it will create nans in the division
        sum_counts[sum_counts <= 0] = 1.0
        mean_E = np.sum(energies, axis=(1, 2)) / sum_counts
        self.evt_mean_E_hist[incident_bin] += mean_E
        self.evt_mean_E_sq_hist[incident_bin] += mean_E**2
        self.pnt_mean_E_sq_hist[incident_bin] += (
            np.sum(energies_sq, axis=(1, 2)) / sum_counts
        )

    def add(self, idxs, energy, events):
        """
        Add events to the accumulator.

        Parameters
        ----------
        idxs: array-like (n_events,)
            Indices of the events, to check for duplicate accumulation
        energy: array-like (n_events,)
            The incident energy of the events
        events: array-like (n_events, max_n_points, 4)
            The events themselves, in xyze format

        """
        assert len(idxs) == len(energy) == len(events)
        if any(idx in self.accumulated_indices for idx in idxs):
            raise ValueError("Index already accumulated")
        self.accumulated_indices.extend(idxs)

        incident_energy_bins = np.digitize(energy, self.incident_bin_boundaries)
        event_counts = np.zeros_like(self.counts_hist[0])
        event_energies = np.zeros_like(self.energy_hist[0])
        event_energies_sq = np.zeros_like(self.energy_hist[0])
        for incident_bin, event in zip(incident_energy_bins, events):
            # remove any padding
            event = event[event[:, 3] > 0]
            self.total_events[incident_bin] += 1
            event_counts[:] = 0
            event_energies[:] = 0
            event_energies_sq[:] = 0
            for layer_n, points_in_layer in enumerate(
                split_to_layers(event, self.layer_bottom, self.cell_thickness)
            ):
                n_pts_hist, energies_hist, energies_hist_sq = self.layer_hists(
                    points_in_layer
                )
                event_counts[layer_n] = n_pts_hist
                event_energies[layer_n] = energies_hist
                event_energies_sq[layer_n] = energies_hist_sq
            self.add_event(
                incident_bin, event_counts, event_energies, event_energies_sq
            )

    @classmethod
    def get_saved_attrs(cls):
        """
        The attrs are the constructor arguments.
        They will be saved to disk whtn the object is saved to enable reconstruction.
        Must be a class method as it is called before the object is created.

        Returns
        -------
        list of str
            The names of the attributes, also the names of the constructor arguments
        """
        attrs = [
            "layer_bottom",
            "cell_thickness",
            "incident_energy_bin_size",
            "lateral_bin_size",
        ]
        for limit in ["min", "max"]:
            for coord in ["X", "Y"]:
                attrs.append(coord + limit)
        return attrs

    def get_saved_datasets(self):
        """
        The datasets are the things we are accumulating.
        They will be saved to disk when the object is saved to disk.

        Returns
        -------
        list of str
            The names of the datasets
        """
        datasets = ["total_events", "accumulated_indices"]
        for name in dir(self):
            if name.endswith("_hist"):
                datasets.append(name)
        return datasets

    def save(self, path):
        """
        Save the accumulator to a file.

        Parameters
        ----------
        path: str
            Path to the file to save

        """
        # save some things as attrs, others as datasets
        with h5py.File(path, "w") as f:
            # save the input arguments
            for attr in self.get_saved_attrs():
                f.attrs[attr] = getattr(self, attr)

            # save the accumulated data
            for dataset in self.get_saved_datasets():
                f.create_dataset(dataset, data=getattr(self, dataset))

    @classmethod
    def load(cls, path, **additional_kwargs):
        """
        Alternative constructor to load an accumulator from a file.

        Parameters
        ----------
        path: str
            Path to the file to load

        Returns
        -------
        StatsAccumulator
            The loaded accumulator
        """
        with h5py.File(path, "r") as f:
            # attrs are all arguments to the constructor
            kwargs = {attr: f.attrs[attr] for attr in cls.get_saved_attrs()}
            kwargs.update(additional_kwargs)
            # create a new accumulator
            new_accumulator = cls(**kwargs)

            # load the accumulated data
            for dataset in new_accumulator.get_saved_datasets():
                setattr(new_accumulator, dataset, f[dataset][()])
            new_accumulator.accumulated_indices = (
                new_accumulator.accumulated_indices.tolist()
            )
        return new_accumulator

    def merge(self, other, seperate_runs=False):
        """
        Merge another accumulator into this one

        Parameters
        ----------
        other: StatsAccumulator or str
            The other accumulator to merge into this one
            ether as a StatsAccumulator object or a path to a file
        seperate_runs: bool, optional
            If the events come from seperate runs, then
            overlapping indices are expected and allowed.
            Default is False
        """
        if isinstance(other, str):
            other = self.load(other)

        for attr in self.get_saved_attrs():
            if np.any(getattr(self, attr) != getattr(other, attr)):
                raise ValueError("Accumulators have incompatible parameters")

        overlap = set(self.accumulated_indices) & set(other.accumulated_indices)
        if seperate_runs and overlap:
            my_max_idx = max(self.accumulated_indices)
            other.accumulated_indices = [
                i + my_max_idx + 1 for i in other.accumulated_indices
            ]
        elif overlap:
            raise ValueError(
                f"{len(overlap)} overlapping indices in these accumulators"
            )

        for dataset in self.get_saved_datasets():
            if dataset == "accumulated_indices":
                getattr(self, dataset).extend(getattr(other, dataset))
            else:
                setattr(self, dataset, getattr(self, dataset) + getattr(other, dataset))

    def _get_bin_centers(self):
        """
        Get the bin centers for the histograms
        """
        x_bin_centers = 0.5 * (
            self.lateral_x_bin_boundaries[:-1] + self.lateral_x_bin_boundaries[1:]
        )
        y_bin_centers = 0.5 * (
            self.lateral_y_bin_boundaries[:-1] + self.lateral_y_bin_boundaries[1:]
        )
        return x_bin_centers, y_bin_centers, self.layer_bottom

    def heatmap_normalised(self, incident_energy, choice, ax=None):
        """
        Plot the normalized histograms

        Parameters
        ----------
        incident_energy: float
            The incident energy to plot
        choice: str
            "counts", "energy", "evt_mean_E", "evt_mean_E_sq", "pnt_mean_E_sq",
            "evt_mean_counts" or "point_energy"

        """
        mask = self.counts_hist > 0
        if ax is None:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection="3d")
        if choice == "point_energy":
            hist = self.energy_hist[:]
            hist[mask] /= self.counts_hist[mask]
        else:
            hist = getattr(self, choice + "_hist")
        incident_energy_bin = np.digitize(incident_energy, self.incident_bin_boundaries)
        hist = hist[incident_energy_bin]
        hist /= np.mean(hist)
        # alpha = np.clip(0.1, 0.8, np.log(hist + 1))
        alpha = np.ones_like(hist)
        alpha[~mask[incident_energy_bin]] = 0.0
        heatmap_stack(
            hist,
            self.layer_bottom,
            self.Xmin,
            self.Xmax,
            self.Ymin,
            self.Ymax,
            ax=ax,
            alpha=alpha,
        )
        energy_min = self.incident_bin_boundaries[incident_energy_bin]
        energy_max = self.incident_bin_boundaries[incident_energy_bin + 1]
        ax.set_title(
            f"{choice.capitalize()} for incident energy {energy_min}-{energy_max} MeV"
        )
        ax.set_xlabel("x (shower coordinates)")
        ax.set_ylabel("y (shower coordinates)")

    def scatter_normalised(self, incident_energy, choice, ax=None):
        """
        Plot the normalized histograms

        Parameters
        ----------
        incident_energy: float
            The incident energy to plot
        choice: str
            "counts", "energy", "evt_mean_E", "evt_mean_E_sq", "pnt_mean_E_sq",
            "evt_mean_counts" or "point_energy"

        """
        if ax is None:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection="3d")
        if choice == "point_energy":
            hist = self.energy_hist / self.counts_hist
        else:
            hist = getattr(self, choice + "_hist")
        incident_energy_bin = np.digitize(incident_energy, self.incident_bin_boundaries)
        hist = hist[incident_energy_bin]
        n_events = self.total_events[incident_energy_bin]
        hist /= n_events
        values = hist.flatten().flatten()
        mask = values > n_events / 1000
        values = values[mask]
        xx, yy, zz = np.meshgrid(*self._get_bin_centers(), indexing="ij")
        xx = xx.flatten()[mask]
        yy = yy.flatten()[mask]
        zz = zz.flatten()[mask]

        log_values = np.log(values + 1)

        colours = plt.cm.viridis(log_values / max(log_values))
        # alpha = (values / max(values)) ** 3
        alpha = np.clip(0, 0.8, log_values)

        ax.scatter(xx, yy, zz, c=colours, alpha=alpha)


class AlignedStatsAccumulator(StatsAccumulator):
    def __init__(self, shift_type, *args, **kwargs):
        arg_order = [
            "Xmin",
            "Xmax",
            "Zmin",
            "Zmax",
            "layer_bottom",
            "cell_thickness",
            "incident_energy_bin_size",
            "lateral_bin_size",
        ]
        for name, arg in zip(arg_order, args):
            kwargs[name] = arg

        # layers will be treated as another continuous binning
        # we want to line up the peak/mean of the events in the layers
        # and so they will be shifted to align them
        if "layer_bottom" not in kwargs:
            n_layers = 50
            kwargs["layer_bottom"] = np.linspace(-2, 2, n_layers + 1)[:-1]
        else:
            n_layers = len(kwargs["layer_bottom"])
        super().__init__(*args, **kwargs)

        self.shift_type = shift_type
        self._set_shift_function(shift_type)

        # then a bit is done to keep track of the shifts
        n_incident_bins = len(self.incident_bin_boundaries) - 1
        self.layer_offset_bins = np.linspace(-1, 1, n_layers + 1)
        self.layer_offset_hist = np.zeros((n_incident_bins + 2, n_layers))
        self.layer_offset_sq_hist = np.zeros((n_incident_bins + 2, n_layers))
        # as each event will not contribute to all layers,
        # we need to keep track of the total number of events at each layer
        # in each incident energy bin
        self.total_events = np.zeros((n_incident_bins + 2, n_layers))

    def _set_shift_function(self, shift_type):
        if shift_type == "mean":

            def shift(events):
                mean_z = np.sum(events[..., 2] * events[..., 3], axis=-1) / np.sum(
                    events[..., 3], axis=-1
                )
                return mean_z[:, None]

        elif shift_type == "peak":
            # TODO wrong + no need for more layers
            def shift(events):
                raise RuntimeError("Not properly implemented")

        else:
            raise ValueError(f"Unknown shift type {shift_type}")
        self.get_shift = shift

    def add(self, idxs, energy, events):
        """
        Add events to the accumulator.

        Parameters
        ----------
        idxs: array-like (n_events,)
            Indices of the events, to check for duplicate accumulation
        energy: array-like (n_events,)
            The incident energy of the events
        events: array-like (n_events, max_n_points, 4)
            The events themselves, in xyze format

        """
        assert len(idxs) == len(energy) == len(events)
        if any(idx in self.accumulated_indices for idx in idxs):
            raise ValueError("Index already accumulated")
        self.accumulated_indices.extend(idxs)

        incident_energy_bins = np.digitize(energy, self.incident_bin_boundaries)
        event_counts = np.zeros_like(self.counts_hist[0])
        event_energies = np.zeros_like(self.energy_hist[0])
        event_energies_sq = np.zeros_like(self.energy_hist[0])

        # calculate and apply shifts
        shifts = self.get_shift(events)
        for inci_bin in set(incident_energy_bins):
            mask = incident_energy_bins == inci_bin
            self.layer_offset_hist[inci_bin] += np.histogram(
                shifts[mask, 0], bins=self.layer_offset_bins
            )[0]
            self.layer_offset_sq_hist[inci_bin] += np.histogram(
                shifts[mask, 0] ** 2, bins=self.layer_offset_bins
            )[0]
        events[..., 2] -= shifts

        for incident_bin, event in zip(incident_energy_bins, events):
            # remove any padding
            event = event[event[:, 3] > 0]
            event_counts[:] = 0
            event_energies[:] = 0
            event_energies_sq[:] = 0
            for layer_n, points_in_layer in enumerate(
                split_to_layers(event, self.layer_bottom, self.cell_thickness)
            ):
                n_pts_hist, energies_hist, energies_hist_sq = self.layer_hists(
                    points_in_layer
                )
                event_counts[layer_n] = n_pts_hist
                event_energies[layer_n] = energies_hist
                event_energies_sq[layer_n] = energies_hist_sq
                if np.sum(n_pts_hist) > 0:
                    self.total_events[incident_bin, layer_n] += 1
            self.add_event(
                incident_bin, event_counts, event_energies, event_energies_sq
            )

    @classmethod
    def load(cls, path, varient=""):
        """
        Alternative constructor to load an accumulator from a file.

        Parameters
        ----------
        path: str
            Path to the file to load

        Returns
        -------
        StatsAccumulator
            The loaded accumulator
        """
        if not varient:
            varient = path.split("/")[-1]
        shift_type = "mean" if "mean" in varient.lower() else "peak"
        return super().load(path, shift_type=shift_type)


def filter_factory(varient):
    """
    Create a filter for the events,
    which will remove some points by setting energy to 0

    Parameters
    ----------
    varient: str
        The varient of the filtering, should be odd or even

    Returns
    -------
    function
        The filter function, with the signature
        filter_events(events, layer_bottom, cell_thickness)
    """

    if varient == "odd":

        def filter_events(events, layer_bottom, cell_thickness):
            floors, ceilings = floors_ceilings(layer_bottom, cell_thickness, 0.5)
            for floor, ceiling in zip(floors[1::2], ceilings[1::2]):
                mask = (events[..., 2] >= floor) & (events[..., 2] < ceiling)
                events[mask] = 0

    elif varient == "even":

        def filter_events(events, layer_bottom, cell_thickness):
            floors, ceilings = floors_ceilings(layer_bottom, cell_thickness, 0.5)
            for floor, ceiling in zip(floors[::2], ceilings[::2]):
                mask = (events[..., 2] >= floor) & (events[..., 2] < ceiling)
                events[mask] = 0

    else:
        raise ValueError(f"Unknown varient {varient}, should be odd or even")
    return filter_events


def read_section_to(
    config, save_to, num_sections, section_number, varient="", batch_size=100
):
    """
    Read a section of the dataset and save the statistics to file.

    Parameters
    ----------
    config: config.Configs
        The configuration object
    save_to: str
        Path to save this section of the statistics
    num_sections: int
        Total number of sections to split the dataset into
    section_number: int
        Section number this call should read and save
    batch_size: int
        Number of events to read at once

    Returns
    -------
    StatsAccumulator
        The statistics for this section

    """
    assert section_number < num_sections
    print(f"Reading section {section_number} of {num_sections}")
    dataset_class = dataset_class_from_config(config)

    # the data will be rescaled to the unit cube, so no
    # need to pass Xmin etc.
    acc = None

    def noop(*args, **kwargs):
        pass

    filter_func = noop

    if not varient:
        acc = StatsAccumulator()
    elif "align" in varient.lower():
        if "mean" in varient.lower():
            acc = AlignedStatsAccumulator("mean")
        elif "peak" in varient.lower():
            acc = AlignedStatsAccumulator("peak")
        if "odd" in varient.lower():
            filter_func = filter_factory("odd")
        elif "even" in varient.lower():
            filter_func = filter_factory("even")

    if acc is None:
        raise ValueError(f"Unknown varient {varient}")

    n_events = np.sum(get_n_events(config.dataset_path, config.n_dataset_files))
    n_events_per_section = n_events // num_sections
    print(f"Will read {n_events_per_section} events of {n_events} in total")
    start = n_events_per_section * section_number
    section_end = n_events_per_section * (section_number + 1)
    for batch_start in tqdm(np.arange(start, section_end, batch_size)):
        pick_events = list(
            range(batch_start, min(batch_start + batch_size, section_end))
        )
        energy, events = read_raw_regaxes(config, pick_events)
        filter_func(events, acc.layer_bottom, acc.cell_thickness)
        # rescaling data into unit cube
        dataset_class.normalize_xyze(events)
        acc.add(pick_events, energy, events)

    if save_to:
        print(f"Saving to {save_to}")
        acc.save(save_to)
    else:
        print("Not saving the results")

    return acc


def save_location(config, num_sections, section_number, varient=""):
    """
    Identify the default path for saving the statistics for a section.
    Create the directory if it does not exist.

    Parameters
    ----------
    config: config.Configs
        The configuration object
    num_sections: int
        Total number of sections to split the dataset into
    section_number: int
        Section number this call should read and save
    """
    path_segments = [config.logdir, "dataset_accumulators"]
    dataset_basename = config.dataset_path.split("/")[-1].split(".")[0]
    if "{" in dataset_basename:
        dataset_basename = dataset_basename.format("All")
    dataset_basename = dataset_basename.replace("_all_steps", "")
    if varient:
        dataset_basename += f"_{varient}"
    if num_sections > 1:
        path_segments.append(dataset_basename)
        path_segments.append(f"{num_sections}_sections")
        path_segments.append(f"section_{section_number}.h5")
    else:
        path_segments.append(dataset_basename[:20])
        path_segments.append(f"{dataset_basename}.h5")
    dir_path = os.path.join(*path_segments[:-1])
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return os.path.join(*path_segments)


def read_section(num_sections, section_number, config=Configs(), varient=""):
    """
    Read a section of the dataset and save the statistics to file.

    Parameters
    ----------
    num_sections: int
        Total number of sections to split the dataset into
    section_number: int
        Section number this call should read and save
    config: config.Configs
        The configuration object
        Optional, the default is the default configuration
    """
    save_to = save_location(config, num_sections, section_number, varient)
    read_section_to(config, save_to, num_sections, section_number, varient)


class HighLevelStats:
    def __init__(self, accumulator, poly_degree):
        self.poly_degree = poly_degree
        self.accumulator = accumulator
        self.incident_energy_bin_centers = 0.5 * (
            accumulator.incident_bin_boundaries[:-1]
            + accumulator.incident_bin_boundaries[1:]
        )
        # incident energies that have at least one recorded event
        self.event_mask = self.accumulator.total_events[1:-1] > 0
        self.incident_energy_bin_centers = self.incident_energy_bin_centers[
            self.event_mask
        ]
        self.total_events_per_incedent_energy = self.accumulator.total_events[1:-1][
            self.event_mask
        ]

    def get_n_pts_vs_incident_energy(self, layer):
        """
        Get the mean number of points per event in a layer
        as a function of the incident energy

        Parameters
        ----------
        layer: int
            The layer to get the number of points for

        Returns
        -------
        n_pts: array
            The number of points in the layer as a function of the incident energy
        """
        n_pts = np.sum(
            self.accumulator.counts_hist[1:-1][self.event_mask, layer], axis=(1, 2)
        )
        return n_pts / self.total_events_per_incedent_energy

    def n_pts(self, layer, no_fit=False):
        """
        Calculate the coefficients for the number of points in each layer with
        the incident energy.
        To be parametrized in the backbone.

        Parameters
        ----------
        layer: int
            The layer to calculate the gradient for

        Returns
        -------
        n_pts_coeffs: array of floats (poly_degree + 1,)
            The coefficients of the polynomial fit to the number of points
            in the layer with the incident energy
        """
        n_pts = self.get_n_pts_vs_incident_energy(layer)
        if no_fit:
            return self.incident_energy_bin_centers, n_pts
        # use polyfit to get a linear fit
        n_pts_coeffs = np.polyfit(
            self.incident_energy_bin_centers, n_pts, self.poly_degree
        )
        return n_pts_coeffs

    def stddev_n_pts(self, layer, no_fit=False):
        """
        Calculate the coefficients for the fit of the standard devation
        of number of points per event in each layer with the incident energy.
        To be parametrized in the backbone.

        Parameters
        ----------
        layer: int
            The layer to calculate the gradient for

        Returns
        -------
        stddev_n_pts_coeffs: array of floats (poly_degree + 1,)
            The coefficients of the standard devation of mean number of points
            in the layer with the incident energy
        """
        mean_n_pts = self.get_n_pts_vs_incident_energy(layer)
        mean_n_pts_sq = (
            self.accumulator.evt_counts_sq_hist[1:-1][self.event_mask, layer]
            / self.total_events_per_incedent_energy
        )
        stddev_n_pts = np.sqrt(mean_n_pts_sq - mean_n_pts**2)
        if no_fit:
            return self.incident_energy_bin_centers, stddev_n_pts
        # use polyfit to get a linear fit
        stddev_n_pts_coeffs = np.polyfit(
            self.incident_energy_bin_centers, stddev_n_pts, self.poly_degree
        )
        return stddev_n_pts_coeffs

    def event_mean_point_energy(self, layer, no_fit=False):
        """
        Calculate the coefficients for th fit of the mean point energy
        per event in each layer with the incident energy.
        To be parametrized in the backbone.

        Parameters
        ----------
        layer: int
            The layer to calculate the gradient for

        Returns
        -------
        energy_mean_coeffs: array of floats (poly_degree + 1,)
            The coeffients of the mean energy per point in the layer
            with the incident energy

        """
        mean_point_E = (
            self.accumulator.evt_mean_E_hist[1:-1][self.event_mask, layer]
            / self.total_events_per_incedent_energy
        )
        if no_fit:
            return self.incident_energy_bin_centers, mean_point_E
        # use polyfit to get a linear fit
        energy_mean_coeffs = np.polyfit(
            self.incident_energy_bin_centers, mean_point_E, self.poly_degree
        )
        return energy_mean_coeffs

    def stddev_event_mean_point_energy(self, layer, no_fit=False):
        """
        Calculate the coeffients for the fit of the standard devation
        of mean point energy per event in each layer with the incident energy.
        To be parametrized in the backbone.

        Parameters
        ----------
        layer: int
            The layer to calculate the gradient for

        Returns
        -------
        stddev_energy_mean_coeffs: array of floats (poly_degree + 1,)
            The coefficient for the fit of the standard devation of mean
            energy per point in the layer with the incident energy
        """
        mean_point_E = (
            self.accumulator.evt_mean_E_hist[1:-1][self.event_mask, layer]
            / self.total_events_per_incedent_energy
        )
        mean_point_E_sq = (
            self.accumulator.evt_mean_E_sq_hist[1:-1][self.event_mask, layer]
            / self.total_events_per_incedent_energy
        )
        stddev_point_E = np.sqrt(mean_point_E_sq - mean_point_E**2)
        if no_fit:
            return self.incident_energy_bin_centers, stddev_point_E
        # use polyfit to get a linear fit
        stddev_energy_mean_coeffs = np.polyfit(
            self.incident_energy_bin_centers, stddev_point_E, self.poly_degree
        )
        return stddev_energy_mean_coeffs

    def stddev_point_energy_in_evt(self, layer, no_fit=False):
        """
        Calculate the mean standard devation of the energy point points
        in each event in each layer with the incident energy.
        To be parametrized by each layer.

        Parameters
        ----------
        layer: int
            The layer to calculate the gradient for

        Returns
        -------
        stddev_energy_coeffs: array of floats (poly_degree + 1,)
            The coeffients for the fit of the standard devation of the energy per point
            in the layer with the incident energy
        """
        pnt_mean_E_sq = (
            self.accumulator.pnt_mean_E_sq_hist[1:-1][self.event_mask, layer]
            / self.total_events_per_incedent_energy
        )
        mean_point_E_sq = (
            self.accumulator.evt_mean_E_sq_hist[1:-1][self.event_mask, layer]
            / self.total_events_per_incedent_energy
        )
        stddev_point_E = np.sqrt(pnt_mean_E_sq - mean_point_E_sq)
        if no_fit:
            return self.incident_energy_bin_centers, stddev_point_E
        # use polyfit to get a linear fit
        stddev_energy_coeffs = np.polyfit(
            self.incident_energy_bin_centers, stddev_point_E, self.poly_degree
        )
        return stddev_energy_coeffs

    def points_covarience_matrix(self, layer):
        """
        Calculate the covarience matrix that defined the bivariet normal distribution
        of the density of points in each layer with the incident energy.
        The matrix is exspressed as gradients and intercepts of the eigenvalues
        and eigenvector angle.

        Parameters
        ----------
        layer: int
            The layer to calculate the gradient for

        Returns
        -------
        eigenvalue_grads: array of floats (2,)
            The gradients of the eigenvalues of the covarience matrix
            with the incident energy
        eigenvalue_intercepts: array of floats (2,)
            The intercepts of the eigenvalues of the covarience matrix
            with the incident energy
        eigenvector_angle_grad: float
            The gradient of the angle of the eigenvector of the covarience matrix
            with the incident energy
        eigenvector_angle_intercept: float
            The intercept of the angle of the eigenvector of the covarience matrix
            with the incident energy
        """
        evt_mean_counts_all_i = (
            self.accumulator.evt_mean_counts_hist[1:-1][self.event_mask, layer].T
            / self.total_events_per_incedent_energy
        ).T
        self.eigenvalue_values = np.zeros((len(self.incident_energy_bin_centers), 2))
        self.eigenvector_angles = np.zeros(len(self.incident_energy_bin_centers))
        # need the location of each bin in x and z
        x_bin_centers, y_bin_centers, _ = self.accumulator._get_bin_centers()
        # then make a meshgrid of the locations
        x_mesh, y_mesh = np.meshgrid(x_bin_centers, y_bin_centers)
        # and zip it together so that when the bin heights are
        # flattened they can be treated as weights
        xy_mesh = np.vstack((x_mesh.flatten(), y_mesh.flatten()))
        for i, evt_mean_counts in enumerate(evt_mean_counts_all_i):
            cov_matrix = np.cov(xy_mesh, aweights=evt_mean_counts.flatten())
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            self.eigenvalue_values[i] = eigenvalues
            self.eigenvector_angles[i] = np.arctan2(*eigenvectors[0])
        # use polyfit to get a linear fit
        eigenvalue_grads = np.zeros((2,))
        eigenvalue_intercepts = np.zeros((2,))
        for i in range(2):
            eigenvalue_grads[i], eigenvalue_intercepts[i] = np.polyfit(
                self.incident_energy_bin_centers, self.eigenvalue_values[:, i], 1
            )
        eigenvector_angle_grad, eigenvector_angle_intercept = np.polyfit(
            self.incident_energy_bin_centers, self.eigenvector_angles, 1
        )
        return (
            eigenvalue_grads,
            eigenvalue_intercepts,
            eigenvector_angle_grad,
            eigenvector_angle_intercept,
        )


class RadialView:
    """
    Create a radial view of a StatsAccumulator
    for fitting the radial profile of the shower.
    """

    def __init__(self, accumulator, x_offset=None, y_offset=None):
        """
        Constructor, calculates the transform imeadiately.

        Parameters
        ----------
        accumulator: StatsAccumulator
            The accumulator with completed detector statistics

        """
        self.accumulator = accumulator
        self.incident_energy_bin_centers = 0.5 * (
            self.accumulator.incident_bin_boundaries[:-1]
            + self.accumulator.incident_bin_boundaries[1:]
        )
        self.raw_energy = self.accumulator.energy_hist[1:-1]
        x_bin_centers, y_bin_centers, _ = self.accumulator._get_bin_centers()
        # the data may not be centered,
        # but the distribution can only fit centered data
        # so calculate the offset, such that the mean value is (0, 0)
        # sum over energies and layers
        all_energy = np.sum(self.raw_energy, axis=(0, 1))
        x_marginal = np.sum(all_energy, axis=1)
        y_marginal = np.sum(all_energy, axis=0)
        if x_offset is None:
            self.x_offset = np.sum(x_marginal * x_bin_centers) / np.sum(x_marginal)
        else:
            self.x_offset = x_offset
        if y_offset is None:
            self.y_offset = np.sum(y_marginal * y_bin_centers) / np.sum(y_marginal)
        else:
            self.y_offset = y_offset
        # now move the bin centers so that the mean value is (0, 0)
        self.x_bin_centers = x_bin_centers - self.x_offset
        self.y_bin_centers = y_bin_centers - self.y_offset
        # calculate the radial positions of the bins from this center
        r_bin_centers = np.sqrt(
            self.x_bin_centers[:, np.newaxis] ** 2
            + self.y_bin_centers[np.newaxis, :] ** 2
        )
        self.r_bin_centers = r_bin_centers.flatten()
        radial_order = np.argsort(self.r_bin_centers)
        self.r_bin_centers = self.r_bin_centers[radial_order]

        n_energies, n_layers, _, _ = self.raw_energy.shape
        # now get the mean energy in each bin
        projected_total_events = self.accumulator.total_events[
            1:-1, np.newaxis, np.newaxis, np.newaxis
        ]
        # bins without any events are meaningless
        projected_total_events[projected_total_events == 0] = 1
        mean_energy = self.raw_energy / projected_total_events
        self.mean_energy = mean_energy.reshape(n_energies, n_layers, -1)[
            :, :, radial_order
        ]
        # and mean squared energy
        mean_sq_energy = self.accumulator.energy_sq_hist[1:-1] / projected_total_events
        mean_sq_energy = mean_sq_energy.reshape(n_energies, n_layers, -1)[
            :, :, radial_order
        ]
        # and the varience of the sample
        var_sample_energy = mean_sq_energy - self.mean_energy**2
        # so the varience of the population
        p_total_events = self.accumulator.total_events[1:-1, np.newaxis, np.newaxis]
        # again, bins without any events are meaningless, but also just one event will
        # cause a nan in the varience, so we need to avoid that
        p_total_events[p_total_events < 2] = 2
        var_pop_energy = var_sample_energy * p_total_events / (p_total_events - 1)
        # negative or zero varience indicates a lack of data
        var_pop_energy[var_pop_energy <= 0] = np.max(var_pop_energy)
        self.std_energy = np.nan_to_num(np.sqrt(var_pop_energy))
        # self.std_energy = std_energy.reshape(n_energies, n_layers, -1)[
        #    :, :, radial_order
        # ]
        # same for the hits
        mean_hits = self.accumulator.counts_hist[1:-1] / projected_total_events
        self.mean_hits = mean_hits.reshape(n_energies, n_layers, -1)[:, :, radial_order]
        mean_sq_hits = self.accumulator.counts_sq_hist[1:-1] / projected_total_events
        mean_sq_hits = mean_sq_hits.reshape(n_energies, n_layers, -1)[
            :, :, radial_order
        ]
        var_sample_hits = mean_sq_hits - self.mean_hits**2
        var_pop_hits = var_sample_hits * p_total_events / (p_total_events - 1)
        # negative or zero varience indicates a lack of data
        var_pop_hits[var_pop_hits <= 0] = np.max(var_pop_hits)
        self.std_hits = np.nan_to_num(np.sqrt(var_pop_hits))
        # self.std_hits = std_hits.reshape(n_energies, n_layers, -1)[:, :, radial_order]

    def fit_to_energy(
        self,
        incedent_energy,
        layer_n,
        radial_function,
        p0=None,
        bounds=None,
        ignore_norm=True,
        **kwargs,
    ):
        """
        Use a customised curve_fit to fit the radial function to the radial profile,
        of the mean energy.
        Optionally, ignore the normamalisation of the function.

        Parameters
        ----------
        incedent_energy: float or int
            The incident energy to fit the radial profile for
            If it's an int, it refers to the bin number in in
            incedent_energy_bin_centers.
        layer_n: int
            The layer to fit the radial profile for
        radial_function: function
            The function to fit to the radial profile
        p0: array-like (n_params,) (optional)
            Initial guess for the parameters
        bounds: array-like (2, n_params) (optional)
            Bounds for the parameters
        ignore_norm: bool (optional)
            Ignore the normalisation of the function
            Effectively adds a normalisation parameter to the fit,
            that will be abandoned.
            Default is True
        **kwargs
            Passed straight to pointcloud.utils.optimise.curve_fit

        Returns
        -------
        popt: array (n_params,)
            The optimal parameters for the fit
        pcov: array (n_params, n_params)
            The covariance matrix of the fit
        """
        r_bins, means, errors = self._select_data(
            incedent_energy, layer_n, self.mean_energy, self.std_energy
        )
        return self._fit(
            r_bins,
            means,
            errors,
            radial_function,
            p0=p0,
            bounds=bounds,
            ignore_norm=ignore_norm,
            **kwargs,
        )

    def fit_to_hits(
        self,
        incedent_energy,
        layer_n,
        radial_function,
        p0=None,
        bounds=None,
        ignore_norm=True,
        **kwargs,
    ):
        """
        Use scipy's curve_fit to fit the radial function to the radial profile,
        of the mean energy.
        Optionally, ignore the normamalisation of the function.

        Parameters
        ----------
        incedent_energy: float or int
            The incident energy to fit the radial profile for
            If it's an int, it refers to the bin number in in
            incedent_energy_bin_centers.
        layer_n: int
            The layer to fit the radial profile for
        radial_function: function
            The function to fit to the radial profile
        p0: array-like (n_params,) (optional)
            Initial guess for the parameters
        bounds: array-like (2, n_params) (optional)
            Bounds for the parameters
        ignore_norm: bool (optional)
            Ignore the normalisation of the function
            Effectively adds a normalisation parameter to the fit,
            that will be abandoned.
            Default is True
        **kwargs
            Passed straight to pointcloud.utils.optimise.curve_fit

        Returns
        -------
        popt: array (n_params,)
            The optimal parameters for the fit
        pcov: array (n_params, n_params)
            The covariance matrix of the fit
        """
        r_bins, means, errors = self._select_data(
            incedent_energy, layer_n, self.mean_hits, self.std_hits
        )

        return self._fit(
            r_bins,
            means,
            errors,
            radial_function,
            p0=p0,
            bounds=bounds,
            ignore_norm=ignore_norm,
            **kwargs,
        )

    def _select_data(
        self,
        incedent_energy,
        layer_n,
        all_means,
        all_stds,
    ):
        """
        Find the r values, mean and errors for a fit, starting from
        a the full set of data.
        Choses bins with corrisponding incident energy and layer,
        and filters out bins with no data.

        Parameters
        ----------
        incedent_energy: float or int
            The incident energy to fit the radial profile for
            If it's an int, it refers to the bin number in in
            incedent_energy_bin_centers.
        layer_n: int
            The layer to fit the radial profile for
        all_means: array (n_energies, n_layers, n_bins)
            The mean values of the radial profile
        all_stds: array (n_energies, n_layers, n_bins)
            The standard deviations of the radial profile

        Returns
        -------
        popt: array (n_params,)
            The optimal parameters for the fit
        pcov: array (n_params, n_params)
            The covariance matrix of the fit
        """
        # get the bin for the incident energy
        if isinstance(incedent_energy, int):
            incident_energy_bin = incedent_energy
        else:
            incident_energy_bin = np.digitize(
                incedent_energy, self.accumulator.incident_bin_boundaries[1:-1]
            )

        # do the fit with the errors
        errors = all_stds[incident_energy_bin, layer_n]
        # no matter what we fit, we only want to fit bins with at least one count
        mean_hits = self.mean_hits[incident_energy_bin, layer_n]
        mask = mean_hits > 0
        if not np.any(mask):
            raise ValueError(
                f"No data for incident energy {incedent_energy} on layer {layer_n}"
            )
        r_bins = self.r_bin_centers[mask]
        errors = errors[mask]
        means = all_means[incident_energy_bin, layer_n][mask]
        return r_bins, means, errors

    def _fit(
        self,
        r_bins,
        means,
        errors,
        radial_function,
        p0=None,
        bounds=None,
        ignore_norm=True,
        **kwargs,
    ):
        """
        Use scipy's curve_fit to fit the radial function to the radial profile.
        Optionally, ignore the normamalisation of the function.

        Parameters
        ----------
        r_bins: array (n_bins,)
            The radial positions of the bins that are used for the fit
        means: array (n_bins,)
            The mean values of the radial profile
        errors: array (n_bins,)
            The errors of the mean values of the radial profile
        radial_function: function
            The function to fit to the radial profile
        p0: array-like (n_params,) (optional)
            Initial guess for the parameters
        bounds: array-like (2, n_params) (optional)
            Bounds for the parameters
        ignore_norm: bool (optional)
            Ignore the normalisation of the function
            Effectively adds a normalisation parameter to the fit,
            that will be abandoned.
            Default is True
        **kwargs
            Passed straight to pointcloud.utils.optimise.curve_fit

        Returns
        -------
        popt: array (n_params,)
            The optimal parameters for the fit
        pcov: array (n_params, n_params)
            The covariance matrix of the fit
        """
        np.random.seed(42)
        if ignore_norm:
            p0 = [1.0] + list(p0)
            bounds = [[0.0] + list(bounds[0]), [np.inf] + list(bounds[1])]

            def to_fit(r, norm, *params):
                found = norm * radial_function(r, *params)
                return found

        else:
            to_fit = radial_function

        if np.any(errors == 0):
            message = (
                "stats_accumulator.RadialView; Some errors are zero, "
                "adding a small value to avoid division by zero"
            )
            warnings.warn(message)
            errors[errors == 0] = 1e-10

        popt, pcov = curve_fit(
            to_fit,
            r_bins,
            means,
            p0=p0,
            bounds=bounds,
            sigma=errors,
            **kwargs,
        )
        if ignore_norm:
            return popt[1:], pcov[1:, 1:]
        return popt, pcov
