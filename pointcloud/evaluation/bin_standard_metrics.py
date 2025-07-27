import numpy as np
import torch
import os
import warnings

from pointcloud.config_varients.wish import Configs
from pointcloud.config_varients.wish_maxwell import Configs as MaxwellConfigs

from pointcloud.utils.metadata import Metadata
from pointcloud.utils import detector_map
from pointcloud.data.conditioning import read_raw_regaxes_withcond, get_cond_dim
from pointcloud.utils import showerflow_utils
from pointcloud.models.wish import Wish
from pointcloud.models.fish import Fish
from pointcloud.models.shower_flow import versions_dict
from pointcloud.models.load import get_model_class
from pointcloud.utils.gen_utils import gen_cond_showers_batch


from pointcloud.config_varients.wish import Configs as WishConfigs
from pointcloud.configs import Configs


def try_mkdir(dir_name):
    """
    Try to make a directory, but don't crash if it already exists.

    Parameters
    ----------
    dir_name : str
        The name of the directory to create.
    """
    try:
        os.mkdir(dir_name)
    except FileExistsError:
        pass


config = Configs()
meta = Metadata(config)
floors, ceilings = detector_map.floors_ceilings(
    meta.layer_bottom_pos_hdf5, meta.cell_thickness_hdf5, 0
)


class BinnedData:
    """
    Agrigate key metrics for calorimeter models
    into bins for easy plotting and comparison.
    """

    true_xyz_limits = [
        [meta.Xmin_global, meta.Xmax_global],
        [meta.Zmin_global, meta.Zmax_global],
        [floors[0], ceilings[-1]],
    ]
    event_center = np.array([-80, -50, 0, 0])
    arg_names = [
        "name",
        "xyz_limits",
        "energy_scale",
        "layer_bottom_pos",
        "cell_thickness",
        "gun_xyz_pos",
    ]

    def __init__(
        self,
        name,
        xyz_limits,
        energy_scale,
        layer_bottom_pos,
        cell_thickness,
        gun_xyz_pos,
        hard_check=True,
        no_box_cut=False,
    ):
        """Construct a BinnedData object.

        Parameters
        ----------
        name : str
            The name of the model, no mechanical significance.
        xyz_limits : list of [float, float]
            The limits of the model in x, y, z.
            Formatted as [[xmin, xmax], [ymin, ymax], [zmin, zmax]].
        energy_scale : float
            How much to divide the observed energy by before binning.
        layer_bottom_pos : np.array
            The y positions of the bottom of each layer, in the coordinates
            the data is given in.
        cell_thickness: float
            The thickness in the y direction of each cell, in the coordinates
            the data is given in.
        gun_xyz_pos : np.array
            The x, y and z positions of the gun, in the coordinates the data is
            given in. The data is shifted so that the gun is at the origin.
        """
        self.hard_check = hard_check
        try:
            self.sanity_layer_box(layer_bottom_pos, xyz_limits)
        except AssertionError as e:
            if hard_check:
                raise
            warnings.warn(f"Layer box sanity check failed: {e}")
        self.name = name
        self.xyz_limits = xyz_limits
        self.energy_scale = energy_scale
        self.layer_bottom_pos = layer_bottom_pos
        self.cell_thickness = cell_thickness
        self.gun_xyz_pos = gun_xyz_pos
        buffed_floors, buffed_ceilings = detector_map.floors_ceilings(
            layer_bottom_pos, cell_thickness, 1.0
        )
        self.layer_bins = np.concatenate([buffed_floors, buffed_ceilings[[-1]]])
        self.gun_shift = self.get_gunshift()

        self.y_labels = []
        self.x_labels = []
        self.bins = []
        self.counts = []

        self.add_hist("sum clusters", "radius [mm]", 0, 500, 32)
        self.add_hist("sum energy [MeV]", "radius [mm]", 0, 500, 32)
        self.add_hist("sum clusters", "layers", 1, 30, 30)
        self.add_hist("sum energy [MeV]", "layers", 1, 30, 30)
        self.add_hist(
            "number of clusters", "cluster energy [MeV]", 0.001, 100, 100, True
        )
        self.add_hist("number of showers", "number of clusters", 0, 10000, 50)
        self.add_hist("number of showers", "energy sum [MeV]", 0, 10000, 50)
        self.add_hist(
            "number of showers",
            "center of gravity X [mm]",
            *self.true_xyz_limits[0],
            2000,
        )
        self.add_hist(
            "number of showers",
            "center of gravity Y [mm]",
            *self.true_xyz_limits[1],
            2000,
        )
        self.add_hist(
            "number of showers",
            "center of gravity Z [layer]",
            *self.true_xyz_limits[2],
            60,
        )
        self.add_hist("mean energy [MeV]", "radius [mm]", 0, 500, 32)
        self.add_hist("mean energy [MeV]", "layers", 1, 30, 30)

        self.max_sample_events = 10
        self.sample_events = []

        if no_box_cut:
            self.box_cut = lambda events: None

    @staticmethod
    def sanity_layer_box(layer_bottom_pos, xyz_limits):
        z_range = xyz_limits[2][1] - xyz_limits[2][0]
        assert (
            abs(layer_bottom_pos[0] - xyz_limits[2][0]) < z_range / 100
        ), f"{layer_bottom_pos[0]=} not close to {xyz_limits[2][0]=}"
        assert (
            abs(layer_bottom_pos[-1] - xyz_limits[2][1]) < z_range / 10
        ), f"{layer_bottom_pos[-1]=} not close to {xyz_limits[2][1]=}"

    def get_gunshift(self):
        """
        Create a fake event that has shape (1, 1, 4), with one point at the
        gun position.  Used to shift the data so that the gun is at the origin,
        becuase it can be projected to the same axis format as events in
        numpy.

        Returns
        -------
        np.array
            The fake event with the gun at the origin.
        """
        fake_events = np.zeros((1, 1, 4))
        fake_events[0, 0, 0] = self.gun_xyz_pos[0]
        fake_events[0, 0, 1] = self.gun_xyz_pos[1]
        fake_events[:, :, 3] = 0  # no energy
        return fake_events

    def recompute_mean_energies(self):
        """
        Using the summed energy and clusters, recompute the mean energy histograms.
        Can be safely called multiple times.
        """
        mask = self.counts[0] > 0
        mean_in_radius = self.counts[1][mask] / self.counts[0][mask]
        hist_idx = self.hist_idx("mean energy [MeV]", "radius [mm]")
        self.counts[hist_idx][mask] = mean_in_radius
        self.counts[hist_idx][~mask] = 0

        mask = self.counts[2] > 0
        mean_in_layers = self.counts[3][mask] / self.counts[2][mask]
        hist_idx = self.hist_idx("mean energy [MeV]", "layers")
        self.counts[hist_idx][mask] = mean_in_layers
        self.counts[hist_idx][~mask] = 0

    def hist_idx(self, y_label, x_label):
        """
        Given an y label (the weight of the bins) and
        an x label (the bin boundrey quantity),
        return the index of the histogram in the binned data.

        Parameters
        ----------
        y_label : str
            The weight of the bins, y axis on the histogram.
        x_label : str
            The bin boundrey quantity, x axis on the histogram.

        Returns
        -------
        int or None
            The index of the histogram in the binned data, or None if it doesn't exist.

        """
        for i, (y_lab, x_lab) in enumerate(zip(self.y_labels, self.x_labels)):
            if x_lab == x_label and y_lab == y_label:
                return i
        return None

    def __len__(self):
        return len(self.y_labels)

    def total_showers(self):
        """
        Number of showers that have been added to the binned data.

        Returns
        -------
        int
            Total showers in the dataset.
        """
        hist_idx = self.hist_idx("number of showers", "number of clusters")
        return np.sum(self.counts[hist_idx])

    def dummy_xs(self, idx):
        """
        Center of each bin in the x axis of the histogram,
        mostly used for plotting.

        Parameters
        ----------
        idx : int
            The index of the histogram in the binned data.

        Returns
        -------
        dummy : np.array
            The center value of each bin in the x axis of the histogram
        """
        bins = self.bins[idx]
        dummy = 0.5 * (bins[1:] + bins[:-1])
        return dummy

    def add_hist(self, y_label, x_label, x_min, x_max, n_bins, logspace=False):
        """
        Add a new histogram to the list of histograms we are collecting data for.
        Mostly called interally, but could in theory be used to collect more data.
        Does not change the behaviour of add_events, so any externally added histograms
        would need to fill themselves.

        Parameters
        ----------
        y_label : str
            The weight of the bins, y axis on the histogram.
        x_label : str
            The bin boundrey quantity, x axis on the histogram.
        x_min : float
            The minimum value of the x axis.
        x_max : float
            The maximum value of the x axis.
        n_bins : int
            The number of bins to use on the x axis.
        logspace : bool, optional
            If True the x axis will be in log space, by default False.

        Raises
        ------
        ValueError
            If a histogram with the same x and y labels already exists

        """
        hist_idx = self.hist_idx(y_label, x_label)
        if hist_idx is not None:
            raise ValueError(f"Already have a hist labeled {x_label} v.s. {y_label}")
        self.y_labels.append(y_label)
        self.x_labels.append(x_label)
        if logspace:
            assert x_min > 0
            assert x_max > 0
            bins = np.logspace(np.log10(x_min), np.log10(x_max), n_bins + 1)
        elif isinstance(x_min, int) and isinstance(x_max, int):
            interval = (x_max - x_min) / n_bins
            bins = np.arange(x_min, x_max + interval, interval)
        else:
            bins = np.linspace(x_min, x_max, n_bins + 1)
        if np.any(np.isnan(bins)):
            raise ValueError(
                f"Problem in x_min {x_min}, x_max {x_max}, n_bins {n_bins}, logspace {logspace}"
            )
        self.bins.append(bins)
        self.counts.append(np.zeros(n_bins))

    def box_cut(self, raw_events):
        """
        Given some events, as produced by the model (not rescaled),
        zero any points outside the xyz limits of the model.
        Acts in place.

        Parameters
        ----------
        raw_events : np.array (n_showers, n_points, 4)
            The events to cut. Will not be modified.
        """
        in_x_range = np.logical_and(
            raw_events[:, :, 0] >= self.xyz_limits[0][0],
            raw_events[:, :, 0] <= self.xyz_limits[0][1],
        )
        in_y_range = np.logical_and(
            raw_events[:, :, 1] >= self.xyz_limits[1][0],
            raw_events[:, :, 1] <= self.xyz_limits[1][1],
        )
        in_z_range = np.logical_and(
            raw_events[:, :, 2] >= self.xyz_limits[2][0],
            raw_events[:, :, 2] <= self.xyz_limits[2][1],
        )
        in_range = np.logical_and(in_x_range, np.logical_and(in_y_range, in_z_range))
        raw_events[~in_range] = 0

    def rescaled_events(self, events):
        """
        Apply the transformations to the event coordinates specified when
        the binned data was created.

        Parameters
        ----------
        events : np.array (n_showers, n_points, 4)
            The events to rescale. Will not be modified.

        Returns
        -------
        rescaled : np.array (n_showers, n_points, 4)
            The rescaled events.
        """
        rescaled = np.copy(events)
        rescaled[:, :, 3] /= self.energy_scale
        if False:  # TODO: fix
            for dim in range(3):
                # for dim in [0, 2]:  # skip the y rescale
                raw = events[:, :, dim]
                unit = (raw - self.xyz_limits[dim][0]) / (
                    self.xyz_limits[dim][1] - self.xyz_limits[dim][0]
                )
                # assert np.all(unit <= 1.), \
                # f"Dim {dim}, limits {self.xyz_limits[dim]}, "\
                # f"Raw min {np.min(raw)}, max {np.max(raw)}, unit {unit}"
                # Models can generate data outside of the range
                rescaled[:, :, dim] = (
                    unit * (self.true_xyz_limits[dim][1] - self.true_xyz_limits[dim][0])
                    + self.true_xyz_limits[dim][0]
                )
        return rescaled

    def add_events(self, events):
        """
        Add events to the histograms that were created in the constructor.

        Parameters
        ----------
        events : np.array (n_showers, n_points, 4)
            The events to add to the histograms.
        """
        if len(self.sample_events) < self.max_sample_events:
            from_batch = list(
                events[: (self.max_sample_events - len(self.sample_events))]
            )
            self.sample_events.extend(from_batch)

        self.box_cut(events)
        mask = events[:, :, 3] > 0
        # sanity check 1
        if len(events) > 4 and not np.any(mask):
            error_message = (
                f"Box cut removed all points from {len(events)} showers, \n"
                f" - box cut had {self.xyz_limits=}"
            )
            if self.hard_check:
                raise ValueError(error_message)
            warnings.warn(error_message)

        raw_zs = events[:, :, 2][mask]

        rescaled = self.rescaled_events(events)

        energy = rescaled[:, :, 3][mask]
        gun_shifted = rescaled - self.gun_shift
        centered = gun_shifted - self.event_center
        radius = np.sqrt(centered[:, :, 0] ** 2 + centered[:, :, 1] ** 2)[mask]

        self.counts[0] += np.histogram(radius, self.bins[0])[0]
        self.counts[1] += np.histogram(radius, self.bins[1], weights=energy)[0]
        self.counts[2] += np.histogram(raw_zs, self.layer_bins)[0]
        self.counts[3] += np.histogram(raw_zs, self.layer_bins, weights=energy)[0]
        self.counts[4] += np.histogram(energy, self.bins[4])[0]

        hits_per_shower = np.sum(mask, axis=1)
        self.counts[5] += np.histogram(hits_per_shower, self.bins[5])[0]

        energy_per_shower = np.sum(rescaled[:, :, 3] * mask, axis=1)
        self.counts[6] += np.histogram(energy_per_shower, self.bins[6])[0]

        with np.errstate(divide="ignore", invalid="ignore"):
            for i in range(3):
                cog = (
                    np.sum((gun_shifted[:, :, i] * gun_shifted[:, :, 3]), axis=1)
                    / energy_per_shower
                )
                self.counts[7 + i] += np.histogram(cog, self.bins[7 + i])[0]
        if len(events) < 4:
            return  # no point checking with too few events.

        # sanity check 2
        percent_populated = np.fromiter(
            (np.sum([b > 0 for b in c]) / len(c) for c in self.counts[:10]), float
        )
        if np.sum(percent_populated > 0.01) > 5:
            return
        error_message = "Some parameters don't seem to fit in hitogram range;\n"
        for i, percent in enumerate(percent_populated):
            if percent > 0.5:
                continue
            error_message += f"{self.y_labels[i]} v.s. {self.x_labels[i]}: {percent:.0%} percent populated\n"
            bin_min = np.min(self.bins[i])
            bin_max = np.max(self.bins[i])
            if i in [2, 3]:
                bin_min = np.min(self.layer_bins)
                bin_max = np.max(self.layer_bins)
            error_message += f"Bin range: {bin_min:.2f} to {bin_max:.2f}\t"
            data = {
                0: radius,
                1: radius,
                2: raw_zs,
                3: raw_zs,
                4: energy,
                5: hits_per_shower,
                6: energy_per_shower,
                7: np.sum((gun_shifted[:, :, 0] * gun_shifted[:, :, 3]), axis=1)
                / energy_per_shower,
                8: np.sum((gun_shifted[:, :, 1] * gun_shifted[:, :, 3]), axis=1)
                / energy_per_shower,
                9: np.sum((gun_shifted[:, :, 2] * gun_shifted[:, :, 3]), axis=1)
                / energy_per_shower,
            }.get(i)
            if len(data):
                error_message += (
                    f"Data range: {np.nanmin(data):.2f} to {np.nanmax(data):.2f}, \n"
                )
            else:
                error_message += "No data to show range, \n"
        error_message += (
            "change construction arguments of BinnedData, or check input data."
        )
        if self.hard_check:
            raise ValueError(error_message)
        warnings.warn(error_message)

    def __str__(self):
        text = ""
        for i in range(len(self)):
            bins = self.bins[i]
            title = f"{self.x_labels[i]} v.s. {self.y_labels[i]};"
            bin_range = f"({bins[0]:.2f}, {bins[-1]:.2f}),"
            line = f"{title:60} {bin_range:30} {np.sum(self.counts[i])}\n"
            text += line
        return text

    def normed(self, idx):
        """
        Return the counts of the histogram at index idx,
        normalized by the total number of showers.

        Parameters
        ----------
        idx : int
            The index of the histogram in the binned data.

        Returns
        -------
        np.array
            The counts of the histogram, normalized by the total number of showers.
        """
        if "mean" in self.y_labels[idx].lower():
            return self.counts[idx]
        counts = self.counts[idx]
        hist_idx = self.hist_idx("number of showers", "number of hits")
        total_showers = np.sum(self.counts[hist_idx])
        return counts / total_showers

    def make_save_dict(self):
        to_save = {}
        for name in self.arg_names:
            to_save[name] = getattr(self, name)
        for i, counts in enumerate(self.counts):
            to_save[f"counts_{i}"] = counts
        to_save["sample_events"] = np.array(self.sample_events)
        return to_save

    def save(self, path):
        """
        Save the binned data to a file.

        Parameters
        ----------
        path : str
            The path to save the binned data to.
        """
        to_save = self.make_save_dict()
        if "sample_projected_idxs" in to_save:
            print("Saving with projections")
        else:
            print("Saving without projections")
        np.savez(path, **to_save)

    @classmethod
    def from_save_dict(cls, save_dict, **kwargs):
        args = {name: save_dict[name] for name in cls.arg_names if name in save_dict}
        for key in kwargs:
            if key not in cls.arg_names:
                raise ValueError(f"Unknown argument {key}")
            if key not in args:
                print(f"Adding {key} to file data")
                args[key] = kwargs[key]
            else:
                print(f"{key} already in file data")
        # don't hard check when loading, the data exists already
        args["hard_check"] = False
        this = cls(**args)
        for i in range(len(this)):
            try:
                this.counts[i] = save_dict[f"counts_{i}"]
            except KeyError:
                print(f"No saved data for counts {i}")
        this.recompute_mean_energies()
        try:
            this.sample_events = save_dict["sample_events"]
        except KeyError:
            print("No saved sample events")
        return this

    @classmethod
    def load(cls, path, **kwargs):
        """
        Load a binned data object from a file, including construction
        arguments and the counts.

        Parameters
        ----------
        path : str
            The path to load the binned data from.

        Returns
        -------
        BinnedData
            The loaded binned data.
        """
        saved = np.load(path, allow_pickle=True)
        return cls.from_save_dict(saved, **kwargs)


class DetectorBinnedData(BinnedData):
    """
    Agrigate key metrics for calorimeter models
    into bins for easy plotting and comparison.
    """

    true_xyz_limits = [
        [meta.Xmin_global, meta.Xmax_global],
        [meta.Zmin_global, meta.Zmax_global],
        [floors[0], ceilings[-1]],
    ]
    arg_names = [
        "name",
        "xyz_limits",
        "energy_scale",
        "MAP",
        "layer_bottom_pos",
        "half_cell_size_global",
        "cell_thickness",
        "gun_xyz_pos",
    ]

    def __init__(
        self,
        name,
        xyz_limits,
        energy_scale,
        MAP,
        layer_bottom_pos,
        half_cell_size_global,
        cell_thickness,
        gun_xyz_pos,
        hard_check=False,
        no_box_cut=True,
    ):
        """Construct a BinnedData object.

        Parameters
        ----------
        name : str
            The name of the model, no mechanical significance.
        energy_scale : float
            How much to divide the cluster energy by before projecting into the detector.
        MAP : list
            list of dictionaries, each containing the grid of cells for a layer
            in global coordinates
            As returned by detector_map.create_map
            Should be already offset, such that the gun fires at the origin.
        layer_bottom_pos : np.array
            The y positions of the bottom of each layer, in the coordinates
            the data is given in.
        half_cell_size_global : float
            Half the size of the cells in the detector,
            perpendicular to the radial direction
        cell_thickness: float
            The thickness in the y direction of each cell, in the coordinates
            the data is given in.
        gun_xyz_pos : np.array
            The x, y and z positions of the gun, in the coordinates the data is
            given in. The data is shifted so that the gun is at the origin.
        """
        self.hard_check = hard_check
        self.name = name
        self.energy_scale = energy_scale
        self.MAP = MAP
        self.layer_bottom_pos = layer_bottom_pos
        self.half_cell_size_global = half_cell_size_global
        self.cell_thickness = cell_thickness
        self.gun_xyz_pos = gun_xyz_pos
        self.xyz_limits = xyz_limits
        buffed_floors, buffed_ceilings = detector_map.floors_ceilings(
            layer_bottom_pos, cell_thickness, 1.0
        )
        self.perpendicular_cell_centers = detector_map.perpendicular_cell_centers(
            self.MAP, self.half_cell_size_global
        )
        for xs, ys in self.perpendicular_cell_centers:
            xs -= BinnedData.event_center[0]
            ys -= BinnedData.event_center[1]
        radial_min, radial_max, n_radial_bins, self.radial_cell_allocations = (
            self.radial_bins_for_cells()
        )
        self.layer_bins = np.concatenate([buffed_floors, buffed_ceilings[[-1]]])
        self.gun_shift = self.get_gunshift()

        self.y_labels = []
        self.x_labels = []
        self.bins = []
        self.counts = []

        self.add_hist(
            "sum active cells", "radius [mm]", radial_min, radial_max, n_radial_bins
        )
        self.add_hist(
            "sum energy [MeV]", "radius [mm]", radial_min, radial_max, n_radial_bins
        )
        self.add_hist("sum active cells", "layers", 1, 30, 30)
        self.add_hist("sum energy [MeV]", "layers", 1, 30, 30)
        self.add_hist("number of cells", "cell energy [MeV]", 0.01, 1100, 100, True)
        self.add_hist("number of showers", "number of active cells", 0, 3000, 50)
        self.add_hist("number of showers", "energy sum [MeV]", 0, 4000, 50)
        self.add_hist(
            "number of showers",
            "center of gravity X [mm]",
            *self.true_xyz_limits[0],
            2000,
        )
        self.add_hist(
            "number of showers",
            "center of gravity Y [mm]",
            *self.true_xyz_limits[1],
            2000,
        )
        self.add_hist(
            "number of showers",
            "center of gravity Z [layer]",
            *self.true_xyz_limits[2],
            60,
        )
        self.add_hist(
            "mean energy [MeV]", "radius [mm]", radial_min, radial_max, n_radial_bins
        )
        self.add_hist("mean energy [MeV]", "layers", 1, 30, 30)

        self.max_sample_events = 10
        self.raw_sample_events = []
        self.sample_events = []
        self.sample_projections = []

        if no_box_cut:
            self.box_cut = lambda events: None

    def radial_bins_for_cells(self):
        radial_locations = []
        radial_max = 0
        for xs, ys in self.perpendicular_cell_centers:
            radial_locations.append(np.sqrt(xs[:, None] ** 2 + ys[None, :] ** 2))
            radial_max = max(radial_max, np.max(radial_locations[-1]))
        radial_max = radial_max + self.half_cell_size_global
        radial_min = -self.half_cell_size_global
        radial_bin_width = 2 * self.half_cell_size_global
        n_radial_bins = int(
            np.floor((radial_max + self.half_cell_size_global) / radial_bin_width)
        )
        radial_bins = np.linspace(radial_min, radial_max, n_radial_bins + 1)
        radial_cell_allocations = []
        for locations in radial_locations:
            radial_cell_allocations.append(np.digitize(locations, radial_bins) - 1)
        return radial_min, radial_max, n_radial_bins, radial_cell_allocations

    def add_events(self, events):
        """
        Add events to the histograms that were created in the constructor.

        Parameters
        ----------
        events : np.array (n_showers, n_points, 4)
            The events to add to the histograms.
        """
        if len(self.raw_sample_events) < self.max_sample_events:
            from_batch = list(
                events[: (self.max_sample_events - len(self.sample_events))]
            )
            self.raw_sample_events.extend(from_batch)

        self.box_cut(events)
        mask = events[:, :, 3] > 0
        # sanity check 1
        if len(events) > 4 and not np.any(mask):
            error_message = (
                f"Box cut removed all points from {len(events)} showers, \n"
                f" - box cut had {self.xyz_limits=}"
            )
            if self.hard_check:
                raise ValueError(error_message)
            warnings.warn(error_message)

        rescaled = self.rescaled_events(events)
        gun_shifted = rescaled - self.gun_shift
        events_as_cells = detector_map.get_projections(
            gun_shifted,
            self.MAP,
            layer_bottom_pos=self.layer_bottom_pos,
            half_cell_size_global=self.half_cell_size_global,
            cell_thickness_global=self.cell_thickness,
            return_cell_point_cloud=False,
            include_artifacts=False,
        )
        detector_map.mip_cut(events_as_cells)

        if len(self.sample_projections) < self.max_sample_events:
            from_batch = list(
                events_as_cells[
                    : (self.max_sample_events - len(self.sample_projections))
                ]
            )
            self.sample_projections.extend(from_batch)
        return self.add_events_as_cells(events_as_cells)

    def add_events_as_cells(self, events_as_cells):
        """
        Add events to the histograms that were created in the constructor.

        Parameters
        ----------
        events : list of list of 2d np.array
            Outer list is events, next list is layers, final list is cell structure
            perpendicular to the layers.
        """
        if len(self.sample_events) < self.max_sample_events:
            for i in range(self.max_sample_events - len(self.sample_events)):
                max_points = 6_000
                event = detector_map.cells_to_points(
                    events_as_cells[i],
                    self.MAP,
                    self.layer_bottom_pos,
                    self.half_cell_size_global,
                    self.cell_thickness,
                    max_points,
                )
                trimmed = np.array([l[:max_points] for l in event])
                self.sample_events.append(trimmed)

        # things can be batch processed if we swap the inner and outer lists
        # and stack the layers
        layers = [np.stack(l) for l in zip(*events_as_cells)]

        hits_per_shower = np.zeros(len(events_as_cells), dtype=int)
        energy_per_shower = np.zeros(len(events_as_cells))
        energy_weighted_position = np.zeros((len(events_as_cells), 3))

        for l, cell_energies in enumerate(layers):
            active_cells = cell_energies > 0
            active_energies = cell_energies[active_cells]
            radial_bins = np.tile(
                self.radial_cell_allocations[l], (active_cells.shape[0], 1, 1)
            )[active_cells]
            # radial counts
            self.counts[0] += np.bincount(
                radial_bins, minlength=self.counts[0].shape[0]
            )
            # radial energies
            self.counts[1] += np.bincount(
                radial_bins, weights=active_energies, minlength=self.counts[1].shape[0]
            )
            # total in layer
            self.counts[2][l] += np.sum(active_cells)
            # energy in layer
            self.counts[3][l] += np.sum(active_energies)
            # energy spectrum
            self.counts[4] += np.histogram(active_energies, self.bins[4])[0]
            # then accumulate the hist and energy for these events
            hits_per_shower += np.sum(active_cells, axis=(1, 2))
            energy_per_shower += np.sum(cell_energies, axis=(1, 2))

            energy_weighted_position[:, 0] += np.sum(
                self.perpendicular_cell_centers[l][0][np.newaxis, :, np.newaxis]
                * cell_energies,
                axis=(1, 2),
            )
            energy_weighted_position[:, 1] += np.sum(
                self.perpendicular_cell_centers[l][1][np.newaxis, np.newaxis, :]
                * cell_energies,
                axis=(1, 2),
            )
            energy_weighted_position[:, 2] += np.sum(
                (l + 0.5) * cell_energies,
                axis=(1, 2),
            )

        self.counts[5] += np.histogram(hits_per_shower, self.bins[5])[0]
        self.counts[6] += np.histogram(energy_per_shower, self.bins[6])[0]

        with np.errstate(divide="ignore", invalid="ignore"):
            for i in range(3):
                cog = energy_weighted_position[:, i] / energy_per_shower
                self.counts[7 + i] += np.histogram(cog, self.bins[7 + i])[0]

        if len(events_as_cells) < 4:
            # no point checking with too few events.
            return

        # sanity check 2
        percent_populated = np.fromiter(
            (np.sum([b > 0 for b in c]) / len(c) for c in self.counts[:10]), float
        )
        if np.sum(percent_populated > 0.01) > 5:
            return
        error_message = "Some parameters don't seem to fit in hitogram range;\n"
        for i, percent in enumerate(percent_populated):
            if percent > 0.5:
                continue
            error_message += f"{self.y_labels[i]} v.s. {self.x_labels[i]}: {percent:.0%} percent populated\n"
            bin_min = np.min(self.bins[i])
            bin_max = np.max(self.bins[i])
            if i in [2, 3]:
                bin_min = np.min(self.layer_bins)
                bin_max = np.max(self.layer_bins)
            error_message += f"Bin range: {bin_min:.2f} to {bin_max:.2f}\t"
        error_message += (
            "change construction arguments of DetectorBinnedData, or check input data."
        )
        if self.hard_check:
            raise ValueError(error_message)
        warnings.warn(error_message)
        return

    def make_save_dict(self):
        save_dict = super().make_save_dict()
        save_dict["raw_sample_events"] = np.array(self.raw_sample_events)
        projected_dict = detector_map.projected_events_to_dict(self.sample_projections)
        for key in projected_dict:
            save_dict["sample_projected_" + key] = projected_dict[key]
        return save_dict

    @classmethod
    def from_save_dict(cls, save_dict, **kwargs):
        this = super().from_save_dict(save_dict, **kwargs)
        projected_dict = {
            key[len("sample_projected_") :]: save_dict[key]
            for key in save_dict
            if key.startswith("sample_projected_")
        }
        this.raw_sample_events = save_dict["raw_sample_events"]
        this.sample_projections = detector_map.dict_to_projected_events(
            projected_dict
        )[1]
        return this


def sample_g4(config, binned, n_events):
    """
    Draw samples from the g4 data and add them to the binned data.

    Parameters
    ----------
    config : Configs
        The configuration used to find the dataset, etc.
    binned : BinnedData
        The binned data to add the samples to, modified inplace.
    n_events : int
        The number of events to sample.
    """
    batch_len = 1000
    batch_starts = np.arange(0, n_events, batch_len)
    n_batches = np.ceil(n_events / batch_len)

    for b, start in enumerate(batch_starts):
        print(f"{b/n_batches:.1%}", end="\r", flush=True)
        cond, events = read_raw_regaxes_withcond(
            config, pick_events=slice(start, start + batch_len)
        )
        binned.add_events(events)
    print()
    print("Done")
    binned.recompute_mean_energies()


def sample_model(config, binned, n_events, model, shower_flow=None):
    """
    Use a model to produce events and add them to the binned data.

    Parameters
    ----------
    config : Configs
        The configuration used to find the dataset, etc.
    binned : BinnedData
        The binned data to add the samples to, modified inplace.
    n_events : int
        The number of events to sample.
    model : torch.nn.Module
        The model to sample from, will be sampled by
        pointcloud.utils.gen_utils.gen_cond_showers_batch.
    shower_flow : distribution, optional
        The flow to sample from, if the model needs it, by default None
        also used by gen_cond_showers_batch.

    Returns
    -------
    cond : np.array (n_events, C)
        The incident conditioning of the particle gun of each event.
    events : np.array (n_events, n_points, 4)
        The events produced by the model
        with the last axis being [x, y, z, energy].

    """
    if n_events == 0:
        return np.zeros(0), np.zeros((0, 0, 4))
    cond, _ = read_raw_regaxes_withcond(
        config,
        total_size=n_events,
        for_model=["showerflow", "diffusion"],
    )
    events = conditioned_sample_model(
        config,
        binned,
        cond,
        model,
        shower_flow=shower_flow,
    )
    return cond, events


def conditioned_sample_model(model_config, binned, cond, model, shower_flow=None):
    """
    Use a model to produce events and add them to the binned data.

    Parameters
    ----------
    model_config : Configs
        The configuration used for model metadata
    binned : BinnedData or DetectorBinnedData
        The binned data to add the samples to, modified inplace.
    model_config : torch.Tensor (n_events, C)
        The conditioning for the sample
    model : torch.nn.Module
        The model to sample from, will be sampled by
        pointcloud.utils.gen_utils.gen_cond_showers_batch.
    shower_flow : distribution, optional
        The flow to sample from, if the model needs it, by default None
        also used by gen_cond_showers_batch.

    Returns
    -------
    events : np.array (n_events, n_points, 4)
        The events produced by the model
        with the last axis being [x, y, z, energy].

    """
    if isinstance(cond, dict):
        n_events = len(next(iter(cond.values())))
    else:
        n_events = len(cond)
    if n_events == 0:
        return np.zeros(0), np.zeros((0, 0, 4))
    if model_config.model_name in ["fish", "wish"]:
        batch_len = min(1000, n_events)
    else:
        batch_len = min(100, n_events)

    batch_starts = np.arange(0, n_events, batch_len)
    n_batches = np.ceil(n_events / batch_len)

    for b, start in enumerate(batch_starts):
        print(f"{b/n_batches:.1%}", end="\r", flush=True)
        if isinstance(cond, dict):
            cond_here = {key: cond[key][start : start + batch_len] for key in cond}
        else:
            cond_here = cond[start : start + batch_len]
        events = gen_cond_showers_batch(
            model, shower_flow, cond_here, bs=batch_len, config=model_config
        )
        binned.add_events(events)

    print()
    print("Done")
    binned.recompute_mean_energies()


def get_path(config, name, detector_projection=False):
    """
    Chose a path to save the binned data to.

    Parameters
    ----------
    config : Configs
        The configuration used to find the logdir.
    name : str
        Name of the binned data, will be used in the file name.

    Returns
    -------
    path : str
        The path to save the binned data to.
    """
    if hasattr(config, "dataset_tag"):
        name += "_" + config.dataset_tag
    if detector_projection:
        name += "_detectorProj"
    log_dir = config.logdir
    binned_metrics_dir = os.path.join(log_dir, "binned_metrics")
    try_mkdir(binned_metrics_dir)
    file_name = name.replace(" ", "_") + ".npz"
    return os.path.join(binned_metrics_dir, file_name)


def get_wish_models(
    wish_path="../point-cloud-diffusion-logs/wish/dataset_accumulators"
    "/p22_th90_ph90_en10-1/wish_poly{}.pt",
    n_poly_degrees=4,
    device="cpu",
):
    """
    Gather a set of models for evaluation.

    Parameters
    ----------
    wish_path : str, optional
        The path to the wish models, with a formatable part for the
        polynomial degree, by default
        "../point-cloud-diffusion-logs/wish/dataset_accumulators"
        "/p22_th90_ph90_en10-1/wish_poly{}.pt"
    n_poly_degrees : int, optional
        The number of polynomial degrees to gather, by default 4
    device : str, optional
        The device to load the models onto, by default "cpu"

    Returns
    -------
    models : dict of {str: (torch.nn.Module, None, Configs)}
        The models to evaluate, with the key being the name of the model.
        The value is a tuple of the model, None (no flow model), and the
        configuration used to create the model.
    """
    models = {}
    for poly_degree in range(1, n_poly_degrees + 1):
        config = WishConfigs()
        config.poly_degree = poly_degree
        config.device = device
        here = Wish.load(wish_path.format(poly_degree))
        models[f"Wish-poly{poly_degree}"] = (here, None, config)
    return models


def get_fish_models(
    fish_path="../point-cloud-diffusion-logs/fish/fish.npz",
    device="cpu",
):
    """
    Gather a set of models for evaluation.

    Parameters
    ----------
    fish_path : str, optional
        The path to the fish model, by default
        "../point-cloud-diffusion-logs/fish/fish.npz"
    device : str, optional
        The device to load the models onto, by default "cpu"

    Returns
    -------
    models : dict of {str: (torch.nn.Module, None, Configs)}
        The models to evaluate, with the key being the name of the model.
        The value is a tuple of the model, None (no flow model), and the
        configuration used to create the model.
    """
    models = {}
    fish_model = Fish.load(fish_path)
    # TODO fish isn't currently a torch Module
    # may need to change this
    # fish_model = fish_model.to(device)
    config = WishConfigs()
    config.model_name = "fish"
    models["Fish"] = (fish_model, None, config)
    return models


def filenames_to_labels(filenames):
    base_names = [os.path.basename(path) for path in filenames]
    assert len(set(base_names)) == len(base_names), "Duplicate filenames"
    common_prefix = os.path.commonprefix(base_names)
    common_suffix = os.path.commonprefix([basename[::-1] for basename in base_names])[
        ::-1
    ]
    stripped_labels = [
        name[len(common_prefix) : -len(common_suffix)] for name in base_names
    ]
    return stripped_labels


def get_caloclouds_models(
    caloclouds_paths="../point-cloud-diffusion-logs/p22_th90_ph90_en10-100/CD_2024_05_24__14_47_04/ckpt_0.000000_990000.pt",
    showerflow_paths="../point-cloud-diffusion-data/showerFlow/ShowerFlow_best.pth",
    device="cpu",
    caloclouds_names="CaloClouds3",
    showerflow_names="",
    config=None,
):
    """
    Gather a set of models for evaluation. Currently just one.

    Parameters
    ----------
    caloclouds_paths : str or list of str, optional
        The path to the caloclouds model, by default
        "../point-cloud-diffusion-logs/p22_th90_ph90_en10-100/CD_2024_05_24__14_47_04/ckpt_0.000000_990000.pt"
    showerflow_paths : str or list of str, optional
        The path to the showerflow model, by default
        "../point-cloud-diffusion-data/showerFlow/ShowerFlow_best.pth"
    device : str, optional
        The device to load the models onto, by default "cpu"
    caloclouds_names: list of str, optional
        The names of the models, by default "CaloClouds3"
        or if multiple paths are given, the filenames will be used.
    showerflow_names: list of str, optional
        The names of the models, by default empty
        or if multiple paths are given, the filenames will be used
    config : Configs, optional
        The configuration used to create the models, by default
        the files global config is used.

    Returns
    -------
    models : dict of {str: (torch.nn.Module, None, Configs)}
        The models to evaluate, with the key being the name of the model.
        The value is a tuple of the model, the flow model, and the
        configuration used to create the model.
    """
    models = {}
    if config is None:
        config = Configs()
    config.device = device
    distillation = getattr(config, "distillation", False)

    if isinstance(caloclouds_paths, str):
        caloclouds_paths = [caloclouds_paths]
        caloclouds_names = [caloclouds_names]
    elif isinstance(caloclouds_names, str):
        caloclouds_names = filenames_to_labels(caloclouds_paths)
    if isinstance(showerflow_paths, str):
        showerflow_paths = [showerflow_paths]
        showerflow_names = [showerflow_names]
    elif isinstance(showerflow_names, str):
        showerflow_names = filenames_to_labels(showerflow_paths)

    for calocloud_name, calocloud_path in zip(caloclouds_names, caloclouds_paths):
        for showerflow_name, showerflow_path in zip(showerflow_names, showerflow_paths):
            model = get_model_class(config)(config, distillation=distillation).to(
                device
            )
            print(calocloud_path)
            model.load_state_dict(
                torch.load(calocloud_path, map_location=device, weights_only=False)[
                    "state_dict"
                ]
            )
            model.eval()
            try:
                showerflow_config = showerflow_utils.config_from_showerflow_path(
                    config, showerflow_path
                )
            except AssertionError:
                warnings.warn(
                    f"Couldn't create config from {showerflow_path}, using input config"
                )
                showerflow_config = config
            input_mask = showerflow_utils.get_input_mask(showerflow_config)
            version = versions_dict[showerflow_config.shower_flow_version]
            flow_model, flow_dist, _ = version(
                num_blocks=showerflow_config.shower_flow_num_blocks,
                num_inputs=np.sum(input_mask),
                num_cond_inputs=get_cond_dim(showerflow_config, "showerflow"),
                device=device,
            )
            print(showerflow_path)
            flow_model.load_state_dict(
                torch.load(showerflow_path, map_location=device, weights_only=False)[
                    "model"
                ]
            )
            flow_model.eval()
            name = (
                f"{calocloud_name}-{showerflow_name}"
                if showerflow_name
                else calocloud_name
            )
            models[name] = (model, flow_dist, showerflow_config)

    return models
