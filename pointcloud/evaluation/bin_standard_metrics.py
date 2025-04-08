import numpy as np
import torch
import os
import warnings

from pointcloud.config_varients.wish import Configs
from pointcloud.config_varients.wish_maxwell import Configs as MaxwellConfigs

from pointcloud.utils.metadata import Metadata
from pointcloud.utils.detector_map import floors_ceilings
from pointcloud.data.conditioning import read_raw_regaxes_withcond, get_cond_dim
from pointcloud.utils import showerflow_utils, precision
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


configs = Configs()
meta = Metadata(configs)
floors, ceilings = floors_ceilings(
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
        buffed_floors, buffed_ceilings = floors_ceilings(
            layer_bottom_pos, cell_thickness, 1.0
        )
        self.layer_bins = np.concatenate([buffed_floors, buffed_ceilings[[-1]]])
        self.gun_shift = self.get_gunshift()

        self.y_labels = []
        self.x_labels = []
        self.bins = []
        self.counts = []

        self.add_hist("sum hits", "radius [mm]", 0, 500, 32)
        self.add_hist("sum energy [MeV]", "radius [mm]", 0, 500, 32)
        self.add_hist("sum hits", "layers", 1, 30, 30)
        self.add_hist("sum energy [MeV]", "layers", 1, 30, 30)
        self.add_hist(
            "number of cells", "visible cell energy [MeV]", 0.000001, 1, 100, True
        )
        self.add_hist("number of showers", "number of hits", 0, 10000, 50)
        self.add_hist("number of showers", "energy sum [MeV]", 0, 10, 50)
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
            "center of gravity Z [mm]",
            *self.true_xyz_limits[2],
            60,
        )
        self.add_hist("mean energy [MeV]", "radius [mm]", 0, 500, 32)
        self.add_hist("mean energy [MeV]", "layers", 1, 30, 30)

        self.max_sample_events = 10
        self.sample_events = []

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
        Using the summed energy and hits, recompute the mean energy histograms.
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
        hist_idx = self.hist_idx("number of showers", "number of hits")
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
        radius = np.sqrt(gun_shifted[:, :, 0] ** 2 + gun_shifted[:, :, 1] ** 2)[mask]

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

    def save(self, path):
        """
        Save the binned data to a file.

        Parameters
        ----------
        path : str
            The path to save the binned data to.
        """
        to_save = {}
        for name in self.arg_names:
            to_save[name] = getattr(self, name)
        for i, counts in enumerate(self.counts):
            to_save[f"counts_{i}"] = counts
        to_save["sample_events"] = np.array(self.sample_events)
        np.savez(path, **to_save)

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
        args = {name: saved[name] for name in cls.arg_names if name in saved}
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
                this.counts[i] = saved[f"counts_{i}"]
            except KeyError:
                print(f"No saved data for counts {i}")
        this.recompute_mean_energies()
        try:
            this.sample_events = saved["sample_events"]
        except KeyError:
            print("No saved sample events")
        return this


def sample_g4(configs, binned, n_events):
    """
    Draw samples from the g4 data and add them to the binned data.

    Parameters
    ----------
    configs : Configs
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
            configs, pick_events=slice(start, start + batch_len)
        )
        binned.add_events(events)
    print()
    print("Done")
    binned.recompute_mean_energies()


def sample_model(configs, binned, n_events, model, shower_flow=None):
    """
    Use a model to produce events and add them to the binned data.

    Parameters
    ----------
    configs : Configs
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
    if configs.model_name in ["fish", "wish"]:
        batch_len = min(1000, n_events)
    else:
        batch_len = min(100, n_events)

    batch_starts = np.arange(0, n_events, batch_len)
    n_batches = np.ceil(n_events / batch_len)

    for b, start in enumerate(batch_starts):
        print(f"{b/n_batches:.1%}", end="\r", flush=True)
        cond, _ = read_raw_regaxes_withcond(
            configs,
            pick_events=slice(start, start + batch_len),
            for_model=["showerflow", "diffusion"],
        )
        events = gen_cond_showers_batch(
            model, shower_flow, cond, bs=batch_len, config=configs
        )
        binned.add_events(events)
    print()
    print("Done")
    binned.recompute_mean_energies()
    return cond, events


def sample_accumulator(configs, binned, acc, n_events):
    """
    For the values for which it's possible, create binned data from an
    accumulator object.

    Parameters
    ----------
    configs : Configs
        The configuration used to find the dataset, etc.
    binned : BinnedData
        The binned data to add the samples to, modified inplace.
    acc : pointcloud.utils.stats_accumulator.StatsAccumulator
        The accumulator to take the data from.
    n_events : int
        Not actually used, but kept for consistency with the other sampling functions.
    """
    for i in range(len(binned)):
        binned.counts[i][:] = np.nan

    x_bin_centers, y_bin_centers, z_bin_centers = acc._get_bin_centers()
    n_x_bins = len(x_bin_centers)
    n_y_bins = len(y_bin_centers)
    n_z_bins = len(z_bin_centers)
    fake_events = np.ones((1, np.max([n_x_bins, n_y_bins, n_z_bins]), 4))
    fake_events[0, :n_x_bins, 0] = x_bin_centers
    fake_events[0, :n_y_bins, 1] = y_bin_centers
    fake_events[0, :n_z_bins, 2] = z_bin_centers
    rescaled = binned.rescaled_events(fake_events)
    x_bin_centers = rescaled[0, :n_x_bins, 0]
    y_bin_centers = rescaled[0, :n_y_bins, 1]
    z_bin_centers = rescaled[0, :n_z_bins, 2]

    gun_shift = binned.get_gunshift()
    x_bin_centers -= gun_shift[0, 0, 0]
    y_bin_centers -= gun_shift[0, 0, 1]
    bin_radii = np.sqrt(
        x_bin_centers[:, np.newaxis] ** 2 + y_bin_centers[np.newaxis, :] ** 2
    )
    flat_radii = bin_radii.flatten()
    flat_hits = np.sum(acc.counts_hist, axis=(0, 1)).flatten()
    flat_energy = np.sum(acc.energy_hist / 1000, axis=(0, 1)).flatten()

    binned.counts[0] = np.histogram(flat_radii, binned.bins[0], weights=flat_hits)[0]
    binned.counts[1] = np.histogram(flat_radii, binned.bins[1], weights=flat_energy)[0]

    layers_hits = np.sum(acc.counts_hist, axis=(0, 2, 3))
    binned.counts[2][:] = layers_hits
    layers_energy = np.sum(acc.energy_hist / 1000, axis=(0, 2, 3))
    binned.counts[3][:] = layers_energy

    print("Done")
    binned.recompute_mean_energies()


def get_path(configs, name):
    """
    Chose a path to save the binned data to.

    Parameters
    ----------
    configs : Configs
        The configuration used to find the logdir.
    name : str
        Name of the binned data, will be used in the file name.

    Returns
    -------
    path : str
        The path to save the binned data to.
    """
    log_dir = configs.logdir
    binned_metrics_dir = os.path.join(log_dir, "binned_metrics")
    try_mkdir(binned_metrics_dir)
    addition = ""
    if hasattr(configs, "diffusion_precision"):
        addition += "_difPrec" + str(configs.diffusion_precision)
    if hasattr(configs, "showerflow_precision"):
        addition += "_sfPrec" + str(configs.showerflow_precision)
    file_name = name.replace(" ", "_") + addition + ".npz"
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
        configs = WishConfigs()
        configs.poly_degree = poly_degree
        configs.device = device
        here = Wish.load(wish_path.format(poly_degree))
        models[f"Wish-poly{poly_degree}"] = (here, None, configs)
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
    configs = WishConfigs()
    configs.model_name = "fish"
    models["Fish"] = (fish_model, None, configs)
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
    configs=None,
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
    configs : Configs, optional
        The configuration used to create the models, by default
        the files global configs is used.

    Returns
    -------
    models : dict of {str: (torch.nn.Module, None, Configs)}
        The models to evaluate, with the key being the name of the model.
        The value is a tuple of the model, the flow model, and the
        configuration used to create the model.
    """
    models = {}
    if configs is None:
        configs = Configs()
    configs.device = device

    diffusion_dtype = precision.get("diffusion", configs)
    showerflow_dtype = precision.get("showerflow", configs)

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
            model = get_model_class(configs)(configs).to(device)
            model.to(configs.device, dtype=diffusion_dtype)
            print(calocloud_path)
            model.load_state_dict(
                torch.load(calocloud_path, map_location=device, weights_only=False)[
                    "state_dict"
                ]
            )
            try:
                showerflow_configs = showerflow_utils.configs_from_showerflow_path(
                    configs, showerflow_path
                )
            except AssertionError:
                warnings.warn(
                    f"Couldn't create configs from {showerflow_path}, using input configs"
                )
                showerflow_configs = configs
            input_mask = showerflow_utils.get_input_mask(showerflow_configs)
            version = versions_dict[showerflow_configs.shower_flow_version]
            flow_model, flow_dist, _ = version(
                num_blocks=showerflow_configs.shower_flow_num_blocks,
                num_inputs=np.sum(input_mask),
                num_cond_inputs=get_cond_dim(showerflow_configs, "showerflow"),
                device=device,
                dtype=showerflow_dtype,
            )
            print(showerflow_path)
            flow_model = flow_model.to(device, dtype=showerflow_dtype)
            flow_model.load_state_dict(
                torch.load(showerflow_path, map_location=device, weights_only=False)[
                    "model"
                ]
            )
            flow_model = flow_model.to(device, dtype=showerflow_dtype)
            #flow_dist.transforms = [
            #    t.to(device, dtype=showerflow_dtype) if hasattr(t, "to") else t
            #    for t in flow_dist.transforms
            #]
            #flow_dist.base_dist.base_dist.loc = flow_dist.base_dist.base_dist.loc.to(
            #    device, dtype=showerflow_dtype
            #)
            #flow_dist.base_dist.base_dist.scale = (
            #    flow_dist.base_dist.base_dist.scale.to(device, dtype=showerflow_dtype)
            #)
            name = (
                f"{calocloud_name}-{showerflow_name}"
                if showerflow_name
                else calocloud_name
            )
            models[name] = (model, flow_dist, showerflow_configs)

    return models
