from torch.utils.data import Dataset
import warnings
import numpy as np
import h5py

from ..configs import Configs
from ..utils.metadata import Metadata
from ..utils.detector_map import floors_ceilings
from .read_write import get_files, events_to_local
from .conditioning import get_cond_features_names, padding_position


class PointCloudDataset(Dataset):
    metadata = Metadata(Configs())
    # these can be accessed without instantiating the class
    energy_scale = 1000  # MeV to GeV
    # The keys that will be draw from the dataset
    # format is {"name in batch": "name on disk"}
    keys_to_include = {"event": "events", "energy": "energy", "points": None}

    def __init__(
        self,
        file_path,
        bs=32,
        max_ds_seq_len=6000,
        quantized_pos=True,
        n_files=0,
    ):
        """
        Base class for point cloud open_files.
        Iterable, torch.utils.data.Dataset subclass.

        Parameters
        ----------
        file_path : str
            Path to the HDF5 file containing the dataset.
            If n_files is > 0, then the file_path should
            contain one or more "{}" to be formatted with
            the file number.
        bs : int, optional
            Batch size, number of events returned in each
            iteration.
            Default is 32.
        max_ds_seq_len : int, optional
            Maximium number of points/hits in each event.
            Data will be trimmed to this length if longer on
            the disk.
            Default is 6000.
        quantized_pos : bool
            If this is True, the positions of the hits
            are left in the grid that they are written
            in the dataset. If False, the positions are
            fuzzed uniformly in the x and z directions
            inside each cell to create continuous
            x and z coordinates.
        n_files : int, optional
            If this is > 0, then the file_path should
            contain one or more "{}" to be formatted with
            the file number.
            If it's 0, then the file_path should be a
            single file.
        """
        self.open_files = self._open_data_files(file_path, n_files)

        event_key = self.keys_to_include["event"]
        self._roll_axis = False
        for open_file in self.open_files:
            shape = open_file[event_key].shape
            if shape[-1] == 4 and shape[-2] != 4:
                # no moves needed.
                break
            elif shape[-1] != 4 and shape[-2] == 4:
                self._roll_axis = True
                break
        else:
            warnings.warn(
                "Can't find xyze axis in data on disk, assuming it's the last axis"
            )
        self._prior_event_axes = self._get_prior_event_axes()

        self.max_ds_seq_len = max_ds_seq_len
        self.index_list = self._make_index_list()
        self.front_padded = self._is_front_padded()
        self.bs = bs

        self.quantized_pos = quantized_pos
        self.offset = 5.0883331298828125 / 6  # size of x36 granular grid

        # avoid repeat calculation
        self._len = len(self.index_list)

    def _open_data_files(self, file_path, n_files):
        """
        Open all the data files, and return them.
        We don't bother closing them, because they are
        read-only and will be closed when the program
        exits.
        N.B. if we end up with memory issues,
        we might need to rethink this.

        Parameters
        ----------
        file_path : str
            Path to the HDF5 file containing the dataset.
            If n_files is > 0, then the file_path should
            contain one or more "{}" to be formatted with
            the file number.
        n_files : int
            If this is > 0, then the file_path should
            contain one or more "{}" to be formatted with
            the file number.
            If it's 0, then the file_path should be a
            single file.

        Returns
        -------
        list
            List of h5py.File objects.
        """
        all_files = [h5py.File(path, "r") for path in get_files(file_path, n_files)]
        if not all_files:
            raise FileNotFoundError(f"No files found at {file_path}")
        return all_files

    def _make_index_list(self):
        """
        To allow the data to be iterated in order of
        number of points if desired, make a list of
        indices that can be used to access the data
        in that order.

        Returns
        -------
        index_list : numpy.ndarray (n_points, 3)
            Array with cols (n_points, file_idx, event_idx)
            that can be used to access the data in order
            of number of points.
        """
        index_list = []
        event_key = self.keys_to_include["event"]
        for file_idx, dataset in enumerate(self.open_files):
            if "n_points" in dataset:
                n_points = dataset["n_points"][:]
            else:
                if self._roll_axis:
                    events = np.moveaxis(dataset[event_key], -1, -2)
                    n_points = self.get_n_points(events)
                else:
                    n_points = self.get_n_points(dataset[event_key])
            n_points[n_points > self.max_ds_seq_len] = self.max_ds_seq_len
            index_list += [(n, file_idx, i) for i, n in enumerate(n_points)]
        # sort the index list by 'n_points'
        index_list.sort(key=lambda x: x[0])
        index_list = np.array(index_list, dtype=int)
        return index_list

    def _get_prior_event_axes(self):
        """
        For each key in keys_to_include,
        find out which axis is the length of th number of events
        in that file.
        Create a number of empty slices to pad out the indexing to that point.
        We make the assumption that this is the same
        for all files in the dataset.
        """
        file_0 = self.open_files[0]
        n_events_in_file0 = file_0[self.keys_to_include["event"]].shape[0]
        axes = {
            name: [slice(None)] * file_0[key].shape.index(n_events_in_file0)
            for name, key in self.keys_to_include.items()
            if key is not None
        }
        return axes

    def _is_front_padded(self, check_file=0):
        event_key = self.keys_to_include["event"]
        padding = padding_position(
            self.open_files[check_file][event_key], self._roll_axis
        )
        if padding == "front":
            is_front_padded = True
        elif padding == "back":
            is_front_padded = False
        elif check_file >= len(self.open_files) - 1:
            # stop before we run out of files to check
            is_front_padded = False
        else:
            is_front_padded = self._is_front_padded(check_file + 1)
        return is_front_padded

    @classmethod
    def get_n_points(cls, data, axis=-1):
        """
        Can operate on an event, or a batch of events.
        """
        n_points_arr = (data[..., axis] != 0.0).sum(1)
        return n_points_arr

    def _choose_idxs(self, idx):
        if idx > self.bs and idx < self.__len__() - self.bs:
            idxs = slice(idx - int(self.bs / 2), idx + int(self.bs / 2))
        elif idx < self.bs:
            idxs = slice(idx, idx + self.bs)
        else:
            idxs = slice(idx - self.bs, idx)
        return idxs

    def _fuzz_parallel(self, event):
        pos_offset_x = np.random.uniform(0, self.offset, 1)
        pos_offset_y = np.random.uniform(0, self.offset, 1)
        event[:, :, 0] = event[:, :, 0] + pos_offset_x
        event[:, :, 1] = event[:, :, 1] + pos_offset_y

    @classmethod
    def normalize_xyze(cls, event, fuzz_perpendicular=False):
        """
        Ensure that the x, y and z go from -1 to 1.
        Also rescale the energy values to by the classes energy_scale.
        Acts in place.

        Parameters
        ----------
        event : np.ndarray, (..., 4)
            One or more events. The last dimension should be 4
            with coordinates x, y, z, and energy.
        fuzz_perpendicular : bool, optional
            If true, the points will be uniformly distirbuted in
            each layer, rather than respecting the relative location
            in the perpendicular direction found in the received data.

        """
        cls.normalize_parallal(event)
        cls.normalize_perpendicular(event, fuzz_perpendicular)
        event[..., 3] *= cls.energy_scale

    @classmethod
    def normalize_parallal(cls, event):
        """
        Normalise in the directions parallel to the layers.
        In this case that's the x and y directions.
        Ensure that the x and y go from -1 to 1.
        Acts in place.

        Parameters
        ----------
        event : np.ndarray, (..., 4)
            One or more events. The last dimension should be 4
            with coordinates x, y, z, and energy.

        """
        assert cls.metadata.orientation[:16] == "hdf5:xyz==local:"
        assert cls.metadata.orientation_global[:17] == "hdf5:xyz==global:"
        local_ori = cls.metadata.orientation[16:]
        global_ori = cls.metadata.orientation_global[17:]
        if global_ori[local_ori.index("x")] == "z":
            assert global_ori[local_ori.index("y")] == "x"
            # rotated coordinates mean local x==global z and local y==global x and local z==global y
            Xmin = cls.metadata.Zmin_global
            Xmax = cls.metadata.Zmax_global
            Ymin = cls.metadata.Xmin_global
            Ymax = cls.metadata.Xmax_global
        elif global_ori[local_ori.index("x")] == "x":
            assert global_ori[local_ori.index("y")] == "z"
            Xmin = cls.metadata.Xmin_global
            Xmax = cls.metadata.Xmax_global
            Ymin = cls.metadata.Zmin_global
            Ymax = cls.metadata.Zmax_global
        else:
            raise NotImplementedError(
                f"Global orientation {global_ori} not supported for local orientation {local_ori}"
            )
        # works on the assumption that the data has been saved with
        # the physical x and z scale unaltered
        event[..., 0] = (
            ((event[..., 0] - Xmin) * 2) / (Xmax - Xmin)
        ) - 1  # x coordinate normalization
        event[..., 1] = (
            ((event[..., 1] - Ymin) * 2) / (Ymax - Ymin)
        ) - 1  # z coordinate normalization
        # ensure padding xyz is 0
        event[event[..., 3] == 0] = 0

    normalised_bounds = np.linspace(-1, 1, len(metadata.layer_bottom_pos_hdf5) + 1)

    @classmethod
    def normalize_perpendicular(cls, event, and_fuzz):
        """
        Normalise in the direction perpendicular to the layers.
        Map the relevant coordinate to a linspace from -1 to 1.
        Gaps between layers are rescaled to have the same size,
        but points do not lose their variation in the perpendicular.
        Acts in place.

        Parameters
        ----------
        event : np.ndarray, (..., 4)
            One or more events. The last dimension should be 4
            with coordinates x, y, z, and energy.
        and_fuzz : bool, optional
            If true, the points will be uniformly distirbuted in
            each layer, rather than respecting the relative location
            in the perpendicular direction found in the received data.

        Raises
        ------
        ValueError
            If some points are outside the layers.

        """
        return cls._normalize_perpendicular(event, and_fuzz, perpendicular_axis=2)

    @classmethod
    def _normalize_perpendicular(cls, event, and_fuzz, perpendicular_axis=2):
        """
        Normalise in the direction perpendicular to the layers.
        Map the relevant coordinate to a linspace from -1 to 1.
        Gaps between layers are rescaled to have the same size,
        but points do not lose their variation in the perpendicular.
        Acts in place.

        Parameters
        ----------
        event : np.ndarray, (..., 4)
            One or more events. The last dimension should be 4
            with coordinates x, y, z, and energy.
        and_fuzz : bool, optional
            If true, the points will be uniformly distirbuted in
            each layer, rather than respecting the relative location
            in the perpendicular direction found in the received data.


        Raises
        ------
        ValueError
            If some points are outside the layers.

        """
        layer_floors, layer_ceilings = floors_ceilings(
            cls.metadata.layer_bottom_pos_hdf5,
            cls.metadata.cell_thickness_hdf5,
            percent_buffer=0,
        )
        rescales = (cls.normalised_bounds[1:] - cls.normalised_bounds[:-1]) / (
            layer_ceilings - layer_floors
        )

        select_from, select_to = floors_ceilings(
            cls.metadata.layer_bottom_pos_hdf5,
            cls.metadata.cell_thickness_hdf5,
            percent_buffer=0.5,
        )

        done = np.zeros(event.shape[:-1], dtype=bool)
        # not real points don't need moving
        done[event[..., 3] <= 0] = True

        for i, (floor, ceiling) in enumerate(zip(layer_floors, layer_ceilings)):
            mask = (event[..., perpendicular_axis] >= select_from[i]) & (
                event[..., perpendicular_axis] < select_to[i]
            )
            new_floor = cls.normalised_bounds[i]
            new_ceiling = cls.normalised_bounds[i + 1]
            if and_fuzz:
                event[..., perpendicular_axis][mask] = np.random.uniform(
                    new_floor, new_ceiling, mask.sum()
                )
            else:
                event[..., perpendicular_axis][mask] = np.clip(
                    ((event[..., perpendicular_axis][mask] - floor) * rescales[i])
                    + new_floor,
                    new_floor,
                    new_ceiling,
                )
            done[mask] = True
        if not np.all(done):
            msg = "Some points appear to be outside all layers"
            percent_outside = (~done).sum() / done.size
            msg += f" ({percent_outside:.2%} of points)\n"
            lowest_point = event[..., perpendicular_axis][~done].min()
            highest_point = event[..., perpendicular_axis][~done].max()
            msg += f"Lowest point outside layers: {lowest_point}\n"
            msg += f"Highest point outside layers: {highest_point}\n"
            lowest_layer = layer_floors[0]
            highest_layer = layer_ceilings[-1]
            msg += f"Layers range from {lowest_layer} to {highest_layer}\n"
            msg += f"Metadata from {cls.metadata.metadata_folder}\n"
            raise ValueError(msg)

    def _event_processing(self, event):
        if self._roll_axis:
            event = np.moveaxis(event, -1, -2)

        # Ensure the shower runs along the z axis
        events_to_local(event, self.metadata.orientation)

        # Trim padding
        max_len = (event[:, :, 3] > 0).sum(axis=1).max()
        trim_len = min(max_len, self.max_ds_seq_len)
        if self.front_padded:
            event = event[:, -trim_len:]
        else:
            event = event[:, :trim_len]

        if not self.quantized_pos:
            self._fuzz_parallel(event)

        self.normalize_xyze(event, fuzz_perpendicular=(not self.quantized_pos))

        return event

    def __getitem__(self, idx):
        idxs = self._choose_idxs(idx)
        batch = {}
        for name_in_batch, name_on_disk in self.keys_to_include.items():
            if name_in_batch == "points":
                continue

            padding = self._prior_event_axes[name_in_batch]
            data = np.array(
                [
                    self.open_files[file_n][name_on_disk][(*padding, event_n)]
                    for n_pts, file_n, event_n in self.index_list[idxs]
                ]
            )

            if name_in_batch == "event":
                data = self._event_processing(data)

            if len(data.shape) == 1:
                data = data[..., np.newaxis]
            batch[name_in_batch] = data

        # special case for points as it's already been processed
        if "points" in self.keys_to_include:
            batch["points"] = self.index_list[idxs, 0, np.newaxis]

        return batch

    def __len__(self):
        return self._len


class PointCloudDatasetUnordered(PointCloudDataset):
    def _choose_idxs(self, idx):
        rng = np.random.default_rng(seed=idx)
        bs = min(self._len, self.bs)
        idxs = rng.choice(self._len, bs, replace=False)
        idxs.sort()
        return idxs


class PointCloudDatasetGH(PointCloudDataset):
    # These should be set from metadata...
    # Ymin, Ymax = 0, 30
    # Xmin, Xmax = 0, 30
    # Zmin, Zmax = 0, 30
    energy_scale = 1  # no conversion needed for GH dataset

    keys_to_include = {"event": "events", "energy": "energy"}

    def __init__(self, file_path, bs=32, max_ds_seq_len=1700, n_files=0):
        self._roll_axis = False  # no need to roll axis for GH dataset
        self.max_ds_seq_len = max_ds_seq_len
        self.open_files = self._open_data_files(file_path, n_files)
        self._prior_event_axes = self._get_prior_event_axes()
        self.index_list = self._make_index_list()
        self.front_padded = self._is_front_padded()
        self.bs = bs

        self.quantized_pos = True  # never fuzz the GH dataset
        # therefore we don't need an offset

        # avoid repeat calculation
        self._len = len(self.index_list)

    def _fuzz_parallel(self, event):
        raise NotImplementedError("Why are you fuzzing the GH dataset?")


# Contains heavy assumptions about the dataset - could draw these from metadata
# might need to also return "points" as "points_per_layer"
class PointCloudAngular(PointCloudDataset):
    keys_to_include = {
        "event": "events",
        "energy": "energy",
        "p_norm_local": "p_norm_local",
    }
    # correct for the sim-E... datasets
    Xmean, Ymean, Zmean = -0.0074305227, -0.21205868, 12.359252
    Xstd, Ystd, Zstd = 21.608465, 22.748442, 5.305082
    Emean, Estd = -1.5300317, 1.2500798

    @classmethod
    def normalize_xyze(cls, event, fuzz_perpendicular=False):
        assert fuzz_perpendicular is False, "Assumption is that this is fuzzed on disk"
        event[..., 3] = (
            (np.log(event[..., 3] + 1e-12) - cls.Emean) / cls.Estd / 2
        )  # energy transformation

        event[..., 0] = (
            (event[..., 0] - cls.Xmean) / cls.Xstd / 2
        )  # x coordinate normalization
        event[..., 1] = (
            (event[..., 1] - cls.Ymean) / cls.Ystd / 2
        )  # y coordinate normalization
        event[..., 2] = (
            (event[..., 2] - cls.Zmean) / cls.Zstd / 2
        )  # z coordinate normalization


class CaloChallangeDataset(Dataset):
    Ymin, Ymax = -17, 17
    Xmin, Xmax = -17, 17
    Zmin, Zmax = 0, 44

    def __init__(self, file_path, cfg, bs=32):
        dataset = h5py.File(file_path, "r")

        # Get the indices and shuffle them
        if cfg.percentage < 1.0:
            tot_len = len(dataset["events"])
            size = int(tot_len * cfg.percentage)
            idx = np.random.choice(np.arange(0, tot_len), size=size, replace=False)
            idx = np.sort(idx).astype(int)
        else:
            cfg.percentage = 1.0
            idx = np.arange(0, len(dataset["events"])).astype(int)
            # idx = np.sort(idx).astype(int)

        self.dataset = {
            "events": dataset["events"][idx],
            "energy": dataset["energy"][idx],
        }
        self.bs = bs
        self.cfg = cfg

    def get_n_points(self, data, axis=-1):
        n_points_arr = (data[..., axis] != 0.0).sum(1)
        return n_points_arr

    @classmethod
    def normalize_xyze(cls, event):
        event[..., 0] = (event[..., 0] - cls.Xmin) * 2 / (
            cls.Xmax - cls.Xmin
        ) - 1  # x coordinate normalization
        event[..., 1] = (event[..., 1] - cls.Ymin) * 2 / (
            cls.Ymax - cls.Ymin
        ) - 1  # y coordinate normalization
        event[..., 2] = (event[..., 2] - cls.Zmin) * 2 / (
            cls.Zmax - cls.Zmin
        ) - 1  # z coordinate normalization

        # event[:, 3, :] = event[:, 3, :] # energy scale
        event[..., 3] = event[..., 3] / 1000  # energy scale
        # event[:, 3, :] = event[:, 3, :] / energy # energy scale E_depos/E_insident

        # ensure padding xyz is 0
        event[event[..., 3] == 0] = 0

    def __getitem__(self, idx):
        if idx > self.bs and idx < self.__len__() - self.bs:
            event = self.dataset["events"][
                idx - int(self.bs / 2) : idx + int(self.bs / 2)
            ].copy()
            energy = self.dataset["energy"][
                idx - int(self.bs / 2) : idx + int(self.bs / 2)
            ].copy()
        elif idx < self.bs:
            event = self.dataset["events"][idx : idx + self.bs].copy()
            energy = self.dataset["energy"][idx : idx + self.bs].copy()
        else:
            event = self.dataset["events"][idx - self.bs : idx].copy()
            energy = self.dataset["energy"][idx - self.bs : idx].copy()

        max_len = (event[:, -1, :] > 0).sum(axis=1).max()
        event = event[:, :, :max_len]

        event = event[:, [0, 1, 2, 3]]
        event = np.moveaxis(event, -1, -2)
        self.normalize_xyze(event)

        # nPoints
        points = self.get_n_points(event, axis=-1).reshape(-1, 1)

        if self.cfg.norm_cond:
            # TODO, this is sort of duplicated from conditioning.normalise_cond_feats
            energy = np.log((energy + 1e-5) / self.cfg.min_energy) / np.log(
                self.cfg.max_energy / self.cfg.min_energy
            )
            points = np.log((points + 1) / self.cfg.min_points) / np.log(
                self.cfg.max_points / self.cfg.min_points
            )

        return {"event": event, "energy": energy, "points": points}

    def __len__(self):
        return len(self.dataset["events"])


def dataset_class_from_config(config):
    """
    Return the correct dataset class based on the config.
    Ensuring that the data box in the metadata is set correctly.

    Parameters
    ----------
    config : config.Configs
        The config object with string attribute `dataset`
        that specifies the dataset.

    Returns
    -------
    correct_class : subclass of Dataset
        The correct dataset class to use for the dataset.

    """
    if config.dataset == "x36_grid" or config.dataset == "clustered":
        if "p_norm_local" in get_cond_features_names(config, "diffusion"):
            correct_class = PointCloudAngular
        else:
            correct_class = PointCloudDataset
    elif config.dataset == "getting_high":
        correct_class = PointCloudDatasetGH
    elif config.dataset == "calo_challenge":
        correct_class = CaloChallangeDataset
    else:
        raise ValueError(f"Don't know how to handle dataset {config.dataset}")
    # ensure the metadata is for the config supplied rather than the default
    correct_class.metadata = Metadata(config)
    return correct_class


if __name__ == "__main__":
    train_dset = PointCloudDataset(
        file_path="/beegfs/desy/user/akorol/data/calo-clouds/hdf5/high_granular_grid"
        + "/train/10-90GeV_x36_grid_regular_524k.hdf5",
        bs=32,
    )
