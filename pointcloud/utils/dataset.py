from torch.utils.data import Dataset
import numpy as np
import h5py


class PointCloudDataset(Dataset):
    # these can be accessed without instantiating the class
    Ymin, Ymax = 0, 30
    Xmin, Xmax = -200, 200
    Zmin, Zmax = -160, 240
    energy_scale = 1000  # MeV to GeV
    # The keys that will be draw from the dataset
    # format is {"name in batch": "name on disk"}
    keys_to_include = {"event": "events", "energy": "energy", "points": None}

    def __init__(
        self, file_path, bs=32, max_ds_seq_len=6000, quantized_pos=True, n_files=0
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
        self.index_list = self._make_index_list()
        self.bs = bs
        self.max_ds_seq_len = max_ds_seq_len

        self.quantized_pos = quantized_pos
        self.offset = 5.0883331298828125 / 6  # size of x36 granular grid

        self._roll_axis = True

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
        if n_files == 0:
            return [h5py.File(file_path, "r")]
        else:
            return [h5py.File(file_path.format(i), "r") for i in range(n_files)]

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
        for file_idx, dataset in enumerate(self.open_files):
            if "n_points" in dataset:
                n_points = dataset["n_points"][:]
            else:
                n_points = self.get_n_points(dataset["events"])
            index_list += [(n, file_idx, i) for i, n in enumerate(n_points)]
        # sort the index list by 'n_points'
        index_list.sort(key=lambda x: x[0])
        index_list = np.array(index_list, dtype=int)
        return index_list

    def get_n_points(self, data, axis=-1):
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

    def _fuzz(self, event):
        pos_offset_x = np.random.uniform(0, self.offset, 1)
        pos_offset_z = np.random.uniform(0, self.offset, 1)
        event[:, :, 0] = event[:, :, 0] + pos_offset_x
        event[:, :, 2] = event[:, :, 2] + pos_offset_z

    def normalize_xyze(self, event):
        event[:, :, 0] = (
            ((event[:, :, 0] - self.Xmin) * 2) / (self.Xmax - self.Xmin)
        ) - 1  # x coordinate normalization
        event[:, :, 1] = (
            ((event[:, :, 1] - self.Ymin) * 2) / (self.Ymax - self.Ymin)
        ) - 1  # y coordinate normalization
        event[:, :, 2] = (
            ((event[:, :, 2] - self.Zmin) * 2) / (self.Zmax - self.Zmin)
        ) - 1  # z coordinate normalization
        event[:, :, 3] = event[:, :, 3] * self.energy_scale
        # ensure padding xyz is 0
        event[event[:, :, -1] == 0] = 0

    def _event_processing(self, event):
        if self._roll_axis:
            event = np.moveaxis(event, -1, -2)

        # Trim padding
        max_len = (event[:, :, -1] > 0).sum(axis=1).max()
        event = event[:, self.max_ds_seq_len - max_len:]

        if not self.quantized_pos:
            self._fuzz(event)

        self.normalize_xyze(event)
        return event

    def __getitem__(self, idx):
        idxs = self._choose_idxs(idx)
        batch = {}
        for name_in_batch, name_on_disk in self.keys_to_include.items():
            if name_in_batch == "points":
                continue
            data = np.array(
                [
                    self.open_files[file_n][name_on_disk][event_n]
                    for n_pts, file_n, event_n in self.index_list[idxs]
                ]
            )
            if name_in_batch == "event":
                data = self._event_processing(data)
            batch[name_in_batch] = data

        # special case for points as it's already been processed
        if "points" in self.keys_to_include:
            batch["points"] = self.index_list[idxs, [0]]

        return batch

    def __len__(self):
        return self._len


class PointCloudDatasetUnordered(PointCloudDataset):
    def _choose_idxs(self, idx):
        rng = np.random.default_rng(seed=idx)
        idxs = rng.choice(self._len, self.bs, replace=False)
        idxs.sort()
        return idxs


class PointCloudDatasetGH(PointCloudDataset):
    Ymin, Ymax = 0, 30
    Xmin, Xmax = 0, 30
    Zmin, Zmax = 0, 30
    energy_scale = 1  # no conversion needed for GH dataset

    keys_to_include = {"event": "events", "energy": "energy"}

    def __init__(self, file_path, bs=32, max_ds_seq_len=1700, n_files=0):
        self.open_files = self._open_data_files(file_path, n_files)
        self.index_list = self._make_index_list()
        self.bs = bs
        self.max_ds_seq_len = max_ds_seq_len

        self.quantized_pos = True  # never fuzz the GH dataset
        # therefore we don't need an offset

        self._roll_axis = False  # no need to roll axis for GH dataset

        # avoid repeat calculation
        self._len = len(self.index_list)

    def _fuzz(self, event):
        raise NotImplementedError("Why are you fuzzing the GH dataset?")


class PointCloudAngular(PointCloudDataset):
    Ymin, Ymax = -250, 250
    Xmin, Xmax = -250, 250
    Zmin, Zmax = 0, 30
    keys_to_include = {"event": "events", "energy": "energy",
                       "p_norm_local": "p_norm_local"}


def dataset_class_from_config(config):
    """
    Return the correct dataset class based on the config.

    Parameters
    ----------
    config : configs.Configs
        The config object with string attribute `dataset`
        that specifies the dataset.

    Returns
    -------
    correct_class : subclass of Dataset
        The correct dataset class to use for the dataset.

    """
    if config.dataset == "x36_grid" or config.dataset == "clustered":
        correct_class = PointCloudDataset
    elif config.dataset == "gettig_high":
        correct_class = PointCloudDatasetGH
    else:
        raise ValueError(f"Don't know how to handle dataset {config.dataset}")
    return correct_class


if __name__ == "__main__":
    train_dset = PointCloudDataset(
        file_path="/beegfs/desy/user/akorol/data/calo-clouds/hdf5/high_granular_grid"
        + "/train/10-90GeV_x36_grid_regular_524k.hdf5",
        bs=32,
    )
