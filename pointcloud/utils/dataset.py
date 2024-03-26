from torch.utils.data import Dataset
import numpy as np
import h5py


class PointCloudDataset(Dataset):
    # these can be accessed without instantiating the class
    Ymin, Ymax = 0, 30
    Xmin, Xmax = -200, 200
    Zmin, Zmax = -160, 240
    energy_scale = 1000  # MeV to GeV
    def __init__(
        self,
        file_path,
        bs=32,
        max_ds_seq_len=6000,
        quantized_pos=True,
    ):
        self.dataset = h5py.File(file_path, "r")
        self.bs = bs
        self.max_ds_seq_len = max_ds_seq_len

        self.quantized_pos = quantized_pos
        self.offset = 5.0883331298828125 / 6  # size of x36 granular grid

        self._include_points = True
        self._roll_axis = True

        # avoid repeat calculation
        self._len = len(self.dataset["events"])

    def get_n_points(self, data, axis=-1):
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

    def __getitem__(self, idx):
        idxs = self._choose_idxs(idx)
        event = self.dataset["events"][idxs]
        energy = self.dataset["energy"][idxs]

        if self._roll_axis:
            event = np.moveaxis(event, -1, -2)

        # Trim padding
        max_len = (event[:, :, -1] > 0).sum(axis=1).max()
        event = event[:, self.max_ds_seq_len - max_len:]

        if not self.quantized_pos:
            self._fuzz(event)

        self.normalize_xyze(event)

        batch = {"event": event, "energy": energy}
        if self._include_points:
            # nPoints
            points = self.get_n_points(event, axis=-1).reshape(-1, 1)
            batch["points"] = points

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
    def __init__(self, file_path, bs=32, max_ds_seq_len=1700):
        self.dataset = h5py.File(file_path, "r")
        self.bs = bs
        self.max_ds_seq_len = max_ds_seq_len

        self.quantized_pos = True  # never fuzz the GH dataset
        # therefore we don't need an offset

        self._include_points = False  # no need to include points for GH dataset
        self._roll_axis = False  # no need to roll axis for GH dataset

        # avoid repeat calculation
        self._len = len(self.dataset["events"])

    def _fuzz(self, event):
        raise NotImplementedError("Why are you fuzzing the GH dataset?")


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
        file_path="/beegfs/desy/user/akorol/data/calo-clouds/hdf5/high_granular_grid/train/10-90GeV_x36_grid_regular_524k.hdf5",
        bs=32,
    )
