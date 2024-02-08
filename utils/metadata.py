import os
import numpy as np

from configs import Configs


def get_metadata_folder(config=Configs()):
    """
    Local folder in which metadata is stored.
    """
    this_dir = os.path.dirname(__file__)
    dataset_filebase = os.path.basename(config.dataset_path).rsplit(".", 1)[0]

    subfolders = os.listdir(os.path.join(this_dir, "../metadata/"))

    if dataset_filebase in subfolders:
        folder = dataset_filebase
    else:
        raise NotImplementedError(
            f"Cannot find metadata for the dataset at {config.dataset_path}."
            + f" Datasets with known metadata: {subfolders}"
            + f" If you have the metadata for this dataset, please add in a subfolder of metadata/."
            + f" If this dataset is equivalent to another dataset, please make a symlink to the equivalent metadata."
        )
    data_dir = os.path.join(this_dir, "../metadata/", folder)
    return data_dir


class Metadata:
    def __init__(self, config=Configs()):
        self.config = config
        self.metadata_folder = get_metadata_folder(config)
        self._load_top_level()

    def _load_top_level(self):
        for file in os.listdir(self.metadata_folder):
            if not file.endswith(".npy"):
                continue
            content = np.load(
                os.path.join(self.metadata_folder, file), allow_pickle=True
            )
            if content.dtype == "O":
                for key, value in content.item().items():
                    assert not hasattr(self, key)
                    setattr(self, key, value)
            else:
                basename = os.path.basename(file)[: -len(".npy")]
                assert not hasattr(self, basename)
                setattr(self, basename, content)

    def load_muon_map(self):
        data_dir = os.path.join(self.metadata_folder, "muon_map")

        self.muon_map_X = np.load(data_dir + "/X.npy")
        self.muon_map_Z = np.load(data_dir + "/Z.npy")
        self.muon_map_Y = np.load(data_dir + "/Y.npy")
        self.muon_map_E = np.load(data_dir + "/E.npy")

        return self.muon_map_X, self.muon_map_Z, self.muon_map_Y, self.muon_map_E
