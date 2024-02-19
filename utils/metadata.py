import os
import numpy as np

from configs import Configs


def get_metadata_folder(config=Configs()):
    """
    Local folder in which metadata is stored.

    Parameters
    ----------
    config : Configs
        Configuration object, with a dataset_path attribute.
        The dataset_path attribute is used to determing the basename for the
        dataset, which is used to find the corresponding metadata folder.

    Returns
    -------
    data_dir : str
        Path to the metadata folder for the dataset.
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
        """
        Object that contains metadata for a dataset.
        Automatically constructed from what is found in the metadata folder.
        """
        self.config = config
        self.metadata_folder = get_metadata_folder(config)
        self._load_top_level()

    def _load_top_level(self):
        """
        Load data saved in the top level of the metadata folder
        in numpy files.

        Simple numpy arrays are loaded as attributes of this object
        of the same name as the file, with the .npy extension removed.

        Numpy files with pickled dictionaries are loaded with
        each key in the dictionary as an attribute of this object.
        """
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
        """
        Load the muon map data from the muon_map subfolder of the metadata folder.
        Only done on request, as it's slightly larger.

        Creates the attributes muon_map_X, muon_map_Y, muon_map_Z, muon_map_E
        and also returns them.
        """
        data_dir = os.path.join(self.metadata_folder, "muon_map")

        self.muon_map_X = np.load(data_dir + "/X.npy")
        self.muon_map_Y = np.load(data_dir + "/Y.npy")
        self.muon_map_Z = np.load(data_dir + "/Z.npy")
        self.muon_map_E = np.load(data_dir + "/E.npy")

        return self.muon_map_X, self.muon_map_Y, self.muon_map_Z, self.muon_map_E
