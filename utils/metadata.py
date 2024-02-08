from configs import Configs


def get_metadata_folder(config=Configs()):
    """
    Local folder in which metadata is stored.
    """
    this_dir = os.path.dirname(__file__)
    dataset_filebase = os.path.basename(config.dataset_path).rsplit(".", 1)[0]
    if dataset_filebase in [
        "10-90GeV_x36_grid_regular_524k",
        "10-90GeV_x36_grid_regular_524k_float32",
    ]:
        folder = "10-90GeV_x36_grid_regular_524k"
    else:
        raise NotImplementedError(
            f"Cannot recognise the dataset at {config.dataset_path}"
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
                    setattr(self, key, value)
            else:
                basename = os.path.basename(file)[: -len(".npy")]
                setattr(self, basename, content)

    def load_muon_map(self):
        data_dir = os.path.join(self.metadata_folder, "muon_map")

        self.muon_map_X = np.load(data_dir + "/X.npy")
        self.muon_map_Z = np.load(data_dir + "/Z.npy")
        self.muon_map_Y = np.load(data_dir + "/Y.npy")
        self.muon_map_E = np.load(data_dir + "/E.npy")

        return self.muon_map_X, self.muon_map_Z, self.muon_map_Y, self.muon_map_E
