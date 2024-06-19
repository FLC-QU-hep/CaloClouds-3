from pointcloud.config_varients.wish import Configs as WishConfigs


class Configs(WishConfigs):
    def __init__(self):
        super().__init__()
        data_name = "sim-photon-showers_10-90GeV_Zpos4"
        self.formatted_tree_base = f"/beegfs/desy/user/dayhallh/data/ILCsoftEvents/formatted_trees_{data_name}"
        self.anomaly_checkpoint = "/beegfs/desy/user/dayhallh/point-cloud-diffusion-data/autoencoder_checkpoints"
        self.storage_base = "/beegfs/desy/user/"
        self.dataset_path_in_storage = True
        self._dataset_path = f"dayhallh/data/ddsim_downsampled/{data_name}_validation_{{}}_all_steps.hdf5"
        
        self.n_dataset_files = 200
        self.Acomment = (
            "Anomaly detection for " + data_name
        )
        self._logdir = (
            f"dayhallh/point-cloud-diffusion-logs/{data_name}/"
        )
        self.device = "cpu"
