from pointcloud.config_varients.wish import Configs as WishConfigs
import os


class Configs(WishConfigs):
    def __init__(self):
        super().__init__()
        self.formatted_tree_base = os.path.join(
            os.environ["HOME"],
            'training',
            'point-cloud-diffusion-data',
            "formatted_trees_p22_r0_th90_ph90_en10-90_downsampled"
        )
        self.dataset_path = os.path.join(
            os.environ["HOME"],
            'training',
            "data_production/ILCsoftEvents/p22_r0_th90_ph90_en10-90_downsampled.h5",
        )
        self.anomaly_checkpoint = os.path.join(
            os.environ["HOME"],
            'training',
            'point-cloud-diffusion-logs',
            'autoencoder_checkpoints'
        )
