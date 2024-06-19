from pointcloud.config_varients import wish
import os


class Configs(wish.Configs):
    def __init__(self):
        super().__init__()

        self.log_comet = False
        self.comet_workspace = "none"
        self.storage_base = "/beegfs/desy/user/"
        self.dataset_path_in_storage = True
        #self._dataset_path = 'akorol/data/calo-clouds/hdf5/high_granular_grid/train/10-90GeV_x36_grid_regular_524k_float32.hdf5'
        self._dataset_path = 'dayhallh/data/ILCsoftEvents/p22_th90_ph90_en10-100_joined/p22_th90_ph90_en10-100_seed{}_all_steps.hdf5'
        self.n_dataset_files = 10

        # Dataloader
        self.workers = 5
        self.max_points = 60_000
        self.val_freq = 10_000  # saving intervall for checkpoints

        self.test_freq = 30 * 1e3
        self.test_size = 400
        self.log_iter = 100  # log every n iterations, default: 100

        data_path = "dayhallh/point-cloud-diffusion-data/"
        self.formatted_tree_base = os.path.join(self.storage_base, data_path, "formatted_trees")
        self.anomaly_checkpoint = os.path.join(self.storage_base, data_path, "autoencoder_checkpoints")
