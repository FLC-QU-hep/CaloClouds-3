from config_varients import wish
import os


class Configs(wish.Configs):
    def __init__(self):
        super().__init__()

        self.log_comet = False
        self.comet_workspace = "none"
        self.dataset_path = "/beegfs/desy/user/akorol/data/calo-clouds/hdf5/high_granular_grid/train/10-90GeV_x36_grid_regular_524k_float32.hdf5"
        self.image_dir = "/beegfs/desy/user/dayhallh/point-cloud-diffusion-images"

        # Dataloader
        self.workers = 5
        self.max_points = 60
        self.val_freq = 10_000  # saving intervall for checkpoints

        self.test_freq = 30 * 1e3
        self.test_size = 400
        self.log_iter = 100  # log every n iterations, default: 100
