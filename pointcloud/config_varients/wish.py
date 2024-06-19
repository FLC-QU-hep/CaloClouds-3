from pointcloud.config_varients import default
import os

class Configs(default.Configs):
    def __init__(self):
        super().__init__()
        
    # Experiment Name
        self.name = 'wish_'  # options: [TEST_, kCaloClouds_, CaloClouds_, CD_]
        self.comet_project = 'wish'   # options: ['k-CaloClouds', 'calo-consistency']
        self.Acomment = 'test of data properties'  # log_iter 100
        self.log_comet = False
        self.comet_workspace = 'henrydayhall'

    # Model arguments
        self.model_name = 'wish'             # choices=['flow', 'AllCond_epicVAE_nFlow_PointDiff', 'epicVAE_nFlow_kDiffusion]
        self.poly_degree = 3
        self.fit_attempts = 100

    # Data
        self.storage_base = '/home/dayhallh/training'
        #self.dataset_path_in_storage = True
        #self._dataset_path = 'akorol/data/calo-clouds/hdf5/high_granular_grid/train/10-90GeV_x36_grid_regular_524k_float32.hdf5'
        self.dataset_path_in_storage = False
        #self._dataset_path = '/home/henry/training/local_data/varied_20_sample.hdf5'
        self._dataset_path = '/home/dayhallh/Data/p22_th90_ph90_en10-100_joined/p22_th90_ph90_en10-100_seed{}_all_steps.hdf5'
        self.n_dataset_files = 10

        self.image_dir = os.environ["HOME"] + '/training/point-cloud-diffusion-images'

    # Dataloader
        self.workers = 1
        #self.train_bs = 64      # k-diffusion: 128 / CD: 256
        self.train_bs = 200      # k-diffusion: 128 / CD: 256
        self.shuffle = True             # choices=[True, False]
        self.max_points = 6_000
        

    # Optimizer and scheduler
        #self.lr = 2e-4              # Caloclouds default: 2e-3, consistency model paper: approx. 1e-5
        self.lr = 1

    # Others
        self.device = 'cpu'
        self.logdir_in_storage = False
        self._logdir = os.environ["HOME"] + '/training/point-cloud-diffusion-logs/wish'
        self.val_freq =  10  #  10_000       # saving intervall for checkpoints

        self.test_freq = 4  # 30 * 1e3   
        self.test_size = 4  # 400
        self.log_iter = 10   # log every n iterations, default: 100
    # for other models
        self.latent_dim = 0

    # for anomaly detection
        self.formatted_tree_base = os.path.join(self.storage_base, "formatted_trees")
        self.anomaly_checkpoint = os.path.join(self.storage_base, "autoencoder_checkpoints")
        self.anomaly_hidden_dim = 8

