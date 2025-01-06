from pointcloud.config_varients import default


class Configs(default.Configs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.device = "cuda"
        self.log_comet = False
        self.latent_dim = 0  # no latent flow in new calocloud
        self.storage_base = "/data/dust/user/"
        #self._dataset_path = 'dayhallh/data/ILCsoftEvents/p22_th90_ph90_en10-100_joined/p22_th90_ph90_en10-100_seed{}_all_steps.hdf5'
        self._dataset_path = "akorol/data/AngularShowers_RegularDetector/hdf5_for_CC/sim-E1261AT600AP180-180_file_{}slcio.hdf5"
        #self._dataset_path = "akorol/data/CaloClouds/hdf5/high_granular_grid/train/10-90GeV_x36_grid_regular_524k_float32.hdf5"
        self.n_dataset_files = 1 #88
        self.Acomment = (
            "Running on the calochallange dataset, first files"
        )
        self._logdir = (
            "dayhallh/point-cloud-diffusion-logs/investigation2/"
        )

        self.workers = 5
        self.max_points = 6_000
        self.val_freq = 1e4  #  1e3          # saving intervall for checkpoints

        #self.cond_features = ["energy" ] #, "points"]
        self.cond_features = ["energy", "p_norm_local"]
        self.shower_flow_cond_features = ["energy", "p_norm_local"]
        #self.shower_flow_cond_features = ["energy"]
        self.shower_flow_fixed_input_norms = True
        #self.shower_flow_fixed_input_norms = False
        self.shower_flow_data_dir = "/data/dust/user/dayhallh/point-cloud-diffusion-data/investigation2/"
        self.shower_flow_version="original"
        self.shower_flow_num_blocks=4

        self.process_kwargs(kwargs)
