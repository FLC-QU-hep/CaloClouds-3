from pointcloud.config_varients import default


class Configs(default.Configs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.log_comet = True
        self.name = "CD_S2P_HDBScan_"
        self.storage_base = "/eos/user/m/mamozzan/"
        self.latent_dim = 0  # no latent flow in new calocloud
        self.dataset_path_in_storage = True
        self.storage_base = "/eos/user/m/mamozzan/"
        self._dataset_path = "/eos/user/m/mamozzan/step2point/outputs/cc3input_hdbscan_ms40_mcs40/input_cc3_file_{}.h5"
        self.metadata_folder = "/eos/user/m/mamozzan/CaloClouds-3/pointcloud/metadata/metadata_p22_th45-135_ph79-109_en5-130"
        self.n_dataset_files = 10
        self.Acomment = (
            "Running on the p22_th45-135_ph79-109_en5-130 dataset, first 10 files"
        )
        self._logdir = "CaloClouds-3/log_dir"

        self.workers = 5
        self.max_points = 20_000
        self.log_iter = 1000

        self.cond_features = 4  # number of conditioning features (i.e. energy+points=2)
        self.cond_features_names = ["energy", "p_norm_local"]
        self.distillation = True
        self.logarithmic_point_energy = True
        self.diffusion_pointwise_hidden_l1 = 32

        self.shower_flow_version = "alt1"  # options: ['original', 'alt1', 'alt2']
        self.shower_flow_cond_features = ["energy", "p_norm_local"]
        self.shower_flow_inputs = [
            "clusters_per_layer",
            "energy_per_layer",
        ]
        self.shower_flow_num_blocks = 2
        self.af_dim = 6
        self.shower_flow_fixed_input_norms = True

        self.process_kwargs(kwargs)

        self.cog_calibration = False
