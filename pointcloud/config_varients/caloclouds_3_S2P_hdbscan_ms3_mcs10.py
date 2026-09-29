from pointcloud.config_varients import default


class Configs(default.Configs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.log_comet = True
        self.name = "CD_S2P_HDBScan_ms3_"
        self.storage_base = "/eos/user/m/mamozzan/"
        self.latent_dim = 0  # no latent flow in new calocloud
        self.dataset_path_in_storage = True
        self._dataset_path = "/eos/project/f/fast/input_cc3/cc3input_hdbscan_ms3_mcs10/input_cc3_file_{}.h5"
        self.metadata_folder = "/eos/user/m/mamozzan/CaloClouds-3/pointcloud/metadata/metadata_p22_th45-135_ph79-109_en5-130"
        self.n_dataset_files = 22
        self.Acomment = (
            "Running on the p22_th45-135_ph79-109_en5-130 dataset, first 10 files"
        )
        self._logdir = "CaloClouds-3/log_dir"

        self.workers = 5
        self.max_points = 4_000  # dataset max 3,823 pts (measured over all 23 files)
        self.log_iter = 1000

        self.cond_features = 4  # number of conditioning features (i.e. energy+points=2)
        self.cond_features_names = ["energy", "p_norm_local"]
        self.distillation = True
        self.logarithmic_point_energy = True
        self.diffusion_pointwise_hidden_l1 = 32

        # ep1500 search trial 10 (best, combined Wasserstein = 13.12).
        # "log1_stable" is registered at runtime by
        # showerflow_utils.ensure_version_registered (pointcloud/models/stable_log1.py).
        self.shower_flow_version = "log1_stable"  # options: ['original', 'alt1', 'alt2', 'log1', 'log1_stable'*]
        self.shower_flow_cond_features = ["energy", "p_norm_local"]
        self.shower_flow_inputs = [
            "clusters_per_layer",
            "energy_per_layer",
        ]
        self.shower_flow_num_blocks = 4
        self.af_dim = 14
        self.shower_flow_fixed_input_norms = True

        self.process_kwargs(kwargs)

        self.cog_calibration = False
