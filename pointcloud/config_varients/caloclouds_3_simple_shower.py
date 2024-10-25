from pointcloud.config_varients import default


class Configs(default.Configs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.log_comet = False
        self.storage_base = "/gpfs/dust/maxwell/user/"
        self.latent_dim = 0  # no latent flow in new calocloud
        self.dataset_path_in_storage = False
        self.storage_base = "/gpfs/dust/maxwell/user/"
        self._dataset_path = "/beegfs/desy/user/akorol/data/AngularShowers_RegularDetector/hdf5_for_CC/sim-E1261AT600AP180-180_file_{}slcio.hdf5"
        self.n_dataset_files = 88
        self.Acomment = (
            "Running on the sim-E1261AT600AP180 dataset, first 10 files"
        )
        self._logdir = (
            "dayhallh/point-cloud-diffusion-logs"
        )

        self.workers = 5
        self.max_points = 6_000

        self.shower_flow_version = 'alt1'  # options: ['original', 'alt1', 'alt2']
        self.shower_flow_cond_features = ["energy", "p_norm_local"]
        self.shower_flow_inputs = [
                "cog_x",
                "cog_y",
                "clusters_per_layer",
                "energy_per_layer",
                ]
        self.shower_flow_num_blocks = 2
        self.shower_flow_fixed_input_norms = True

        self.process_kwargs(kwargs)
