from pointcloud.config_varients import default


class Configs(default.Configs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.log_comet = False
        self.storage_base = "/data/dust/user/"
        self.latent_dim = 0  # no latent flow in new calocloud
        self.storage_base = "/data/dust/user/"
        self._dataset_path = "akorol/data/AngularShowers_RegularDetector/hdf5_for_CC/sim-E1261AT600AP180-180_file_{}slcio.hdf5"
        self.n_dataset_files = 88
        self.Acomment = "Running on the sim-E1261AT600AP180 dataset, first 10 files"
        self._logdir = "dayhallh/point-cloud-diffusion-logs"
        # self._dataset_path = 'dayhallh/data/ILCsoftEvents/p22_th90_ph90_en10-100_joined/p22_th90_ph90_en10-100_seed{}_all_steps.hdf5'
        # self.n_dataset_files = 10
        # self.Acomment = (
        #    "Running on the p22_th90_ph90_en10-100 dataset, first 10 files"
        # )
        # self._logdir = (
        #    "dayhallh/point-cloud-diffusion-logs/p22_th90_ph90_en10-100"
        # )

        self.workers = 5
        self.max_points = 10_000

        self.cond_features = 2  # number of conditioning features (i.e. energy+points=2)
        self.cond_features_names = ["energy", "points"]

        self.shower_flow_version = "original"  # options: ['original', 'alt1', 'alt2']
        self.shower_flow_cond_features = ["energy"]
        self.shower_flow_inputs = [
            "total_clusters",
            "total_energy",
            "cog_x",
            "cog_y",
            "cog_z",
            "clusters_per_layer",
            "energy_per_layer",
        ]
        self.shower_flow_fixed_input_norms = False
        self.process_kwargs(kwargs)

        # only use if projecting into a detector
        self.shower_flow_n_scaling = False
        #self.shower_flow_coef_real = [
        #    2.42091454e-09,
        #    -2.72191705e-05,
        #    2.95613817e-01,
        #    4.88328360e01,
        #]
        #self.shower_flow_coef_fake = [
        #    -9.02997505e-07,
        #    2.82747963e-03,
        #    1.01417267e00,
        #    1.64829018e02,
        #]
