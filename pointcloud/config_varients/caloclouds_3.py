from pointcloud.config_varients import default


class Configs(default.Configs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.log_comet = False
        self.storage_base = "/gpfs/dust/maxwell/user/"
        self.latent_dim = 0  # no latent flow in new calocloud
        self.storage_base = "/gpfs/dust/maxwell/user/"
        self._dataset_path = 'dayhallh/data/ILCsoftEvents/p22_th90_ph90_en10-100_joined/p22_th90_ph90_en10-100_seed{}_all_steps.hdf5'
        self.n_dataset_files = 10
        self.Acomment = (
            "Running on the p22_th90_ph90_en10-100 dataset, first 10 files"
        )
        self._logdir = (
            "dayhallh/point-cloud-diffusion-logs/p22_th90_ph90_en10-100"
        )

        self.workers = 5
        self.max_points = 6_000
        self.process_kwargs(kwargs)
