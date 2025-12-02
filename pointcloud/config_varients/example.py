from pointcloud.config_varients import caloclouds_3


class Configs(caloclouds_3.Configs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.Acomment = "Test run"
        self.device = "cpu"
        self.log_comet = False
        self.storage_base = "/data/dust/user/"
        # inside the storage_base
        self._logdir = "dayhallh/point-cloud-diffusion-runtest"
        self.dataset_path_in_storage = True
        self._dataset_path = "dayhallh/point-cloud-diffusion-runtest/p22_th45-135_ph79-109_en10-127_seed0_ip.hdf5"
