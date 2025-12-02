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
        # only determines where the distillation model reads the teacher model from
        # model writing location is chosen automatically by the checkpoint manager
        # specify location from within the logdir
        self.model_path = "CD_2025_11_27__10_51_23/ckpt_0.000000_410000.pt"
