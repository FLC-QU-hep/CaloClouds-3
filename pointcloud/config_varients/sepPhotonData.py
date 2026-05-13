import os

from pointcloud.config_varients import caloclouds_3

user_name = os.getenv("USER", os.getenv("USERNAME", ""))


class Configs(caloclouds_3.Configs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.Acomment = "Test run"
        self.device = "cpu"
        self.log_comet = False
        self.storage_base = "/eos/user/m/mamozzan/"
        # inside the storage_base
        self._logdir = "log_dir/"
        # should eht storage_base be prepended to the dataset path?
        self.dataset_path_in_storage = False
        # dataset path (without the first part, if dataset_path_in_storage is True)
        self.n_dataset_files = 1
        self._dataset_path = "/eos/user/m/mamozzan/step2point/outputs/cc3input_withincell/input_cc3_file_{}.h5"
        # path to stor files assocated eith the shower flow
        self.shower_flow_data_dir = self.storage_base + "shower_flow_data/"
        # only determines where the distillation model reads the teacher model from
        # model writing location is chosen automatically by the checkpoint manager
        # specify location from within the logdir
        self.model_path = ""
        self.metadata_folder = "metadata_p22_th45-135_ph79-109_en5-130"
