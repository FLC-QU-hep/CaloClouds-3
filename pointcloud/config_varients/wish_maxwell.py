from pointcloud.config_varients import caloclouds_3


class Configs(caloclouds_3.Configs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.storage_base = "/eos/user/m/mamozzan/"
        self._logdir = "CaloClouds-3/log_dir"
        self.logdir_in_storage = True

        self.dataset_path_in_storage = False
        self._dataset_path = "/eos/user/m/mamozzan/step2point/outputs/cc3input_hdbscan/input_cc3_file_{}.h5"
        self.n_dataset_files = 4
        self.metadata_folder = "/eos/user/m/mamozzan/CaloClouds-3/pointcloud/metadata/metadata_p22_th45-135_ph79-109_en5-130"

        self.process_kwargs(kwargs)
