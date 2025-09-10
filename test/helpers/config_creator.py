"""
Some common setting for testing config, regardless of the underlying config version.
"""
import os
import shutil
from . import example_paths
from pointcloud.config_varients import default, caloclouds_2, caloclouds_3

dict_of_config = {
    "default": default,
    "caloclouds_2": caloclouds_2,
    "caloclouds_3": caloclouds_3,
}


def make(module_name="default", start_from=None, my_tmpdir=None):
    if start_from is not None:
        config = start_from
    else:
        config = dict_of_config[module_name].Configs()
    # no logging for tests, as we would need a comet key
    config.log_comet = False
    config.max_iters = 2
    config.device = "cpu"
    config.fit_attempts = 2
    if my_tmpdir is not None:
        my_tmpdir = str(my_tmpdir)
        config.storage_base = my_tmpdir
        config.logdir = os.path.join(my_tmpdir, "logs")
        os.makedirs(config.logdir, exist_ok=True)
        config.logdir_uda = os.path.join(my_tmpdir, "logs_uda")
        os.makedirs(config.logdir_uda, exist_ok=True)
        model_path = example_paths.mini_ckpt_sample
        # actually put the model in the logdir
        shutil.copy(model_path, config.logdir)
        # and in the uda logdir
        shutil.copy(model_path, config.logdir_uda)
        config.model_path = os.path.basename(model_path)
    config.dataset_path_in_storage = False
    config.dataset_path = example_paths.mini_data_sample
    return config
