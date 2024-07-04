# get the folder above on the path
import os
import glob
import pytest
from pointcloud.config_varients import default, wish, configs_calotransf
from scripts.main import main


def customise_configs(configs, my_tmpdir):
    configs.log_comet = False
    test_dir = os.path.dirname(os.path.realpath(__file__))
    configs.dataset_path = os.path.join(test_dir, "mini_data_sample.hdf5")
    configs.max_iters = 2
    configs.logdir = my_tmpdir.mkdir("logs")
    configs.device = "cpu"
    configs.fit_attempts = 2


def test_calotranf(tmpdir):
    cfg = configs_calotransf.Configs()
    # no logging for tests, as we would need a comet key
    cfg.log_comet = False
    test_dir = os.path.dirname(os.path.realpath(__file__))
    cfg.dataset_path = os.path.join(test_dir, 'mini_data_sample.hdf5')
    cfg.max_iters = 2
    cfg.logdir = tmpdir.mkdir("logs")
    cfg.logdir_uda = tmpdir.mkdir("logs_uda")
    #cfg.logdir = "tmp/logs"
    #cfg.logdir_uda = "tmp/logs"
    cfg.device = 'cpu'
    # run from scratch
    # cfg.model_path = ""
    main(cfg)
    # assert glob.glob(f"{cfg.logdir_uda}/{cfg.name}*/ckpt_*.pt")
    # the calotransf needs to check that a recent checkpoint can be loaded.
    # cfg.model_path = "pointcloud/calotransfer/pretrained/ckpt_0.000000_2000000.pt"
    # shutil.copy(os.path.join(test_dir, cfg.model_path), cfg.logdir_uda)
    # print(os.listdir(cfg.logdir_uda))
    # main(cfg)
    # check the model ckpt was created
    # assert glob.glob(f"{cfg.logdir}/{cfg.name}*/ckpt_*.pt")



# The user warning is about the number of workers,
# but this number works well on our setup
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_main_default(tmpdir):
    # set a test config
    configs = default.Configs()
    customise_configs(configs, tmpdir)
    main(configs)
    # check the model ckpt was created
    assert glob.glob(f"{configs.logdir}/{configs.name}*/ckpt_*.pt")


def test_main_wish(tmpdir):
    # set a test config
    configs = wish.Configs()
    customise_configs(configs, tmpdir)
    main(configs)
    # check the model ckpt was created
    assert glob.glob(f"{configs.logdir}/{configs.name}*/ckpt_*.pt")
