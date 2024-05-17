# get the folder above on the path
import os
import glob
import pytest
from pointcloud.config_varients.default import Configs
from scripts.main import main


def test_main(tmpdir):
#def test_main():
    # set a test config
    cfg = Configs()
    # no logging for tests, as we would need a comet key
    cfg.log_comet = False
    test_dir = os.path.dirname(os.path.realpath(__file__))
    cfg.dataset_path = os.path.join(test_dir, 'mini_data_sample.hdf5')
    cfg.max_iters = 2
    cfg.logdir = tmpdir.mkdir("logs")
    #cfg.logdir = "tmp/logs"
    cfg.device = 'cpu'
    main(cfg)
    # check the model ckpt was created
    assert glob.glob(f"{cfg.logdir}/{cfg.name}*/ckpt_*.pt")


