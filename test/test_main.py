# get the folder above on the path
import sys
from pathlib import Path
path_root1 = Path(__file__).parents[1]
sys.path.append(str(path_root1))

import os
import pytest
from configs import Configs
from main import main


def test_main(tmpdir):
    # set a test config
    cfg = Configs()
    # no logging for tests, as we would need a comet key
    cfg.log_comet = False
    cfg.dataset_path = os.path.join(path_root1, 'test', 'mini_data_sample.hdf5')
    cfg.max_iters = 2
    cfg.logdir = tmpdir.mkdir("logs")
    main(cfg)


