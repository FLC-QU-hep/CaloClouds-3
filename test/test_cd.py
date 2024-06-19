import os
import shutil
import pytest
from pointcloud.config_varients.default import Configs
from scripts.cd import main


# The user warning is about the number of workers, but this number works well on our setup
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_main(tmpdir):
    # set a test config
    cfg = Configs()
    # no logging for tests, as we would need a comet key
    cfg.log_comet = False
    test_dir = os.path.dirname(os.path.abspath(__file__))
    cfg.dataset_path = os.path.join(test_dir, 'mini_data_sample.hdf5')
    cfg.max_iters = 2
    cfg.logdir = tmpdir.mkdir("logs")
    cfg.model_path = "mini_ckpt_sample.pt"
    cfg.device = 'cpu'
    # actually put the model in the logdir
    shutil.copy(os.path.join(test_dir, cfg.model_path), cfg.logdir)
    main(cfg)


