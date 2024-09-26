# get the folder above on the path
import os
import pytest
from scripts.ShowerFlow import main
from helpers import config_creator



# The user warning is about the number of workers,
# but this number works well on our setup
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_main_default(tmpdir):
    # set a test config
    configs = config_creator.make("default", my_tmpdir=tmpdir)
    data_dir = os.path.join(configs.storage_base, "point-cloud-diffusion-data")
    os.makedirs(data_dir, exist_ok=True)
    best_model_path, best_data_path, history_path = main(configs, total_epochs=1)
    assert os.path.exists(best_model_path)
    assert os.path.exists(best_data_path)
    assert os.path.exists(history_path)

