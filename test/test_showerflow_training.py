"""
Test the showerflow_training module.
"""
import os
import numpy as np
from helpers import config_creator


from pointcloud.utils import showerflow_training


def test_get_to_factory(tmpdir):
    # bundle the tests for the gets and the function factory
    # together as the function factory needs the output of the gets
    configs = config_creator.make(my_tmpdir=tmpdir)
    showerflow_dir = os.path.join(str(tmpdir), "showerflow")
    os.makedirs(showerflow_dir)

    # start tests with the get functions
    pointsE_path = showerflow_training.get_incident_npts_visible(
        configs, showerflow_dir
    )
    assert os.path.exists(pointsE_path)
    # get the file's tiem stamp
    pointsE_time = os.path.getmtime(pointsE_path)
    # doing it again should not cause it to be recreated
    pointsE_path = showerflow_training.get_incident_npts_visible(
        configs, showerflow_dir
    )
    new_pointsE_time = os.path.getmtime(pointsE_path)
    assert pointsE_time == new_pointsE_time
    loaded = np.load(pointsE_path)
    n_events = len(loaded["energy"])
    assert np.all(loaded["energy"] > 0)
    assert np.all(loaded["visible_energy"] >= 0)
    assert np.all(loaded["visible_energy"] <= loaded["energy"])
    assert loaded["visible_energy"].shape == (n_events,)
    assert np.all(loaded["num_points"] >= 0)
    assert loaded["num_points"].shape == (n_events,)

    direction_path = showerflow_training.get_gun_direction(configs, showerflow_dir)
    assert os.path.exists(direction_path)
    # do it again to check it is not recreated
    time = os.path.getmtime(direction_path)
    direction_path = showerflow_training.get_gun_direction(configs, showerflow_dir)
    assert time == os.path.getmtime(direction_path)

    loaded = np.load(direction_path)
    assert loaded.shape == (n_events, 3)

    clusters_path = showerflow_training.get_clusters_per_layer(configs, showerflow_dir)
    assert os.path.exists(clusters_path)
    # do it again to check it is not recreated
    time = os.path.getmtime(clusters_path)
    clusters_path = showerflow_training.get_clusters_per_layer(configs, showerflow_dir)
    assert time == os.path.getmtime(clusters_path)

    loaded = np.load(clusters_path)
    assert loaded["clusters_per_layer"].shape == (n_events, 30)
    assert np.all(loaded["clusters_per_layer"] >= 0)
    assert loaded["rescaled_clusters_per_layer"].shape == (n_events, 30)
    assert np.all(loaded["rescaled_clusters_per_layer"] >= 0)

    energy_path = showerflow_training.get_energy_per_layer(configs, showerflow_dir)
    # do it again to check it is not recreated
    time = os.path.getmtime(energy_path)
    energy_path = showerflow_training.get_energy_per_layer(configs, showerflow_dir)
    assert time == os.path.getmtime(energy_path)
    assert os.path.exists(energy_path)
    loaded = np.load(energy_path)
    assert loaded["energy_per_layer"].shape == (n_events, 30)
    assert np.all(loaded["energy_per_layer"] >= 0)
    assert loaded["rescaled_energy_per_layer"].shape == (n_events, 30)
    assert np.all(loaded["rescaled_energy_per_layer"] >= 0)

    cog_path, cog_sample = showerflow_training.get_cog(
        configs, showerflow_dir
    )
    assert os.path.exists(cog_path)
    # do it again to check it is not recreated
    time = os.path.getmtime(cog_path)
    cog_path, cog_sample = showerflow_training.get_cog(
        configs, showerflow_dir
    )
    assert time == os.path.getmtime(cog_path)

    loaded = np.load(cog_path)
    assert loaded.shape == (n_events, 3)
    assert len(cog_sample) == 3
    # check the cog sample is in the loaded data
    for row in zip(*cog_sample):
        row = np.array(row)
        distances = np.sum(np.abs(loaded - row), axis=1)
        assert np.any(distances < 1e-4)

    # finally, we have all the data created, so we can test the factory
    factory = showerflow_training.train_ds_function_factory(
        pointsE_path, cog_path, clusters_path, energy_path, configs
    )
    assert callable(factory)
    dataset = factory(0, 3)
    item_0 = dataset[0]
    assert len(item_0) == 66
