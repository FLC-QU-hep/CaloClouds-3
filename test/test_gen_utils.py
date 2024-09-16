"""
Test the gen_utils (generator utilities) module.
"""
import numpy as np
import torch
from numpy import testing as npt
from pointcloud.utils import gen_utils, metadata

from pointcloud.models import wish
from pointcloud.evaluation import generate
from pointcloud.models.shower_flow import compile_HybridTanH_model
from pointcloud.utils.stats_accumulator import HighLevelStats

from helpers import sample_accumulator, config_creator


def test_get_cog():
    # with just one particle, the cog is always at the particle's position
    found = gen_utils.get_cog(np.zeros(1), np.zeros(1), np.zeros(1), np.ones(1))
    npt.assert_allclose(found, np.zeros(3))
    found = gen_utils.get_cog(np.ones(1), np.ones(1), np.ones(1), np.ones(1))
    npt.assert_allclose(found, np.ones(3))
    found = gen_utils.get_cog(np.ones(1), -np.ones(1), -np.ones(1), np.ones(1))
    npt.assert_allclose(found, [1, -1, -1])

    # with two particles, the cog is at the weighted center
    found = gen_utils.get_cog(np.zeros(2), np.zeros(2), np.zeros(2), np.ones(2))
    npt.assert_allclose(found, np.zeros(3))
    found = gen_utils.get_cog(np.arange(2), -np.arange(2), np.arange(2), np.ones(2))
    npt.assert_allclose(found, [0.5, -0.5, 0.5])
    found = gen_utils.get_cog(
        np.arange(2), -np.arange(2), np.arange(2), np.arange(1, 3)
    )
    npt.assert_allclose(found, [2 / 3, -2 / 3, 2 / 3])


# TODO test get_scale_factor, but I don't really know what is expected of it.


def test_cond_batcher():
    # don't choke on empty input
    for batch in gen_utils.cond_batcher(torch.ones(0), 1):
        pass
    # sort the input
    data = torch.tensor([[3, 7], [1, 2], [5, 8], [4, 6]])
    expected = np.array([[[1, 2], [4, 6]], [[3, 7], [5, 8]]])
    for i, batch in enumerate(gen_utils.cond_batcher(data, 2)):
        npt.assert_allclose(batch, expected[i])


def test_truescale_showerflow_output():
    config = config_creator.make()
    meta = metadata.Metadata(config)
    mean_cog_x, mean_cog_y, _ = meta.mean_cog
    std_cog_x, std_cog_y, _ = meta.std_cog
    # showerflow output is an N by 65 element arrat
    # start with an array full of zeros
    sample = np.zeros((3, 65))
    (
        num_clusters,
        energies,
        cog_x,
        cog_y,
        clusters_per_layer,
        e_per_layer,
    ) = gen_utils.truescale_showerflow_output(sample, config)
    npt.assert_allclose(num_clusters, np.ones((3, 1)))
    npt.assert_allclose(energies, 40 * np.ones((3, 1)))  # clip to a min energy
    npt.assert_allclose(cog_x, mean_cog_x)
    npt.assert_allclose(cog_y, mean_cog_y)
    npt.assert_allclose(clusters_per_layer, 0)
    npt.assert_allclose(e_per_layer, 0)

    # now try a random array with a large number of clusters
    sample = np.random.rand(100, 65)
    sample[:, [2, 3, 4]] -= 0.5
    max_points = config.max_points
    sample[:, [0, 1]] *= max_points * 10
    sample[:, 5:] *= max_points * 10
    (
        num_clusters,
        energies,
        cog_x,
        cog_y,
        clusters_per_layer,
        e_per_layer,
    ) = gen_utils.truescale_showerflow_output(sample, config)
    assert np.all(num_clusters >= 1)
    assert np.all(num_clusters <= max_points)
    assert np.all(energies >= 0)
    npt.assert_allclose(np.mean(cog_x), mean_cog_x, atol=std_cog_x * 3)
    npt.assert_allclose(np.mean(cog_y), mean_cog_y, atol=std_cog_y * 3)
    assert np.all(clusters_per_layer >= 0)
    assert np.all(clusters_per_layer <= 1)
    assert np.all(e_per_layer >= 0)
    assert np.all(e_per_layer <= 1)


class TestGenMethods:
    configs = []
    models = []
    model_names = {}

    @classmethod
    def setup_class(cls):
        """
        Make a list of models to test.
        """
        # make ourselves a simple model
        configs = config_creator.make("wish")
        wish_model = wish.Wish(configs)

        # give it some reasonable parameters
        acc = sample_accumulator.make(add_varients=True)
        hls = HighLevelStats(acc, wish_model.poly_degree)
        wish_model.set_from_stats(hls)

        cls.configs.append(configs)
        cls.models.append((wish_model,))
        cls.model_names["wish"] = 0

        # now make a caloclouds/showerflow model
        configs = config_creator.make()
        params_dict = generate.make_params_dict()
        # make it short for testing
        params_dict["n_events"] = 10
        params_dict["batch_size"] = 2

        # fake the flow model
        flow, distribution = compile_HybridTanH_model(
            num_blocks=configs.shower_flow_num_blocks,
            num_inputs=65,
            num_cond_inputs=1,
            device=configs.device,
        )

        diff_model, coef_real, coef_fake, n_splines = generate.load_diffusion_model(
            configs, "cm", model_path="test/example_cm_model.pt"
        )

        cls.configs.append(configs)
        cls.models.append((diff_model, distribution))
        cls.model_names["diffusion"] = 1

    @classmethod
    def teardown_class(cls):
        cls.configs.clear()
        cls.models.clear()

    def test_get_shower(self):
        num_points = 100
        bs = 1
        energy = 50
        cond_N = 50
        model = self.models[self.model_names["diffusion"]][0]
        for bs in [0, 1, 2]:
            found = gen_utils.get_shower(model, num_points, energy, cond_N, bs)
            assert found.shape == (bs, num_points, 4)
            # there is actually nothing at this stage to prevet the model
            # from producing unphysical values. It's a raw diffusion model.

    def test_gen_showers_batch(self):
        num = 6
        bs = 2
        e_min = 40
        e_max = 100
        # shoud work for all model/config pairs
        for config, models in zip(self.configs, self.models):
            found, found_cond = gen_utils.gen_showers_batch(
                *models, e_min, e_max, num, bs, config
            )
            assert found.shape == (num, config.max_points, 4)
            assert found_cond.shape == (num, 1)
            assert (found[:, :, 3] >= 0).all()

    def test_gen_cond_showers_batch(self):
        cond = (torch.arange(10).reshape(-1, 1) * 10).float()
        # cond as a double
        for config, models in zip(self.configs, self.models):
            found = gen_utils.gen_cond_showers_batch(*models, cond, config=config)
            assert found.shape == (10, config.max_points, 4)
            assert np.all(found[:, :, 3] >= 0)

    def test_gen_wish_inner_batch(self):
        model = self.models[self.model_names["wish"]][0]
        config = self.configs[self.model_names["wish"]]
        destination_array = np.zeros((10, config.max_points, 4))
        cond = (torch.arange(1, 3).reshape(-1, 1) * 10).float()
        first_index = 2
        gen_utils.gen_wish_inner_batch(
            cond, destination_array, first_index, model
        )
        assert np.all(destination_array[:2] == 0)
        assert np.all(destination_array[4:] == 0)
        assert np.all(destination_array[2:4, :, 3] >= 0)

    def test_gen_v1_inner_batch(self):
        model, shower_flow = self.models[self.model_names["diffusion"]]
        config = self.configs[self.model_names["diffusion"]]
        destination_array = np.zeros((10, config.max_points, 4))
        first_index = 2
        cond = (torch.arange(1, 3).reshape(-1, 1) * 10).float()
        gen_utils.gen_v1_inner_batch(
            cond, destination_array, first_index, model, shower_flow, config
        )
        assert np.all(destination_array[:2] == 0)
        assert np.all(destination_array[4:] == 0)
        assert np.all(destination_array[2:4, :, 3] >= 0)
