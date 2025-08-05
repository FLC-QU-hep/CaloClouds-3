"""
Tests for our custom optimisers
"""
import numpy as np
from numpy import testing as npt
from pointcloud.utils import optimisers


def simple_callable(x):
    return x**2


def callable_with_args(x, a, b):
    return a * x + b


def test_get_p0_n_bounds():
    # try with minimal information
    (
        found_p0,
        found_n,
        found_lower_bound,
        found_upper_bound,
    ) = optimisers.get_p0_n_bounds(simple_callable, [])
    npt.assert_allclose(found_p0, 1)
    assert found_n == 0
    npt.assert_allclose(found_lower_bound, [])
    npt.assert_allclose(found_upper_bound, [])

    # try with full information
    (
        found_p0,
        found_n,
        found_lower_bound,
        found_upper_bound,
    ) = optimisers.get_p0_n_bounds(simple_callable, p0=[], bounds=(0, []))
    npt.assert_allclose(found_p0, 2)
    assert found_n == 0

    # try callable with args
    (
        found_p0,
        found_n,
        found_lower_bound,
        found_upper_bound,
    ) = optimisers.get_p0_n_bounds(callable_with_args)
    npt.assert_allclose(found_p0, [1, 1])
    assert found_n == 2
    npt.assert_allclose(found_lower_bound, [-np.inf, -np.inf])
    npt.assert_allclose(found_upper_bound, [np.inf, np.inf])

    # try callable with args and bounds
    (
        found_p0,
        found_n,
        found_lower_bound,
        found_upper_bound,
    ) = optimisers.get_p0_n_bounds(callable_with_args, bounds=[(0, 2), (10, 5)])
    npt.assert_allclose(found_p0, [5, 3.5])
    assert found_n == 2
    npt.assert_allclose(found_lower_bound, [0, 2])
    npt.assert_allclose(found_upper_bound, [10, 5])


def test_evaluator_factory():
    # start with the simplest case of a callable with no additional arguments
    evaluator = optimisers.evaluator_factory(simple_callable, np.ones(1), np.ones(1))
    # should always return 0
    npt.assert_allclose(evaluator([]), 0)
    # or if teh x data cannot match the y data
    evaluator = optimisers.evaluator_factory(
        simple_callable, np.ones(1), np.ones(1) * 2
    )
    npt.assert_allclose(evaluator([]), 1)
    # now try with params
    perfect_a = 2
    perfect_b = 3
    xs = np.linspace(0, 10, 100)
    ys = callable_with_args(xs, perfect_a, perfect_b)
    evaluator = optimisers.evaluator_factory(callable_with_args, xs, ys)
    found = evaluator([perfect_a, perfect_b])
    npt.assert_allclose(found, 0)
    # or if the params are wrong
    found_1 = evaluator([perfect_a + 1, perfect_b + 1])
    assert found_1 > 0
    found_2 = evaluator([perfect_a + 2, perfect_b + 2])
    assert found_2 > found_1
    # we should be able to specify sigma
    sigma = np.eye(100)
    evaluator = optimisers.evaluator_factory(callable_with_args, xs, ys, sigma=sigma)
    found = evaluator([perfect_a, perfect_b])
    npt.assert_allclose(found, 0)
    # or if the params are wrong
    found_1s = evaluator([perfect_a + 1, perfect_b + 1])
    assert found_1s > 0


def test_chose_trials():
    # with one trial, p0 should always be used
    p0 = np.array([1, 2])
    found = optimisers.chose_trials(1, p0, [-np.inf, -np.inf], [np.inf, np.inf])
    assert found.shape == (1, 2)
    npt.assert_allclose(found[0], p0)
    found = optimisers.chose_trials(1, p0, [0, 0.5], [10, 5])
    assert found.shape == (1, 2)
    npt.assert_allclose(found[0], p0)
    # more trials should still include p0
    found = optimisers.chose_trials(100, p0, [-np.inf, -np.inf], [np.inf, np.inf])
    assert found.shape == (100, 2)
    assert np.all(found[0] == p0)
    # bout they should be different
    assert not np.all(p0 == found)
    varied = np.unique(found, axis=0)
    assert varied.shape[0] > 90
    # and within bounds
    found = optimisers.chose_trials(100, p0, [0, 0.5], [10, 5])
    assert found.shape == (100, 2)
    npt.assert_allclose(found[0], p0)
    assert np.all(found >= [0, 0.5])
    assert np.all(found <= [10, 5])
    varied = np.unique(found, axis=0)
    assert varied.shape[0] > 90


def test_curve_fit():
    """
    This is a big complex function, and exspensive to run,
    so only a basic test here
    """
    xs = np.linspace(0, 10, 100)
    found_popt, found_pcov = optimisers.curve_fit(
        callable_with_args,
        xs,
        callable_with_args(xs, 2, 3),
        5,
        p0=[1, 1],
    )
    npt.assert_allclose(found_popt, [2, 3], atol=0.1)
    assert found_pcov.shape == (2, 2)
