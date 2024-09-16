import numpy as np
from pointcloud.data import trees


def empty(root_xz=np.array([0, 0]), n_layers=2, max_skips=6):
    fake_empty = np.array([]).reshape(0, 4)
    layers = [fake_empty] * n_layers
    empty_tree = trees.Tree(layers, root_xz, max_skips)
    return empty_tree


def simple(root_xz=np.array([0, 0])):
    fake_layer0 = np.array([[0.0, 0.0, 0.0, 1.01]])
    fake_layer1 = np.array([[0.0, 0.0, 1.0, 1.02], [0.0, 0.3, 1.0, 1.03]])
    fake_empty = np.array([]).reshape(0, 4)
    fake_layer2 = np.array(
        [[0.0, 0.0, 2.0, 1.04], [0.0, 1.0, 2.0, 1.05], [1.0, 0.0, 2.0, 1.06]]
    )
    fake_layers = [fake_layer0, fake_layer1, fake_empty, fake_layer2]
    tree = trees.Tree(fake_layers, root_xz, 2)
    return tree
