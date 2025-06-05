"""
Use me like this:
from pointcloud.configs import Configs
config = Configs()
format_save(config)
"""

import numpy as np
import torch
import copy
from torch_geometric.data import Data

from ..data.trees import DataAsTrees


def format_data(data_as_trees, sample_idxs):
    """
    Convert the trees to a format that works as input for a graph neural network.

    Parameters
    ----------
    data_as_trees : pointcloud.data.trees.DataAsTrees
        A dataset that will read from the disk and convert internally
        to a tree structure before returning the data.
    sample_idxs : list of int
        The indices of the trees to convert.

    Returns
    -------
    features : list of np.ndarray (n_events, n_pnts, n_features)
        The features of the nodes in the trees.
        Features are [x, y, z (layer num), energy, incident, children1, children2, ...]
        where children1, children2, ... are the number of children the corrisponding 
        number of layers ahead of the node.
        The total number of children features is the max number of skips in the
        tree + 1.
    edges : list of np.ndarray (n_events, 2, n_pts-1)
        The edges of the trees, in the format [[parents....], [children....]].

    """
    features = []
    edges = []
    trees = data_as_trees.get(sample_idxs)
    # this is not a hard max, as empty layers didn't count when constructing the tree.
    max_child_projection = trees[0].max_skips + 1
    for i, (idx, tree) in enumerate(zip(sample_idxs, trees)):
        # simply energy
        energy = tree.energy
        # all nodes have the same incident energy
        incident = np.full_like(energy, tree.incident)
        # node can make children up to max_child_projection layers ahead of them
        child_projection = np.zeros(
            (energy.shape[0], max_child_projection), dtype=np.float32
        )
        projections = np.clip(
            tree.edges[:, 1] - tree.edges[:, 0] - 1, 0, max_child_projection - 1
        )
        for parent, projection in zip(tree.edges[:, 0], projections):
            child_projection[parent, projection] += 1
        # put that all together
        feats = (
            np.vstack(
                [
                    tree.xy[:, 0],
                    tree.xy[:, 1],
                    tree.layer,
                    energy,
                    incident,
                    *child_projection.T,
                ]
            )
            .astype(np.float32)
            .T
        )
        features.append(feats)
        # and keep track of the graph
        edges.append(tree.edges.T)
    return features, edges


def direct_data(config, file_path):
    """
    Quick and direct convertion of a single file to pytorch geometric data.
    """
    config_copy = copy.deepcopy(config)
    config_copy.datset_path = file_path
    config_copy.n_dataset_files = 0
    data_as_trees = DataAsTrees(config_copy, quiet=True)
    features, edges = format_data(data_as_trees, list(range(len(data_as_trees))))
    data = []
    for i, (feat, edge) in enumerate(zip(features, edges)):
        feat = torch.from_numpy(feat).contiguous().to(config.device)
        edge = torch.from_numpy(edge).contiguous().to(config.device)
        data.append(Data(x=feat, edge_index=edge, tree_idx=i))
    return data


def format_save(config, max_trees_in_memory=1000):
    save_base = config.formatted_tree_base
    features = []
    edges = []
    data = DataAsTrees(config, max_trees_in_memory, quiet=True)
    total_trees = len(data)
    for start_idx in range(0, total_trees, max_trees_in_memory):
        print(f"{start_idx/total_trees:.1%}", end="\r", flush=True)
        idxs = list(range(start_idx, min(start_idx + max_trees_in_memory, total_trees)))
        feat, edg = format_data(data, idxs)
        features += feat
        edges += edg
        if start_idx / max_trees_in_memory % 10 == 0:
            print("Saving...")
            np.savez(save_base + "_features_chpt.npz", *features)
            np.savez(save_base + "_edges_chpt.npz", *edges)
    np.savez(save_base + "_features.npz", *features)
    np.savez(save_base + "_edges.npz", *edges)
