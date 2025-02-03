"""
Return data as a tree structure.
"""

import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from .read_write import read_raw_regaxes, get_n_events
from ..utils import precision
from ..utils.metadata import Metadata
from ..utils.detector_map import split_to_layers


# TODO impement a flyweight stategy to the data.
class Tree:
    """
    A tree structure to hold the data.

    Properties
    ----------
    n_layers : int
        The number of layers in the event, not including
        the layer for the root node.
    max_skips : int
        Most occupied layers a parent child relation can pass through.
    occupied_layers : np.ndarray of int
        Indices of the layers that have points in them,
        including the layer for the root node.
    total_points : int
        The total number of points the tree is connecting,
        including the root node.
    xy : np.ndarray of float32 (n_points, 2)
        The xy positions of the points including the root node.
    energy : np.ndarray of float32 (n_points)
        The energy of the points including the root node.
    edges : np.ndarray of int32 (n_points, 2)
        Connected pairs of points by parent, child indices,
        including the root node.
    layer : np.ndarray of int (n_points)
        Layer index for each point, including the root node.

    """

    def __init__(self, event_as_layers, root_xy, max_skips=5, dtype=np.float32):
        """
        Construct the tree from the event data.
        Put a root node bofore the first layer.

        Parameters
        ----------
        event_as_layers : list of np.ndarray (n_points, 4)
            The event data split into layers. Format is x, y, z, energy.
        root_xy : np.ndarray of float32 (2)
            The xy position of the root node, presumably the particle gun.
        max_skips : int
            The maximum number of occupied layers a parent child relation can skip.

        """
        self.max_skips = max_skips
        self._setup_layers(event_as_layers)
        self._setup_root(root_xy)
        self.dtype = dtype
        for _ in self.occupied_layers[2:]:
            self._assign_next_layer()

    def _setup_layers(self, event_as_layers):
        """
        Work out the layer structure, with one root node.
        Don't actually assign parents yet, just create the array for them.

        Parameters
        ----------
        event_as_layers : list of np.ndarray (n_points, 4)
            The event data split into layers. Format is x, y, z, energy.
        """
        root_point = np.array([[0, 0, 0, 1]], dtype=self.dtype)
        self._event_as_layers = [root_point] + event_as_layers
        self._layer_masks = [layer[:, 3] > 0 for layer in self._event_as_layers]
        self.n_layers = len(self._event_as_layers)
        n_points_in_layer = [mask.sum() for mask in self._layer_masks]
        self.total_points = sum(n_points_in_layer)
        self.occupied_layers = np.where(n_points_in_layer)[0]
        cumulative_points_in_layer = np.cumsum(n_points_in_layer)
        # store the xy positions of the points
        self.xy = np.zeros((self.total_points, 2), dtype=self.dtype)
        self.energy = np.zeros(self.total_points, dtype=self.dtype)
        for layer_n, layer in enumerate(self._event_as_layers[1:], 1):
            layer_slice = slice(
                cumulative_points_in_layer[layer_n - 1],
                cumulative_points_in_layer[layer_n],
            )
            layer_e = layer[:, 3]
            layer_mask = layer_e > 0
            if np.any(layer_mask):
                self.energy[layer_slice] = layer_e[layer_mask]
                self.xy[layer_slice] = layer[layer_mask][:, [0, 1]]

        self.layer = np.repeat(np.arange(self.n_layers), n_points_in_layer)
        # allocate storage for the edges,
        # 2 columns, one for the parent, one for the child
        # each point asside from the root must have exactly one parent,
        # points in the first layer all have the root parent
        self.edges = -np.ones((self.total_points - 1, 2), dtype=self.dtype)

    def _setup_root(self, root_xy):
        """
        Set up the root node, and assign its parents.

        Parameters
        ----------
        root_xy : np.ndarray of float32 (2)
            The xy position of the root node, presumably the particle gun.
        """
        self.energy[0] = 1.
        self.xy[0] = root_xy
        self._event_as_layers[0][0] = np.array(
            [[root_xy[0], root_xy[1], 0, 1]], dtype=self.dtype
        )
        # every point in layer one has the root as parent
        if len(self.occupied_layers) == 1:
            return
        init_layer_indices = np.where(self.layer == self.occupied_layers[1])[0]
        self.edges[init_layer_indices - 1, 0] = 0
        self.edges[init_layer_indices - 1, 1] = init_layer_indices
        self._occ_idx = 1

    def project_positions(self, to_occ_layer_idx):
        """
        Find the projected positions of the points in the next layer.
        No internal state is changed. (static method)

        Parameters
        ----------
        to_occ_layer_idx : int
            The layer we want to find the positions on.
        """
        current_idxs = []
        current_xy = []
        to_layer = self.occupied_layers[to_occ_layer_idx]
        for back in range(1, min(self.max_skips + 1, to_occ_layer_idx)):
            previous_layer = self.occupied_layers[to_occ_layer_idx - back - 1]
            from_layer = self.occupied_layers[to_occ_layer_idx - back]
            from_idxs = np.where(self.layer == from_layer)[0]
            current_idxs.append(from_idxs)
            previous = self.edges[from_idxs - 1, 0]
            xy_previous = self.xy[previous]
            xy_from = self.xy[from_idxs]
            gap1 = from_layer - previous_layer
            gap2 = to_layer - from_layer
            projected_xy = xy_from + (gap2 / gap1) * (xy_from - xy_previous)
            current_xy.append(projected_xy)
        current_idxs = np.concatenate(current_idxs)
        projected_xy = np.concatenate(current_xy)
        return current_idxs, projected_xy

    def _assign_next_layer(self):
        """
        Assign parents to the points in the next layer
        Fill the edges array with the parent child relations.
        """
        self._occ_idx += 1
        next_occupied_layer = self.occupied_layers[self._occ_idx]
        parent_idxs, parent_xy = self.project_positions(self._occ_idx)
        parent_energies = self.energy[parent_idxs]
        child_idxs = np.where(self.layer == next_occupied_layer)[0]
        child_xy = self.xy[child_idxs]
        child_energies = self.energy[child_idxs]
        parent_child_distances = akt_distance(
            parent_xy, parent_energies, child_xy, child_energies
        )
        closest_parents = np.argmin(parent_child_distances, axis=1)
        self.edges[child_idxs - 1, 0] = parent_idxs[closest_parents]
        self.edges[child_idxs - 1, 1] = child_idxs


def check_connected(tree):
    """
    Check if a tree is fully connected.

    Parameters
    ----------
    tree : Tree
        The tree to check.

    Returns
    -------
    bool
        True if the tree is fully connected, False otherwise.
    list of int
        The indices of the points that are connected to the root.
    """
    # start with everything connected to the root
    stack = [child for parent, child in tree.edges if parent == 0]
    in_graph = [0] + stack
    while stack:
        parent = stack.pop()
        children = [child for p, child in tree.edges if p == parent]
        # to prevent looping, only add children that are not already in the graph
        stack += [child for child in children if child not in in_graph]
        in_graph += children
    is_connected = len(in_graph) == tree.total_points
    return is_connected, in_graph


def akt_distance(parent_xys, parent_energies, child_xys, child_energies):
    """
    Calculate an anti-kT style distance between the first and
    the second set of points.

    Parameters
    ----------
    parent_xys : np.ndarray of float32 (n_parents, 2)
        The xy positions of the first set of points.
    parent_energies : np.ndarray of float32 (n_parents)
        The energies of the first set of points.
    child_xys : np.ndarray of float32 (n_children, 2)
        The xy positions of the second set of points.
    child_energies : np.ndarray of float32 (n_children)
        The energies of the second set of points.

    Returns
    -------
    distances : np.ndarray of float32 (n_parents, n_children)
        The distances between the points.

    """
    if len(parent_xys) == 0 or len(child_xys) == 0:
        return np.array([]).reshape(len(parent_xys), len(child_xys))
    eucideans = distance_matrix(child_xys, parent_xys)
    akt_factors = np.maximum(parent_energies, child_energies[:, np.newaxis])
    akt_factors = np.clip(akt_factors, 1e-6, None)
    return eucideans / akt_factors


class DataAsTrees:
    """
    Lazy generation of trees from the data.
    Will hold a certain number of generated trees in memory to reduce recomputing.
    """

    def __init__(self, configs, max_trees_in_memory=1000, quiet=False):
        """
        Intialize the data as trees object.

        Parameters
        ----------
        configs : pointcloud.configs.Configs
            The configuration object, which indicates where the dataset is.
        max_trees_in_memory : int
            The maximum number of trees to hold in memory, to reduce recomputing.
        quiet : bool
            If True, don't print progress updates when constructing trees.
        """
        self.quiet = quiet
        self.configs = configs
        data_lengths = get_n_events(configs.dataset_path, configs.n_dataset_files)
        self.__len = np.sum(data_lengths)
        self._max_trees_in_memory = max_trees_in_memory
        self._held_idxs = []
        self._held_trees = []

        metadata = Metadata(configs)
        self._layer_bottom_pos = metadata.layer_bottom_pos_hdf5
        self._cell_thickness_global = (
            self._layer_bottom_pos[1] - self._layer_bottom_pos[0]
        ) / 2
        self.root_xy = metadata.gun_xyz_pos_hdf5[:2]

    def __len__(self):
        return self.__len

    def get(self, idxs):
        """
        Return the trees for the given indices.

        Parameters
        ----------
        idxs : list of int
            The indices of the events to return.

        Returns
        -------
        list of Tree
            The trees for the given indices.
        """
        need_trees = sorted(idx for idx in idxs if idx not in self._held_idxs)
        num_trees = len(idxs)
        incidents, events = read_raw_regaxes(self.configs, need_trees)
        results = []
        to_hold = []
        for i, idx in enumerate(idxs):
            if i % 100 == 0 and not self.quiet:
                print(f"{i/num_trees:.2%}", end="\r", flush=True)
            if idx in self._held_idxs:
                results.append(self._held_trees[self._held_idxs.index(idx)])
            else:
                event = events[need_trees.index(idx)]
                tree = self._generate_tree(event)
                # leave the incident in the tree for now
                incident = incidents[need_trees.index(idx)]
                tree.incident = incident
                to_hold.append((idx, tree))
                results.append(tree)
        # can't mess with the held trees while fetching results
        for idx, tree in to_hold:
            self._hold_tree(idx, tree)
        return results

    def _generate_tree(self, event):
        event_as_layers = list(
            split_to_layers(event, self._layer_bottom_pos, self._cell_thickness_global)
        )
        dtype = precision.get("anomaly_tree_feat", self.configs)
        return Tree(event_as_layers, self.root_xy, dtype=dtype)

    def _hold_tree(self, idx, tree):
        if len(self._held_idxs) >= self._max_trees_in_memory:
            self._held_idxs.pop(0)
            self._held_trees.pop(0)
        self._held_idxs.append(idx)
        self._held_trees.append(tree)


def plot_tree(
    config_or_trees,
    idx,
    ax=None,
    xlims=(-150, 150),
    ylims=(-150, 150),
    zlims=(-2, 31),
    line_alpha=0.5,
    show=True,
):
    if isinstance(config_or_trees, DataAsTrees):
        tree_data = config_or_trees
    else:
        tree_data = DataAsTrees(config_or_trees)
    tree = tree_data.get([idx])[0]

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    if line_alpha > 0:
        n_children = np.fromiter(
            (np.sum(tree.edges[:, 0] == i) for i in range(tree.total_points)),
            dtype=np.float32,
        )
        n_children /= np.max(n_children)
        cmap = plt.get_cmap("viridis")
        line_segments = []
        colours = []
        for edge in tree.edges:
            line = np.empty((3, 2))
            line[0] = tree.xy[edge][:, 0]
            line[1] = tree.xy[edge][:, 1]
            line[2] = tree.layer[edge]
            line_segments.append(line.T)
            colour = cmap(1 - n_children[edge[0]], alpha=line_alpha)
            colours.append(colour)

        # Create a LineCollection object
        lc = Line3DCollection(line_segments, color=colours, linewidths=1)
        ax.add_collection3d(lc)
    reg_energy = tree.energy / np.max(tree.energy)
    ax.scatter(
        tree.xy[:, 0],
        tree.layer,
        tree.xy[:, 1],
        s=20 * reg_energy,
        c=[(0, 0, 0, 0.5)],
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("layers")
    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)
    ax.set_zlim(*zlims)

    if show:
        plt.ion()
        plt.show()
    return fig, ax
