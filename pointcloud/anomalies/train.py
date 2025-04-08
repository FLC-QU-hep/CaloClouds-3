"""
Loading data and training autoencoder
"""

import numpy as np
import os
import datetime
import torch.nn as nn
import torch.optim as optim
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from . import autoencoder
from ..configs import Configs


def checkpoint_location(loss, folder_name):
    # Get the folder name, and recursively create the directory if needed
    folder_name = os.path.dirname(folder_name)
    os.system(f"mkdir -p {folder_name}")
    now = datetime.datetime.now()
    time_str = now.strftime("%y-%m-%d.%H")
    file_base = os.path.join(folder_name, f"{time_str}_{int(loss)}.pt")
    return file_base


class TreeDataset(Dataset):
    def __init__(self, configs):
        super(TreeDataset, self).__init__()
        base = configs.formatted_tree_base
        self.features = np.load(base + "_features.npz")
        try:
            self.n_features = self.features["arr_0"].shape[1]
        except KeyError:
            self.n_features = 1
        self.edges = np.load(base + "_edges.npz")
        self.device = configs.device

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[f"arr_{idx}"]
        feat = torch.from_numpy(feat).contiguous().to(self.device).float()
        edge = self.edges[f"arr_{idx}"]
        edge = torch.from_numpy(edge).contiguous().to(self.device)
        data = Data(x=feat, edge_index=edge, tree_idx=idx)
        return data


# Train function
def epoch(model, dataloader, optimizer, criterion):
    """
    Use all the data once
    """
    total_loss = 0
    for data in dataloader:
        optimizer.zero_grad()
        reconstructions = model(data)
        loss = criterion(reconstructions, data.x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss


def get_criterion():
    return nn.MSELoss(reduction="mean")


def train(configs=Configs(), last_chpt=None, num_epochs=50, batch_size=32):
    dataset = TreeDataset(configs)

    if last_chpt is None:
        model = autoencoder.GraphAutoencoder(
            dataset.n_features, configs.anomaly_hidden_dim
        )
    else:
        model = autoencoder.GraphAutoencoder.load(last_chpt)
    model.train()

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    length = len(train_loader)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = get_criterion()

    for epoch_n in range(num_epochs):
        total_loss = epoch(model, train_loader, optimizer, criterion)
        checkpoint = checkpoint_location(total_loss, configs.anomaly_checkpoint)
        model.save(checkpoint)
        print(f"Epoch {epoch_n + 1}, Loss: {total_loss / length}")
