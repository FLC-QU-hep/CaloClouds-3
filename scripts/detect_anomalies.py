import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader

from pointcloud.config_varients.wish_maxwell import Configs
from pointcloud.data.trees import DataAsTrees
from pointcloud.anomalies import autoencoder

config = Configs()
data = DataAsTrees(config)

print(f"Have {len(data)} trees in the dataset.")

# Parameters
num_samples = min(5000, len(data))
input_dim = 4
hidden_dim = 8
num_epochs = 50
batch_size = 32
threshold = 0.1

# Draw data
print(f"Drawing {num_samples} samples from the dataset.")
data_list = autoencoder.format_data(data, num_samples)
print("Data formatted.")

# Create data loaders
train_data = data_list[:]
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Initialize model, optimizer, and loss function
model = autoencoder.GraphAutoencoder(input_dim, hidden_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss(reduction="mean")

# Train the model
print(f"Training model for {num_epochs} epochs.")
autoencoder.train(model, train_loader, optimizer, criterion, num_epochs)
print("Model trained. Finding anomalies.")
del train_data, train_loader

# Detect anomalies in data
anomalies = autoencoder.detect_anomalies(model, data, threshold, 10000)
print(f"Detected {len(anomalies)} anomalies. Saving to anomalies.npy")
autoencoder.save_anomalies("anomalies.npy", anomalies)
