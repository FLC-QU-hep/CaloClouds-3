import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


# Define a simple graph autoencoder model
class GraphAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GraphAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder = GCNConv(input_dim, hidden_dim)
        self.decoder = GCNConv(hidden_dim, input_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.encoder(x, edge_index)
        x = torch.relu(x)
        x = self.decoder(x, edge_index)
        return x

    def save(self, path):
        """
        Save the model, along side it's input and hidden dimensions
        """
        state_dict = self.state_dict()
        state_dict["input_dim"] = self.input_dim
        state_dict["hidden_dim"] = self.hidden_dim
        torch.save(state_dict, path)

    @classmethod
    def load(cls, path):
        """
        As the save files contain the input and hidden dimensions, we can
        load the model without needing to specify them
        """
        state_dict = torch.load(path, weights_only=False)
        model = cls(state_dict["input_dim"], state_dict["hidden_dim"])
        del state_dict["input_dim"]
        del state_dict["hidden_dim"]
        model.load_state_dict(state_dict)
        return model
