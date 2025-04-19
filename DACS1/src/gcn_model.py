import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class GCNRecommendationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNRecommendationModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def load_graph_data(graph_path):
    return torch.load(graph_path, weights_only=False)

