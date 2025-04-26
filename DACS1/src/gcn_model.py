import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNRecommendationModel(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=64):
        super(GCNRecommendationModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def get_embedding(self, data):
        #Trả về embedding đầu ra của tất cả nodes (phim)
        self.eval()
        with torch.no_grad():
            out = self.forward(data.x, data.edge_index)
        return out

def load_graph_data(graph_path="data/movie_graph.pt"):
    data = torch.load(graph_path)
    return data
