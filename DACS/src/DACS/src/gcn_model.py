import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNRecommendationModel(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=64, dropout=0.5):
        """
        Khởi tạo mô hình GCN cho hệ thống gợi ý phim.

        Args:
            in_channels (int): Số chiều của vector đặc trưng đầu vào.
            hidden_channels (int): Số chiều của tầng ẩn.
            out_channels (int): Số chiều của embedding đầu ra.
            dropout (float): Tỷ lệ dropout để tránh overfitting.
        """
        super(GCNRecommendationModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        """
        Forward pass của mô hình GCN.

        Args:
            x (torch.Tensor): Tensor đặc trưng của các node (phim).
            edge_index (torch.Tensor): Tensor chứa thông tin cạnh của đồ thị.

        Returns:
            torch.Tensor: Embedding của các node sau khi qua GCN.
        """
        # Kiểm tra đầu vào
        if x is None or edge_index is None:
            raise ValueError("Input 'x' và 'edge_index' không được để trống.")
        if x.shape[0] != edge_index.max().item() + 1:
            raise ValueError("Số lượng node trong 'x' không khớp với chỉ số trong 'edge_index'.")

        # Tầng GCN đầu tiên
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Tầng GCN thứ hai
        x = self.conv2(x, edge_index)
        return x

    def get_embedding(self, data):
        """
        Trả về embedding của tất cả node (phim) trong đồ thị.

        Args:
            data (torch_geometric.data.Data): Đối tượng đồ thị chứa 'x' và 'edge_index'.

        Returns:
            torch.Tensor: Embedding của các node.
        """
        self.eval()
        with torch.no_grad():
            if not hasattr(data, 'x') or not hasattr(data, 'edge_index'):
                raise ValueError("Đối tượng 'data' phải chứa 'x' và 'edge_index'.")
            embeddings = self.forward(data.x, data.edge_index)
        print(f"Đã tạo embedding với kích thước: {embeddings.shape}")
        return embeddings

def load_graph_data(graph_path="data/movie_graph.pt"):
    """
    Tải dữ liệu đồ thị từ file.

    Args:
        graph_path (str): Đường dẫn đến file đồ thị (.pt).

    Returns:
        torch_geometric.data.Data: Đối tượng đồ thị.
    """
    try:
        data = torch.load(graph_path, weights_only=False)
        print(f"Đã tải đồ thị từ {graph_path}. Số lượng node: {data.x.shape[0]}, Số lượng cạnh: {data.edge_index.shape[1]}")
        return data
    except FileNotFoundError:
        print(f"Không tìm thấy file tại {graph_path}. Vui lòng kiểm tra đường dẫn.")
        exit(1)