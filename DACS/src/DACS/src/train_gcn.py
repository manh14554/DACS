import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm
import torch.optim as optim
import os
import random
from gcn_model import load_graph_data

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        """Khởi tạo mô hình GCN với BatchNorm và Dropout."""
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def contrastive_loss(out, edge_index, num_nodes, margin=1.0, num_neg_samples=100):
    """
    Contrastive loss: đẩy các positive pairs lại gần và random chọn negative pairs ra xa.
    """
    pos_pairs = edge_index.t()  # [num_edges, 2]
    pos_dist = (out[pos_pairs[:, 0]] - out[pos_pairs[:, 1]]).pow(2).sum(1)

    # Random negative sampling (tránh dùng toàn bộ similarity matrix)
    neg_samples = []
    for _ in range(num_neg_samples):
        i = random.randint(0, num_nodes - 1)
        j = random.randint(0, num_nodes - 1)
        # Đảm bảo không chọn cặp trùng hoặc cặp đã kết nối
        while i == j or ((edge_index[0] == i) & (edge_index[1] == j)).any():
            j = random.randint(0, num_nodes - 1)
        neg_samples.append((i, j))

    neg_samples = torch.tensor(neg_samples, device=out.device)
    neg_dist = (out[neg_samples[:, 0]] - out[neg_samples[:, 1]]).pow(2).sum(1)

    loss = pos_dist.mean() + F.relu(margin - neg_dist).mean()
    return loss

def train(model, data, epochs=200, lr=0.001, margin=1.0, num_neg_samples=100, save_path="data/gcn_model.pth"):
    """
    Huấn luyện mô hình GCN với contrastive loss.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Đang train trên thiết bị: {device}")
    model = model.to(device)
    data = data.to(device)

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    num_nodes = data.x.size(0)

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)

        loss = contrastive_loss(out, data.edge_index, num_nodes, margin, num_neg_samples)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Đã lưu mô hình vào: {save_path}")

    return model

if __name__ == "__main__":
    # Load graph data
    data = load_graph_data("data/movie_graph.pt")

    # Initialize model
    model = GCN(
        in_channels=data.x.size(1),
        hidden_channels=128,
        out_channels=64,
        dropout=0.5
    )

    # Train model
    trained_model = train(
        model,
        data,
        epochs=200,
        lr=0.001,
        margin=1.0,
        num_neg_samples=500,  # tăng số lượng negative samples
        save_path="data/gcn_model.pth"
    )