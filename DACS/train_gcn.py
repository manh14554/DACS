import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

# 1. Load graph data
data: Data = torch.load("movie_graph.pt", weights_only=False)
print(f"Graph loaded: {data}")

# 2. Tạo nhãn (label)
labels = data.y.float()

# Chia tập train/test
num_nodes = data.num_nodes
idx = list(range(num_nodes))
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)
train_idx = torch.tensor(train_idx, dtype=torch.long)
test_idx = torch.tensor(test_idx, dtype=torch.long)

# 3. Định nghĩa mô hình GCN
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# 4. Khởi tạo mô hình
model = GCN(in_channels=data.x.shape[1], hidden_channels=64, out_channels=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

# 5. Huấn luyện
num_epochs = 200
for epoch in range(1, num_epochs + 1):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index).squeeze()

    loss = loss_fn(out[train_idx], labels[train_idx])
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            test_loss = loss_fn(out[test_idx], labels[test_idx])
        print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

# 6. Lưu mô hình (state_dict)
torch.save(model.state_dict(), "gcn_model.pth")
print("Mô hình đã được lưu vào gcn_model.pth")

# 7. (Tùy chọn) Lưu thêm checkpoint đầy đủ
torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': loss.item(),
}, "gcn_checkpoint.pth")
print("Checkpoint đầy đủ đã được lưu vào gcn_checkpoint.pth")
