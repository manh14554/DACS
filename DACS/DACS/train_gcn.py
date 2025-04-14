import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Tải dữ liệu đồ thị từ file đã lưu
data: Data = torch.load("movie_graph.pt", weights_only=False)
print(f"Graph loaded: {data}")

# 2. Chuẩn bị nhãn (điểm đánh giá trung bình các phim)
# Chuyển đổi nhãn sang kiểu float để phù hợp với bài toán hồi quy
labels = data.y.float()

# 3. Chia dữ liệu thành các tập train, validation và test
# Tỷ lệ chia: 70% train, 20% validation, 10% test
num_nodes = data.num_nodes
idx = list(range(num_nodes))
# Lần chia thứ nhất: tách 30% làm tập test và validation
train_idx, temp_idx = train_test_split(idx, test_size=0.3, random_state=42)
# Lần chia thứ hai: tách 66% trong 30% ở trên để có ~20% validation và ~10% test
val_idx, test_idx = train_test_split(temp_idx, test_size=0.66, random_state=42)

# Chuyển các chỉ số sang tensor để phù hợp với PyTorch
train_idx = torch.tensor(train_idx, dtype=torch.long)
val_idx = torch.tensor(val_idx, dtype=torch.long)
test_idx = torch.tensor(test_idx, dtype=torch.long)

# 4. Định nghĩa kiến trúc mô hình GCN
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        # Lớp tích chập đồ thị đầu tiên
        self.conv1 = GCNConv(in_channels, hidden_channels)
        # Lớp dropout để tránh overfitting với tỷ lệ 30%
        self.dropout = torch.nn.Dropout(p=0.3)
        # Lớp tích chập đồ thị thứ hai
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # Lan truyền tiến qua lớp GCN thứ nhất
        x = self.conv1(x, edge_index)
        # Áp dụng hàm kích hoạt ReLU
        x = F.relu(x)
        # Áp dụng dropout
        x = self.dropout(x)
        # Lan truyền tiến qua lớp GCN thứ hai
        x = self.conv2(x, edge_index)
        return x

# 5. Khởi tạo mô hình và các thành phần huấn luyện
# Tạo mô hình với số kênh đầu vào bằng số đặc trưng của nút
model = GCN(in_channels=data.x.shape[1], hidden_channels=64, out_channels=1)
# Sử dụng bộ tối ưu Adam với tốc độ học 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# Sử dụng hàm loss SmoothL1Loss (Huber loss) ít nhạy với ngoại lệ hơn MSE
loss_fn = torch.nn.SmoothL1Loss()

# 6. Quá trình huấn luyện mô hình với early stopping
num_epochs = 200  # Số epoch tối đa
best_val_loss = float("inf")  # Khởi tạo loss tốt nhất
patience = 20  # Số epoch chờ đợi nếu không cải thiện
counter = 0  # Đếm số epoch không cải thiện

# Lưu lại lịch sử loss để vẽ đồ thị
train_losses, val_losses = [], []

for epoch in range(1, num_epochs + 1):
    # Chế độ huấn luyện
    model.train()
    # Xóa gradient từ bước trước
    optimizer.zero_grad()
    # Lan truyền tiến
    out = model(data.x, data.edge_index).squeeze()
    # Tính toán loss trên tập train
    loss = loss_fn(out[train_idx], labels[train_idx])
    # Lan truyền ngược
    loss.backward()
    # Cập nhật trọng số
    optimizer.step()

    # Đánh giá trên tập validation
    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(out[val_idx], labels[val_idx])

    # Lưu lại các giá trị loss
    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    # In thông tin mỗi 20 epoch
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

    # Kiểm tra điều kiện dừng sớm
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        # Lưu lại mô hình tốt nhất
        torch.save(model.state_dict(), "best_gcn_model.pth")
    else:
        counter += 1
        if counter >= patience:
            print("Kích hoạt dừng sớm do loss không cải thiện.")
            break

# 7. Đánh giá mô hình trên tập test
# Nạp lại mô hình tốt nhất
model.load_state_dict(torch.load("best_gcn_model.pth"))
model.eval()
with torch.no_grad():
    final_out = model(data.x, data.edge_index).squeeze()
    test_loss = loss_fn(final_out[test_idx], labels[test_idx])
print(f"Test Loss: {test_loss.item():.4f}")

# 8. Lưu embedding các nút (phim) để sử dụng cho hệ thống gợi ý
with torch.no_grad():
    node_embeddings = model(data.x, data.edge_index)
    torch.save(node_embeddings, "movie_embeddings.pt")
print("Đã lưu embedding các phim vào movie_embeddings.pt")

# 9. Lưu trọng số mô hình và checkpoint
# Lưu trọng số cuối cùng
torch.save(model.state_dict(), "final_gcn_model.pth")
# Lưu checkpoint để có thể tiếp tục huấn luyện sau
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': loss.item(),
}, "gcn_checkpoint.pth")
print("Đã lưu mô hình và checkpoint.")

# 10. Vẽ và lưu đồ thị quá trình huấn luyện
plt.figure(figsize=(10, 5))
# Vẽ đường loss tập train
plt.plot(train_losses, label="Train Loss")
# Vẽ đường loss tập validation
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Đồ thị Loss trong quá trình huấn luyện")
plt.legend()
plt.grid(True)
plt.savefig("loss_plot.png")
print("Đã lưu đồ thị loss vào loss_plot.png")