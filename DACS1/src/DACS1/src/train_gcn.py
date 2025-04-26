import os
import torch
import torch.nn as nn
import torch.optim as optim
from gcn_model import GCNRecommendationModel, load_graph_data

def train(model, data, epochs=100, lr=0.01):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.x)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    return model

if __name__ == "__main__":
    data = load_graph_data("data/movie_graph.pt")
    model = GCNRecommendationModel(
        in_channels=data.x.size(1),
        hidden_channels=128,
        out_channels=64
    )

    print("Bắt đầu huấn luyện GCN...")
    trained_model = train(model, data, epochs=100, lr=0.01)

    os.makedirs("data", exist_ok=True)
    torch.save(trained_model.state_dict(), "data/gcn_model.pth")
    print("Đã huấn luyện và lưu mô hình GCN.")
