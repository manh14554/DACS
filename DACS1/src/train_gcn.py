from gcn_model import GCNRecommendationModel, load_graph_data
import torch
import torch.nn as nn
import torch.optim as optim

def train(model, data, epochs=100, lr=0.001, save_path="gcn_model.pt"):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.x)  # reconstruction loss
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Đã lưu model tại: {save_path}")

if __name__ == "__main__":
    graph_path = "data/movie_graph.pt"
    data = load_graph_data(graph_path)
    print("Băt đầu load dữ liệu graph")
    model = GCNRecommendationModel(
        input_dim=data.x.size(1),
        hidden_dim=128,
        output_dim=data.x.size(1)  # bắt buộc khớp để dùng MSE
    )
    print("Bắt đầu train model")
    train(model, data, epochs=100, lr=0.001, save_path="gcn_model.pt")
