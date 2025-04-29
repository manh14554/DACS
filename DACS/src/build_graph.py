import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import faiss
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def create_edges_from_genres(df):
    """Tạo cạnh giữa các phim có cùng thể loại."""
    edges = set()
    genre_to_movies = {}

    for idx, genres in enumerate(df['genres']):
        for genre in genres:
            genre_to_movies.setdefault(genre, []).append(idx)

    for genre, movie_list in genre_to_movies.items():
        if len(movie_list) > 500:  # Giới hạn 500 phim mỗi thể loại
            movie_list = movie_list[:500]
        for i in range(len(movie_list)):
            for j in range(i + 1, len(movie_list)):
                edges.add((movie_list[i], movie_list[j]))
                edges.add((movie_list[j], movie_list[i]))

    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    print(f"Số lượng cạnh từ thể loại: {edge_index.shape[1]}")
    return edge_index

def create_edges_from_similarity(features, k=10):
    """
    Tạo cạnh dựa trên độ tương đồng (sử dụng FAISS để tìm k-NN).

    Args:
        features (torch.Tensor): Tensor đặc trưng của các phim.
        k (int): Số lượng hàng xóm gần nhất cho mỗi node.

    Returns:
        torch.Tensor: Tensor chứa các cạnh (edge_index).
    """
    features_np = features.numpy().astype(np.float32)
    num_movies = features_np.shape[0]

    # Xây dựng FAISS index
    dim = features_np.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(features_np)
    print(f"FAISS index built với {num_movies} vectors.")

    # Tìm k-NN cho mỗi phim
    distances, indices = index.search(features_np, k + 1)  # +1 để loại bỏ chính nó
    edges = []

    for i in range(num_movies):
        neighbors = indices[i][1:]  # Bỏ qua chính node i
        for j in neighbors:
            if j != i:  # Đảm bảo không tạo cạnh với chính nó
                edges.append((i, j))
                edges.append((j, i))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    print(f"Số lượng cạnh từ độ tương đồng (FAISS): {edge_index.shape[1]}")
    return edge_index

def build_graph(df_path="data/movies_df.pkl", features_path="data/features.pt",
                output_path="data/movie_graph.pt", k_neighbors=10):
    """Xây dựng đồ thị từ dữ liệu phim."""
    # Load dữ liệu đã xử lý trước
    try:
        df = pd.read_pickle(df_path)
    except FileNotFoundError:
        print(f"Không tìm thấy file tại {df_path}. Vui lòng kiểm tra đường dẫn.")
        exit(1)
    
    try:
        features = torch.load(features_path)
    except FileNotFoundError:
        print(f"Không tìm thấy file tại {features_path}. Vui lòng kiểm tra đường dẫn.")
        exit(1)

    # Kiểm tra số lượng phim
    if len(df) != features.shape[0]:
        print(f"Lỗi: Số lượng phim trong DataFrame ({len(df)}) không khớp với số lượng node trong features ({features.shape[0]}).")
        exit(1)

    print(f"Số lượng phim: {len(df)}")

    # Tạo cạnh từ thể loại
    edge_index_genre = create_edges_from_genres(df)

    # Tạo cạnh từ độ tương đồng (FAISS)
    edge_index_sim = create_edges_from_similarity(features, k=k_neighbors)

    # Hợp nhất các cạnh
    edge_index = torch.cat([edge_index_genre, edge_index_sim], dim=1)
    edge_index = torch.unique(edge_index, dim=1)  # Loại bỏ cạnh trùng lặp
    print(f"Tổng số lượng cạnh trong đồ thị: {edge_index.shape[1]}")

    # Tạo ánh xạ từ chỉ số node trong đồ thị sang chỉ số trong DataFrame
    node_to_movie_idx = torch.tensor(df.index.tolist(), dtype=torch.long)

    # Tạo đồ thị
    data = Data(x=features, edge_index=edge_index)
    data.node_to_movie_idx = node_to_movie_idx  # Lưu ánh xạ

    # Lưu đồ thị
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(data, output_path)
    print(f"Đã lưu xong movie_graph.pt vào: {output_path}")

if __name__ == "__main__":
    build_graph(k_neighbors=10)