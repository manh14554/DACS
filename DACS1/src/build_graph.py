import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
import os

def create_edges_from_genres(df):
    """Tạo cạnh giữa các phim có cùng thể loại."""
    edges = set()
    genre_to_movies = {}

    for idx, genres in enumerate(df['genres']):
        for genre in genres:
            genre_to_movies.setdefault(genre, []).append(idx)

    for genre, movie_list in genre_to_movies.items():
        # Giới hạn số lượng phim tối đa trong mỗi thể loại để giảm cạnh
        if len(movie_list) > 200:  # Giới hạn 200 phim mỗi thể loại
            movie_list = movie_list[:50]
        for i in range(len(movie_list)):
            for j in range(i + 1, len(movie_list)):
                edges.add((movie_list[i], movie_list[j]))
                edges.add((movie_list[j], movie_list[i]))

    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    print(f"Số lượng cạnh từ thể loại: {edge_index.shape[1]}")
    return edge_index

def create_edges_from_similarity(features, threshold=None, max_edges_per_node=50):
    """
    Tạo cạnh dựa trên cosine similarity giữa feature vectors, giới hạn số cạnh tối đa mỗi node.

    Args:
        features (torch.Tensor): Tensor đặc trưng của các phim.
        threshold (float, optional): Ngưỡng độ tương đồng. Nếu None, tự động chọn phân vị 98%.
        max_edges_per_node (int): Số lượng cạnh tối đa cho mỗi node.

    Returns:
        torch.Tensor: Tensor chứa các cạnh (edge_index).
    """
    sim_matrix = cosine_similarity(features.numpy())
    num_movies = sim_matrix.shape[0]

    # Tự động chọn ngưỡng nếu không được cung cấp
    if threshold is None:
        sim_values = sim_matrix[np.triu_indices(num_movies, k=1)]
        threshold = np.percentile(sim_values, 98)  # Tăng từ 90% lên 98%
        print(f"Ngưỡng độ tương đồng tự động: {threshold:.3f}")

    # Tạo danh sách cạnh, giới hạn số lượng cạnh mỗi node
    edges = []
    edge_count_per_node = np.zeros(num_movies, dtype=int)

    # Tạo ma trận chỉ số được sắp xếp theo độ tương đồng giảm dần
    sorted_indices = np.argsort(-sim_matrix, axis=1)  # Sắp xếp giảm dần

    for i in range(num_movies):
        for j_idx in sorted_indices[i]:
            j = j_idx.item()
            if i == j:
                continue  # Bỏ qua trường hợp i == j
            if sim_matrix[i, j] < threshold:
                break  # Dừng nếu độ tương đồng nhỏ hơn ngưỡng
            if edge_count_per_node[i] >= max_edges_per_node or edge_count_per_node[j] >= max_edges_per_node:
                continue  # Bỏ qua nếu node đã đủ số cạnh tối đa
            edges.append((i, j))
            edges.append((j, i))
            edge_count_per_node[i] += 1
            edge_count_per_node[j] += 1

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    print(f"Số lượng cạnh từ độ tương đồng: {edge_index.shape[1]}")
    return edge_index

def build_graph(df_path="data/movies_df.pkl", features_path="data/features.pt",
                output_path="data/movie_graph.pt", use_similarity=True, sim_threshold=None):
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

    # Tạo cạnh từ độ tương đồng nếu được yêu cầu
    if use_similarity:
        edge_index_sim = create_edges_from_similarity(features, threshold=sim_threshold)
        # Hợp nhất các cạnh
        edge_index = torch.cat([edge_index_genre, edge_index_sim], dim=1)
        edge_index = torch.unique(edge_index, dim=1)  # Loại bỏ cạnh trùng lặp
    else:
        edge_index = edge_index_genre

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
    build_graph()