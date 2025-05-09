import torch
import pandas as pd
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity

def create_edges_from_genres(df, max_per_genre=500):
    """Tạo edge giữa các phim có cùng thể loại (giới hạn số lượng mỗi thể loại)."""
    edges = set()
    genre_to_movies = {}

    for idx, genres in enumerate(df['genres']):
        for genre in genres:
            genre_to_movies.setdefault(genre, []).append(idx)

    for movie_list in genre_to_movies.values():
        # Giới hạn số lượng phim mỗi thể loại để tránh tràn bộ nhớ
        movie_list = movie_list[:max_per_genre]
        for i in range(len(movie_list)):
            for j in range(i + 1, len(movie_list)):
                edges.add((movie_list[i], movie_list[j]))
                edges.add((movie_list[j], movie_list[i]))

    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    return edge_index

def create_edges_from_similarity(features, threshold=0.85):
    """Tạo edge dựa trên cosine similarity giữa feature vectors."""
    sim_matrix = cosine_similarity(features.numpy())
    edges = []

    num_movies = sim_matrix.shape[0]
    for i in range(num_movies):
        for j in range(i + 1, num_movies):  # j > i để tránh duplicate
            if sim_matrix[i, j] > threshold:
                edges.append((i, j))
                edges.append((j, i))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

def build_graph():
    # Load dữ liệu đã xử lý trước
    df = pd.read_pickle("data/movies_df.pkl")
    features = torch.load("data/features.pt")

    # Chọn cách tạo edge
    edge_index = create_edges_from_genres(df, max_per_genre=500)  # Giới hạn phim mỗi thể loại
    # edge_index = create_edges_from_similarity(features, threshold=0.85)  # Nếu muốn dùng cosine

    # Tạo graph
    data = Data(x=features, edge_index=edge_index)
    data.df = df  # Lưu DataFrame vào graph để sử dụng sau này

    # Lưu graph
    torch.save(data, "data/movie_graph.pt")
    print("Đã lưu xong movie_graph.pt")

if __name__ == "__main__":
    build_graph()
