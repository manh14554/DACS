import torch
import pandas as pd
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
import os

def create_edges_from_genres(df):
    #Tạo edge giữa các phim có cùng thể loại.
    edges = set()
    genre_to_movies = {}

    for idx, genres in enumerate(df['genres']):
        for genre in genres:
            genre_to_movies.setdefault(genre, []).append(idx)

    for movie_list in genre_to_movies.values():
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
        for j in range(i + 1, num_movies):
            if sim_matrix[i, j] > threshold:
                edges.append((i, j))
                edges.append((j, i))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

def build_graph(df_path="data/movies_df.pkl", features_path="data/features.pt",
                output_path="data/movie_graph.pt", use_similarity=False, sim_threshold=0.85):
    # Load dữ liệu đã xử lý trước
    df = pd.read_pickle(df_path)
    features = torch.load(features_path)

    # Chọn cách tạo edge
    if use_similarity:
        edge_index = create_edges_from_similarity(features, threshold=sim_threshold)
    else:
        edge_index = create_edges_from_genres(df)

    # Tạo graph
    data = Data(x=features, edge_index=edge_index)
    data.df = df  # Đính kèm DataFrame để sử dụng sau này

    # Lưu graph
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(data, output_path)
    print(f"Đã lưu xong movie_graph.pt vào: {output_path}")

if __name__ == "__main__":
    build_graph()
