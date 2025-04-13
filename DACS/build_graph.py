# build_graph.py

import torch
from torch_geometric.data import Data
from itertools import combinations
from data_preprocessing import load_and_process_movies, encode_genre_features, normalize_year

def build_edges_from_genres(df):
    genre_to_movies = {}

    for idx, genres in enumerate(df['genre_names']):
        for genre in genres:
            genre_to_movies.setdefault(genre, []).append(idx)

    edge_set = set()
    for movies in genre_to_movies.values():
        for i, j in combinations(movies, 2):
            edge_set.add((i, j))
            edge_set.add((j, i))

    edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()
    return edge_index

def main():
    print("Loading & processing movie data...")
    df = load_and_process_movies()
    print(f"Loaded {len(df)} movies")

    print("Encoding genre features...")
    x = encode_genre_features(df)
    print(f"Feature tensor shape: {x.shape}")

    df = normalize_year(df)
    print("Encoding release year...")
    x_year = torch.tensor(df['release_year'].values, dtype=torch.float).unsqueeze(1)
    
    print("Building graph edges...")
    edge_index = build_edges_from_genres(df)
    print(f"Total edges: {edge_index.shape[1]}")

    # Add vote_average as a target label
    if 'vote_average' in df.columns:
        y = torch.tensor(df['vote_average'].values, dtype=torch.float)
    else:
        raise ValueError("Column 'vote_average' is missing from the DataFrame")

    data = Data(x=x, edge_index=edge_index, y=y)
    torch.save(data, "movie_graph.pt")
    print("Saved movie_graph.pt")

if __name__ == "__main__":
    main()
