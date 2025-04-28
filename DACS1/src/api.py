from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
import pandas as pd
import os
from gcn_model import load_graph_data
from train_gcn import GCN

app = Flask(__name__)
CORS(app)

# Load dữ liệu và mô hình

# Load graph data
graph_data_path = "data/movie_graph.pt"
data = load_graph_data(graph_data_path)

# Khởi tạo model
model = GCN(
    in_channels=data.x.size(1),
    hidden_channels=128,
    out_channels=64,
    dropout=0.5
)

# Load trọng số
model_path = "data/gcn_model.pth"
try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
except FileNotFoundError:
    exit(1)
except RuntimeError:
    exit(1)

model.eval()

# Load DataFrame phim
movies_path = "data/movies_df.pkl"
try:
    df_movies = pd.read_pickle(movies_path)
except FileNotFoundError:
    exit(1)

# Kiểm tra độ khớp
if len(df_movies) != data.x.size(0):
    exit(1)

# Tính embedding và ma trận tương đồng

with torch.no_grad():
    movie_embeddings = model(data.x, data.edge_index)

sim_matrix = F.cosine_similarity(movie_embeddings.unsqueeze(1), movie_embeddings.unsqueeze(0), dim=2)

# Hàm tiện ích

def get_movie_info(row):
    return {
        "title": row['title'],
        "image": row.get('poster_path', ""),
        "rating": row.get('vote_average', None)
    }

# Các API

@app.route("/api/recommend/by_title", methods=["POST"])
def recommend_by_title_api():
    data_json = request.get_json()
    title = data_json.get("title", "").strip()
    top_k = data_json.get("top_k", 5)

    if not title:
        return jsonify({"error": "Missing 'title' field."}), 400

    matched = df_movies[df_movies['title'].str.lower().str.contains(title.lower(), na=False)]

    if matched.empty:
        return jsonify({
            "selected_movie": None,
            "recommendations": [],
            "message": "Không tìm thấy phim phù hợp với tên đã nhập."
        })

    movie_idx = matched.index[0]
    node_idx = (data.node_to_movie_idx == movie_idx).nonzero(as_tuple=True)[0].item()

    sims = sim_matrix[node_idx]
    top_indices = torch.topk(sims, top_k + 1).indices.tolist()
    top_indices = [i for i in top_indices if i != node_idx][:top_k]

    df_indices = [data.node_to_movie_idx[i].item() for i in top_indices]
    recommendations = df_movies.iloc[df_indices]

    return jsonify({
        "selected_movie": get_movie_info(df_movies.iloc[movie_idx]),
        "recommendations": [get_movie_info(row) for _, row in recommendations.iterrows()]
    })


@app.route("/api/recommend/by_genre", methods=["POST"])
def recommend_by_genre_api():
    data_json = request.get_json()
    genres = data_json.get("genres", [])
    top_k = data_json.get("top_k", 5)

    if not genres or not isinstance(genres, list):
        return jsonify({"error": "Missing or invalid 'genres' field."}), 400

    filtered = df_movies[df_movies['genres'].apply(
        lambda x: any(g.lower() in [genre.lower() for genre in x] for g in genres)
    )]

    if filtered.empty:
        return jsonify({
            "recommendations": [],
            "message": "Không tìm thấy phim phù hợp với thể loại đã chọn."
        })

    node_indices = [(data.node_to_movie_idx == idx).nonzero(as_tuple=True)[0].item() for idx in filtered.index.tolist()]
    genre_embedding = movie_embeddings[node_indices].mean(dim=0)

    sims = F.cosine_similarity(genre_embedding.unsqueeze(0), movie_embeddings)
    top_indices = torch.topk(sims, top_k).indices.tolist()

    df_indices = [data.node_to_movie_idx[i].item() for i in top_indices]
    recommendations = df_movies.iloc[df_indices]

    return jsonify({
        "recommendations": [get_movie_info(row) for _, row in recommendations.iterrows()]
    })


@app.route("/api/recommend/by_history", methods=["POST"])
def recommend_by_history_api():
    data_json = request.get_json()
    watched_titles = data_json.get("watched_titles", [])
    top_k = data_json.get("top_k", 5)

    if not watched_titles or not isinstance(watched_titles, list):
        return jsonify({"error": "Missing or invalid 'watched_titles' field."}), 400

    matched_indices = []
    for title in watched_titles:
        matched = df_movies[df_movies['title'].str.lower().str.contains(title.lower(), na=False)]
        if not matched.empty:
            movie_idx = matched.index[0]
            node_idx = (data.node_to_movie_idx == movie_idx).nonzero(as_tuple=True)[0].item()
            matched_indices.append(node_idx)

    if not matched_indices:
        return jsonify({
            "recommendations": [],
            "message": "Không tìm thấy phim nào từ lịch sử đã xem."
        })

    history_embedding = movie_embeddings[matched_indices].mean(dim=0)

    sims = F.cosine_similarity(history_embedding.unsqueeze(0), movie_embeddings)
    top_indices = torch.topk(sims, top_k).indices.tolist()

    df_indices = [data.node_to_movie_idx[i].item() for i in top_indices]
    recommendations = df_movies.iloc[df_indices]

    return jsonify({
        "recommendations": [get_movie_info(row) for _, row in recommendations.iterrows()]
    })


# Run server
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)
