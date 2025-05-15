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

# Tính embedding
with torch.no_grad():
    movie_embeddings = model(data.x, data.edge_index)

# Hàm tiện ích

def get_movie_info(row):
    return {
        "title": str(row['title']),
        "image": str(row.get('poster_path', "")),
        "rating": float(row.get('vote_average', 0)) if row.get('vote_average') is not None else None,
        "id": int(row.get('id', 0)) if row.get('id') is not None else None
    }


def get_top_k_similar(embedding, all_embeddings, top_k, exclude_idx=None):
    """
    Trả về top_k chỉ số có cosine similarity cao nhất với embedding truyền vào.
    Nếu exclude_idx được chỉ định, sẽ loại bỏ nó khỏi kết quả.
    """
    sims = F.cosine_similarity(embedding.unsqueeze(0), all_embeddings).squeeze()
    if exclude_idx is not None:
        sims[exclude_idx] = -1e9  # loại bỏ chính nó khỏi top_k
    top_indices = torch.topk(sims, top_k).indices.tolist()
    return top_indices

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

    top_indices = get_top_k_similar(movie_embeddings[node_idx], movie_embeddings, top_k, exclude_idx=node_idx)

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

    top_indices = get_top_k_similar(genre_embedding, movie_embeddings, top_k)
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

    top_indices = get_top_k_similar(history_embedding, movie_embeddings, top_k)
    df_indices = [data.node_to_movie_idx[i].item() for i in top_indices]
    recommendations = df_movies.iloc[df_indices]

    return jsonify({
        "recommendations": [get_movie_info(row) for _, row in recommendations.iterrows()]
    })


# Run server
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)
