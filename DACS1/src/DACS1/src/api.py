from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from gcn_model import GCNRecommendationModel
import pandas as pd
import os
from difflib import get_close_matches

# Flask setup
app = Flask(__name__)
CORS(app)

# Load data
print("Đang tải dữ liệu và mô hình...")
data = torch.load("data/movie_graph.pt", weights_only=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)

df = data.df
df = df.reset_index(drop=True)

model = GCNRecommendationModel(
    in_channels=data.x.shape[1],
    hidden_channels=128,
    out_channels=64
).to(device)
model.load_state_dict(torch.load("data/gcn_model.pth", map_location=device))
model.eval()

# khởi tạo mô hình và tính toán embedding cho tất cả các phim
with torch.no_grad():
    hidden_embedding = model.extract_features(data.x, data.edge_index)
print("Khởi tạo hoàn tất!")



def find_best_match(title, titles):
    matches = get_close_matches(title.lower(), [t.lower() for t in titles], n=1, cutoff=0.6)
    if matches:
        return df[df['title'].str.lower() == matches[0]].index[0]
    return None

def recommend_by_embedding(query_vector, exclude_index=None, top_k=5):
    similarity = F.cosine_similarity(query_vector, hidden_embedding)
    if exclude_index is not None:
        similarity[exclude_index] = -1
    top_indices = similarity.topk(top_k).indices.tolist()
    return top_indices


@app.route('/api/recommend/by_title', methods=['POST'])
def recommend_by_title():
    data_json = request.json
    title = data_json.get('title', '').strip()
    top_k = data_json.get('top_k', 5)

    index = find_best_match(title, df['title'])
    if index is None:
        return jsonify({"error": "Không tìm thấy phim phù hợp"}), 404

    selected_movie = {
        "title": df.iloc[index]['title'],
        "genres": df.iloc[index]['genre_names'],
        "rating": float(df.iloc[index]['vote_average'])
    }

    indices = recommend_by_embedding(hidden_embedding[index].unsqueeze(0), exclude_index=index, top_k=top_k)
    recommendations = [ 
        {
            "title": df.iloc[i]['title'],
            "genres": df.iloc[i]['genre_names'],
            "rating": float(df.iloc[i]['vote_average'])
        } for i in indices
    ]

    return jsonify({
        "selected_movie": selected_movie,
        "recommendations": recommendations
    })

@app.route('/api/recommend/by_genre', methods=['POST'])
def recommend_by_genre():
    data_json = request.json
    genres = data_json.get('genres', [])  # Expect list
    top_k = data_json.get('top_k', 10)

    genres = [g.strip().lower() for g in genres if g.strip() != '']

    if not genres:
        return jsonify({"error": "Bạn cần nhập ít nhất 1 thể loại"}), 400

    available_genres = sorted({g for genres_list in df['genre_names'] for g in genres_list})
    matched = df[df['genre_names'].apply(lambda movie_genres: any(
        genre in item.lower() for genre in genres for item in movie_genres
    ))]

    if matched.empty:
        close = get_close_matches(genres[0], [g.lower() for g in available_genres], n=1)
        if close:
            return jsonify({
                "error": f"Không tìm thấy thể loại '{genres[0]}'",
                "suggestion": f"Bạn có muốn thử với '{close[0]}' không?"
            }), 404
        else:
            return jsonify({"error": "Không tìm thấy thể loại phù hợp"}), 404

    indices = matched.index.tolist()
    query_vec = hidden_embedding[indices].mean(dim=0).unsqueeze(0)
    rec_indices = recommend_by_embedding(query_vec, top_k=top_k)

    recommendations = [ 
        {
            "title": df.iloc[i]['title'],
            "genres": df.iloc[i]['genre_names'],
            "rating": float(df.iloc[i]['vote_average'])
        } for i in rec_indices
    ]

    return jsonify({
        "recommendations": recommendations
    })

@app.route('/api/genres', methods=['GET'])
def get_all_genres():
    genres = sorted({g for genres in df['genre_names'] for g in genres})
    return jsonify({"genres": genres})

@app.route('/api/movies', methods=['GET'])
def get_all_movies():
    movies = [{"title": row['title'], "genres": row['genre_names']} for _, row in df.iterrows()]
    return jsonify({"movies": movies})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
