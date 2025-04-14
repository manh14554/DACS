from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from data_preprocessing import load_and_process_movies, encode_genre_features
from build_graph import build_edges_from_genres
from difflib import get_close_matches
import pandas as pd
from flask_cors import CORS

# Khởi tạo Flask app
app = Flask(__name__)
CORS(app)  # Cho phép truy cập từ frontend (React, v.v.)

# Định nghĩa kiến trúc mô hình GCN
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)  # Lớp 1
        self.conv2 = GCNConv(hidden_channels, out_channels)  # Lớp 2

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)  # Truyền qua lớp GCN đầu
        x = F.relu(x)  # Kích hoạt ReLU
        x = self.conv2(x, edge_index)  # Truyền qua lớp GCN thứ 2
        return x

# Tải dữ liệu phim và xử lý đặc trưng
print("Đang tải dữ liệu và mô hình...")
df = load_and_process_movies()  # Load dữ liệu phim từ MongoDB hoặc file
x = encode_genre_features(df)   # Mã hóa đặc trưng thể loại thành tensor
df = df.reset_index(drop=True)

# Tạo graph từ thông tin thể loại (edge_index)
edge_index = build_edges_from_genres(df)

# Tạo đối tượng Data cho PyG
y = torch.tensor(df['vote_average'].values, dtype=torch.float)
data = Data(x=x, edge_index=edge_index, y=y)

# Khởi tạo và tải mô hình GCN đã huấn luyện
model = GCN(in_channels=data.x.shape[1], hidden_channels=64, out_channels=1)
model.load_state_dict(torch.load("gcn_model.pth"))
model.eval()  # Đặt mô hình ở chế độ suy luận

# Trích xuất embedding ẩn từ lớp GCN đầu tiên để dùng cho gợi ý
with torch.no_grad():
    hidden_embedding = model.conv1(data.x, data.edge_index)
    hidden_embedding = F.relu(hidden_embedding)
print("Khởi tạo hoàn tất!")



# Tìm tiêu đề phim gần đúng nhất
def find_best_match(title, titles):
    matches = get_close_matches(title.lower(), [t.lower() for t in titles], n=1, cutoff=0.6)
    if matches:
        return df[df['title'].str.lower() == matches[0]].index[0]
    return None

# Gợi ý phim dựa trên độ tương đồng cosine giữa vector embedding
def recommend_by_embedding(query_vector, exclude_index=None, top_k=5):
    similarity = F.cosine_similarity(query_vector, hidden_embedding)  # Tính độ tương đồng cosine
    if exclude_index is not None:
        similarity[exclude_index] = -1  # Loại bỏ phim đầu vào khỏi danh sách gợi ý
    top_indices = similarity.topk(top_k).indices.tolist()  # Lấy top-k phim tương tự nhất
    return top_indices



# Gợi ý phim dựa trên tiêu đề
@app.route('/api/recommend/by_title', methods=['POST'])
def recommend_by_title():
    data = request.json
    title = data.get('title', '').strip()
    top_k = data.get('top_k', 5)

    index = find_best_match(title, df['title'])
    if index is None:
        return jsonify({"error": "Không tìm thấy phim phù hợp"}), 404

    # Phim được chọn làm trung tâm gợi ý
    selected_movie = {
        "title": df.iloc[index]['title'],
        "genres": df.iloc[index]['genre_names'],
        "rating": float(df.iloc[index]['vote_average'])
    }

    # Lấy danh sách gợi ý phim tương tự
    indices = recommend_by_embedding(hidden_embedding[index].unsqueeze(0), exclude_index=index, top_k=top_k)
    recommendations = []
    for i in indices:
        recommendations.append({
            "title": df.iloc[i]['title'],
            "genres": df.iloc[i]['genre_names'],
            "rating": float(df.iloc[i]['vote_average'])
        })

    return jsonify({
        "selected_movie": selected_movie,
        "recommendations": recommendations
    })

# Gợi ý phim dựa trên thể loại
@app.route('/api/recommend/by_genre', methods=['POST'])
def recommend_by_genre():
    data = request.json
    genre = data.get('genre', '').strip().lower()
    top_k = data.get('top_k', 10)

    # Danh sách tất cả thể loại có trong tập phim
    available_genres = sorted({g for genres in df['genre_names'] for g in genres})

    # Lọc ra các phim thuộc thể loại người dùng nhập
    matched = df[df['genre_names'].apply(lambda g: any(genre in item.lower() for item in g))]

    if matched.empty:
        close = get_close_matches(genre, [g.lower() for g in available_genres], n=1)
        if close:
            return jsonify({
                "error": f"Không tìm thấy thể loại '{genre}'",
                "suggestion": f"Bạn có muốn thử với '{close[0]}' không?"
            }), 404
        else:
            return jsonify({"error": "Không tìm thấy thể loại phù hợp"}), 404

    # Trung bình embedding của tất cả phim trong thể loại để gợi ý
    indices = matched.index.tolist()
    query_vec = hidden_embedding[indices].mean(dim=0).unsqueeze(0)
    rec_indices = recommend_by_embedding(query_vec, top_k=top_k)

    recommendations = []
    for i in rec_indices:
        recommendations.append({
            "title": df.iloc[i]['title'],
            "genres": df.iloc[i]['genre_names'],
            "rating": float(df.iloc[i]['vote_average'])
        })

    return jsonify({
        "genre": genre,
        "recommendations": recommendations
    })

# Lấy danh sách tất cả thể loại hiện có
@app.route('/api/genres', methods=['GET'])
def get_all_genres():
    genres = sorted({g for genres in df['genre_names'] for g in genres})
    return jsonify({"genres": genres})

# Lấy danh sách tất cả phim (dùng cho frontend dropdown chẳng hạn)
@app.route('/api/movies', methods=['GET'])
def get_all_movies():
    movies = [{"title": row['title'], "genres": row['genre_names']} 
        for _, row in df.iterrows()]
    return jsonify({"movies": movies})

# Khởi chạy Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
