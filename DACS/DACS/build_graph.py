import torch
from torch_geometric.data import Data
from itertools import combinations
from data_preprocessing import (
    load_and_process_movies,
    encode_genre_features,
    extract_tfidf_features,
    normalize_numeric_features,
    combine_all_features,
    normalize_year
)

# Hàm xây dựng cạnh (edges) giữa các phim có cùng thể loại (genre)
def build_edges_from_genres(df):
    genre_to_movies = {}  # Từ điển lưu các thể loại và danh sách các phim thuộc thể loại đó

    # Duyệt qua tất cả các phim và gán chỉ số của phim vào thể loại tương ứng
    for idx, genres in enumerate(df['genre_names']):#enumerate để lấy chỉ số và giá trị của genres
        for genre in genres:
            genre_to_movies.setdefault(genre, []).append(idx) # Nếu genre chưa có trong từ điển, khởi tạo danh sách rỗng

    edge_set = set()  # Dùng set để lưu các cặp cạnh duy nhất

    # Tạo các cạnh giữa các phim có cùng thể loại
    for movies in genre_to_movies.values():
        for i, j in combinations(movies, 2):
            edge_set.add((i, j))  # Thêm cạnh từ i đến j
            edge_set.add((j, i))  # Thêm cạnh ngược lại (do đồ thị vô hướng)

    # Chuyển đổi các cạnh từ dạng set sang tensor
    edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()  # Chuyển sang định dạng (2, num_edges)
    return edge_index

def main():
    # Bắt đầu quá trình tải và xử lý dữ liệu
    print("Loading & processing movie data...")
    df = load_and_process_movies()  # Tải dữ liệu phim từ file hoặc database
    print(f"Loaded {len(df)} movies")  # In số lượng phim đã tải

    # Tiến hành mã hóa các đặc trưng của phim
    print("Encoding features...")
    genre_feat = encode_genre_features(df)  # Mã hóa đặc trưng thể loại thành tensor
    tfidf_feat = extract_tfidf_features(df)  # Mã hóa TF-IDF cho các mô tả phim
    numeric_feat = normalize_numeric_features(df, ['popularity', 'vote_count', 'runtime'])  # Chuẩn hóa các đặc trưng số
    df = normalize_year(df)  # Chuẩn hóa năm phát hành
    year_feat = torch.tensor(df['release_year'].values, dtype=torch.float).unsqueeze(1)  # Đặc trưng năm phát hành

    # Gộp tất cả các đặc trưng thành một tensor đặc trưng cuối cùng
    x = combine_all_features(genre_feat, tfidf_feat, numeric_feat, year_feat)
    print(f"Feature tensor shape: {x.shape}")  # In kích thước của tensor đặc trưng

    # Xây dựng đồ thị các cạnh từ thể loại phim
    print("Building graph edges based on genres...")
    edge_index = build_edges_from_genres(df)  # Tạo các cạnh giữa các phim có cùng thể loại
    print(f"Total edges created: {edge_index.shape[1]}")  # In số lượng cạnh được tạo ra

    # Kiểm tra và tạo nhãn cho các phim (ở đây là điểm trung bình đánh giá)
    if 'vote_average' in df.columns:
        y = torch.tensor(df['vote_average'].values, dtype=torch.float)  # Chuyển 'vote_average' thành tensor
    else:
        raise ValueError("'vote_average' column missing!")  # Nếu thiếu cột 'vote_average', báo lỗi

    # Tạo đối tượng Data cho PyTorch Geometric (đồ thị dữ liệu)
    data = Data(x=x, edge_index=edge_index, y=y)

    torch.save(data, "movie_graph.pt")
    print("Graph saved to movie_graph.pt")  


if __name__ == "__main__":
    main()
