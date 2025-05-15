import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sentence_transformers import SentenceTransformer
import torch
import os
from torch.utils.data import DataLoader

def load_data_from_csv(csv_path):
    """Tải dữ liệu từ file CSV."""
    try:
        df = pd.read_csv(csv_path)
        print(f"Đã tải thành công dữ liệu từ {csv_path}. Số lượng phim: {len(df)}")
        return df
    except FileNotFoundError:
        print(f"Không tìm thấy file tại {csv_path}. Vui lòng kiểm tra đường dẫn.")
        exit(1)

def process_genres(df):
    """Chuyển cột genres từ chuỗi thành list các thể loại."""
    def parse_genres(genre_str):
        if pd.isna(genre_str) or not isinstance(genre_str, str):
            return []
        return [g.strip() for g in genre_str.split(",") if g.strip()]
    
    df['genres'] = df['genres'].apply(parse_genres)
    df = df[df['genres'].apply(lambda x: len(x) > 0)]
    print(f"Số lượng phim sau khi xử lý genres: {len(df)}")
    return df

def encode_genres(df):
    """Mã hóa các thể loại thành tensor nhị phân."""
    mlb = MultiLabelBinarizer()
    genre_encoded = mlb.fit_transform(df['genres'])
    genre_tensor = torch.tensor(genre_encoded, dtype=torch.float)
    print(f"Kích thước genre_tensor: {genre_tensor.shape}")
    return genre_tensor, mlb.classes_

def encode_titles(df, model_name='sentence-transformers/all-MiniLM-L6-v2', batch_size=32, device="cpu"):
    """Mã hóa tiêu đề phim thành embeddings sử dụng mô hình SentenceTransformer."""
    model = SentenceTransformer(model_name, device=device)  # Đảm bảo model sử dụng đúng thiết bị
    titles = df['title'].fillna("").astype(str).tolist()
    dataloader = DataLoader(titles, batch_size=batch_size)
    embeddings = []
    for batch in dataloader:
        embeddings.append(model.encode(batch, convert_to_tensor=True))
    embeddings = torch.cat(embeddings, dim=0).to(device)  # Đảm bảo embeddings nằm trên đúng thiết bị
    print(f"Kích thước title_embeddings: {embeddings.shape}")
    return embeddings

def concatenate_features(genre_tensor, title_embeddings):
    """Nối các đặc trưng thành một tensor duy nhất."""
    # Đảm bảo cả hai tensor nằm trên cùng thiết bị
    device = title_embeddings.device  # Lấy thiết bị của title_embeddings làm chuẩn
    genre_tensor = genre_tensor.to(device)
    # Kết hợp hai tensor
    features = torch.cat((title_embeddings, genre_tensor), dim=1)
    return features

def preprocess(csv_path, device="cpu"):
    """Chức năng chính để xử lý dữ liệu."""
    df = load_data_from_csv(csv_path)
    df = df.dropna(subset=['title'])  # Loại bỏ các phim không có tiêu đề
    print(f"Số lượng phim sau khi loại bỏ title NaN: {len(df)}")
    df = process_genres(df)
    genre_tensor, genre_classes = encode_genres(df)
    title_embeddings = encode_titles(df, device=device)
    features = concatenate_features(genre_tensor, title_embeddings)
    return df, features, genre_classes

if __name__ == "__main__":
    # Đặt thiết bị: GPU hoặc CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Đường dẫn output
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    # Xử lý dữ liệu
    df, features, genre_classes = preprocess("../datas/tmdb_5000_movies.csv", device=device)

    # Lưu kết quả
    torch.save(features, os.path.join(output_dir, "features.pt"))
    df.to_pickle(os.path.join(output_dir, "movies_df.pkl"))
    with open(os.path.join(output_dir, "genre_classes.txt"), "w") as f:
        f.write("\n".join(genre_classes))

    print("Đã xử lý và lưu xong dữ liệu.")