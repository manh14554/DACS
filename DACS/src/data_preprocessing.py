import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sentence_transformers import SentenceTransformer
import torch
import os

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

def encode_titles(df, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """Chuyển đổi tiêu đề thành tensor embedding."""
    model = SentenceTransformer(model_name)
    titles = df['title'].fillna("").astype(str).tolist()
    embeddings = model.encode(titles, convert_to_tensor=True)
    print(f"Kích thước title_embeddings: {embeddings.shape}")
    return embeddings

def concatenate_features(genre_tensor, title_embeddings):
    """Nối các đặc trưng thành một tensor duy nhất."""
    if genre_tensor.shape[0] != title_embeddings.shape[0]:
        raise ValueError("Số lượng phim không khớp giữa genre_tensor và title_embeddings.")
    features = torch.cat((title_embeddings, genre_tensor), dim=1)
    print(f"Kích thước features sau khi nối: {features.shape}")
    return features

def preprocess(csv_path):
    """Chức năng chính để xử lý dữ liệu."""
    df = load_data_from_csv(csv_path)
    df = df.dropna(subset=['title'])
    print(f"Số lượng phim sau khi loại bỏ title NaN: {len(df)}")
    df = process_genres(df)
    genre_tensor, genre_classes = encode_genres(df)
    title_embeddings = encode_titles(df)
    features = concatenate_features(genre_tensor, title_embeddings)
    return df, features, genre_classes

if __name__ == "__main__":
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    df, features, genre_classes = preprocess("../datas/TMDB_movie_dataset_v11.csv")

    torch.save(features, os.path.join(output_dir, "features.pt"))
    df.to_pickle(os.path.join(output_dir, "movies_df.pkl"))
    with open(os.path.join(output_dir, "genre_classes.txt"), "w") as f:
        f.write("\n".join(genre_classes))

    print("Đã xử lý và lưu xong dữ liệu.")
