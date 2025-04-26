import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from sentence_transformers import SentenceTransformer
import torch
import os

def load_data_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df

def process_genres(df):
    # Chuyển cột genres từ chuỗi thành list các thể loại
    df['genres'] = df['genres'].fillna("").apply(lambda x: [genre.strip() for genre in x.split(',') if genre.strip()])
    return df

def encode_genres(df):
    mlb = MultiLabelBinarizer() # mlb là MultiLabelBinarizer để mã hóa nhiều nhãn
    genre_encoded = mlb.fit_transform(df['genres'])# Chuyển đổi genres thành các nhãn nhị phân
    genre_tensor = torch.tensor(genre_encoded, dtype=torch.float)# Chuyển đổi thành tensor PyTorch 
    return genre_tensor, mlb.classes_

def encode_titles(df, model_name='sentence-transformers/all-MiniLM-L6-v2'):# Chọn mô hình SentenceTransformer
    model = SentenceTransformer(model_name)# Tải mô hình
    embeddings = model.encode(df['title'].astype(str).tolist(), convert_to_tensor=True)# Chuyển đổi tiêu đề thành tensor
    return embeddings

def concatenate_features(genre_tensor, title_embeddings):# Nối các đặc trưng lại với nhau
    return torch.cat((genre_tensor, title_embeddings), dim=1)# Nối genre_tensor và title_embeddings theo chiều 1 (cột)

def preprocess(csv_path):# Chức năng chính để xử lý dữ liệu
    df = load_data_from_csv(csv_path)
    df = process_genres(df)  # <-- Thêm dòng này để xử lý genres trước khi encode
    genre_tensor, genre_classes = encode_genres(df)# Chuyển đổi genres thành tensor
    title_embeddings = encode_titles(df)# Chuyển đổi tiêu đề thành tensor
    features = concatenate_features(genre_tensor, title_embeddings)# Nối các đặc trưng lại với nhau

if __name__ == "__main__":
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    df, features, genre_classes = preprocess("../data/TMDB_movie_dataset_v11.csv")

    torch.save(features, os.path.join(output_dir, "features.pt"))
    df.to_pickle(os.path.join(output_dir, "movies_df.pkl"))

    print("Đã xử lý và lưu xong dữ liệu.")
