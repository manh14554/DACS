import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from sentence_transformers import SentenceTransformer
import torch
import os

def load_and_merge_data(movies_path, credits_path):
    # Đọc dữ liệu từ 2 file
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    # Tránh trùng cột 'title' từ credits (nếu có)
    if 'title' in credits.columns:
        credits = credits.drop(columns=['title'])

    # Gộp dữ liệu theo movie_id
    df = movies.merge(credits, left_on='id', right_on='movie_id')

    return df

def process_genres(df):
    # Chuyển cột genres từ chuỗi JSON -> list các thể loại
    df['genres'] = df['genres'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)])
    return df

def encode_genres(df):
    # One-hot encoding cho genre
    mlb = MultiLabelBinarizer()
    genre_encoded = mlb.fit_transform(df['genres'])
    genre_tensor = torch.tensor(genre_encoded, dtype=torch.float)
    return genre_tensor, mlb.classes_

def encode_titles(df, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    # Encode tiêu đề phim bằng SentenceTransformer
    model = SentenceTransformer(model_name)
    embeddings = model.encode(df['title'].tolist(), convert_to_tensor=True)
    return embeddings

def concatenate_features(genre_tensor, title_embeddings):
    # Nối genre one-hot + embedding tiêu đề → làm input cho GCN
    return torch.cat((genre_tensor, title_embeddings), dim=1)

def preprocess(movies_path, credits_path):
    df = load_and_merge_data(movies_path, credits_path)
    df = process_genres(df)
    genre_tensor, genre_classes = encode_genres(df)
    title_embeddings = encode_titles(df)
    features = concatenate_features(genre_tensor, title_embeddings)
    return df, features, genre_classes

if __name__ == "__main__":
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    df, features, genre_classes = preprocess(
        "../data/tmdb_5000_movies.csv", "../data/tmdb_5000_credits.csv"
    )

    torch.save(features, os.path.join(output_dir, "features.pt"))
    df.to_pickle(os.path.join(output_dir, "movies_df.pkl"))

    print("Đã lưu xong features và dataframe.")
