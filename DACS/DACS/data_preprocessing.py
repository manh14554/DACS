import json
import pandas as pd
import torch
from pymongo import MongoClient
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
#sklearn dùng để xử lý TF-IDF, MinMaxScaler dùng để chuẩn hóa dữ liệu
# PyTorch dùng để xử lý tensor và mô hình hóa

# Xử lý JSON
def json_parse(s):
    if isinstance(s, list): return s
    if pd.isna(s): return []
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        try:
            return json.loads(s.replace("'", '"'))
        except Exception:
            return []
    except Exception:
        return []


# Load và xử lý phim
def load_and_process_movies():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["tmdb"]
    movies_col = db["movies"]

    movies_data = list(movies_col.find())
    df = pd.DataFrame(movies_data)

    json_cols = ['genres', 'keywords', 'production_companies', 'spoken_languages']
    for col in json_cols:
        if col in df.columns:
            df[col] = df[col].apply(json_parse)

    df['genre_names'] = df['genres'].apply(lambda g: [x['name'] for x in g if 'name' in x])
    df['keyword_names'] = df['keywords'].apply(lambda k: [x['name'] for x in k if 'name' in x])

    return df

# Tạo one-hot cho genres
def encode_genre_features(df):
    mlb = MultiLabelBinarizer()
    genre_encoded = mlb.fit_transform(df['genre_names'])
    return torch.tensor(genre_encoded, dtype=torch.float)


# TF-IDF cho overview
def extract_tfidf_features(df, field='overview', max_features=100):
    df[field] = df[field].fillna("").astype(str)
    tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df[field])
    return torch.tensor(tfidf_matrix.toarray(), dtype=torch.float)

# Chuẩn hóa numeric feature
def normalize_numeric_features(df, cols):
    df[cols] = df[cols].fillna(0)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[cols])
    return torch.tensor(scaled, dtype=torch.float)


# Chuẩn hóa năm
def normalize_year(df):
    scaler = MinMaxScaler()
    years = df['release_date'].fillna('1900-01-01').str[:4].astype(int)
    df['release_year'] = scaler.fit_transform(years.values.reshape(-1, 1))
    return df  


# Kết hợp tất cả feature
def combine_all_features(genre_tensor, tfidf_tensor, numeric_tensor, year_tensor):
    return torch.cat([genre_tensor, tfidf_tensor, numeric_tensor, year_tensor], dim=1)