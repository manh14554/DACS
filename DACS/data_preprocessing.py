# data_preprocessing.py

import json
import pandas as pd
import torch
from pymongo import MongoClient
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

def json_parse(s):
    if isinstance(s, list): return s
    if pd.isna(s): return []
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        try:
            return json.loads(s.replace("'", "\""))
        except Exception:
            return []
    except Exception:
        return []

def load_and_process_movies():
    # Kết nối MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["tmdb"]
    movies_col = db["movies"]
    
    # Load dữ liệu
    movies_data = list(movies_col.find())
    df = pd.DataFrame(movies_data)

    # Parse các trường JSON
    json_cols = ['genres', 'keywords', 'production_companies', 'spoken_languages']
    for col in json_cols:
        if col in df.columns:
            df[col] = df[col].apply(json_parse)

    # Trích xuất danh sách genre
    df['genre_names'] = df['genres'].apply(lambda g: [x['name'] for x in g if 'name' in x])

    return df

def encode_genre_features(df):
    mlb = MultiLabelBinarizer()
    genre_encoded = mlb.fit_transform(df['genre_names'])
    x = torch.tensor(genre_encoded, dtype=torch.float)
    return x

def normalize_year(df):
    scaler = MinMaxScaler()
    years = df['release_date'].fillna('1900-01-01').str[:4].astype(int)
    df['release_year'] = scaler.fit_transform(years.values.reshape(-1, 1))
    return df