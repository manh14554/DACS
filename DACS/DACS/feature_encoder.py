from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# Khởi tạo model embedding
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

def encode_combined_features(df: pd.DataFrame):
    # 1. Encode thể loại (genre)
    mlb = MultiLabelBinarizer()
    genre_encoded = mlb.fit_transform(df['genre_names'])
    genre_tensor = torch.tensor(genre_encoded, dtype=torch.float)

    # 2. Encode tiêu đề phim (title)
    title_embeddings = bert_model.encode(df['title'].tolist(), convert_to_tensor=True)

    # 3. Kết hợp cả 2
    combined = torch.cat([genre_tensor, title_embeddings], dim=1)
    return combined
