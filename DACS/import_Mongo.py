import pandas as pd
from pymongo import MongoClient

# Kết nối MongoDB (mặc định là localhost)
client = MongoClient("mongodb://localhost:27017/")

# Tạo database và collection
db = client["tmdb"]  # tên database
movies_col = db["movies"]
credits_col = db["credits"]

# Đọc file CSV
movies_df = pd.read_csv("C:/Users/Admin/Downloads/archive/tmdb_5000_movies.csv")
credits_df = pd.read_csv("C:/Users/Admin/Downloads/archive/tmdb_5000_credits.csv")

# Chuyển về dict (records) và insert
movies_data = movies_df.to_dict(orient="records")
credits_data = credits_df.to_dict(orient="records")

# Xoá dữ liệu cũ (nếu có) rồi insert
movies_col.delete_many({})
credits_col.delete_many({})

movies_col.insert_many(movies_data)
credits_col.insert_many(credits_data)

print(" Import thành công!")

