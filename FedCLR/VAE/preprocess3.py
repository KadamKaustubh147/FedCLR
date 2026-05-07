import numpy as np
import pandas as pd
import random

random.seed(42)
np.random.seed(42)

# Load .dat files (tab separated, no header, cols: user_id, item_id, rating)
movies = pd.read_csv("../../Data_GA_DTCDR/douban_movie/ratings.dat", sep="\t", header=None, names=["user_id","item_id","rating"])
music  = pd.read_csv("../../Data_GA_DTCDR/douban_book/ratings.dat",  sep="\t", header=None, names=["user_id","item_id","rating"])

# Overlapping users
movie_users = set(movies["user_id"].unique())
music_users = set(music["user_id"].unique())
common_users = movie_users & music_users
print(f"Overlapping users: {len(common_users)}")  # should be 1666

# Filter to overlapping users only
movies = movies[movies["user_id"].isin(common_users)]
music  = music[music["user_id"].isin(common_users)]

# Reindex users
user_map = {u: i for i, u in enumerate(sorted(common_users))}
movies["user_id"] = movies["user_id"].map(user_map)
music["user_id"]  = music["user_id"].map(user_map)
num_users = len(user_map)

# Reindex items
movie_item_map = {m: i for i, m in enumerate(sorted(movies["item_id"].unique()))}
music_item_map = {m: i for i, m in enumerate(sorted(music["item_id"].unique()))}
movies["item_id"] = movies["item_id"].map(movie_item_map)
music["item_id"]  = music["item_id"].map(music_item_map)
num_movies = len(movie_item_map)
num_music  = len(music_item_map)

print(f"Users: {num_users}, Movies: {num_movies}, Music: {num_music}")

# Binarize — keep only positive interactions
movies = movies[movies["rating"] >= 3]
music  = music[music["rating"]  >= 3]

# Cold-start split — 20% test
all_user_ids = list(range(num_users))
random.shuffle(all_user_ids)
split = int(num_users * 0.8)
train_users     = set(all_user_ids[:split])
coldstart_users = set(all_user_ids[split:])
print(f"Train users: {len(train_users)}, Cold-start users: {len(coldstart_users)}")

# Build matrices
X_source      = np.zeros((num_users, num_movies), dtype=np.float32)
X_target      = np.zeros((num_users, num_music),  dtype=np.float32)
X_target_test = np.zeros((num_users, num_music),  dtype=np.float32)

for _, row in movies.iterrows():
    X_source[int(row["user_id"]), int(row["item_id"])] = 1.0

for _, row in music.iterrows():
    uid = int(row["user_id"])
    if uid in train_users:
        X_target[uid, int(row["item_id"])] = 1.0
    else:
        X_target_test[uid, int(row["item_id"])] = 1.0

print(f"X_source shape: {X_source.shape}, sparsity: {(X_source==0).mean():.4f}")
print(f"X_target shape: {X_target.shape}, sparsity: {(X_target==0).mean():.4f}")

# Save
np.save("X_source.npy",      X_source)
np.save("X_target.npy",      X_target)
np.save("X_target_test.npy", X_target_test)
np.save("train_users.npy",   np.array(list(train_users)))
np.save("cold_start_users.npy", np.array(list(coldstart_users)))

print("Done!")