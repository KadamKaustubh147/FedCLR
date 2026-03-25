import pandas as pd
import numpy as np


def preprocess():
    print("Loading raw data...")

    movies = pd.read_csv("../douban_dataset/moviereviews_cleaned.txt", sep="\t")
    music = pd.read_csv("../douban_dataset/musicreviews_cleaned.txt", sep="\t")

    # removing quotes from column names
    movies.columns = [c.strip('"') for c in movies.columns]
    music.columns = [c.strip('"') for c in music.columns]

    # removing everything else and keeping three important columns
    movies = movies[["user_id", "movie_id", "rating"]]
    music = music[["user_id", "music_id", "rating"]]

    # Convert types
    movies["user_id"] = movies["user_id"].astype(int)
    movies["movie_id"] = movies["movie_id"].astype(int)
    movies["rating"] = movies["rating"].astype(float)

    music["user_id"] = music["user_id"].astype(int)
    music["music_id"] = music["music_id"].astype(int)
    music["rating"] = music["rating"].astype(float)

    # Binarize
    movies["rating"] = (movies["rating"] >= 3).astype(int)
    music["rating"] = (music["rating"] >= 3).astype(int)

    # Keep overlapping users
    common_users = set(movies["user_id"]).intersection(set(music["user_id"]))
    movies = movies[movies["user_id"].isin(common_users)]
    music = music[music["user_id"].isin(common_users)]

    print(f"Overlapping users: {len(common_users)}")

    # Reindex users
    user_map = {u: i for i, u in enumerate(sorted(common_users))}
    movies["user_id"] = movies["user_id"].map(user_map)
    music["user_id"] = music["user_id"].map(user_map)

    # Reindex items
    
    # user_id: 843, 12903, 999999
    # movie_id: 21292, 34584
    
    movie_map = {m: i for i, m in enumerate(movies["movie_id"].unique())}
    music_map = {m: i for i, m in enumerate(music["music_id"].unique())}

    movies["movie_id"] = movies["movie_id"].map(movie_map)
    music["music_id"] = music["music_id"].map(music_map)

    num_users = len(user_map)
    num_movies = len(movie_map)
    num_music = len(music_map)

    print(f"Users: {num_users}, Movies: {num_movies}, Music: {num_music}")

    # Build matrices
    X_source = np.zeros((num_users, num_movies), dtype=np.float32)
    X_target = np.zeros((num_users, num_music), dtype=np.float32)

    for _, row in movies.iterrows():
        X_source[row["user_id"], row["movie_id"]] = row["rating"]

    for _, row in music.iterrows():
        X_target[row["user_id"], row["music_id"]] = row["rating"]

    # Save
    np.save("X_source.npy", X_source)
    np.save("X_target.npy", X_target)

    print("Saved X_source.npy and X_target.npy")


if __name__ == "__main__":
    preprocess()

