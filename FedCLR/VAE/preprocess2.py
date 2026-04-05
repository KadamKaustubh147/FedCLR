import pandas as pd
import numpy as np
import random

def k_core_filter(df, user_col, item_col, rating_col, k=5):
    """
    Iteratively filter until all users and items have >= k interactions.
    Must be iterative, not one-pass.
    """
    while True:
        before = len(df)
        
        # Filter items with less than k interactions
        item_counts = df.groupby(item_col)[rating_col].count()
        valid_items = item_counts[item_counts >= k].index
        df = df[df[item_col].isin(valid_items)]
        
        # Filter users with less than k interactions
        user_counts = df.groupby(user_col)[rating_col].count()
        valid_users = user_counts[user_counts >= k].index
        df = df[df[user_col].isin(valid_users)]
        
        after = len(df)
        
        # If nothing was removed this iteration, we've converged
        if before == after:
            break
    
    return df


def preprocess():
    print("Loading raw data...")

    # ============================================================
    # CHANGE THESE PATHS TO MATCH YOUR ACTUAL FILE NAMES
    # ============================================================
    movies = pd.read_csv("../douban_dataset/moviereviews_cleaned.txt", sep="\t")
    music  = pd.read_csv("../douban_dataset/musicreviews_cleaned.txt",  sep="\t")

    # Strip quotes from column names if present
    movies.columns = [c.strip('"') for c in movies.columns]
    music.columns  = [c.strip('"') for c in music.columns]

    print("Movie columns:", movies.columns.tolist())
    print("Music columns:", music.columns.tolist())

    # ============================================================
    # CHANGE THESE COLUMN NAMES TO MATCH YOUR ACTUAL COLUMNS
    # ============================================================
    movies = movies[["user_id", "movie_id", "rating"]].copy()
    music  = music[["user_id",  "music_id", "rating"]].copy()

    # Rename item columns to generic name for filtering function
    movies = movies.rename(columns={"movie_id": "item_id"})
    music  = music.rename(columns={"music_id":  "item_id"})

    # Convert types
    movies["user_id"] = movies["user_id"].astype(int)
    movies["item_id"] = movies["item_id"].astype(int)
    movies["rating"]  = movies["rating"].astype(float)

    music["user_id"]  = music["user_id"].astype(int)
    music["item_id"]  = music["item_id"].astype(int)
    music["rating"]   = music["rating"].astype(float)

    # Drop duplicates
    movies = movies.drop_duplicates(subset=["user_id", "item_id"])
    music  = music.drop_duplicates(subset=["user_id",  "item_id"])

    print(f"\nBefore filtering:")
    print(f"  Movies: {movies['user_id'].nunique()} users, {movies['item_id'].nunique()} items")
    print(f"  Music:  {music['user_id'].nunique()} users,  {music['item_id'].nunique()} items")

    # =========================
    # STEP 1 — K-CORE FILTERING (k=5)
    # =========================
    K = 5
    print(f"\nApplying k={K} core filtering...")
    movies = k_core_filter(movies, "user_id", "item_id", "rating", k=K)
    music  = k_core_filter(music,  "user_id", "item_id", "rating", k=K)

    print(f"\nAfter k={K} core filtering:")
    print(f"  Movies: {movies['user_id'].nunique()} users, {movies['item_id'].nunique()} items")
    print(f"  Music:  {music['user_id'].nunique()} users,  {music['item_id'].nunique()} items")

    # =========================
    # STEP 2 — KEEP ONLY OVERLAPPING USERS (appear in BOTH domains)
    # For Movie&Music CDR task → target ~1666 overlapping users
    # =========================
    movie_users = set(movies["user_id"].unique())
    music_users = set(music["user_id"].unique())

    common_users = movie_users & music_users
    print(f"\nOverlapping users (Movie & Music): {len(common_users)}")
    print(f"  Target from paper: ~1666")

    movies = movies[movies["user_id"].isin(common_users)]
    music  = music[music["user_id"].isin(common_users)]

    print(f"\nAfter overlap filter:")
    print(f"  Movies: {movies['user_id'].nunique()} users, {movies['item_id'].nunique()} items")
    print(f"  Music:  {music['user_id'].nunique()} users,  {music['item_id'].nunique()} items")
    print(f"  Target items — Movies: ~9565, Music: ~5567")

    # =========================
    # STEP 3 — REINDEX USERS
    # =========================
    user_map = {u: i for i, u in enumerate(sorted(common_users))}

    movies["user_id"] = movies["user_id"].map(user_map)
    music["user_id"]  = music["user_id"].map(user_map)

    num_users = len(user_map)

    # =========================
    # STEP 4 — REINDEX ITEMS
    # =========================
    movie_item_map = {m: i for i, m in enumerate(sorted(movies["item_id"].unique()))}
    music_item_map = {m: i for i, m in enumerate(sorted(music["item_id"].unique()))}

    movies["item_id"] = movies["item_id"].map(movie_item_map)
    music["item_id"]  = music["item_id"].map(music_item_map)

    num_movies = len(movie_item_map)
    num_music  = len(music_item_map)

    print(f"\nFinal dataset:")
    print(f"  Users:  {num_users}  (target: ~1666)")
    print(f"  Movies: {num_movies} (target: ~9565)")
    print(f"  Music:  {num_music}  (target: ~5567)")

    # After STEP 4 (reindexing), add this STEP 4.5 before building matrices

    # # =========================
    # # STEP 4.5 — COLD-START SPLIT
    # # 20% of users = cold-start (test), 80% = train
    # # =========================
    # all_user_ids = list(range(num_users))
    # random.seed(42)
    # random.shuffle(all_user_ids)

    # split = int(num_users * 0.8)
    # train_users = set(all_user_ids[:split])
    # cold_start_users = set(all_user_ids[split:])

    # print(f"\nCold-start split:")
    # print(f"  Train users: {len(train_users)}")
    # print(f"  Cold-start (test) users: {len(cold_start_users)}")

    # np.save("train_users.npy", np.array(list(train_users)))
    # np.save("cold_start_users.npy", np.array(list(cold_start_users)))

    # =========================
    # STEP 5 — BINARIZE RATINGS
    # =========================
    movies["rating"] = (movies["rating"] >= 3).astype(np.float32)
    music["rating"]  = (music["rating"]  >= 3).astype(np.float32)

    # =========================
    # STEP 6 — BUILD MATRICES
    # Movie = Source domain (rich data)
    # Music = Target domain (sparse data)
    # =========================
    X_source = np.zeros((num_users, num_movies), dtype=np.float32)
    X_target = np.zeros((num_users, num_music),  dtype=np.float32)

    for _, row in movies.iterrows():
        X_source[int(row["user_id"]), int(row["item_id"])] = row["rating"]

    for _, row in music.iterrows():
        X_target[int(row["user_id"]), int(row["item_id"])] = row["rating"]

    print(f"\nMatrix shapes:")
    print(f"  X_source: {X_source.shape}")
    print(f"  X_target: {X_target.shape}")
    print(f"  X_source sparsity: {(X_source == 0).mean():.4f} (target: ~0.9564)")
    print(f"  X_target sparsity: {(X_target == 0).mean():.4f} (target: ~0.9954)")

    # =========================
    # STEP 7 — SAVE
    # =========================
    np.save("X_source.npy", X_source)
    np.save("X_target.npy", X_target)
    np.save("user_map.npy",
            np.array(list(user_map.items()), dtype=object),
            allow_pickle=True)

    print("\nSaved X_source.npy, X_target.npy, user_map.npy")
    print("\nDone!")


if __name__ == "__main__":
    preprocess()