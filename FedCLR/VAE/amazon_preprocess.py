# amazon_preprocess.py
# ===============================
# Imports & Config
# ===============================

import os
import numpy as np
import pandas as pd
import random

random.seed(42)
np.random.seed(42)

BASE_PATH  = r"C:\dev_stuff\FedCLR_implementation\datasets\dual-user-inter\dataset"
DOMAIN     = "game_video"   # "game_video" → source=game, target=video
                             # "video_game" → source=video, target=game
                             # "cloth_sport", "sport_cloth" also work
OUT_DIR    = os.path.join("output", DOMAIN)
os.makedirs(OUT_DIR, exist_ok=True)

# ===============================
# Load train / valid / test splits
# ===============================

def load_split(domain, split):
    path = os.path.join(BASE_PATH, domain, f"{split}.txt")
    df   = pd.read_csv(path, sep="\t", header=None, names=["user_id", "item_id"])
    print(f"  [{domain}] {split}: {df.shape[0]} rows")
    return df

print(f"\nLoading DOMAIN: {DOMAIN}")
src_train = load_split(DOMAIN, "train")
src_valid = load_split(DOMAIN, "valid")
src_test  = load_split(DOMAIN, "test")

# For the target domain we use the reversed pair
# e.g. if DOMAIN="game_video", reversed is "video_game"
parts    = DOMAIN.split("_")
rev_domain = f"{parts[1]}_{parts[0]}"   # "game_video" → "video_game"
print(f"\nLoading REVERSE (target) DOMAIN: {rev_domain}")
tgt_train = load_split(rev_domain, "train")
tgt_valid = load_split(rev_domain, "valid")
tgt_test  = load_split(rev_domain, "test")

# ===============================
# Pool all rows to find overlapping users
# ===============================

src_all = pd.concat([src_train, src_test], ignore_index=True)
tgt_all = pd.concat([tgt_train, tgt_test], ignore_index=True)

src_users = set(src_all["user_id"].unique())
tgt_users = set(tgt_all["user_id"].unique())
common_users = src_users & tgt_users
print(f"\nSource users   : {len(src_users)}")
print(f"Target users   : {len(tgt_users)}")
print(f"Overlapping    : {len(common_users)}")

# Filter to common users
src_all = src_all[src_all["user_id"].isin(common_users)].copy()
tgt_all = tgt_all[tgt_all["user_id"].isin(common_users)].copy()

# ===============================
# Reindex users & items
# ===============================

user_map = {u: i for i, u in enumerate(sorted(common_users))}
src_all["user_id"] = src_all["user_id"].map(user_map)
tgt_all["user_id"] = tgt_all["user_id"].map(user_map)

src_item_map = {m: i for i, m in enumerate(sorted(src_all["item_id"].unique()))}
tgt_item_map = {m: i for i, m in enumerate(sorted(tgt_all["item_id"].unique()))}
src_all["item_id"] = src_all["item_id"].map(src_item_map)
tgt_all["item_id"] = tgt_all["item_id"].map(tgt_item_map)

num_users     = len(user_map)
num_src_items = len(src_item_map)
num_tgt_items = len(tgt_item_map)
print(f"\nUsers       : {num_users}")
print(f"Source items: {num_src_items}")
print(f"Target items: {num_tgt_items}")

# ===============================
# Cold-start split (80 / 20 on users)
# ===============================

all_user_ids = list(range(num_users))
random.shuffle(all_user_ids)
split_idx       = int(num_users * 0.8)
train_users     = set(all_user_ids[:split_idx])
coldstart_users = set(all_user_ids[split_idx:])
print(f"\nTrain users     : {len(train_users)}")
print(f"Cold-start users: {len(coldstart_users)}")

# ===============================
# Build interaction matrices
# ===============================

# Source matrix — all users (used for cross-domain transfer signal)
X_source = np.zeros((num_users, num_src_items), dtype=np.float32)
for _, row in src_all.iterrows():
    X_source[int(row["user_id"]), int(row["item_id"])] = 1.0

# Target train — only train_users have labels visible during training
X_target = np.zeros((num_users, num_tgt_items), dtype=np.float32)
for _, row in tgt_all.iterrows():
    uid = int(row["user_id"])
    if uid in train_users:
        X_target[uid, int(row["item_id"])] = 1.0

# Target test — only cold-start users; used for evaluation
X_target_test = np.zeros((num_users, num_tgt_items), dtype=np.float32)
for _, row in tgt_all.iterrows():
    uid = int(row["user_id"])
    if uid in coldstart_users:
        X_target_test[uid, int(row["item_id"])] = 1.0

# ===============================
# Sparsity report
# ===============================

print(f"\nX_source shape     : {X_source.shape}  sparsity: {(X_source == 0).mean():.4f}")
print(f"X_target shape     : {X_target.shape}  sparsity: {(X_target == 0).mean():.4f}")
print(f"X_target_test shape: {X_target_test.shape}  sparsity: {(X_target_test == 0).mean():.4f}")

# ===============================
# Save
# ===============================

np.save(os.path.join(OUT_DIR, "X_source.npy"),          X_source)
np.save(os.path.join(OUT_DIR, "X_target.npy"),          X_target)
np.save(os.path.join(OUT_DIR, "X_target_test.npy"),     X_target_test)
np.save(os.path.join(OUT_DIR, "train_users.npy"),       np.array(list(train_users)))
np.save(os.path.join(OUT_DIR, "cold_start_users.npy"),  np.array(list(coldstart_users)))

print(f"\nSaved to: {OUT_DIR}")
print("Done!")