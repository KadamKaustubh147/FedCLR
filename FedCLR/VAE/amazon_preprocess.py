# amazon_preprocess.py
# ===============================
# Imports & Config
# ===============================

import os
import numpy as np
import pandas as pd
import random
from scipy.sparse import lil_matrix, csr_matrix

# Keep things fully reproducible
random.seed(42)
np.random.seed(42)

BASE_PATH  = r"C:\dev_stuff\FedCLR_implementation\datasets\dual-user-inter\dataset"
DOMAIN     = "cloth_sport"   # "game_video" → source=game, target=video
                             # "video_game" → source=video, target=game
                             # "cloth_sport", "sport_cloth" also work
OUT_DIR = "."

# ===============================
# Load train / test splits (valid excluded)
# ===============================

def load_split(domain, split):
    path = os.path.join(BASE_PATH, domain, f"{split}.txt")
    df   = pd.read_csv(path, sep="\t", header=None, names=["user_id", "item_id"])
    print(f"  [{domain}] {split}: {df.shape[0]} rows")
    return df

print(f"\nLoading DOMAIN: {DOMAIN}")
src_train = load_split(DOMAIN, "train")
src_test  = load_split(DOMAIN, "test")

# reversed pair is the target domain
parts      = DOMAIN.split("_")
rev_domain = f"{parts[1]}_{parts[0]}"
print(f"\nLoading REVERSE (target) DOMAIN: {rev_domain}")
tgt_train = load_split(rev_domain, "train")
tgt_test  = load_split(rev_domain, "test")

# ===============================
# Overlapping users & 10% Sampling
# ===============================

src_all = pd.concat([src_train, src_test], ignore_index=True)
tgt_all = pd.concat([tgt_train, tgt_test], ignore_index=True)

src_users    = set(src_all["user_id"].unique())
tgt_users    = set(tgt_all["user_id"].unique())
common_users = list(src_users & tgt_users)  # Converted to list for sampling

print(f"\nTotal Source users : {len(src_users)}")
print(f"Total Target users : {len(tgt_users)}")
print(f"Total Overlapping  : {len(common_users)}")

# --- Downsample to 10% of users ---
sample_size  = int(len(common_users) * 0.10)
sampled_users = set(random.sample(sorted(common_users), sample_size))
print(f"Sampled 10% users  : {len(sampled_users)}")

# Filter datasets down to just the sampled users
src_all = src_all[src_all["user_id"].isin(sampled_users)].copy()
tgt_all = tgt_all[tgt_all["user_id"].isin(sampled_users)].copy()

# ===============================
# Reindex users & items
# ===============================

# Reindex using only our 10% subset
user_map = {u: i for i, u in enumerate(sorted(sampled_users))}
src_all["user_id"] = src_all["user_id"].map(user_map)
tgt_all["user_id"] = tgt_all["user_id"].map(user_map)

src_item_map = {m: i for i, m in enumerate(sorted(src_all["item_id"].unique()))}
tgt_item_map = {m: i for i, m in enumerate(sorted(tgt_all["item_id"].unique()))}
src_all["item_id"] = src_all["item_id"].map(src_item_map)
tgt_all["item_id"] = tgt_all["item_id"].map(tgt_item_map)

num_users     = len(user_map)
num_src_items = len(src_item_map)
num_tgt_items = len(tgt_item_map)

print(f"\nDownsampled Users   : {num_users}")
print(f"Downsampled Src items: {num_src_items}")
print(f"Downsampled Tgt items: {num_tgt_items}")

# ===============================
# Cold-start split (80 / 20 on sampled users)
# ===============================

all_user_ids    = list(range(num_users))
random.shuffle(all_user_ids)
split_idx       = int(num_users * 0.8)
train_users     = set(all_user_ids[:split_idx])
coldstart_users = set(all_user_ids[split_idx:])

print(f"\nTrain users (80%)     : {len(train_users)}")
print(f"Cold-start users (20%): {len(coldstart_users)}")

# ===============================
# Build SPARSE interaction matrices
# ===============================

print("\nBuilding sparse matrices...")

# Source — all sampled users
X_source = lil_matrix((num_users, num_src_items), dtype=np.float32)
for _, row in src_all.iterrows():
    X_source[int(row["user_id"]), int(row["item_id"])] = 1.0
X_source = csr_matrix(X_source)

# Target train — only train users have labels visible during training
X_target = lil_matrix((num_users, num_tgt_items), dtype=np.float32)
for _, row in tgt_all.iterrows():
    uid = int(row["user_id"])
    if uid in train_users:
        X_target[uid, int(row["item_id"])] = 1.0
X_target = csr_matrix(X_target)

# Target test — only cold-start users; used for evaluation
X_target_test = lil_matrix((num_users, num_tgt_items), dtype=np.float32)
for _, row in tgt_all.iterrows():
    uid = int(row["user_id"])
    if uid in coldstart_users:
        X_target_test[uid, int(row["item_id"])] = 1.0
X_target_test = csr_matrix(X_target_test)

# ===============================
# Sparsity report
# ===============================

def sparsity(m):
    return 1 - m.nnz / (m.shape[0] * m.shape[1])

def mem_mb(m):
    return (m.data.nbytes + m.indices.nbytes + m.indptr.nbytes) / 1e6

print(f"\nX_source      shape: {X_source.shape}  sparsity: {sparsity(X_source):.4f}  mem: {mem_mb(X_source):.2f} MB")
print(f"X_target      shape: {X_target.shape}  sparsity: {sparsity(X_target):.4f}  mem: {mem_mb(X_target):.2f} MB")
print(f"X_target_test shape: {X_target_test.shape}  sparsity: {sparsity(X_target_test):.4f}  mem: {mem_mb(X_target_test):.2f} MB")

# ===============================
# Save
# ===============================

np.save(os.path.join(OUT_DIR, "X_source.npy"), X_source.toarray())
np.save(os.path.join(OUT_DIR, "X_target.npy"), X_target.toarray())
np.save(os.path.join(OUT_DIR, "X_target_test.npy"), X_target_test.toarray())
np.save(os.path.join(OUT_DIR, "train_users.npy"),       np.array(list(train_users)))
np.save(os.path.join(OUT_DIR, "cold_start_users.npy"),  np.array(list(coldstart_users)))

print(f"\nSaved downsampled datasets to: {OUT_DIR}")
print("Done!")