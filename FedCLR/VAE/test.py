import numpy as np
import torch
import random

from model import VAE
from train import CrossDomainDataset


# =========================
# SPLIT DATA BY USER
# =========================
def split_by_user(dataset):
    user_to_indices = {}

    for idx in range(len(dataset)):
        _, _, user_id = dataset[idx]

        if user_id not in user_to_indices:
            user_to_indices[user_id] = []

        user_to_indices[user_id].append(idx)

    return user_to_indices


# =========================
# TRAIN TEST SPLIT (SAME AS TRAINING)
# =========================
def train_test_split_users(user_to_indices, test_ratio=0.2):

    all_users = list(user_to_indices.keys())
    random.shuffle(all_users)

    split = int(len(all_users) * (1 - test_ratio))

    train_users = all_users[:split]
    test_users = all_users[split:]

    return train_users, test_users


# =========================
# METRICS
# =========================
def precision_at_k(pred_items, true_items, k):
    pred_k = set(pred_items[:k])
    true_set = set(np.where(true_items > 0)[0])

    return len(pred_k & true_set) / k if k > 0 else 0.0


def recall_at_k(pred_items, true_items, k):
    pred_k = set(pred_items[:k])
    true_set = set(np.where(true_items > 0)[0])

    return len(pred_k & true_set) / len(true_set) if len(true_set) > 0 else 0.0


def ndcg_at_k(pred_items, true_items, k):
    pred_k = pred_items[:k]
    true_set = set(np.where(true_items > 0)[0])

    dcg = 0.0
    for i, item in enumerate(pred_k):
        if item in true_set:
            dcg += 1 / np.log2(i + 2)

    ideal_hits = min(len(true_set), k)
    idcg = sum([1 / np.log2(i + 2) for i in range(ideal_hits)])

    return dcg / idcg if idcg > 0 else 0.0


# =========================
# EVALUATION
# =========================
def evaluate(model, dataset, test_users, device, k_list=[50, 100]):

    results = {}

    for k in k_list:
        all_prec, all_rec, all_ndcg = [], [], []

        for user_id in test_users:

            # get ONE user row (full vector)
            x_s, x_t, _ = dataset[user_id]

            x_s = x_s.unsqueeze(0).to(device)

            with torch.no_grad():
                logits, _, _, _ = model(x_s)

            scores = logits.squeeze().cpu().numpy()

            # IMPORTANT: no masking (cold-start)
            pred_items = np.argsort(-scores)

            true_items = x_t.numpy()

            all_prec.append(precision_at_k(pred_items, true_items, k))
            all_rec.append(recall_at_k(pred_items, true_items, k))
            all_ndcg.append(ndcg_at_k(pred_items, true_items, k))

        results[k] = {
            "precision": np.mean(all_prec),
            "recall": np.mean(all_rec),
            "ndcg": np.mean(all_ndcg),
        }

    return results


# =========================
# MAIN
# =========================
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = CrossDomainDataset("X_source.npy", "X_target.npy")

    # SAME split as training
    # user_to_indices = split_by_user(dataset)
    # _, test_users = train_test_split_users(user_to_indices, test_ratio=0.2)

    test_users = np.load("test_users.npy")

    print(f"Testing on {len(test_users)} users (loaded from training)")

    # print(f"Testing on {len(test_users)} users")

    # Load trained model
    model = VAE(dataset.Xs.shape[1], dataset.Xt.shape[1]).to(device)
    model.load_state_dict(torch.load("fedclr_model.pth", map_location=device))
    model.eval()

    print("\n📊 Evaluating...\n")

    results = evaluate(model, dataset, test_users, device, k_list=[50, 100])

    # =========================
    # SAVE RESULTS
    # =========================
    with open("results.txt", "w") as f:

        f.write("===== Evaluation Results =====\n\n")

        for k in results:
            line = (
                f"@{k}\n"
                f"Precision: {results[k]['precision']:.4f}\n"
                f"Recall:    {results[k]['recall']:.4f}\n"
                f"NDCG:      {results[k]['ndcg']:.4f}\n\n"
            )

            print(line)
            f.write(line)

    print("✅ Results saved to results.txt")


if __name__ == "__main__":
    main()