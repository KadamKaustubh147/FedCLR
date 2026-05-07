import numpy as np
import torch
from model import VAE
from train import CrossDomainDataset


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
    dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(pred_k) if item in true_set)
    ideal_hits = min(len(true_set), k)
    idcg = sum(1 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(model, X_source, X_target_test, cold_start_users, device, k_list=[50, 100]):

    results = {}
    model.eval()

    for k in k_list:
        all_prec, all_rec, all_ndcg = [], [], []

        skipped = 0
        for user_id in cold_start_users:

            true_items = X_target_test[user_id]

            # skip users with no test interactions
            if true_items.sum() == 0:
                skipped += 1
                continue

            x_s = torch.tensor(X_source[user_id], dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                logits, _, _, _ = model(x_s)

            scores = logits.squeeze().cpu().numpy()
            pred_items = np.argsort(-scores)

            all_prec.append(precision_at_k(pred_items, true_items, k))
            all_rec.append(recall_at_k(pred_items, true_items, k))
            all_ndcg.append(ndcg_at_k(pred_items, true_items, k))

        print(f"  Skipped {skipped} cold-start users with no test interactions at @{k}")

        results[k] = {
            "precision": np.mean(all_prec),
            "recall":    np.mean(all_rec),
            "ndcg":      np.mean(all_ndcg),
        }

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load matrices
    X_source      = np.load("X_source.npy")
    X_target_test = np.load("X_target_test.npy")  # cold-start ground truth

    # Load cold-start users
    cold_start_users = np.load("cold_start_users.npy").tolist()
    print(f"Evaluating on {len(cold_start_users)} cold-start users")

    # Load model
    dataset = CrossDomainDataset("X_source.npy", "X_target.npy")
    model = VAE(dataset.Xs.shape[1], dataset.Xt.shape[1]).to(device)
    model.load_state_dict(torch.load("fedclr_model.pth", map_location=device))
    model.eval()

    results = evaluate(model, X_source, X_target_test, cold_start_users, device, k_list=[50, 100])

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