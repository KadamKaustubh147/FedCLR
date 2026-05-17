"""
Collective Matrix Factorization (CMF) - Evaluation Script
Mirrors the Fed-CLR evaluate.py structure exactly.

Loads a trained CMF model (cmf_model.npz), scores target-domain items
for every cold-start user, and reports Precision@K, Recall@K, NDCG@K.

Scoring: X_target ≈ U · Vt^T  (identity link, paper Section 7.1.1 CMF-Identity)
         For cold-start users, source-domain signal is baked into U via the
         shared factor learned during joint factorisation on the training split.
"""

import numpy as np
import os


# ---------------------------------------------------------------------------
# Metric functions  (identical to Fed-CLR test script)
# ---------------------------------------------------------------------------

def precision_at_k(pred_items, true_items, k):
    pred_k   = set(pred_items[:k])
    true_set = set(np.where(true_items > 0)[0])
    return len(pred_k & true_set) / k if k > 0 else 0.0


def recall_at_k(pred_items, true_items, k):
    pred_k   = set(pred_items[:k])
    true_set = set(np.where(true_items > 0)[0])
    return len(pred_k & true_set) / len(true_set) if len(true_set) > 0 else 0.0


def ndcg_at_k(pred_items, true_items, k):
    pred_k   = pred_items[:k]
    true_set = set(np.where(true_items > 0)[0])
    dcg      = sum(1 / np.log2(i + 2)
                   for i, item in enumerate(pred_k) if item in true_set)
    ideal_hits = min(len(true_set), k)
    idcg     = sum(1 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# CMF loader  (standalone -- no dependency on cmf_train.py needed)
# ---------------------------------------------------------------------------

class CMFModel:
    """Thin wrapper that loads saved CMF factors and scores target items."""

    def __init__(self, path):
        d = np.load(path)
        self.U  = d["U"]    # (m  x k)
        self.Vs = d["Vs"]   # (n_s x k)  kept for completeness
        self.Vt = d["Vt"]   # (n_t x k)

    def score_target(self, user_ids):
        """
        Return predicted target-domain scores for the given user indices.

        Scores = U[user_ids] @ Vt^T   (identity link, paper Section 7.1.1)

        For cold-start users, their source-domain interactions were included
        during training (Ws=ones covers all source entries), so their U rows
        are informed by source-domain signal even though their target
        interactions were masked (Wt[cold_start] = 0).

        Parameters
        ----------
        user_ids : array-like of int

        Returns
        -------
        scores : np.ndarray (len(user_ids) x n_t)
        """
        return self.U[user_ids] @ self.Vt.T


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(model, X_source, X_target_test, cold_start_users,
             k_list=(50, 100)):
    """
    Evaluate CMF on cold-start users.

    Parameters
    ----------
    model            : CMFModel instance
    X_source         : (m x n_s) source interaction matrix  [kept for API parity]
    X_target_test    : (m x n_t) ground-truth target interactions for test users
    cold_start_users : list of user indices to evaluate
    k_list           : cutoffs for top-K metrics

    Returns
    -------
    results : dict  {k: {"precision": ..., "recall": ..., "ndcg": ...}}
    """
    all_scores = model.score_target(cold_start_users)   # (n_cold x n_t)

    results = {}
    for k in k_list:
        all_prec, all_rec, all_ndcg = [], [], []
        skipped = 0

        for idx, user_id in enumerate(cold_start_users):
            true_items = X_target_test[user_id]

            if true_items.sum() == 0:
                skipped += 1
                continue

            scores     = all_scores[idx]
            pred_items = np.argsort(-scores)

            all_prec.append(precision_at_k(pred_items, true_items, k))
            all_rec.append(recall_at_k(pred_items, true_items, k))
            all_ndcg.append(ndcg_at_k(pred_items, true_items, k))

        print(f"  Skipped {skipped} cold-start users with no test interactions at @{k}")
        results[k] = {
            "precision": float(np.mean(all_prec)),
            "recall":    float(np.mean(all_rec)),
            "ndcg":      float(np.mean(all_ndcg)),
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    X_SOURCE_PATH      = "../../FedCLR/VAE/X_source.npy"
    X_TARGET_TEST_PATH = "../../FedCLR/VAE/X_target_test.npy"
    COLD_START_PATH    = "../../FedCLR/VAE/cold_start_users.npy"
    MODEL_PATH         = "cmf_model.npz"
    OUT_RESULTS        = "cmf_results.txt"
    K_LIST             = [50, 100]

    print("Loading data...")
    X_source       = np.load(X_SOURCE_PATH).astype(np.float64)
    X_target_test  = np.load(X_TARGET_TEST_PATH).astype(np.float64)
    cold_start_users = np.load(COLD_START_PATH).tolist()

    print(f"  X_source      : {X_source.shape}")
    print(f"  X_target_test : {X_target_test.shape}")
    print(f"  Cold-start users: {len(cold_start_users)}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            f"Run cmf_train.py first to generate it."
        )
    model = CMFModel(MODEL_PATH)
    print(f"  Loaded CMF factors: U{model.U.shape}, Vt{model.Vt.shape}")

    print(f"\nEvaluating on {len(cold_start_users)} cold-start users ...")
    results = evaluate(model, X_source, X_target_test,
                       cold_start_users, k_list=K_LIST)

    lines = ["===== CMF Evaluation Results =====\n"]
    for k in results:
        line = (
            f"@{k}\n"
            f"Precision: {results[k]['precision']:.4f}\n"
            f"Recall:    {results[k]['recall']:.4f}\n"
            f"NDCG:      {results[k]['ndcg']:.4f}\n"
        )
        print(line)
        lines.append(line)

    with open(OUT_RESULTS, "w") as f:
        f.write("\n".join(lines))
    print(f"✅ Results saved to {OUT_RESULTS}")


if __name__ == "__main__":
    main()