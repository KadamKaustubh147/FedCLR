import numpy as np
import torch
from model import VAE


# =========================
# LOAD DATA
# =========================
def load_data():
    Xs = np.load("X_source.npy")
    Xt = np.load("X_target.npy")
    return Xs, Xt


# =========================
# LOAD MODEL
# =========================
def load_model(input_dim_source, input_dim_target, device):
    model = VAE(input_dim_source, input_dim_target).to(device)
    model.load_state_dict(torch.load("vae_best.pth", map_location=device))
    model.eval()
    return model


# =========================
# LOAD & REVERSE MAP
# =========================
def load_reverse_map(path):
    mapping = np.load(path, allow_pickle=True).item()
    reverse_map = {v: k for k, v in mapping.items()}
    return reverse_map


# =========================
# RECOMMENDATION FUNCTION
# =========================
def recommend(model, user_id, Xs, Xt=None, top_k=10, device="cpu"):
    model.eval()

    x_s = torch.tensor(Xs[user_id], dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _, _, _ = model(x_s)

    scores = logits.squeeze().cpu().numpy()

    # Remove already interacted items
    if Xt is not None:
        seen_items = Xt[user_id] > 0
        scores[seen_items] = -1e9

    # Top-K indices
    top_items = np.argsort(-scores)[:top_k]

    return top_items


# =========================
# MAIN
# =========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    Xs, Xt = load_data()

    # Load model
    model = load_model(Xs.shape[1], Xt.shape[1], device)

    # Load reverse maps
    reverse_music_map = load_reverse_map("music_map.npy")
    reverse_user_map = load_reverse_map("user_map.npy")

    # Choose internal user index
    user_id = 10

    # Recommend
    top_items = recommend(model, user_id, Xs, Xt, top_k=10, device=device)

    print(f"\nUser (internal index): {user_id}")

    # Convert to original user ID
    original_user_id = reverse_user_map[user_id]
    print(f"Original User ID: {original_user_id}")

    print("\nTop-K recommendations (internal indices):")
    print(top_items)

    # Convert to original music IDs
    original_ids = [reverse_music_map[i] for i in top_items]

    print("\nTop-K recommended music IDs (original):")
    print(original_ids)


if __name__ == "__main__":
    main()