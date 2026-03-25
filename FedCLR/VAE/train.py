import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import VAE


# =========================
# DATASET
# =========================
class CrossDomainDataset(Dataset):
    def __init__(self, source_path, target_path):
        self.Xs = np.load(source_path)
        self.Xt = np.load(target_path)

        assert self.Xs.shape[0] == self.Xt.shape[0]

    def __len__(self):
        return self.Xs.shape[0]

    def __getitem__(self, idx):
        x_s = torch.tensor(self.Xs[idx], dtype=torch.float32)
        x_t = torch.tensor(self.Xt[idx], dtype=torch.float32)
        return x_s, x_t


# =========================
# LOSS FUNCTIONS
# =========================
def reconstruction_loss(logits, x_t):
    log_softmax = F.log_softmax(logits, dim=1)
    loss = -torch.sum(x_t * log_softmax, dim=1)
    return loss.mean()


def kl_loss(mu, logvar):
    return -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    )


def vae_loss(logits, x_t, mu, logvar, beta=1.0):
    rec = reconstruction_loss(logits, x_t)
    kl = kl_loss(mu, logvar)
    return rec + beta * kl, rec, kl


# =========================
# TRAINING
# =========================
def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for x_s, x_t in dataloader:
        x_s = x_s.to(device)
        x_t = x_t.to(device)

        optimizer.zero_grad()

        logits, mu, logvar, _ = model(x_s)
        loss, rec, kl = vae_loss(logits, x_t, mu, logvar)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# =========================
# MAIN
# =========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = CrossDomainDataset("X_source.npy", "X_target.npy")

    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True
    )

    input_dim_source = dataset.Xs.shape[1]
    input_dim_target = dataset.Xt.shape[1]

    # Model + optimizer
    model = VAE(input_dim_source, input_dim_target).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 200
    best_loss = float("inf")

    print("🚀 Starting training...\n")

    for epoch in range(epochs):
        loss = train(model, dataloader, optimizer, device)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

        # ✅ Save best model
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), "vae_best.pth")
            print("✅ Saved BEST model")

        # (optional) checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss
            }, f"checkpoint_epoch_{epoch+1}.pth")
            print(f"📦 Saved checkpoint at epoch {epoch+1}")

    print("\n🎯 Training complete")
    print(f"Best Loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()