import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import VAE
import torch.nn as nn   # ✅ added


# =========================
# XAVIER INITIALIZATION
# =========================
def init_weights(m):   # ✅ added
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# =========================
# DATASET --> douban
# =========================
# class CrossDomainDataset(Dataset):
#     def __init__(self, source_path, target_path):
#         self.Xs = np.load(source_path)
#         self.Xt = np.load(target_path)

#         assert self.Xs.shape[0] == self.Xt.shape[0]

#     def __len__(self):
#         return self.Xs.shape[0]

#     def __getitem__(self, user_id):
#         x_s = torch.tensor(self.Xs[user_id], dtype=torch.float32)
#         x_t = torch.tensor(self.Xt[user_id], dtype=torch.float32)
#         # user_id is the index -->
#         return x_s, x_t, user_id   # ✅ return user_id

from scipy.sparse import load_npz

class CrossDomainDataset(Dataset):
    def __init__(self, source_path, target_path):
        self.Xs = np.load(source_path)  # loads as csr_matrix
        self.Xt = np.load(target_path)

        assert self.Xs.shape[0] == self.Xt.shape[0]

    def __len__(self):
        return self.Xs.shape[0]

    def __getitem__(self, user_id):
        # .toarray() converts sparse row → dense numpy array
        # .squeeze() removes the extra dimension (1, num_items) → (num_items,)
        # x_s = torch.tensor(self.Xs[user_id].toarray().squeeze(), dtype=torch.float32)
        # x_t = torch.tensor(self.Xt[user_id].toarray().squeeze(), dtype=torch.float32)

        x_s = torch.tensor(self.Xs[user_id], dtype=torch.float32)
        x_t = torch.tensor(self.Xt[user_id], dtype=torch.float32)
        return x_s, x_t, user_id

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
# CONTRASTIVE
# =========================
def similarity(z1, z2):
    return torch.sum(z1 * z2, dim=1)


def fedclr_contrastive_loss(sim_inn, sim_int, temperature=0.3):
    exp_int = torch.exp(sim_int / temperature)
    exp_inn = torch.exp(sim_inn / temperature)
    loss = -torch.log(exp_int / (exp_int + exp_inn + 1e-8))
    return loss.mean()


# =========================
# TRAINING
# =========================
def train(model, dataloader, optimizer, device, prev_z_memory, z_glob_memory, epoch):
    model.train()
    total_loss = 0

    for x_s, x_t, user_id in dataloader:
        x_s = x_s.to(device)
        x_t = x_t.to(device)

        optimizer.zero_grad()

        logits, mu, logvar, z = model(x_s)

        loss, rec, kl = vae_loss(logits, x_t, mu, logvar)

        # =========================
        # Contrastive part
        # =========================
        if epoch > 0 and all(prev_z_memory[i.item()] is not None for i in user_id):
            z = F.normalize(z, dim=1)

            prev_z = torch.stack([prev_z_memory[i.item()] for i in user_id]).to(device)
            prev_z = F.normalize(prev_z, dim=1)

            # Inner model similarity
            sim_inn = similarity(z, prev_z)

            z_global = torch.stack([z_glob_memory[i.item()] for i in user_id]).to(device)


            sim_int = similarity(z, z_global)

            con_loss = fedclr_contrastive_loss(sim_inn, sim_int)

            loss = loss + 40 * con_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        

        # update memory correctly (per user)
        for i, user_idx in enumerate(user_id):
            prev_z_memory[user_idx.item()] = F.normalize(z, dim=1)[i].detach().cpu()

    return total_loss / len(dataloader)


# =========================
# MAIN (OPTIONAL - CENTRALIZED)
# =========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CrossDomainDataset("X_source.npy", "X_target.npy")

    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False
    )

    input_dim_source = dataset.Xs.shape[1]
    input_dim_target = dataset.Xt.shape[1]

    model = VAE(input_dim_source, input_dim_target).to(device)

    model.apply(init_weights)   # ✅ APPLY XAVIER HERE

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    num_users = len(dataset)
    prev_z_memory = [None] * num_users

    epochs = 200

    for epoch in range(epochs):
        loss = train(model, dataloader, optimizer, device, prev_z_memory, {}, epoch)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")


if __name__ == "__main__":
    main()