import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from model import VAE
from train import CrossDomainDataset, train


# =========================
# SPLIT DATA INTO CLIENTS
# =========================
def split_dataset(dataset, num_clients):
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    return np.array_split(indices, num_clients)


# =========================
# LOCAL TRAINING (CLIENT)
# =========================
def local_training(global_model, dataset, indices, device, local_epochs):

    dataloader = DataLoader(
        Subset(dataset, indices),
        batch_size=128,
        shuffle=False
    )

    # clone global model
    local_model = VAE(
        global_model.encoder[0].in_features,
        global_model.decoder[-1].out_features
    ).to(device)

    local_model.load_state_dict(global_model.state_dict())

    optimizer = torch.optim.Adam(local_model.parameters(), lr=1e-3)

    # each client has its own memory
    prev_z_memory = [None] * len(dataset)

    for epoch in range(local_epochs):
        train(local_model, dataloader, optimizer, device, prev_z_memory, epoch)

    return local_model.state_dict(), len(indices)


# =========================
# AGGREGATION (FEDAVG)
# =========================
def aggregate(global_model, client_states, client_sizes):

    new_state = {}
    total = sum(client_sizes)

    for key in global_model.state_dict().keys():
        new_state[key] = sum(
            client_states[i][key] * (client_sizes[i] / total)
            for i in range(len(client_states))
        )

    global_model.load_state_dict(new_state)


# =========================
# FEDERATED TRAINING
# =========================
def federated_training():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CrossDomainDataset("X_source.npy", "X_target.npy")

    input_dim_source = dataset.Xs.shape[1]
    input_dim_target = dataset.Xt.shape[1]

    global_model = VAE(input_dim_source, input_dim_target).to(device)

    num_clients = 5
    global_rounds = 10
    local_epochs = 3

    client_splits = split_dataset(dataset, num_clients)

    print("🚀 Starting Federated Training...\n")

    for round_idx in range(global_rounds):

        print(f"\n🌍 Global Round {round_idx+1}")

        client_states = []
        client_sizes = []

        # simulate all clients (no sampling yet)
        for i in range(num_clients):

            print(f"Client {i} training...")

            state_dict, size = local_training(
                global_model,
                dataset,
                client_splits[i],
                device,
                local_epochs
            )

            client_states.append(state_dict)
            client_sizes.append(size)

        # aggregate
        aggregate(global_model, client_states, client_sizes)

        print("✅ Aggregation done")

    torch.save(global_model.state_dict(), "fedclr_model.pth")

    print("\n🎯 Federated Training Complete")


if __name__ == "__main__":
    federated_training()