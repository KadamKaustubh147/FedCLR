import numpy as np
import torch
import random
import copy
from torch.utils.data import DataLoader, Subset

from model import VAE
from train import CrossDomainDataset, train


# =========================
# SPLIT DATA BY USER
# =========================
def split_by_user(dataset):
    # user_to_indices builts a hashmap of a single user that maps to all the row indices that belong to that user --> all the row data
    
    user_to_indices = {}

    for idx in range(len(dataset)):
        # how the fuck the first column is user_id
        # train.py:25
        _, _, user_id = dataset[idx]

        if user_id not in user_to_indices:
            user_to_indices[user_id] = []

        user_to_indices[user_id].append(idx)

    return user_to_indices


# =========================
# LOCAL TRAINING (CLIENT)
# =========================


# What exactly is prev_z_memory??
def local_training(global_model, dataset, indices, device, local_epochs, prev_z_memory):

    dataloader = DataLoader(
        Subset(dataset, indices),
        batch_size=min(128, len(indices)),
        shuffle=False
    )

    # clone global model
    local_model = copy.deepcopy(global_model)

    optimizer = torch.optim.Adam(local_model.parameters(), lr=1e-3)

    for epoch in range(local_epochs):
        train(local_model, dataloader, optimizer, device, prev_z_memory, epoch)

    return local_model.state_dict(), len(indices)


# =========================
# AGGREGATION (FEDAVG)
# =========================

'''
⚠️ Subtle but important difference
model.parameters()
just values
NO structure
model.state_dict()
values + names + full model structure
'''

# format of client state

'''

client_states
[
    model1.state_dict(),
    model2.state_dict(),
    model3.state_dict()
]

Each looks like:

{
  "encoder.weight": tensor(...),
  "encoder.bias": tensor(...),
  ...
}

'''
# 
def aggregate(global_model, client_states, client_sizes):

    new_state = {}
    total = sum(client_sizes)

    # wtf is state_dict? --> a dictionary that maps each layer to its parameters (weights and biases)
    for key in global_model.state_dict().keys():
        # new state is a single global model state
        new_state[key] = sum(
            client_states[i][key] * (client_sizes[i] / total)
            for i in range(len(client_states))
        )

    # update global model
    # we override previous state --> that is why we use conntrastive to bring in the stability
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

    # 🔥 user-based clients
    client_splits = split_by_user(dataset)
    all_users = list(client_splits.keys())

    # 🔥 global memory
    # because the dataset is user item one hot encoding, the user_id is the index of the dataset
    num_users = len(dataset)
    prev_z_memory = [None] * num_users

    # these factors below aren't specified -
    global_rounds = 50
    local_epochs = 3
    C = 0.1  # fraction of clients

    print("🚀 Starting Federated Training...\n")

    for round_idx in range(global_rounds):

        print(f"\n🌍 Global Round {round_idx+1}")

        num_selected = max(1, int(C * len(all_users)))
        selected_users = random.sample(all_users, num_selected)

        client_states = []
        client_sizes = []

        for user_id in selected_users:

            print(f"Client (User) {user_id} training...")

            indices = client_splits[user_id]

            state_dict, size = local_training(
                global_model,
                dataset,
                indices,
                device,
                local_epochs,
                prev_z_memory
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