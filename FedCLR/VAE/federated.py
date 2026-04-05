import numpy as np
import torch
import random
import copy
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from model import VAE
# imports dataset and train function --> local training function
from train import CrossDomainDataset, train, init_weights

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# =========================
# SPLIT DATA BY USER
# =========================
# this is kinda of a useless function cuz the dataset is already split by user. --> later will optimise if needed
# 1 row --> 1 user
def split_by_user(dataset):
    # user_to_indices builts a hashmap of a single user that maps to all the row indices that belong to that user --> all the row data
    
    user_to_indices = {}

    for idx in range(len(dataset)):
        # train.py:25
        _, _, user_id = dataset[idx]

        if user_id not in user_to_indices:
            user_to_indices[user_id] = []

        user_to_indices[user_id].append(idx)

    return user_to_indices


# =========================
# TRAIN TEST SPLIT (USER LEVEL)
# =========================
def train_test_split_users(user_to_indices, test_ratio=0.2):

    all_users = list(user_to_indices.keys())
    random.shuffle(all_users)

    split = int(len(all_users) * (1 - test_ratio))

    train_users = all_users[:split]
    test_users = all_users[split:]

    return train_users, test_users


# =========================
# LOCAL TRAINING for a SINGLE USER
# =========================


# What exactly is prev_z_memory??

def local_training(global_model, dataset, indices, device, local_epochs, prev_z_memory):

    # data loader
    dataloader = DataLoader(
        Subset(dataset, indices),
        # find whether this min is required or not
        batch_size=min(128, len(indices)),
        shuffle=False
    )

    # ✅ compute z_glob from global model BEFORE cloning
    # we do a forward pass --> global_model function
    # eval() turns off dropout layer.
    global_model.eval()
    z_glob_memory = {}
    with torch.no_grad():
        for x_s, x_t, user_id in dataloader:
            # user_id --> batch of 128 users???
            print(user_id)
            x_s = x_s.to(device)
            _, _, _, z = global_model(x_s)
            z_norm = F.normalize(z, dim=1)
            for i, uid in enumerate(user_id):
                # .detach() --> disconnect from computation graph
                # .cpu() --> store the z_glob_memory vector in RAM
                # from the forward pass of the global model we get single z --> z = mu + epislon*sigma (distribution mean and std is there) and we sample z --> that latent representation 
                # dim(z) = 100
                z_glob_memory[uid.item()] = z_norm[i].detach().cpu()

    # clone global model
    local_model = copy.deepcopy(global_model)

    optimizer = torch.optim.Adam(local_model.parameters(), lr=1e-3)

    for epoch in range(local_epochs):
        loss = train(local_model, dataloader, optimizer, device, prev_z_memory, z_glob_memory, epoch)

        print(f"    📉 Local Epoch {epoch+1}/{local_epochs} | Loss: {loss:.4f}")


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
    print(device)

    dataset = CrossDomainDataset("X_source.npy", "X_target.npy")

    input_dim_source = dataset.Xs.shape[1]
    input_dim_target = dataset.Xt.shape[1]

    # This code creates a brand new untrained model --> weights are random 
    global_model = VAE(input_dim_source, input_dim_target).to(device)
    # xavier intialisation
    global_model.apply(init_weights)   

    # 🔥 user-based clients
    client_splits = split_by_user(dataset)
    all_users = list(client_splits.keys())

    # =========================
    # APPLY TRAIN TEST SPLIT
    # =========================
    train_users, test_users = train_test_split_users(client_splits, test_ratio=0.2)

    print(f"Train users: {len(train_users)}, Test users: {len(test_users)}")

    # =========================
    # SAVE SPLIT (IMPORTANT)
    # =========================
    np.save("train_users.npy", np.array(train_users))
    np.save("test_users.npy", np.array(test_users))

    print("💾 Saved train/test split")

    # 🔥 global memory
    # because the dataset is user item one hot encoding, the user_id is the index of the dataset
    num_users = len(dataset)
    prev_z_memory = [None] * num_users

    # these factors below aren't specified -
    global_rounds = 50
    local_epochs = 3
    C = 0.05  # fraction of clients

    print("🚀 Starting Federated Training...\n")

    for round_idx in range(global_rounds):

        print(f"\n🌍 Global Round {round_idx+1}")

        # =========================
        # SAMPLE ONLY FROM TRAIN USERS
        # =========================
        num_selected = max(1, int(C * len(train_users)))
        selected_users = random.sample(train_users, num_selected)

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