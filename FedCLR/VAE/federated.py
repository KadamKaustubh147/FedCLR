import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import copy
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from test1 import evaluate
# from scipy.sparse import load_npz


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

    optimizer = torch.optim.Adam(local_model.parameters(), lr=0.05)

    for epoch in range(local_epochs):
        loss = train(local_model, dataloader, optimizer, device, prev_z_memory, z_glob_memory, epoch)

        print(f"    📉 Local Epoch {epoch+1}/{local_epochs} | Loss: {loss:.4f}")


    return local_model.state_dict(), len(indices), loss


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




def federated_training():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    dataset = CrossDomainDataset("X_source.npy", "X_target.npy")

    input_dim_source = dataset.Xs.shape[1]
    input_dim_target = dataset.Xt.shape[1]

    global_model = VAE(input_dim_source, input_dim_target).to(device)
    global_model.apply(init_weights)

    client_splits = split_by_user(dataset)

    # =========================
    # LOAD SPLIT FROM PREPROCESSING
    # don't re-split here — use the same split as preprocessing
    # =========================
    train_users = np.load("train_users.npy").tolist()
    cold_start_users = np.load("cold_start_users.npy").tolist()

    print(f"Train users: {len(train_users)}, Cold-start users: {len(cold_start_users)}")

    num_users = len(dataset)
    prev_z_memory = [None] * num_users

    global_rounds = 50
    local_epochs = 5
    C = 0.1

    print("🚀 Starting Federated Training...\n")
    
    # for plotting purposes
    global_losses = []
    precision_50_history = []
    precision_100_history = []

    recall_50_history = []
    recall_100_history = []

    ndcg_50_history = []
    ndcg_100_history = []

    for round_idx in range(global_rounds):
        print(f"\n🌍 Global Round {round_idx+1}")

        # sample ONLY from train users
        num_selected = max(1, int(C * len(train_users)))
        # num_selected = 50
        selected_users = random.sample(train_users, num_selected)

        client_states = []
        client_sizes = []

        round_loss = 0
        
        for user_id in selected_users:
            print(f"Client (User) {user_id} training...")
            indices = client_splits[user_id]
            state_dict, size, loss = local_training(
                global_model, dataset, indices,
                device, local_epochs, prev_z_memory
            )
            client_states.append(state_dict)
            client_sizes.append(size)
            round_loss += loss
        
        
        # Evaluation
        avg_round_loss = round_loss / len(selected_users)
        
        global_losses.append(avg_round_loss)
        print(f"📊 Global Round Loss {round_idx+1}: {avg_round_loss:.4f}")
        aggregate(global_model, client_states, client_sizes)
        print("✅ Aggregation done")
        
        X_source = np.load("X_source.npy")
        X_target_test = np.load("X_target_test.npy")
        
        results = evaluate(
            global_model,
            X_source,
            X_target_test,
            cold_start_users,
            device,
            k_list=[50, 100]
        )

        # =========================
        # STORE METRICS
        # =========================
        precision_50_history.append(
            results[50]["precision"]
        )

        precision_100_history.append(
            results[100]["precision"]
        )

        recall_50_history.append(
            results[50]["recall"]
        )

        recall_100_history.append(
            results[100]["recall"]
        )

        ndcg_50_history.append(
            results[50]["ndcg"]
        )

        ndcg_100_history.append(
            results[100]["ndcg"]
        )

        # =========================
        # PRINT METRICS
        # =========================
        print(
            f"📈 @50 | "
            f"P: {results[50]['precision']:.4f} "
            f"R: {results[50]['recall']:.4f} "
            f"NDCG: {results[50]['ndcg']:.4f}"
        )

        print(
            f"📈 @100 | "
            f"P: {results[100]['precision']:.4f} "
            f"R: {results[100]['recall']:.4f} "
            f"NDCG: {results[100]['ndcg']:.4f}"
        )

    torch.save(global_model.state_dict(), "fedclr_model.pth")
    print("\n🎯 Federated Training Complete")
    

    rounds = range(1, global_rounds + 1)

    # -------------------------
    # LOSS
    # -------------------------
    plt.figure(figsize=(8, 5))

    plt.plot(
        rounds,
        global_losses,
        marker='o'
    )

    plt.xlabel("Global Rounds")
    plt.ylabel("Loss")

    plt.title(
        "Federated Training Loss"
    )

    plt.grid(True)

    plt.show()

    # -------------------------
    # PRECISION
    # -------------------------
    plt.figure(figsize=(8, 5))

    plt.plot(
        rounds,
        precision_50_history,
        marker='o',
        label='Precision@50'
    )

    plt.plot(
        rounds,
        precision_100_history,
        marker='o',
        label='Precision@100'
    )

    plt.xlabel("Global Rounds")
    plt.ylabel("Precision")

    plt.title(
        "Precision Convergence"
    )

    plt.legend()

    plt.grid(True)

    plt.show()

    # -------------------------
    # RECALL
    # -------------------------
    plt.figure(figsize=(8, 5))

    plt.plot(
        rounds,
        recall_50_history,
        marker='o',
        label='Recall@50'
    )

    plt.plot(
        rounds,
        recall_100_history,
        marker='o',
        label='Recall@100'
    )

    plt.xlabel("Global Rounds")
    plt.ylabel("Recall")

    plt.title(
        "Recall Convergence"
    )

    plt.legend()

    plt.grid(True)

    plt.show()

    # -------------------------
    # NDCG
    # -------------------------
    plt.figure(figsize=(8, 5))

    plt.plot(
        rounds,
        ndcg_50_history,
        marker='o',
        label='NDCG@50'
    )

    plt.plot(
        rounds,
        ndcg_100_history,
        marker='o',
        label='NDCG@100'
    )

    plt.xlabel("Global Rounds")
    plt.ylabel("NDCG")

    plt.title(
        "NDCG Convergence"
    )

    plt.legend()

    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    federated_training()