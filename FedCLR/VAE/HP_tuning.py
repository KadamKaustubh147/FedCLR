import optuna
import torch
import numpy as np
import random
import copy
import logging # Added for logging
from torch.utils.data import DataLoader
from model import VAE
from train import CrossDomainDataset, train, init_weights
from test1 import evaluate
import torch.nn.functional as F
from federated import split_by_user

# =========================
# LOGGING CONFIGURATION
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("movie_music_logs.txt"),
        logging.StreamHandler() # This keeps logs visible in the terminal too
    ]
)
logger = logging.getLogger(__name__)

# =========================
# CONSTANTS FROM THE PAPER
# =========================
ALPHA = 40          
TEMPERATURE = 0.3   
GLOBAL_ROUNDS = 10  
LOCAL_EPOCHS = 5
BATCH_SIZE = 128    

def objective(trial):
    lr = trial.suggest_categorical("lr", [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5])
    dropout = trial.suggest_float("dropout", 0.0, 1.0, step=0.1)

    logger.info(f"Starting Trial #{trial.number} | lr: {lr}, dropout: {dropout}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = CrossDomainDataset("X_source.npy", "X_target.npy")
    train_users = np.load("train_users.npy").tolist()
    cold_start_users = np.load("cold_start_users.npy").tolist()
    client_splits = split_by_user(dataset)
    
    input_dim_source = dataset.Xs.shape[1]
    input_dim_target = dataset.Xt.shape[1]

    global_model = VAE(input_dim_source, input_dim_target, dropout=dropout).to(device)
    global_model.apply(init_weights)

    num_users = len(dataset)
    prev_z_memory = [None] * num_users
    best_recall_50 = 0.0

    for round_idx in range(GLOBAL_ROUNDS):
        num_selected = max(1, int(0.1 * len(train_users)))
        selected_users = random.sample(train_users, num_selected)
        
        client_states = []
        client_sizes = []

        for user_id in selected_users:
            indices = client_splits[user_id]
            state_dict, size, _ = local_training_optimized(
                global_model, dataset, indices, device, 
                LOCAL_EPOCHS, prev_z_memory, lr, ALPHA, TEMPERATURE
            )
            client_states.append(state_dict)
            client_sizes.append(size)

        aggregate_params(global_model, client_states, client_sizes)

        # Evaluation
        X_source = np.load("X_source.npy")
        X_target_test = np.load("X_target_test.npy")
        
        results = evaluate(
            global_model, X_source, X_target_test, 
            cold_start_users, device, k_list=[50]
        )
        
        current_recall = results[50]["recall"]
        best_recall_50 = max(best_recall_50, current_recall)

        # Log round progress
        logger.info(f"Trial {trial.number} - Round {round_idx}: Recall@50 = {current_recall:.4f}")

        trial.report(current_recall, round_idx)
        if trial.should_prune():
            logger.warning(f"Trial {trial.number} pruned at round {round_idx}.")
            raise optuna.exceptions.TrialPruned()

    logger.info(f"Finished Trial {trial.number} | Best Recall@50: {best_recall_50:.4f}")
    return best_recall_50

# ... [local_training_optimized and aggregate_params remain unchanged] ...

def local_training_optimized(global_model, dataset, indices, device, local_epochs, prev_z_memory, lr, alpha, tau):
    from torch.utils.data import Subset
    dataloader = DataLoader(Subset(dataset, indices), batch_size=BATCH_SIZE, shuffle=False)
    
    global_model.eval()
    z_glob_memory = {}
    with torch.no_grad():
        for x_s, _, user_id in dataloader:
            _, _, _, z = global_model(x_s.to(device))
            z_norm = F.normalize(z, dim=1)
            for i, uid in enumerate(user_id):
                z_glob_memory[uid.item()] = z_norm[i].cpu()

    local_model = copy.deepcopy(global_model)
    local_model.train()
    optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)

    for epoch in range(local_epochs):
        _ = train(local_model, dataloader, optimizer, device, prev_z_memory, z_glob_memory, epoch)

    return local_model.state_dict(), len(indices), None

def aggregate_params(global_model, client_states, client_sizes):
    new_state = {}
    total = sum(client_sizes)
    for key in global_model.state_dict().keys():
        new_state[key] = sum(client_states[i][key] * (client_sizes[i] / total) for i in range(len(client_states)))
    global_model.load_state_dict(new_state)

if __name__ == "__main__":
    search_space = {
        "lr": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
        "dropout": [round(x * 0.1, 1) for x in range(0, 11)] 
    }

    sampler = optuna.samplers.GridSampler(search_space)
    
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, 
        n_warmup_steps=2, 
        interval_steps=1
    )

    study = optuna.create_study(
        sampler=sampler, 
        pruner=pruner, 
        direction="maximize"
    )

    # Route Optuna's internal logs to our logger as well
    optuna.logging.enable_propagation() 
    optuna.logging.disable_default_handler() 

    logger.info("Starting Grid Search Optimization...")
    study.optimize(objective, n_trials=100) 

    logger.info("--- Optimization Complete ---")
    logger.info(f"Best Trial Value: {study.best_trial.value}")
    logger.info(f"Best Params: {study.best_params}")
    
    print("\nCheck 'movie_book_logs.txt' for detailed logs.")