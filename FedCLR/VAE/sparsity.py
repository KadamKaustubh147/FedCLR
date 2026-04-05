import numpy as np
Xs = np.load("X_source.npy")
Xt = np.load("X_target.npy")
print(Xs.shape, Xt.shape)
print("X_source sparsity:", (Xs == 0).mean())
print("X_target sparsity:", (Xt == 0).mean())