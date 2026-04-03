import numpy as np

X = np.load("X_source.npy")

print(X.shape)        # VERY IMPORTANT
print(X.dtype)
print(X[:5])          # first 5 rows