All three are latent vectors (`z`), but from **different points in time**:

---

**`z_glob`** — what the global model thinks RIGHT NOW

```
Global model (trained by all previous rounds)
    ↓ forward pass on user's data
    z_glob  ← "this is the global knowledge about this user"
```
Captured BEFORE cloning. Represents the server's current understanding.

---

**`prev_z`** — what the local model thought LAST EPOCH

```
Local model at epoch 1
    ↓ forward pass
    z  ← stored in prev_z_memory

Local model at epoch 2
    ↓ forward pass
    z  ← current z, compared against prev_z
```
Captured at the end of each local epoch. Represents how the local model is drifting epoch by epoch.

---

**`z_glob_memory`** — just a dictionary storing `z_glob` per user

```python
z_glob_memory = {
    42:  tensor([0.1, 0.3, ...]),   # user 42's z_glob
    87:  tensor([0.2, 0.1, ...]),   # user 87's z_glob
    156: tensor([0.5, 0.2, ...]),   # user 156's z_glob
}
```
It's not a new concept — just a container so you can look up `z_glob` by user ID during training.

---

**How they work together in the contrastive loss:**

```
z_cur   = local model NOW          (current epoch)
prev_z  = local model LAST EPOCH   (inner-model)
z_glob  = global model             (inter-model)

sim_inn = similarity(z_cur, prev_z)   # how much did local model change?
sim_int = similarity(z_cur, z_glob)   # how close is local to global?

loss = -log( exp(sim_int) / (exp(sim_int) + exp(sim_inn)) )
```

The loss **maximizes** `sim_int` (local should stay close to global) and **minimizes** `sim_inn` (local should keep changing, not stagnate). This is exactly what prevents local models from drifting away from the global model during federated training.