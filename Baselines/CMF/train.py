"""
Collective Matrix Factorization (CMF) - Training Script
Implements: Singh & Gordon, "Relational Learning via Collective Matrix Factorization", KDD 2008.

Schema:  E1 (users) ~ E2 (items_source) via X_source  [m x n_s]
         E1 (users) ~ E3 (items_target) via X_target   [m x n_t]
Shared factor: U  (users, shape m x k)
Item factors:  Vs (source items, n_s x k), Vt (target items, n_t x k)

Objective (Eq. 3 in paper, squared loss / identity link):
    L = alpha * [½‖Ws ⊙ (Xs - U Vs^T)‖²_F  +  lam_v/2·‖Vs‖²]
      + (1-alpha) * [½‖Wt ⊙ (Xt - U Vt^T)‖²_F  +  lam_v/2·‖Vt‖²]
      + lam_u/2 * ‖U‖²_F          ← U participates in both, so total weight = alpha+(1-alpha)=1

Key paper settings (Section 7.1.1):
    k=20, G(U)=1e5·‖U‖²/2 => lam = 1/1e5 = 1e-5
    alpha=0.5 (Section 7.2), identity link, squared loss
    Ws = ones (source matrix is FULLY OBSERVED binary; 0 is a real observation, not missing)
    Wt = indicator of observed entries (cold-start users masked to 0 for training)
    Run for fixed MAX_ITER=50 iterations, no early stopping

FIX NOTES vs previous version:
    1. Ws = np.ones_like(X_source)  — NOT (X_source > 0).
       The source matrix is a fully-observed binary matrix per Fed-CLR Eq.(1):
       x^S[i,j] in {0,1} for all (i,j). Treating 0-entries as missing (weight=0)
       discards real negative feedback and biases U/Vs toward predicting only 1s.

    2. Regulariser on Vs is scaled by alpha; on Vt by (1-alpha).
       The paper places the Vs regulariser inside L1 (scaled by alpha) and
       Vt regulariser inside L2 (scaled by 1-alpha) — see Eq. 3 and Section 4.1.
       The Newton Hessians for item factors must therefore be:
           H_Vs = alpha*(U^T D U) + alpha*lam_v*I
           H_Vt = (1-alpha)*(U^T D U) + (1-alpha)*lam_v*I
       The previous code used lam_v*I (unscaled), over-regularising both item
       factors by a factor of 1/alpha and 1/(1-alpha) respectively.
"""

import numpy as np
import time
import json
import os


# ---------------------------------------------------------------------------
# CMF model
# ---------------------------------------------------------------------------

class CMF:
    """
    Collective Matrix Factorization with two relation matrices sharing a
    common user factor U.

    Attributes
    ----------
    U   : np.ndarray (m x k)   user latent factors
    Vs  : np.ndarray (n_s x k) source-item latent factors
    Vt  : np.ndarray (n_t x k) target-item latent factors
    """

    def __init__(self, m, n_s, n_t, k=20, alpha=0.5,
                 lam_u=1e-5, lam_v=1e-5, seed=42):
        """
        Parameters
        ----------
        m     : number of users
        n_s   : number of source-domain items
        n_t   : number of target-domain items
        k     : latent dimension (paper Section 7: k=20)
        alpha : weight on source-domain reconstruction loss  (paper Section 7.2: 0.5)
        lam_u : L2 regularisation on U  (paper Section 7.1.1: G(U)=1e5·‖U‖²/2 => lam=1e-5)
        lam_v : L2 regularisation on Vs and Vt  (same as lam_u per paper)
        seed  : random seed for Xavier-style initialisation (paper: Xavier init)
        """
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (k + 1))           # Xavier-ish (paper Section 4, Fed-CLR Sec IV-A)
        self.U  = rng.normal(0, scale, (m,   k))
        self.Vs = rng.normal(0, scale, (n_s, k))
        self.Vt = rng.normal(0, scale, (n_t, k))

        self.k     = k
        self.alpha = alpha
        self.lam_u = lam_u
        self.lam_v = lam_v

    # -----------------------------------------------------------------------
    # Loss  (paper Eq. 3, expanded with regularisers inside each sub-loss)
    # -----------------------------------------------------------------------

    def loss(self, Xs, Xt, Ws, Wt):
        """
        Compute total weighted squared-error loss (paper Eq. 3).

        L = alpha * [½‖Ws⊙(Xs - U Vs^T)‖²  +  lam_v/2·‖Vs‖²]
          + (1-alpha) * [½‖Wt⊙(Xt - U Vt^T)‖²  +  lam_v/2·‖Vt‖²]
          + lam_u/2 * ‖U‖²

        Note: the U regulariser effectively has coefficient
        [alpha + (1-alpha)] * lam_u = lam_u, since U appears in both sub-losses.
        """
        res_s = Ws * (Xs - self.U @ self.Vs.T)
        res_t = Wt * (Xt - self.U @ self.Vt.T)
        L_s   = 0.5 * np.sum(res_s ** 2) + (self.lam_v / 2) * np.sum(self.Vs ** 2)
        L_t   = 0.5 * np.sum(res_t ** 2) + (self.lam_v / 2) * np.sum(self.Vt ** 2)
        reg_u = (self.lam_u / 2) * np.sum(self.U ** 2)
        return self.alpha * L_s + (1 - self.alpha) * L_t + reg_u

    # -----------------------------------------------------------------------
    # One alternating-projection sweep  (paper Section 4.1)
    # -----------------------------------------------------------------------

    def _update_U(self, Xs, Xt, Ws, Wt):
        """
        Update all rows of U (user factors) — paper Eq. 4 Newton step.

        Hessian (block-diagonal, one k×k block per user row):
            H_U[i] = alpha * Vs^T diag(Ws[i]) Vs
                   + (1-alpha) * Vt^T diag(Wt[i]) Vt
                   + lam_u * I          ← U regulariser (total weight = 1 across both sub-losses)
        RHS:
            rhs[i] = alpha * (Xs[i] * Ws[i]) @ Vs
                   + (1-alpha) * (Xt[i] * Wt[i]) @ Vt
        """
        m = self.U.shape[0]
        for i in range(m):
            WsVs = Ws[i][:, None] * self.Vs   # (n_s x k)  diag(Ws[i]) · Vs
            WtVt = Wt[i][:, None] * self.Vt   # (n_t x k)  diag(Wt[i]) · Vt

            H = (self.alpha         * self.Vs.T @ WsVs
               + (1 - self.alpha)   * self.Vt.T @ WtVt
               + self.lam_u         * np.eye(self.k))

            rhs = (self.alpha       * (Xs[i] * Ws[i]) @ self.Vs
                 + (1 - self.alpha) * (Xt[i] * Wt[i]) @ self.Vt)

            self.U[i] = np.linalg.solve(H, rhs)

    def _update_Vs(self, Xs, Ws):
        """
        Update all rows of Vs (source-item factors) — paper Eq. 4 / Section 4.1.

        Vs[j] lives inside L1 (scaled by alpha). Hessian:
            H_Vs[j] = alpha * U^T diag(Ws[:,j]) U  +  alpha * lam_v * I
                                                        ^^^^^^^^^^^^^^
                                                        regulariser ALSO scaled by alpha
                                                        (Vs regulariser is inside L1)
        RHS:
            rhs[j] = alpha * (Xs[:,j] * Ws[:,j]) @ U
        """
        n_s = self.Vs.shape[0]
        for j in range(n_s):
            Wj  = Ws[:, j]                        # (m,)
            Xj  = Xs[:, j]                        # (m,)
            WU  = Wj[:, None] * self.U            # (m x k)  diag(Wj) · U
            H   = self.alpha * (self.U.T @ WU) + self.alpha * self.lam_v * np.eye(self.k)
            rhs = self.alpha * (Xj * Wj) @ self.U
            self.Vs[j] = np.linalg.solve(H, rhs)

    def _update_Vt(self, Xt, Wt):
        """
        Update all rows of Vt (target-item factors) — paper Eq. 6 / Section 4.1.

        Vt[j] lives inside L2 (scaled by 1-alpha). Hessian:
            H_Vt[j] = (1-alpha) * U^T diag(Wt[:,j]) U  +  (1-alpha) * lam_v * I
                                                             ^^^^^^^^^^^^^^^^^
                                                             regulariser ALSO scaled by (1-alpha)
        RHS:
            rhs[j] = (1-alpha) * (Xt[:,j] * Wt[:,j]) @ U
        """
        n_t = self.Vt.shape[0]
        for j in range(n_t):
            Wj  = Wt[:, j]                        # (m,)
            Xj  = Xt[:, j]                        # (m,)
            WU  = Wj[:, None] * self.U            # (m x k)
            H   = (1 - self.alpha) * (self.U.T @ WU) + (1 - self.alpha) * self.lam_v * np.eye(self.k)
            rhs = (1 - self.alpha) * (Xj * Wj) @ self.U
            self.Vt[j] = np.linalg.solve(H, rhs)

    # -----------------------------------------------------------------------
    # Training loop  (paper Section 4.1 alternating Newton projections)
    # -----------------------------------------------------------------------

    def fit(self, Xs, Xt, Ws=None, Wt=None,
            max_iter=50, verbose=True, log_every=5):
        """
        Alternating Newton projections (paper Section 4.1).

        Parameters
        ----------
        Xs       : (m x n_s) source interaction matrix — binary {0,1}, FULLY OBSERVED
        Xt       : (m x n_t) target interaction matrix
        Ws       : (m x n_s) weight matrix for source.
                   MUST be np.ones_like(Xs) — source is fully observed;
                   0-entries are real negative observations, not missing data.
        Wt       : (m x n_t) weight matrix for target.
                   0 for unobserved/cold-start entries, 1 for observed.
        max_iter : number of alternating-projection cycles to run (fixed, no early stop)
        verbose  : print progress
        log_every: print every N iterations
        """
        # ------------------------------------------------------------------
        # Weight matrices
        # ------------------------------------------------------------------
        if Ws is None:
            # Paper: source matrix is FULLY OBSERVED binary.
            # Every (user, item) pair has a known value (0 or 1).
            # Weight = 1 for all entries.
            Ws = np.ones_like(Xs)
        if Wt is None:
            # Target: only observed (positive) interactions are known during training.
            Wt = (Xt > 0).astype(float)

        history = []

        if verbose:
            print(f"CMF training | m={Xs.shape[0]}, n_s={Xs.shape[1]}, "
                  f"n_t={Xt.shape[1]}, k={self.k}, alpha={self.alpha}")
            print(f"lam_u={self.lam_u}, lam_v={self.lam_v}, max_iter={max_iter}")
            print(f"Ws all-ones: {np.all(Ws == 1)}  (should be True)")
            print("-" * 60)

        for it in range(1, max_iter + 1):
            t0 = time.time()

            # Alternating projection order: U -> Vs -> Vt  (paper Section 4.1)
            self._update_U(Xs, Xt, Ws, Wt)
            self._update_Vs(Xs, Ws)
            self._update_Vt(Xt, Wt)

            L = self.loss(Xs, Xt, Ws, Wt)
            history.append(float(L))
            elapsed = time.time() - t0

            if verbose and (it % log_every == 0 or it == 1):
                print(f"  Iter {it:3d}/{max_iter} | Loss={L:.4f} | "
                      f"Time/iter={elapsed:.1f}s")

        if verbose:
            print("-" * 60)
            print(f"Training complete. Final loss: {history[-1]:.4f}")

        return history

    # -----------------------------------------------------------------------
    # Predict
    # -----------------------------------------------------------------------

    def predict_target(self, user_ids=None):
        """
        Predict target-domain scores.  X_target ≈ U · Vt^T (identity link).
        Returns (m x n_t) score matrix, or subset for user_ids.
        """
        U = self.U if user_ids is None else self.U[user_ids]
        return U @ self.Vt.T

    # -----------------------------------------------------------------------
    # Save / Load
    # -----------------------------------------------------------------------

    def save(self, path):
        np.savez(path,
                 U=self.U, Vs=self.Vs, Vt=self.Vt,
                 k=np.array(self.k),
                 alpha=np.array(self.alpha),
                 lam_u=np.array(self.lam_u),
                 lam_v=np.array(self.lam_v))
        print(f"Model saved to {path}.npz")

    @classmethod
    def load(cls, path):
        d = np.load(path)
        obj = cls.__new__(cls)
        obj.U     = d["U"]
        obj.Vs    = d["Vs"]
        obj.Vt    = d["Vt"]
        obj.k     = int(d["k"])
        obj.alpha = float(d["alpha"])
        obj.lam_u = float(d["lam_u"])
        obj.lam_v = float(d["lam_v"])
        return obj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------
    # Config — all values from CMF paper (Singh & Gordon, KDD 2008)
    # ------------------------------------------------------------------
    X_SOURCE_PATH   = "../../FedCLR/VAE/X_source.npy"
    X_TARGET_PATH   = "../../FedCLR/VAE/X_target.npy"
    COLD_START_PATH = "../../FedCLR/VAE/cold_start_users.npy"
    OUT_MODEL       = "cmf_model"       # .npz appended by save()
    OUT_LOG         = "cmf_train_log.json"

    # ---------------------------------------------------------------
    # Hyperparameters — exactly as in CMF paper Section 7.1.1 / 7.2:
    #   k=20              (Section 7, embedding dimension)
    #   alpha=0.5         (Section 7.2, equal weighting of both domains)
    #   lam_u = lam_v = 1e-5  (Section 7.1.1: G(U)=1e5·‖U‖²/2 => lam=1/1e5=1e-5)
    #   tol=0.05          (Section 7.1.1: stop when |ΔL|/L < 5%)
    #   Identity link + squared loss (Section 7.1.1, CMF-Identity variant)
    # ---------------------------------------------------------------
    K        = 20
    ALPHA    = 0.5
    LAM_U    = 1e-5
    LAM_V    = 1e-5
    MAX_ITER = 50        # run all 50 iterations, no early stopping
    LOG_EVERY = 5
    SEED     = 42

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading data...")
    X_source = np.load(X_SOURCE_PATH).astype(np.float64)
    X_target = np.load(X_TARGET_PATH).astype(np.float64)

    m, n_s = X_source.shape
    _,  n_t = X_target.shape
    print(f"  X_source: {X_source.shape}")
    print(f"  X_target: {X_target.shape}")

    # ------------------------------------------------------------------
    # Weight matrices
    # ------------------------------------------------------------------
    # SOURCE: fully-observed binary matrix — ALL entries weighted equally.
    # Paper treats x^S[i,j]=0 as a real negative observation, not missing.
    # Using (X_source > 0) would silently discard all zero-entries,
    # making U/Vs blind to negative feedback and biasing scores upward.
    Ws = np.ones_like(X_source)          # <-- FIX: was (X_source > 0) in old code

    # TARGET (training): mask cold-start users completely (held out for test).
    # For non-cold-start users, only positive interactions are observed.
    cold_start_users = []
    if os.path.exists(COLD_START_PATH):
        cold_start_users = np.load(COLD_START_PATH).tolist()
        print(f"  Cold-start users: {len(cold_start_users)}")
    else:
        print("  No cold_start_users.npy found — using all users for training")

    Wt = (X_target > 0).astype(np.float64)
    if cold_start_users:
        Wt[cold_start_users] = 0.0   # mask cold-start target interactions during training

    # ------------------------------------------------------------------
    # Build and train model
    # ------------------------------------------------------------------
    model = CMF(m, n_s, n_t,
                k=K, alpha=ALPHA,
                lam_u=LAM_U, lam_v=LAM_V,
                seed=SEED)

    t_start = time.time()
    history = model.fit(X_source, X_target, Ws, Wt,
                        max_iter=MAX_ITER,
                        verbose=True,
                        log_every=LOG_EVERY)
    total_time = time.time() - t_start
    print(f"\nTotal training time: {total_time:.1f}s")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    model.save(OUT_MODEL)

    log = {
        "k": K, "alpha": ALPHA, "lam_u": LAM_U, "lam_v": LAM_V,
        "max_iter": MAX_ITER, "seed": SEED,
        "loss_history": history,
        "total_time_s": total_time,
        "final_loss": history[-1],
        "n_iters": len(history),
        "Ws_all_ones": True,
        "paper": "Singh & Gordon, KDD 2008",
    }
    with open(OUT_LOG, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Training log saved to {OUT_LOG}")


if __name__ == "__main__":
    main()