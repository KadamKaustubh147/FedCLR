import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_dim_source, input_dim_target,
                 hidden_dim=300, latent_dim=100):
        super().__init__()

        # Encoder (source domain)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim_source, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh()
        )

        # why not hidden to latent
        self.mu_layer = nn.Linear(latent_dim, latent_dim)
        self.logvar_layer = nn.Linear(latent_dim, latent_dim)

        # Decoder (target domain)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim_target)
        )

    def encode(self, x_s):
        hidden = self.encoder(x_s)
        mu = self.mu_layer(hidden)
        logvar = self.logvar_layer(hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Because predicting log variance is numerically stable and unconstrained.
        # σ=e^(0.5 * logvar) beacuse we need log of standard deviation, not variance, and logvar is log(σ^2) = 2*log(σ)
        log_std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(log_std)
        return mu + eps * log_std

    def decode(self, z):
        logits = self.decoder(z)
        return logits  # NO sigmoid (important)

    def forward(self, x_s):
        mu, logvar = self.encode(x_s)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)

        return logits, mu, logvar, z