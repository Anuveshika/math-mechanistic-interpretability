import torch
import torch.nn as nn
import torch.optim as optim

class SparseAutoencoder(nn.Module):
    def __init__(self, dim, hidden_dim, l1_coeff=1e-3):
        super().__init__()
        self.encoder = nn.Linear(dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, dim)
        self.l1_coeff = l1_coeff

    def forward(self, x):
        z = torch.relu(self.encoder(x))
        x_hat = self.decoder(z)
        return x_hat, z

def train_sae(X, hidden_dim=1024, epochs=10):
    X = torch.tensor(X, dtype=torch.float32)
    sae = SparseAutoencoder(X.shape[1], hidden_dim)
    opt = optim.Adam(sae.parameters(), lr=1e-3)

    for _ in range(epochs):
        opt.zero_grad()
        x_hat, z = sae(X)
        loss = ((X - x_hat) ** 2).mean() + sae.l1_coeff * z.abs().mean()
        loss.backward()
        opt.step()

    return sae
