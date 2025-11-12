import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, Dataset
from scipy import sparse
import numpy as np

# -----------------------
# Vector Quantizer
# -----------------------
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=256, embedding_dim=256, beta=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z_e):
        # z_e: (B, D)
        dist = (
            torch.sum(z_e ** 2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight ** 2, dim=1)
            - 2 * torch.matmul(z_e, self.embeddings.weight.t())
        )
        encoding_inds = torch.argmin(dist, dim=1)
        z_q = self.embeddings(encoding_inds)
        commitment_loss = F.mse_loss(z_e.detach(), z_q)
        codebook_loss = F.mse_loss(z_e, z_q.detach())
        loss = codebook_loss + self.beta * commitment_loss
        z_q = z_e + (z_q - z_e).detach()
        return z_q, encoding_inds, loss


# -----------------------
# Visit-level VQ-VAE
# -----------------------
class VisitVQVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=128, num_embeddings=512, beta=0.25):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.vq = VectorQuantizer(num_embeddings, latent_dim, beta)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, indices, vq_loss = self.vq(z_e)
        x_rec = self.decoder(z_q)
        return x_rec, indices, vq_loss


# -----------------------
# Sparse dataset loader
# -----------------------
class SparseDataset(Dataset):
    def __init__(self, sparse_matrix):
        self.sparse_matrix = sparse_matrix.tocsr()

    def __len__(self):
        return self.sparse_matrix.shape[0]

    def __getitem__(self, idx):
        row = self.sparse_matrix[idx].toarray().ravel().astype(np.float32)
        return torch.from_numpy(row)


# -----------------------
# Preprocessing
# -----------------------
df = pd.read_csv("oasis3_clinical.csv", low_memory=False)
id_cols = ["subject", "session_id", "visit_date"]
df = df.drop(columns=[c for c in id_cols if c in df.columns], errors="ignore")

categorical_cols = [c for c in df.columns if df[c].dtype == "object"]
numeric_cols = [c for c in df.columns if c not in categorical_cols]

df[categorical_cols] = df[categorical_cols].fillna("missing")
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

from sklearn.preprocessing import RobustScaler

preprocessor = ColumnTransformer([
    ("num", RobustScaler(quantile_range=(5,95)), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical_cols)
])

X = preprocessor.fit_transform(df)
print("Matrix type:", type(X))
print("Shape:", X.shape)
print("Density:", X.nnz / (X.shape[0] * X.shape[1]))

# -----------------------
# Dataloader + Training
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = SparseDataset(X)
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

model = VisitVQVAE(input_dim=X.shape[1], latent_dim=256, num_embeddings=256).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

for epoch in range(10):
    total_loss = 0
    model.train()
    for batch in loader:
        batch = batch.to(device)
        x_rec, _, vq_loss = model(batch)
        recon_loss = F.l1_loss(x_rec, batch)
        loss = recon_loss + vq_loss

        if torch.isnan(loss):
            print("NaN detected, skipping batch")
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * batch.size(0)

    print(f"Epoch {epoch}: total_loss={total_loss/len(dataset):.4f}")


# -----------------------
# Token extraction
# -----------------------
all_tokens = []
model.eval()
with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)
        z_e = model.encoder(batch)
        _, token_ids, _ = model.vq(z_e)
        all_tokens.extend(token_ids.cpu().numpy())

tokens = np.array(all_tokens)
print(tokens.shape, tokens[:10])
