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
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from tqdm import tqdm

df = pd.read_csv("oasis3_clinical.csv", low_memory=False)

# --- Identify meta columns
time_col_candidates = ["days_to_visit"]
patient_col_candidates = ["OASIS_session_label", "patient_id"]

time_col = next((c for c in time_col_candidates if c in df.columns), None)
pid_col  = next((c for c in patient_col_candidates if c in df.columns), None)

if pid_col is None:
    raise ValueError("No patient ID column found.")

if time_col is None:
    print("⚠️ No time column found — using row index as temporal order")
    df["__t__"] = df.groupby(pid_col).cumcount()
    time_col = "__t__"

# --- Drop rows with no patient ID
df = df.dropna(subset=[pid_col])

# --- Fill missing data carefully
df = df.copy()
df = df.fillna("missing")

# --- Remove ID/time columns from features
feature_cols = [c for c in df.columns if c not in [pid_col, time_col]]

# --- Final structure: dict[patient_id] = list of events
patient_events = {}

for pid, group in tqdm(df.groupby(pid_col)):
    group = group.sort_values(time_col)

    events = []
    for _, row in group.iterrows():
        t = row[time_col]

        for col in feature_cols:
            value = row[col]
            events.append((t, col, value))

    patient_events[pid] = events

# Example output:
# patient_events["S1"] → list of (t, variate_name, variate_value)
print("Example:", patient_events[next(iter(patient_events))][:10])
print(patient_events)

exit()

# Keep IDs for temporal grouping
id_cols = ["patient_id", "session_id", "days_to_visit"]
existing_cols = [c for c in id_cols if c in df.columns]
if existing_cols:
    meta_df = df[existing_cols].copy()
else:
    # Fallback: create dummy subject/session ids
    meta_df = pd.DataFrame({
        "patient_id": np.arange(len(df)),
        "session_id": np.zeros(len(df), dtype=int),
        "days_to_visit": np.arange(len(df))
    })
df = df.drop(columns=[c for c in id_cols if c in df.columns], errors="ignore")

categorical_cols = [c for c in df.columns if df[c].dtype == "object"]
numeric_cols = [c for c in df.columns if c not in categorical_cols]

df[categorical_cols] = df[categorical_cols].fillna("missing")
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

preprocessor = ColumnTransformer([
    ("num", RobustScaler(quantile_range=(5,95)), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical_cols)
])

X = preprocessor.fit_transform(df)
print("Matrix type:", type(X))
print("Shape:", X.shape)
print("Density:", X.nnz / (X.shape[0] * X.shape[1]))

# -----------------------
# VisitVQVAE Training
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

class SparseDataset(Dataset):
    def __init__(self, X):
        self.X = X
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx].toarray(), dtype=torch.float32).squeeze(0)

dataset = SparseDataset(X)
loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)

# Define VisitVQVAE (assumed already implemented)
model = VisitVQVAE(input_dim=X.shape[1], latent_dim=256, num_embeddings=256).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

for epoch in range(20):
    total_loss = 0
    model.train()
    for batch in loader:
        batch = batch.to(device)
        batch = torch.nan_to_num(batch, nan=0.0, posinf=1.0, neginf=-1.0)

        x_rec, _, vq_loss = model(batch)
        if not torch.isfinite(x_rec).all() or not torch.isfinite(vq_loss):
            print("Non-finite values detected in model outputs — skipping batch")
            continue

        recon_loss = F.l1_loss(x_rec, batch)
        loss = recon_loss + vq_loss

        if not torch.isfinite(loss):
            print("NaN detected, skipping batch")
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * batch.size(0)

    print(f"Epoch {epoch}: total_loss={total_loss/len(dataset):.4f}")

# -----------------------
# Token Extraction
# -----------------------
all_tokens = []
model.eval()
with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)
        z_e = model.encoder(batch)
        _, token_ids, _ = model.vq(z_e)
        all_tokens.extend(token_ids.cpu().numpy())

tokens = np.array(all_tokens, dtype=np.int64)

# Shift by +1 to reserve PAD_ID=0
tokens += 1
PAD_ID = 0

num_tokens = int(tokens.max()) + 1  # include PAD
print("tokens max:", tokens.max())  # should be 257 or something reasonable
print("num_tokens:", num_tokens)    # must be >= tokens.max() + 1
print(tokens.shape, tokens[:10])

# Attach back to subject/session IDs
meta_df["token_id"] = tokens

# -----------------------
# Group by Subject for Temporal Modeling
# -----------------------
seqs = []
for sid, group in meta_df.groupby("patient_id"):
    group = group.sort_values("days_to_visit")
    seqs.append(group["token_id"].tolist())

from torch.nn.utils.rnn import pad_sequence

PAD_ID = 0
seq_tensors = [torch.tensor(s, dtype=torch.long) for s in seqs if len(s) > 1]
padded = pad_sequence(seq_tensors, batch_first=True, padding_value=PAD_ID)
print(padded.max() < num_tokens)
attention_masks = (padded != PAD_ID).long()

print("Number of sequences:", len(seqs))
print("Padded shape:", padded.shape)

# -----------------------
# TemporalTOTEM Model
# -----------------------
import torch.nn as nn

class TemporalTOTEM(nn.Module):
    def __init__(self, num_tokens, token_dim=128, n_layers=4, n_heads=4, max_len=32):
        super().__init__()
        self.token_embed = nn.Embedding(num_tokens, token_dim)
        self.pos_embed = nn.Embedding(max_len, token_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=token_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(token_dim, num_tokens)

    def forward(self, tokens, mask=None):
        seq_len = tokens.size(1)
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        x = self.token_embed(tokens) + self.pos_embed(positions)
        x = self.transformer(x, src_key_padding_mask=(mask == 0) if mask is not None else None)
        logits = self.fc_out(x)
        return logits

# -----------------------
# Train TOTEM Temporal Transformer
# -----------------------
num_tokens = int(tokens.max()) + 1  # include PAD
temporal_model = TemporalTOTEM(num_tokens=num_tokens).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
optimizer_t = torch.optim.Adam(temporal_model.parameters(), lr=1e-4)

padded = padded.to(device)
attention_masks = attention_masks.to(device)

for epoch in range(10):
    temporal_model.train()
    total_loss = 0
    for seq, mask in zip(padded, attention_masks):
        seq = seq.unsqueeze(0)
        mask = mask.unsqueeze(0)

        logits = temporal_model(seq[:, :-1], mask=mask[:, :-1])
        target = seq[:, 1:]

        loss = criterion(logits.reshape(-1, num_tokens), target.reshape(-1))
        if torch.isnan(loss):
            print("NaN detected, skipping batch")
            continue

        optimizer_t.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(temporal_model.parameters(), 1.0)
        optimizer_t.step()
        total_loss += loss.item()

    print(f"[Temporal] Epoch {epoch}: loss={total_loss/len(padded):.4f}")

