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

# --- Identify meta columns (FORCE grouping by patient_id)
pid_col = "patient_id"
if pid_col not in df.columns:
    raise ValueError("Expected a 'patient_id' column in the CSV.")

# Prefer days_to_visit as the time column; fallback to artificial index
if "days_to_visit" in df.columns:
    time_col = "days_to_visit"
else:
    print("⚠️ No days_to_visit column found — using row index as temporal order")
    df["__t__"] = df.groupby(pid_col).cumcount()
    time_col = "__t__"

# --- Drop rows with no patient ID
df = df.dropna(subset=[pid_col])

# --- Fill missing data carefully
df = df.copy()
df = df.fillna("missing")

# --- Remove ID/time columns from features
feature_cols = [c for c in df.columns if c not in [pid_col, time_col]]

# Drop session-like IDs from features if present
drop_from_features = ["OASIS_session_label", "session_id"]
feature_cols = [c for c in feature_cols if c not in drop_from_features]



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

# ==========================
# Build events DataFrame from patient_events
# ==========================
rows = []
for pid, events in patient_events.items():
    for (t, col, value) in events:
        rows.append({
            pid_col: pid,      # e.g. "patient_id" or "OASIS_session_label"
            time_col: t,       # e.g. "days_to_visit" or "__t__"
            "variate": col,
            "value": value
        })

events_df = pd.DataFrame(rows)

# Ensure sorted by patient + time
events_df = events_df.sort_values([pid_col, time_col]).reset_index(drop=True)

print("Events df head:")
print(events_df.head())

# ==========================
# STEP 1: Variate embedding (e_variate)
# ==========================

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

# Build variate vocabulary
unique_variates = sorted(events_df["variate"].unique())
variate2id = {v: i for i, v in enumerate(unique_variates)}
id2variate = {i: v for v, i in variate2id.items()}
num_variates = len(unique_variates)
print("num_variates:", num_variates)

# Add integer ID for each variate
events_df["variate_id"] = events_df["variate"].map(variate2id).astype(int)

class VariateEmbedding(nn.Module):
    def __init__(self, num_variates, d_model):
        super().__init__()
        self.variate_embed = nn.Embedding(num_variates, d_model)

    def forward(self, variate_ids):
        """
        variate_ids: LongTensor (B, T) or (N,)
        returns:     FloatTensor (B, T, d_model) or (N, d_model)
        """
        return self.variate_embed(variate_ids)

# ==========================
# STEP 2: Time-delta embedding (e_time)
# ==========================

# Compute time delta per event within each patient
events_df["delta_t"] = events_df.groupby(pid_col)[time_col].diff()
events_df["delta_t"] = events_df["delta_t"].fillna(0.0)

# Optional: normalize delta_t (helps stability)
max_dt = events_df["delta_t"].max()
if max_dt == 0 or pd.isna(max_dt):
    events_df["delta_t_norm"] = 0.0
else:
    events_df["delta_t_norm"] = events_df["delta_t"] / max_dt

class TimeDeltaEmbedding(nn.Module):
    def __init__(self, d_model, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, delta_t):
        """
        delta_t: FloatTensor (B, T) or (N,)
        returns: FloatTensor (B, T, d_model) or (N, d_model)
        """
        if delta_t.dim() == 1:
            delta_t = delta_t.unsqueeze(-1)           # (N, 1)
        elif delta_t.dim() == 2:
            delta_t = delta_t.unsqueeze(-1)           # (B, T, 1)
        else:
            raise ValueError("delta_t must be 1D or 2D")
        return self.mlp(delta_t)

# ==========================
# STEP 3: Classify variates by type (numeric / categorical / text)
# ==========================

# Heuristic: figure out which feature columns are numeric based on df before casting
numeric_variates = []
categorical_variates = []
text_variates = []

for col in feature_cols:
    # Try to see if this column is mostly numeric
    col_series = df[col].replace("missing", np.nan)
    col_num = pd.to_numeric(col_series, errors="coerce")
    frac_numeric = col_num.notna().mean()

    if frac_numeric > 0.9:
        numeric_variates.append(col)
    else:
        categorical_variates.append(col)

print("numeric_variates:", numeric_variates[:10], "...")
print("categorical_variates:", categorical_variates[:10], "...")
print("text_variates:", text_variates)

# ---- Numeric values: convert to float and compute per-variate stats
events_df["value_num"] = pd.to_numeric(
    events_df["value"].replace("missing", np.nan), errors="coerce"
)

num_stats = (
    events_df[events_df["variate"].isin(numeric_variates)]
    .groupby("variate")["value_num"]
    .agg(["mean", "std"])
    .reset_index()
)

# Ensure no zero std to avoid division by zero
num_stats["std"] = num_stats["std"].replace(0, 1.0).fillna(1.0)
num_stats["mean"] = num_stats["mean"].fillna(0.0)

# Build tensors aligned with variate_id indices: for each variate_id, store mean/std and type
VAR_TYPE_NUM = 0
VAR_TYPE_CAT = 1
VAR_TYPE_TEXT = 2

variate_type = torch.full((num_variates,), VAR_TYPE_CAT, dtype=torch.long)  # default categorical
numeric_means = torch.zeros(num_variates, dtype=torch.float32)
numeric_stds  = torch.ones(num_variates,  dtype=torch.float32)

for _, row in num_stats.iterrows():
    vname = row["variate"]
    if vname not in variate2id:
        continue
    vid = variate2id[vname]
    variate_type[vid] = VAR_TYPE_NUM
    numeric_means[vid] = float(row["mean"])
    numeric_stds[vid]  = float(row["std"])

for vname in text_variates:
    if vname in variate2id:
        variate_type[variate2id[vname]] = VAR_TYPE_TEXT

# ---- Categorical values: build a global vocabulary over (variate, value) pairs
cat_rows = events_df[events_df["variate"].isin(categorical_variates)][["variate", "value"]].drop_duplicates()
cat_rows = cat_rows.reset_index(drop=True)
cat_rows["cat_id"] = np.arange(len(cat_rows), dtype=np.int64)

pair2cat_id = {(r.variate, r.value): int(r.cat_id) for r in cat_rows.itertuples()}

events_df["cat_id"] = -1
mask_cat = events_df["variate"].isin(categorical_variates)
events_df.loc[mask_cat, "cat_id"] = events_df.loc[mask_cat].apply(
    lambda r: pair2cat_id[(r["variate"], r["value"])], axis=1
)

num_cat_tokens = len(cat_rows)
print("num_cat_tokens:", num_cat_tokens)


# ==========================
# STEP 4: Event value embedding module (e_value)
# ==========================

class EventValueEmbedding(nn.Module):
    def __init__(
        self,
        d_model,
        num_variates,
        variate_type_tensor,
        numeric_means,
        numeric_stds,
        num_cat_tokens,
        text_encoder=None,  # optional: plug in a clinical LM wrapper later
    ):
        super().__init__()
        self.d_model = d_model

        # Register buffers so they move with .to(device) and are saved with state_dict
        self.register_buffer("variate_type", variate_type_tensor.clone())
        self.register_buffer("numeric_means", numeric_means.clone())
        self.register_buffer("numeric_stds",  numeric_stds.clone())

        # Numeric value MLP: Case A (floats)
        self.numeric_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, d_model),
        )

        # Categorical value embedding: Case B (bool/categorical)
        if num_cat_tokens > 0:
            self.cat_embed = nn.Embedding(num_cat_tokens, d_model)
        else:
            self.cat_embed = None

        # Text encoder: Case C (clinical notes) – you can wire in a pre-trained LM here
        self.text_encoder = text_encoder  # should map a batch of text -> (B, T, d_model) or similar

        # Type constants
        self.VAR_TYPE_NUM = VAR_TYPE_NUM
        self.VAR_TYPE_CAT = VAR_TYPE_CAT
        self.VAR_TYPE_TEXT = VAR_TYPE_TEXT

    def forward(self, variate_ids, value_num, cat_ids, texts=None):
        """
        variate_ids : LongTensor (B, T)
        value_num   : FloatTensor (B, T) - raw or prefilled floats (0.0 for non-numeric)
        cat_ids     : LongTensor (B, T) - categorical IDs, -1 for non-categorical
        texts       : optional nested list/array [[str_or_None]*T]*B for text features

        returns:
            e_value : FloatTensor (B, T, d_model)
        """
        B, T = variate_ids.shape
        device = variate_ids.device

        # Initialize with zeros
        e_value = torch.zeros(B, T, self.d_model, device=device)

        # Lookup per-token variate types
        var_types = self.variate_type[variate_ids]  # (B, T)

        # ----- Case A: numeric -----
        mask_num = (var_types == self.VAR_TYPE_NUM)

        if mask_num.any():
            # gather means/stds for each event
            mu = self.numeric_means[variate_ids]     # (B, T)
            sigma = self.numeric_stds[variate_ids]   # (B, T)
            # normalize
            v = (value_num - mu) / sigma
            v = v.unsqueeze(-1)                      # (B, T, 1)
            e_num_full = self.numeric_mlp(v)         # (B, T, d_model)
            e_value[mask_num] = e_num_full[mask_num]

        # ----- Case B: categorical / boolean -----
        if self.cat_embed is not None:
            mask_cat = (var_types == self.VAR_TYPE_CAT) & (cat_ids >= 0)
            if mask_cat.any():
                # cat_ids is (B, T)
                cat_ids_clamped = cat_ids.clone()
                cat_ids_clamped[cat_ids_clamped < 0] = 0  # avoid -1 indexing
                e_cat_full = self.cat_embed(cat_ids_clamped)  # (B, T, d_model)
                e_value[mask_cat] = e_cat_full[mask_cat]

        # ----- Case C: text (optional, stub) -----
        # If you have free-text notes and a text encoder, you can handle them here.
        # Example (pseudo-code):
        # if self.text_encoder is not None and texts is not None:
        #     mask_text = (var_types == self.VAR_TYPE_TEXT)
        #     # Pack texts[batch][time] where mask_text is True, run encoder, then scatter back.

        return e_value
    

# ==========================
# STEP 5: Build sequences for variate_id, delta_t_norm, value_num and cat_id
# ==========================

from torch.nn.utils.rnn import pad_sequence

seqs_variate   = []   # List[Tensor(T,)]
seqs_deltat    = []   # List[Tensor(T,)]
seqs_value_num = []   # List[Tensor(T,)]
seqs_cat_id    = []   # List[Tensor(T,)]

for pid, group in events_df.groupby(pid_col):
    group = group.sort_values(time_col)

    var_ids = torch.tensor(group["variate_id"].values, dtype=torch.long)
    dt_norm = torch.tensor(group["delta_t_norm"].values, dtype=torch.float32)

    # numeric values: float32; NaN -> 0.0 (we'll also use per-variate means/std in the module)
    v_num = torch.tensor(
        group["value_num"].fillna(0.0).values, dtype=torch.float32
    )

    # categorical ids: int64; -1 for non-categorical
    v_cat = torch.tensor(
        group["cat_id"].fillna(-1).values, dtype=torch.long
    )

    if len(var_ids) == 0:
        continue

    seqs_variate.append(var_ids)
    seqs_deltat.append(dt_norm)
    seqs_value_num.append(v_num)
    seqs_cat_id.append(v_cat)

PAD_VAR_ID  = 0
PAD_DT      = 0.0
PAD_NUMVAL  = 0.0
PAD_CAT_ID  = -1

padded_variate_ids = pad_sequence(seqs_variate,    batch_first=True, padding_value=PAD_VAR_ID)
padded_delta_t      = pad_sequence(seqs_deltat,     batch_first=True, padding_value=PAD_DT)
padded_value_num    = pad_sequence(seqs_value_num,  batch_first=True, padding_value=PAD_NUMVAL)
padded_cat_id       = pad_sequence(seqs_cat_id,     batch_first=True, padding_value=PAD_CAT_ID)

print("padded_variate_ids shape:", padded_variate_ids.shape)
print("padded_delta_t shape:", padded_delta_t.shape)
print("padded_value_num shape:", padded_value_num.shape)
print("padded_cat_id shape:", padded_cat_id.shape)

# ==========================
# Instantiate the embedding modules and test shapes
# ==========================

d_model = 128  # or whatever dimension you want

var_embedder  = VariateEmbedding(num_variates=num_variates, d_model=d_model)
time_embedder = TimeDeltaEmbedding(d_model=d_model, hidden_dim=64)

value_embedder = EventValueEmbedding(
    d_model=d_model,
    num_variates=num_variates,
    variate_type_tensor=variate_type,
    numeric_means=numeric_means,
    numeric_stds=numeric_stds,
    num_cat_tokens=num_cat_tokens,
    text_encoder=None,  # later
)

event_layernorm = nn.LayerNorm(d_model)

BATCH_PATIENTS = 8

with torch.no_grad():
    v_ids   = padded_variate_ids[:BATCH_PATIENTS]
    dt_norm = padded_delta_t[:BATCH_PATIENTS]
    v_num   = padded_value_num[:BATCH_PATIENTS]
    v_cat   = padded_cat_id[:BATCH_PATIENTS]

    e_variate = var_embedder(v_ids)                 # (B, T, d_model)
    e_time    = time_embedder(dt_norm)              # (B, T, d_model)
    e_value   = value_embedder(v_ids, v_num, v_cat) # (B, T, d_model)

    e_event = event_layernorm(e_variate + e_time + e_value)

print("e_event shape:", e_event.shape)




################################################################################################
######### TESTING OUTPUT OF PREPROCESSING STEP ONLY ############################################################################################################################
################################################################################################
# Example output:
# patient_events["S1"] → list of (t, variate_name, variate_value)


# print("Example:", patient_events[next(iter(patient_events))][:10])

# rows = []
# for pid, events in patient_events.items():
#     for (t, col, value) in events:
#         rows.append({
#             pid_col: pid,          # e.g. "patient_id" or "OASIS_session_label"
#             time_col: t,           # e.g. "days_to_visit" or "__t__"
#             "variate": col,
#             "value": value
#         })

# events_df = pd.DataFrame(rows)
# events_df.to_csv("patient_events.csv", index=False)
# print("Saved events to patient_events.csv")




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