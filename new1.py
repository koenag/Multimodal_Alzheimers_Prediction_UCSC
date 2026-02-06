import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, Optional, Callable
from tqdm import tqdm
import math

# ==========================================
# PART 1: YOUR ORIGINAL DATA PREPROCESSING
# ==========================================

# ... [Copying your embedding classes here for completeness] ...

class VariateEmbedding(nn.Module):
    def __init__(self, num_variates, d_model):
        super().__init__()
        self.variate_embed = nn.Embedding(num_variates, d_model)

    def forward(self, variate_ids):
        return self.variate_embed(variate_ids)

class TimeDeltaEmbedding(nn.Module):
    def __init__(self, d_model, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, delta_t):
        if delta_t.dim() == 1:
            delta_t = delta_t.unsqueeze(-1)
        elif delta_t.dim() == 2:
            delta_t = delta_t.unsqueeze(-1)
        return self.mlp(delta_t)

class EventValueEmbedding(nn.Module):
    def __init__(self, d_model, num_variates, variate_type_tensor, numeric_means, numeric_stds, num_cat_tokens, text_encoder=None):
        super().__init__()
        self.d_model = d_model
        self.register_buffer("variate_type", variate_type_tensor.clone())
        self.register_buffer("numeric_means", numeric_means.clone())
        self.register_buffer("numeric_stds",  numeric_stds.clone())

        self.numeric_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, d_model),
        )

        if num_cat_tokens > 0:
            self.cat_embed = nn.Embedding(num_cat_tokens, d_model)
        else:
            self.cat_embed = None
            
        self.VAR_TYPE_NUM = 0
        self.VAR_TYPE_CAT = 1
        self.VAR_TYPE_TEXT = 2
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, variate_ids, value_num, cat_ids, texts=None):
        B, T = variate_ids.shape
        device = variate_ids.device
        e_value = torch.zeros(B, T, self.d_model, device=device)
        var_types = self.variate_type[variate_ids] 

        mask_num = (var_types == self.VAR_TYPE_NUM)
        if mask_num.any():
            mu = self.numeric_means[variate_ids] 
            sigma = self.numeric_stds[variate_ids] 
            v = (value_num - mu) / (sigma + 1e-6) # Added epsilon for safety
            v = v.unsqueeze(-1)
            e_num_full = self.numeric_mlp(v)
            # Safe assignment using boolean masking
            B_idx, T_idx = torch.where(mask_num)
            e_value[B_idx, T_idx] = e_num_full[B_idx, T_idx]
            
        if self.cat_embed is not None:
            mask_cat = (var_types == self.VAR_TYPE_CAT) & (cat_ids >= 0)
            if mask_cat.any():
                cat_ids_clamped = cat_ids.clone()
                cat_ids_clamped[cat_ids_clamped < 0] = 0
                e_cat_full = self.cat_embed(cat_ids_clamped)
                B_idx, T_idx = torch.where(mask_cat)
                e_value[B_idx, T_idx] = e_cat_full[B_idx, T_idx]

        return self.layernorm(e_value)

# --- LOAD DATA ---
# (Make sure the path is correct for your system)
print("Loading Data...")
# NOTE: Update this path to match your file location exactly
import os

# Get the folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Build the full path to the csv
csv_path = os.path.join(script_dir, "oasis3_clinical_modified1.csv")

df = pd.read_csv(csv_path, low_memory=False)
pid_col = "patient_id"
if "days_to_visit" in df.columns:
    time_col = "days_to_visit"
else:
    df["__t__"] = df.groupby(pid_col).cumcount()
    time_col = "__t__"

df = df.dropna(subset=[pid_col]).copy()
df = df.fillna("missing")

feature_cols = [c for c in df.columns if c not in [pid_col, time_col, "OASIS_session_label", "session_id"]]

patient_events = {}
for pid, group in tqdm(df.groupby(pid_col), desc="Grouping Events"):
    group = group.sort_values(time_col)
    events = []
    for _, row in group.iterrows():
        t = row[time_col]
        for col in feature_cols:
            value = row[col]       
            if value == "missing": continue
            events.append((t, col, value))
    patient_events[pid] = events

rows = []
for pid, events in patient_events.items():
    for (t, col, value) in events:
        rows.append({pid_col: pid, time_col: t, "variate": col, "value": value})

events_df = pd.DataFrame(rows).sort_values([pid_col, time_col]).reset_index(drop=True)
unique_variates = sorted(events_df["variate"].unique())
variate2id = {v: i for i, v in enumerate(unique_variates)}
num_variates = len(unique_variates)
events_df["variate_id"] = events_df["variate"].map(variate2id).astype(int)
events_df["delta_t"] = events_df.groupby(pid_col)[time_col].diff().fillna(0.0)
max_dt = events_df["delta_t"].max()
events_df["delta_t_norm"] = events_df["delta_t"] / (max_dt if max_dt > 0 else 1.0)

# --- STATS CALCULATION ---
numeric_variates = []
categorical_variates = []
for col in feature_cols:
    col_series = df[col].replace("missing", np.nan)
    col_num = pd.to_numeric(col_series, errors="coerce")
    if col_num.notna().mean() > 0.9:
        numeric_variates.append(col)
    else:
        categorical_variates.append(col)

events_df["value_num"] = pd.to_numeric(events_df["value"].replace("missing", np.nan), errors="coerce")

num_stats = events_df[events_df["variate"].isin(numeric_variates)].groupby("variate")["value_num"].agg(["mean", "std"]).reset_index()
num_stats["std"] = num_stats["std"].replace(0, 1.0).fillna(1.0)
num_stats["mean"] = num_stats["mean"].fillna(0.0)

VAR_TYPE_NUM = 0
VAR_TYPE_CAT = 1
VAR_TYPE_TEXT = 2

variate_type = torch.full((num_variates,), VAR_TYPE_CAT, dtype=torch.long)
numeric_means = torch.zeros(num_variates, dtype=torch.float32)
numeric_stds  = torch.ones(num_variates,  dtype=torch.float32)

for _, row in num_stats.iterrows():
    vid = variate2id[row["variate"]]
    variate_type[vid] = VAR_TYPE_NUM
    numeric_means[vid] = float(row["mean"])
    numeric_stds[vid]  = float(row["std"])

cat_rows = events_df[events_df["variate"].isin(categorical_variates)][["variate", "value"]].drop_duplicates().reset_index(drop=True)
cat_rows["cat_id"] = np.arange(len(cat_rows), dtype=np.int64)
pair2cat_id = {(r.variate, r.value): int(r.cat_id) for r in cat_rows.itertuples()}

events_df["cat_id"] = -1
mask_cat = events_df["variate"].isin(categorical_variates)
events_df.loc[mask_cat, "cat_id"] = events_df.loc[mask_cat].apply(lambda r: pair2cat_id.get((r["variate"], r["value"]), -1), axis=1)

num_cat_tokens = len(cat_rows)

# --- SEQUENCE PADDING ---
seqs_variate, seqs_deltat, seqs_value_num, seqs_cat_id = [], [], [], []

print("Building Sequences...")
for pid, group in events_df.groupby(pid_col):
    group = group.sort_values(time_col)
    if len(group) == 0: continue
    seqs_variate.append(torch.tensor(group["variate_id"].values, dtype=torch.long))
    seqs_deltat.append(torch.tensor(group["delta_t_norm"].values, dtype=torch.float32))
    seqs_value_num.append(torch.tensor(group["value_num"].fillna(0.0).values, dtype=torch.float32))
    seqs_cat_id.append(torch.tensor(group["cat_id"].fillna(-1).values, dtype=torch.long))

padded_variate_ids = pad_sequence(seqs_variate, batch_first=True, padding_value=0)
padded_delta_t     = pad_sequence(seqs_deltat,  batch_first=True, padding_value=0.0)
padded_value_num   = pad_sequence(seqs_value_num, batch_first=True, padding_value=0.0)
padded_cat_id      = pad_sequence(seqs_cat_id,  batch_first=True, padding_value=-1)

# --- CREATE EMBEDDINGS (e_event) ---
d_model = 128
var_embedder  = VariateEmbedding(num_variates=num_variates, d_model=d_model)
time_embedder = TimeDeltaEmbedding(d_model=d_model, hidden_dim=64)
value_embedder = EventValueEmbedding(d_model, num_variates, variate_type, numeric_means, numeric_stds, num_cat_tokens)
event_layernorm = nn.LayerNorm(d_model)

print("generating e_event embeddings...")
# Run on CPU for safety or move to GPU if fits
with torch.no_grad():
    e_variate = var_embedder(padded_variate_ids)
    e_time    = time_embedder(padded_delta_t)
    e_value   = value_embedder(padded_variate_ids, padded_value_num, padded_cat_id)
    e_event   = event_layernorm(e_variate + e_time + e_value)

print("e_event shape:", e_event.shape) 
# e_event is now ready! (Num_Patients, Max_Seq_Len, 128)


# ==========================================
# PART 2: THE NEW TIME-SERIES MODEL
# ==========================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return self.dropout(x)

class ContinuousTransformerGPT(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(d_model, d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = self.pos_encoder(src)
        output = self.transformer_decoder(
            tgt=src, 
            memory=src, 
            tgt_mask=src_mask,
            memory_mask=src_mask,
            tgt_key_padding_mask=src_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        prediction = self.output_head(output)
        return prediction

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# --- TRAINING SETUP ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {DEVICE}")

model = ContinuousTransformerGPT(d_model=d_model, nhead=4, num_layers=4).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# Prepare Data for Training
padding_mask = (padded_variate_ids == 0) # Boolean mask: True where padding exists

# Create TensorDataset
dataset = torch.utils.data.TensorDataset(e_event.detach(), padding_mask)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

print("Starting Time-Series Training...")

for epoch in range(10):
    model.train()
    total_loss = 0
    count = 0
    
    for batch_events, batch_pad_mask in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
        batch_events = batch_events.to(DEVICE)       # (B, T, 128)
        batch_pad_mask = batch_pad_mask.to(DEVICE)   # (B, T)
        
        B, T, D = batch_events.shape
        
        # Causal Mask (Teacher Forcing)
        causal_mask = generate_square_subsequent_mask(T).to(DEVICE)
        
        # Forward Pass
        predictions = model(
            src=batch_events, 
            src_mask=causal_mask,
            src_key_padding_mask=batch_pad_mask
        )
        
        # Shift for Teacher Forcing:
        # Preds: t=0..T-1 (predicting next step)
        preds_shifted = predictions[:, :-1, :]
        # Targets: t=1..T (actual next step)
        targets_shifted = batch_events[:, 1:, :]
        
        # Mask for Loss
        # We need to ignore loss where targets are padding
        # batch_pad_mask is True for padding. We want True for VALID data for masking.
        valid_mask = ~batch_pad_mask[:, 1:] 
        
        # Apply mask
        loss = criterion(preds_shifted[valid_mask], targets_shifted[valid_mask])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        count += 1

    print(f"Epoch {epoch}: Avg MSE Loss {total_loss/count:.6f}")

print("Training Complete!")
torch.save(model.state_dict(), "continuous_transformer_gpt.pt")