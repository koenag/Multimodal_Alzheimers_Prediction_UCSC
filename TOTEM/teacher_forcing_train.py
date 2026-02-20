
import os
import math
from typing import Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ==========================================
# PART 1: DATA PREPROCESSING (EVENTS -> TIMESTEPS -> TENSORS)
# ==========================================

class VariateEmbedding(nn.Module):
    def __init__(self, num_variates: int, d_model: int):
        super().__init__()
        self.variate_embed = nn.Embedding(num_variates, d_model)

    def forward(self, variate_ids: torch.LongTensor) -> torch.Tensor:
        return self.variate_embed(variate_ids)

class TimeDeltaEmbedding(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        if delta_t.dim() == 1:
            delta_t = delta_t.unsqueeze(-1)
        elif delta_t.dim() == 2:
            delta_t = delta_t.unsqueeze(-1)
        return self.mlp(delta_t)

class EventValueEmbedding(nn.Module):
    VAR_TYPE_NUM = 0
    VAR_TYPE_CAT = 1
    VAR_TYPE_TEXT = 2

    def __init__(self, d_model: int, variate_type_tensor: torch.LongTensor, numeric_means: torch.FloatTensor, numeric_stds: torch.FloatTensor, num_cat_tokens: int):
        super().__init__()
        self.d_model = d_model
        self.register_buffer("variate_type", variate_type_tensor.clone().long())
        self.register_buffer("numeric_means", numeric_means.clone().float())
        self.register_buffer("numeric_stds", numeric_stds.clone().float())

        self.numeric_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, d_model),
        )

        self.cat_embed = nn.Embedding(num_cat_tokens, d_model) if num_cat_tokens > 0 else None
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, variate_ids: torch.LongTensor, value_num: torch.FloatTensor, cat_ids: torch.LongTensor) -> torch.Tensor:
        B, L = variate_ids.shape
        device = variate_ids.device
        e_value = torch.zeros(B, L, self.d_model, device=device)
        var_types = self.variate_type[variate_ids]
        
        mask_num = (var_types == self.VAR_TYPE_NUM)
        if mask_num.any():
            mu = self.numeric_means[variate_ids]
            sigma = self.numeric_stds[variate_ids]
            v = (value_num - mu) / (sigma + 1e-6)
            e_num = self.numeric_mlp(v.unsqueeze(-1))
            b_idx, l_idx = torch.where(mask_num)
            e_value[b_idx, l_idx] = e_num[b_idx, l_idx]

        if self.cat_embed is not None:
            mask_cat = (var_types == self.VAR_TYPE_CAT) & (cat_ids >= 0)
            if mask_cat.any():
                cat_ids_clamped = cat_ids.clamp(min=0)
                e_cat = self.cat_embed(cat_ids_clamped)
                b_idx, l_idx = torch.where(mask_cat)
                e_value[b_idx, l_idx] = e_cat[b_idx, l_idx]

        return self.layernorm(e_value)

def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    keep = (~mask).unsqueeze(-1).float()  # (...,K,1)
    summed = (x * keep).sum(dim=dim)
    denom = keep.sum(dim=dim).clamp(min=1.0)
    return summed / denom

# --- LOAD DATA ---
print("Loading Data...")
script_dir = os.path.dirname(os.path.abspath(__file__))
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

# Build event table: each row = (pid, t, variate, value)
patient_events = {}
for pid, group in tqdm(df.groupby(pid_col), desc="Grouping Events"):
    group = group.sort_values(time_col)
    events = []
    for _, row in group.iterrows():
        t = row[time_col]
        for col in feature_cols:
            value = row[col]
            if value == "missing":
                continue
            events.append((t, col, value))
    patient_events[pid] = events

rows = []
for pid, events in patient_events.items():
    for (t, col, value) in events:
        rows.append({pid_col: pid, time_col: t, "variate": col, "value": value})

events_df = pd.DataFrame(rows).sort_values([pid_col, time_col]).reset_index(drop=True)

# Map variates to IDs
unique_variates = sorted(events_df["variate"].unique())
variate2id = {v: i for i, v in enumerate(unique_variates)}
num_variates = len(unique_variates)
events_df["variate_id"] = events_df["variate"].map(variate2id).astype(int)

# ---- Identify numeric vs categorical variates ----
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

# ---- GROUP EVENTS BY TIMESTEP (pid, time_col) ----
# Computes delta_t between timesteps (not between events inside a timestep).
# Then each event inside a timestep shares that timestep's delta_t_norm.
print("Grouping into timesteps...")

# Compute global max delta between timesteps for normalization
dt_all = []
for pid, g in events_df.groupby(pid_col):
    times = np.array(sorted(g[time_col].unique()))
    if len(times) <= 1:
        continue
    dt = np.diff(times).astype(np.float32)
    dt_all.append(dt)
max_dt = float(np.max(np.concatenate(dt_all))) if len(dt_all) else 1.0
if max_dt <= 0:
    max_dt = 1.0

patients = sorted(events_df[pid_col].unique())

# First pass to get Tmax and Kmax
Tmax = 0
Kmax = 0
pid_to_steps = {}

for pid, gpid in tqdm(events_df.groupby(pid_col), desc="Building step lists"):
    gpid = gpid.sort_values(time_col)
    times = list(sorted(gpid[time_col].unique()))
    Tmax = max(Tmax, len(times))

    # dt per timestep
    prev_t = None
    step_rows = []
    for t in times:
        gt = gpid[gpid[time_col] == t]
        k = len(gt)
        Kmax = max(Kmax, k)

        if prev_t is None:
            dt_norm = 0.0
        else:
            dt_norm = float((t - prev_t) / max_dt)
        prev_t = t

        step_rows.append((t, dt_norm, gt))
    pid_to_steps[pid] = step_rows

print(f"Max timesteps per patient (Tmax): {Tmax}")
print(f"Max events per timestep (Kmax): {Kmax}")

# Allocate padded tensors: (P, Tmax, Kmax)
P = len(patients)
PAD_VAR_ID = 0
PAD_CAT_ID = -1

x_var = torch.full((P, Tmax, Kmax), PAD_VAR_ID, dtype=torch.long)
x_dt  = torch.zeros((P, Tmax, Kmax), dtype=torch.float32)
x_val = torch.zeros((P, Tmax, Kmax), dtype=torch.float32)
x_cat = torch.full((P, Tmax, Kmax), PAD_CAT_ID, dtype=torch.long)

# masks
ev_pad_mask = torch.ones((P, Tmax, Kmax), dtype=torch.bool)  # True = pad
ts_pad_mask = torch.ones((P, Tmax), dtype=torch.bool)        # True = pad timestep

print("Filling padded timestep tensors...")
for p_idx, pid in enumerate(tqdm(patients, desc="Padding")):
    steps = pid_to_steps[pid]
    for t_idx, (t, dt_norm, gt) in enumerate(steps):
        ts_pad_mask[p_idx, t_idx] = False
        # Fill events
        vids = torch.tensor(gt["variate_id"].values, dtype=torch.long)
        vnum = torch.tensor(gt["value_num"].fillna(0.0).values, dtype=torch.float32)
        cids = torch.tensor(gt["cat_id"].fillna(-1).values, dtype=torch.long)

        k = len(vids)
        x_var[p_idx, t_idx, :k] = vids
        x_val[p_idx, t_idx, :k] = vnum
        x_cat[p_idx, t_idx, :k] = cids
        x_dt[p_idx, t_idx, :k]  = dt_norm
        ev_pad_mask[p_idx, t_idx, :k] = False

# ==========================================
# PART 2: DECODER-ONLY TIME-SERIES TRANSFORMER (TIMESTEP-LEVEL TEACHER FORCING)
# ==========================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(1), :].unsqueeze(0)
        return self.dropout(x)

def generate_causal_mask(T: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

class DecoderOnlyTimeSeriesTransformer(nn.Module):
    def __init__(self, d_model: int = 128, nhead: int = 4, num_layers: int = 4, dropout: float = 0.1, dim_ff: int = 512):
        super().__init__()
        self.pos = PositionalEncoding(d_model, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, d_model)

    def forward(self, z_in: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        z = self.pos(z_in)
        T_in = z.size(1)
        cmask = generate_causal_mask(T_in, z.device)
        h = self.tr(z, mask=cmask, src_key_padding_mask=key_padding_mask)
        return self.head(h)

# ---- Build per-event embeddings, then pool into timestep embeddings z_t ----
print("Generating per-event embeddings and pooling to timesteps...")

d_model = 128
var_embedder = VariateEmbedding(num_variates=num_variates, d_model=d_model)
time_embedder = TimeDeltaEmbedding(d_model=d_model, hidden_dim=64)
value_embedder = EventValueEmbedding(
    d_model=d_model,
    variate_type_tensor=variate_type,
    numeric_means=numeric_means,
    numeric_stds=numeric_stds,
    num_cat_tokens=num_cat_tokens,
)
event_layernorm = nn.LayerNorm(d_model)

with torch.no_grad():
    # Flatten (P,T,K) -> (P*T, K)
    PT = P * Tmax
    var_flat = x_var.view(PT, Kmax)
    dt_flat  = x_dt.view(PT, Kmax)
    val_flat = x_val.view(PT, Kmax)
    cat_flat = x_cat.view(PT, Kmax)

    e_var = var_embedder(var_flat)                  # (PT,K,D)
    e_time = time_embedder(dt_flat)                 # (PT,K,D)
    e_val  = value_embedder(var_flat, val_flat, cat_flat)  # (PT,K,D)
    e_event = event_layernorm(e_var + e_time + e_val)       # (PT,K,D)
    e_event = e_event.view(P, Tmax, Kmax, d_model)          # (P,T,K,D)

    # Pool events within each timestep -> z_t (P, T, D)
    z = masked_mean(e_event, ev_pad_mask, dim=2)

print("z (timestep embedding) shape:", tuple(z.shape))  # (P, Tmax, 128)

# ==========================================
# TRAINING: TIMESTEP-LEVEL TEACHER FORCING
# ==========================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {DEVICE}")

model = DecoderOnlyTimeSeriesTransformer(d_model=d_model, nhead=4, num_layers=4, dropout=0.1, dim_ff=512).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# Dataset: z and timestep padding mask
dataset = TensorDataset(z.detach(), ts_pad_mask)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

print("Starting Decoder-Only Time-Series Training (teacher forcing over timesteps)...")

for epoch in range(10):
    model.train()
    total_loss = 0.0
    steps = 0
    loss_sum_t = None
    loss_cnt_t = None

    for batch_z, batch_ts_mask in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
        batch_z = batch_z.to(DEVICE)                # (B, T, D)
        batch_ts_mask = batch_ts_mask.to(DEVICE)    # (B, T) True=PAD timestep

        B, T, D = batch_z.shape
        if T < 2:
            continue

        # initialize per-timestep accumulators once we know sequence length
        if loss_sum_t is None:
            loss_sum_t = torch.zeros(T - 1, device=DEVICE)
            loss_cnt_t = torch.zeros(T - 1, device=DEVICE)

        # Teacher forcing shift at timestep level:
        z_in = batch_z[:, :-1, :]                   # (B, T-1, D) uses steps 1..T-1
        z_tgt = batch_z[:, 1:, :]                   # (B, T-1, D) targets steps 2..T
        mask_in = batch_ts_mask[:, :-1]             # (B, T-1)
        mask_tgt = batch_ts_mask[:, 1:]             # (B, T-1)

        # Forward: ALL positions computed at the same time, causal mask inside model
        pred = model(z_in, key_padding_mask=mask_in)  # (B, T-1, D)

        # Loss only on valid target timesteps
        valid = ~mask_tgt  # (B, T-1)

        # Per-timestep MSE (computed in parallel across all positions)
        mse_t = (pred - z_tgt).pow(2).mean(dim=-1)  # (B, T-1)
        # accumulate sums/counts per timestep (ignore padded target timesteps)
        v = valid.float()
        loss_sum_t += (mse_t * v).sum(dim=0)
        loss_cnt_t += v.sum(dim=0)

        if valid.any():
            loss = criterion(pred[valid], z_tgt[valid])
        else:
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.item())
        steps += 1

    avg = total_loss / max(1, steps)
    print(f"Epoch {epoch}: Avg MSE Loss {avg:.6f}")
    if loss_sum_t is not None and loss_cnt_t is not None:
        per_t = (loss_sum_t / (loss_cnt_t + 1e-9)).detach().cpu().numpy()
        print("  Per-timestep MSE (pos0 predicts t=2, pos1 predicts t=3, ...):")
        print("  ", np.round(per_t, 6).tolist())

print("Training Complete!")
torch.save(model.state_dict(), "decoder_only_timestep_tf.pt")
