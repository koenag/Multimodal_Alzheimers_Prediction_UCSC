import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, Optional, Callable
from tqdm import tqdm

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=256, embedding_dim=256, beta=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z_e):
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

class SparseDataset(Dataset):
    def __init__(self, sparse_matrix):
        self.sparse_matrix = sparse_matrix.tocsr()

    def __len__(self):
        return self.sparse_matrix.shape[0]

    def __getitem__(self, idx):
        row = self.sparse_matrix[idx].toarray().ravel().astype(np.float32)
        return torch.from_numpy(row)
    
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

        self.register_buffer("variate_type", variate_type_tensor.clone())
        self.register_buffer("numeric_means", numeric_means.clone())
        self.register_buffer("numeric_stds",  numeric_stds.clone())

        self.numeric_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, d_model),
        )

        if num_cat_tokens > 0:
            self.cat_embed = nn.Embedding(num_cat_tokens, d_model)
        else:
            self.cat_embed = None
        self.text_encoder = text_encoder  # should map a batch of text -> (B, T, d_model) or similar

        self.VAR_TYPE_NUM = VAR_TYPE_NUM
        self.VAR_TYPE_CAT = VAR_TYPE_CAT
        self.VAR_TYPE_TEXT = VAR_TYPE_TEXT

    def forward(self, variate_ids, value_num, cat_ids, texts=None):
        B, T = variate_ids.shape
        device = variate_ids.device
        e_value = torch.zeros(B, T, self.d_model, device=device)
        var_types = self.variate_type[variate_ids]  # (B, T)

        mask_num = (var_types == self.VAR_TYPE_NUM)

        if mask_num.any():
            mu = self.numeric_means[variate_ids]     # (B, T)
            sigma = self.numeric_stds[variate_ids]   # (B, T)
            v = (value_num - mu) / sigma
            v = v.unsqueeze(-1)                      # (B, T, 1)
            e_num_full = self.numeric_mlp(v)         # (B, T, d_model)
            e_value[mask_num] = e_num_full[mask_num]
            
        if self.cat_embed is not None:
            mask_cat = (var_types == self.VAR_TYPE_CAT) & (cat_ids >= 0)
            if mask_cat.any():
                cat_ids_clamped = cat_ids.clone()
                cat_ids_clamped[cat_ids_clamped < 0] = 0  # avoid -1 indexing
                e_cat_full = self.cat_embed(cat_ids_clamped)  # (B, T, d_model)
                e_value[mask_cat] = e_cat_full[mask_cat]

        return e_value
    
class EventValueEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_variates: int,
        variate_type_tensor: torch.Tensor,
        numeric_means: torch.Tensor,
        numeric_stds: torch.Tensor,
        num_cat_tokens: int,
        text_encoder: Optional[Callable[[list], torch.Tensor]] = None,
        numeric_mode: str = "mlp",   # "mlp" or "quantize"
        num_quantize_bins: int = 100,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.register_buffer("variate_type", variate_type_tensor.clone().long())
        self.register_buffer("numeric_means", numeric_means.clone().float())
        self.register_buffer("numeric_stds", numeric_stds.clone().float())
        self.VAR_TYPE_NUM = VAR_TYPE_NUM
        self.VAR_TYPE_CAT = VAR_TYPE_CAT
        self.VAR_TYPE_TEXT = VAR_TYPE_TEXT

        # numeric
        self.numeric_mode = numeric_mode
        if numeric_mode == "mlp":
            self.numeric_mlp = nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Linear(64, d_model)
            )
        elif numeric_mode == "quantize":
            self.num_quantize_bins = num_quantize_bins
            self.quantize_embed = nn.Embedding(num_quantize_bins, d_model)
        else:
            raise ValueError("numeric_mode must be 'mlp' or 'quantize'")

        # categorical embedding (case B)
        self.cat_embed = nn.Embedding(num_cat_tokens if num_cat_tokens>0 else 1, d_model) if num_cat_tokens>0 else None
        if self.cat_embed is not None:
            nn.init.normal_(self.cat_embed.weight, mean=0.0, std=0.02)
            
        self.text_encoder = text_encoder
        self.text_project = None
        if text_encoder is not None:
            self._text_project_created = False
            
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, variate_ids: torch.LongTensor, value_num: torch.FloatTensor, cat_ids: torch.LongTensor, text_list: Optional[list] = None, numeric_bin_ids: Optional[torch.LongTensor] = None):
        B, T = variate_ids.shape
        device = variate_ids.device
        e_value = torch.zeros(B, T, self.d_model, device=device, dtype=torch.float32)

        # get types (B, T)
        var_types = self.variate_type[variate_ids]  # broadcasting indexing -> (B, T)

        # ---- numeric MLP ----
        mask_num = (var_types == self.VAR_TYPE_NUM)
        if mask_num.any():
            if self.numeric_mode == "mlp":
                # gather means and stds, normalize
                mu = self.numeric_means[variate_ids]    # (B, T)
                sigma = self.numeric_stds[variate_ids]  # (B, T)
                v = (value_num - mu) / (sigma + 1e-12)
                v_in = v.unsqueeze(-1)  # (B, T, 1)
                e_num_full = self.numeric_mlp(v_in)  # (B, T, d_model)

                # masked assign safely:
                m = mask_num.unsqueeze(-1).expand(-1, -1, self.d_model)  # (B,T,d_model)
                e_value = e_value.masked_scatter(m, e_num_full[m])
            else:
                # expect numeric_bin_ids provided
                if numeric_bin_ids is None:
                    raise ValueError("numeric_bin_ids must be provided when numeric_mode == 'quantize'")
                # clamp to valid range and gather embedding
                b = numeric_bin_ids.clone()
                b[b < 0] = 0
                e_num_full = self.quantize_embed(b)  # (B,T,d_model)
                m = mask_num.unsqueeze(-1).expand(-1, -1, self.d_model)
                e_value = e_value.masked_scatter(m, e_num_full[m])

        if self.cat_embed is not None:
            mask_cat = (var_types == self.VAR_TYPE_CAT) & (cat_ids >= 0)
            if mask_cat.any():
                cat_ids_clamped = cat_ids.clone()
                cat_ids_clamped[cat_ids_clamped < 0] = 0
                e_cat_full = self.cat_embed(cat_ids_clamped)  # (B,T,d_model)
                m = mask_cat.unsqueeze(-1).expand(-1, -1, self.d_model)
                e_value = e_value.masked_scatter(m, e_cat_full[m])

        if (self.text_encoder is not None) and (text_list is not None):
            mask_text = (var_types == self.VAR_TYPE_TEXT)
            if mask_text.any():
                flat_texts = []
                positions = []  # tuples (b,t)
                for b in range(B):
                    for t in range(T):
                        if mask_text[b, t]:
                            txt = text_list[b][t]
                            if txt is None:
                                txt = ""
                            flat_texts.append(txt)
                            positions.append((b, t))
                enc = self.text_encoder(flat_texts)  # Tensor (M, enc_dim)
                if not self._text_project_created:
                    enc_dim = enc.shape[-1]
                    self.text_project = nn.Linear(enc_dim, self.d_model).to(device)
                    self._text_project_created = True
                e_txt_flat = self.text_project(enc)  # (M, d_model)
                for i, (b, t) in enumerate(positions):
                    e_value[b, t, :] = e_txt_flat[i]

        return self.layernorm(e_value)
    
class TextWithEventsDataset(Dataset):
    def __init__(self, items, tokenizer, max_len=512):
        self.items = items
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        return it
    
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
        else:
            raise ValueError("delta_t must be 1D or 2D")
        return self.mlp(delta_t)

df = pd.read_csv("oasis3_clinical.csv", low_memory=False)

pid_col = "patient_id"
if pid_col not in df.columns:
    raise ValueError("Expected a 'patient_id' column in the CSV.")

if "days_to_visit" in df.columns:
    time_col = "days_to_visit"
else:
    df["__t__"] = df.groupby(pid_col).cumcount()
    time_col = "__t__"

df = df.dropna(subset=[pid_col])

df = df.copy()
df = df.fillna("missing")

feature_cols = [c for c in df.columns if c not in [pid_col, time_col]]

drop_from_features = ["OASIS_session_label", "session_id"]
feature_cols = [c for c in feature_cols if c not in drop_from_features]

df = df.copy()
df = df.fillna("missing")

feature_cols = [c for c in df.columns if c not in [pid_col, time_col]]

patient_events = {}

for pid, group in tqdm(df.groupby(pid_col)):
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
        rows.append({
            pid_col: pid,
            time_col: t,
            "variate": col,
            "value": value
        })

events_df = pd.DataFrame(rows)
events_df = events_df.sort_values([pid_col, time_col]).reset_index(drop=True)
unique_variates = sorted(events_df["variate"].unique())
variate2id = {v: i for i, v in enumerate(unique_variates)}
id2variate = {i: v for v, i in variate2id.items()}
num_variates = len(unique_variates)
events_df["variate_id"] = events_df["variate"].map(variate2id).astype(int)
events_df["delta_t"] = events_df.groupby(pid_col)[time_col].diff()
events_df["delta_t"] = events_df["delta_t"].fillna(0.0)
max_dt = events_df["delta_t"].max()
if max_dt == 0 or pd.isna(max_dt):
    events_df["delta_t_norm"] = 0.0
else:
    events_df["delta_t_norm"] = events_df["delta_t"] / max_dt

numeric_variates = []
categorical_variates = []
text_variates = []

for col in feature_cols:
    col_series = df[col].replace("missing", np.nan)
    col_num = pd.to_numeric(col_series, errors="coerce")
    frac_numeric = col_num.notna().mean()

    if frac_numeric > 0.9:
        numeric_variates.append(col)
    else:
        categorical_variates.append(col)

events_df["value_num"] = pd.to_numeric(
    events_df["value"].replace("missing", np.nan), errors="coerce"
)

num_stats = (
    events_df[events_df["variate"].isin(numeric_variates)]
    .groupby("variate")["value_num"]
    .agg(["mean", "std"])
    .reset_index()
)

num_stats["std"] = num_stats["std"].replace(0, 1.0).fillna(1.0)
num_stats["mean"] = num_stats["mean"].fillna(0.0)

VAR_TYPE_NUM = 0
VAR_TYPE_CAT = 1
VAR_TYPE_TEXT = 2

variate_type = torch.full((num_variates,), VAR_TYPE_CAT, dtype=torch.long)
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

seqs_variate   = []
seqs_deltat    = []
seqs_value_num = []
seqs_cat_id    = []

for pid, group in events_df.groupby(pid_col):
    group = group.sort_values(time_col)

    var_ids = torch.tensor(group["variate_id"].values, dtype=torch.long)
    dt_norm = torch.tensor(group["delta_t_norm"].values, dtype=torch.float32)
    v_num = torch.tensor(
        group["value_num"].fillna(0.0).values, dtype=torch.float32
    )
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

d_model = 128

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

#BATCH_PATIENTS = 64

with torch.no_grad():
    v_ids   = padded_variate_ids#[:BATCH_PATIENTS]
    dt_norm = padded_delta_t#[:BATCH_PATIENTS]
    v_num   = padded_value_num#[:BATCH_PATIENTS]
    v_cat   = padded_cat_id#[:BATCH_PATIENTS]

    e_variate = var_embedder(v_ids)
    e_time    = time_embedder(dt_norm)
    e_value   = value_embedder(v_ids, v_num, v_cat)

    e_event = event_layernorm(e_variate + e_time + e_value)

print("e_event shape:", e_event.shape)

# ---- build one item per patient from e_event ----
seq_lengths = [len(s) for s in seqs_variate]   # list of length 1378

items = []
num_patients = e_event.shape[0]

for i in range(num_patients):
    L_i = seq_lengths[i]          # number of real events for patient i
    ev_i = e_event[i, :L_i].clone()   # (L_i, d_model=128)

    items.append({
        "input_text": "Patient history: ",       # or any prompt you like
        "target_text": "Likely diagnosis: ...",  # currently unused
        "event_embeddings": ev_i,
    })

print("Number of items built from e_event:", len(items))


def compute_variate_quantile_bins(events_df, numeric_variates, num_bins=100, q_min=0.001, q_max=0.999):
    bins = {}
    for v in numeric_variates:
        vals = pd.to_numeric(events_df.loc[events_df['variate']==v, 'value'].replace('missing', np.nan), errors='coerce').dropna().astype(float).values
        if len(vals) == 0:
            edges = np.linspace(0., 1., num_bins+1)[1:-1] 
            bins[v] = edges
            continue
        q = np.linspace(q_min, q_max, num_bins-1)
        edges = np.quantile(vals, q)
        bins[v] = edges.astype(float)
    return bins

def discretize_value_into_bins(value_array, variate_array, variate2bin_edges, variate2id):
    out = np.full(len(value_array), -1, dtype=np.int64)
    for i, (vname, val) in enumerate(zip(variate_array, value_array)):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            out[i] = -1
            continue
        edges = variate2bin_edges.get(vname, None)
        if edges is None or len(edges) == 0:
            out[i] = -1
            continue
        out[i] = np.digitize(val, edges, right=False)
    return out
    
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

HF_MODEL = "amusktweewt/tiny-model-500M-chat-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 4
LR = 1e-4
EPOCHS = 50

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(HF_MODEL).to(DEVICE)
embed_dim = model.get_input_embeddings().embedding_dim
print("model embed dim:", embed_dim)

d_model = 128
projector = nn.Linear(d_model, embed_dim).to(DEVICE)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

def collate_fn(batch):
    prompts = [b["input_text"] for b in batch]

    # Tokenize prompts once
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)

    input_ids = encoded.input_ids              # (B, T_txt)
    attention_mask = encoded.attention_mask    # (B, T_txt)

    # Labels must be same shape as input_ids for CausalLM
    labels = input_ids.clone()                 # (B, T_txt)

    # ----- event embeddings (same as before) -----
    evs = [b["event_embeddings"] for b in batch]
    T_ev = max(e.shape[0] for e in evs)
    ev_padded = []
    ev_mask = []

    for e in evs:
        T_i, D = e.shape
        pad_len = T_ev - T_i
        if pad_len > 0:
            pad_block = torch.zeros((pad_len, D), dtype=e.dtype)
            e_p = torch.cat([e, pad_block], dim=0)
            mask = torch.cat([torch.ones(T_i), torch.zeros(pad_len)])
        else:
            e_p = e
            mask = torch.ones(T_i)
        ev_padded.append(e_p)
        ev_mask.append(mask)

    ev_padded = torch.stack(ev_padded, dim=0)  # (B, T_ev, d_model)
    ev_mask   = torch.stack(ev_mask,   dim=0)  # (B, T_ev)

    batch_data = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "event_embeddings": ev_padded,
        "event_mask": ev_mask,
    }

    # 🔥 no patient_id here anymore

    return batch_data




optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

def train_epoch(dataloader):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(DEVICE)
        attn = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        evs = batch["event_embeddings"].to(DEVICE)
        ev_mask = batch["event_mask"].to(DEVICE)
        B, T_ev, _ = evs.shape
        evs_proj = projector(evs.view(-1, d_model)).view(B, T_ev, embed_dim)
        token_embeds = model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([evs_proj, token_embeds], dim=1)
        ev_prefix_mask = ev_mask.long()
        extended_attn = torch.cat([ev_prefix_mask, attn], dim=1)
        pad_prefix = torch.full((B, T_ev), -100, dtype=torch.long, device=labels.device)
        labels_padded = torch.cat([pad_prefix, labels], dim=1)
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attn,
            labels=labels_padded,
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item() * input_ids.size(0)
    return total_loss / len(dataloader.dataset)

import pickle

@torch.no_grad()
def extract_event_prefix_embeddings(dataloader, model, projector, device, d_model, embed_dim, pool="mean"):
    model.eval()
    projector.eval()

    all_embs = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attn      = batch["attention_mask"].to(device)
        evs       = batch["event_embeddings"].to(device)
        ev_mask   = batch["event_mask"].to(device)

        B, T_ev, _ = evs.shape

        evs_proj = projector(evs.view(-1, d_model)).view(B, T_ev, embed_dim)
        token_embeds = model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([evs_proj, token_embeds], dim=1)

        ev_prefix_mask = ev_mask.long()
        extended_attn = torch.cat([ev_prefix_mask, attn], dim=1)

        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attn,
            output_hidden_states=True,
        )

        last_hidden = outputs.hidden_states[-1]  # (B, T_ev+T_txt, hidden_size)
        ev_hidden = last_hidden[:, :T_ev, :]     # (B, T_ev, hidden_size)

        mask = ev_mask.unsqueeze(-1)
        ev_hidden_masked = ev_hidden * mask

        lengths = ev_mask.sum(dim=1, keepdim=True).clamp(min=1)

        if pool == "mean":
            pooled = ev_hidden_masked.sum(dim=1) / lengths
        elif pool == "max":
            neg_inf = torch.finfo(ev_hidden_masked.dtype).min
            ev_hidden_masked_masked = ev_hidden.masked_fill(mask == 0, neg_inf)
            pooled, _ = ev_hidden_masked_masked.max(dim=1)
        else:
            raise ValueError("pool must be 'mean' or 'max'")

        all_embs.append(pooled.cpu())

    all_embs = torch.cat(all_embs, dim=0)   # (N, hidden_size)
    return all_embs




# use real items built from e_event
ds = TextWithEventsDataset(items, tokenizer)
dl = DataLoader(ds, batch_size=BATCH, collate_fn=collate_fn)


for epoch in range(EPOCHS):
    loss = train_epoch(dl)
    print("epoch", epoch, "loss", loss)

model.save_pretrained("./peft_lora_multimodal")
tokenizer.save_pretrained("./peft_lora_multimodal")

# optional: save projector
torch.save(projector.state_dict(), "projector.pt")

# extract embeddings (no ids now)
embs = extract_event_prefix_embeddings(
    dl,
    model=model,
    projector=projector,
    device=DEVICE,
    d_model=d_model,
    embed_dim=embed_dim,
    pool="mean",
)

print("Extracted embeddings:", embs.shape)  # (N, hidden_size)

# save as torch tensor
torch.save(embs, "event_prefix_embeddings.pt")

# or also as pickle with numpy
import numpy as np
import pickle

embs_np = embs.numpy()
with open("event_prefix_embeddings.pkl", "wb") as f:
    pickle.dump({"embeddings": embs_np}, f)

print("Saved event embeddings to event_prefix_embeddings.pt / .pkl")
