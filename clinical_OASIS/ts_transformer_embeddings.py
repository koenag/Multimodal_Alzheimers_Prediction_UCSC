#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-series Transformer for clinical embeddings.

Inputs per patient (from patients.pt built earlier):
  - X_timestep: [T, D]  (required)  — clinical embedding per visit
  - delta_t:   [T]      (optional)  — days since previous visit
  - t:         [T]      (optional)  — absolute days_to_visit

This script provides:
  - ClinicalSeqDataset: loads variable-length sequences from patients.pt
  - collate_pad: pads to max T in batch and builds masks
  - TimeSeriesTransformer: TransformerEncoder over time with Δt embedding
  - Classification head for downstream tasks (num_classes >= 2)

Usage example is at the bottom.
"""

import argparse
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Dataset & collate (variable T)
# -----------------------------

class ClinicalSeqDataset(Dataset):
    """
    Expects a dict saved by your previous step:
      patients: Dict[str, Dict[str, Tensor]]
        {
          "X_timestep": [T, D] (float32),
          "delta_t":    [T]    (float32)   # optional but recommended
          "t":          [T]    (float32)   # optional
          # Optionally, you can add "label" (int) for supervised training.
        }
    """
    def __init__(self, patients_path: str, label_map: Optional[Dict[str, int]] = None):
        super().__init__()
        payload: Dict[str, Dict[str, torch.Tensor]] = torch.load(patients_path, map_location="cpu")
        self.keys = list(payload.keys())
        self.items = payload
        self.label_map = label_map or {}  # patient_id -> int label (optional)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        pid = self.keys[idx]
        item = self.items[pid]

        X = item["X_timestep"].float()  # [T, D]
        delta_t = item.get("delta_t", torch.zeros(X.shape[0])).float()  # [T]
        t_abs = item.get("t", torch.arange(X.shape[0], dtype=torch.float32))  # [T]
        label = self.label_map.get(pid, -1)  # -1 if unknown / unsupervised

        return {
            "pid": pid,
            "X": X,                # [T, D]
            "delta_t": delta_t,    # [T]
            "t_abs": t_abs,        # [T]
            "label": torch.tensor(label, dtype=torch.long)
        }


def collate_pad(batch: List[Dict]):
    """
    Pads variable-length sequences to max_T in the batch.
    Returns:
      X_pad: [B, T_max, D]
      dt_pad: [B, T_max]
      t_pad: [B, T_max]
      key_padding_mask: [B, T_max] (True = PAD/MASK)
      labels: [B] (or None)
      pids: list[str]
      lengths: [B] (original T per item)
    """
    B = len(batch)
    lengths = [b["X"].shape[0] for b in batch]
    T_max = max(lengths)
    D = batch[0]["X"].shape[1]

    X_pad = torch.zeros(B, T_max, D, dtype=torch.float32)
    dt_pad = torch.zeros(B, T_max, dtype=torch.float32)
    t_pad  = torch.zeros(B, T_max, dtype=torch.float32)
    key_padding_mask = torch.ones(B, T_max, dtype=torch.bool)  # start masked

    labels = torch.stack([b["label"] for b in batch], dim=0)

    for i, b in enumerate(batch):
        T_i = b["X"].shape[0]
        X_pad[i, :T_i] = b["X"]
        dt_pad[i, :T_i] = b["delta_t"]
        t_pad[i, :T_i] = b["t_abs"]
        key_padding_mask[i, :T_i] = False  # unmask actual tokens

    pids = [b["pid"] for b in batch]

    return {
        "X": X_pad,                         # [B, T, D]
        "delta_t": dt_pad,                  # [B, T]
        "t_abs": t_pad,                     # [B, T]
        "key_padding_mask": key_padding_mask,  # [B, T] (True = pad)
        "labels": labels,                   # [B]
        "pids": pids,
        "lengths": torch.tensor(lengths)
    }


# -----------------------------
# Time encodings
# -----------------------------

class Time2Vec(nn.Module):
    """
    Simple time2vec-style embedding for Δt or absolute time.
    v(t) = [w0 * t + b0, sin(w1 * t + b1), ..., sin(wk * t + bk)]
    """
    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim
        self.w = nn.Parameter(torch.randn(out_dim))
        self.b = nn.Parameter(torch.zeros(out_dim))

    def forward(self, t: torch.Tensor):
        # t: [B, T] (or [T])
        x = t.unsqueeze(-1)  # [..., 1]
        # linear term in first channel, sin for others
        lin = self.w[0] * x + self.b[0]
        if self.out_dim == 1:
            return lin.squeeze(-1)
        sin_part = torch.sin(self.w[1:] * x + self.b[1:])  # [..., out_dim-1]
        out = torch.cat([lin, sin_part], dim=-1)
        return out


class DeltaTEmbedding(nn.Module):
    """
    Embed Δt (and optionally absolute t) then project to model_dim and add to X.
    """
    def __init__(self, model_dim: int, use_abs_time: bool = False, time2vec_dim: int = 8):
        super().__init__()
        self.use_abs_time = use_abs_time
        self.t2v_dt = Time2Vec(time2vec_dim)
        self.t2v_abs = Time2Vec(time2vec_dim) if use_abs_time else None
        in_dim = time2vec_dim * (2 if use_abs_time else 1)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, delta_t: torch.Tensor, t_abs: Optional[torch.Tensor] = None):
        # delta_t, t_abs: [B, T]
        dt_e = self.t2v_dt(torch.log1p(torch.clamp(delta_t, min=0)))  # log1p stabilizes large gaps
        if self.use_abs_time and t_abs is not None:
            ta_e = self.t2v_abs(torch.log1p(torch.clamp(t_abs - t_abs.min(dim=1, keepdim=True).values, min=0)))
            e = torch.cat([dt_e, ta_e], dim=-1)  # [B, T, 2*time2vec_dim]
        else:
            e = dt_e  # [B, T, time2vec_dim]
        return self.proj(e)  # [B, T, model_dim]


# --------------------------------
# Time-series Transformer backbone
# --------------------------------

class TimeSeriesTransformer(nn.Module):
    """
    Transformer encoder over time with:
      - optional CLS token
      - Δt embedding added to input features
      - masked pooling
      - classification head

    Inputs:
      X: [B, T, D]           (clinical embeddings per visit)
      delta_t: [B, T]        (optional; can pass zeros)
      t_abs: [B, T]          (optional absolute time)
      key_padding_mask: [B, T]  True = pad  (as in nn.TransformerEncoder)
    """
    def __init__(
        self,
        input_dim: int,
        model_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
        add_cls: bool = True,
        use_abs_time: bool = False,
        time2vec_dim: int = 8,
        num_classes: int = 2,           # set to 0 if you don't want a head
        pooling: str = "cls",           # "cls" or "mean"
    ):
        super().__init__()
        self.add_cls = add_cls
        self.pooling = pooling
        self.model_dim = model_dim

        # Project input embeddings to model_dim if needed
        self.in_proj = nn.Linear(input_dim, model_dim) if input_dim != model_dim else nn.Identity()

        # Δt / time embeddings
        self.t_proj = DeltaTEmbedding(model_dim, use_abs_time=use_abs_time, time2vec_dim=time2vec_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim)) if add_cls else None

        # Positional encoding (simple learnable absolute)
        self.pos_emb = nn.Parameter(torch.zeros(1, 4096, model_dim))  # supports up to 4096 timesteps

        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,  # [B, T, D]
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(model_dim)

        # Classification head
        self.num_classes = num_classes
        self.head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_classes),
        ) if num_classes and num_classes > 0 else None

        self._reset_parameters()

    def _reset_parameters(self):
        if isinstance(self.in_proj, nn.Linear):
            nn.init.xavier_uniform_(self.in_proj.weight)
            nn.init.zeros_(self.in_proj.bias)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.01)

    def forward(
        self,
        X: torch.Tensor,                    # [B, T, D_in]
        delta_t: Optional[torch.Tensor],    # [B, T]
        t_abs: Optional[torch.Tensor],      # [B, T]
        key_padding_mask: torch.Tensor      # [B, T], True = pad
    ) -> Dict[str, torch.Tensor]:
        B, T, _ = X.shape

        x = self.in_proj(X)                                  # [B, T, model_dim]
        tenc = self.t_proj(delta_t if delta_t is not None else torch.zeros(B, T, device=X.device),
                           t_abs=t_abs)                       # [B, T, model_dim]
        x = x + tenc

        # Add CLS at front if requested
        if self.add_cls:
            cls_tok = self.cls_token.expand(B, -1, -1)       # [B, 1, model_dim]
            x = torch.cat([cls_tok, x], dim=1)               # [B, T+1, model_dim]

            # Build masks for the added CLS token
            # CLS is never padded
            cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            key_padding_mask = torch.cat([cls_pad, key_padding_mask], dim=1)  # [B, T+1]

            # Positional embedding (0..T for CLS..)
            pos = self.pos_emb[:, : T + 1, :]                 # [1, T+1, D]
        else:
            pos = self.pos_emb[:, : T, :]                     # [1, T, D]

        x = x + pos

        # Transformer expects key_padding_mask: True at PAD positions
        enc = self.encoder(x, src_key_padding_mask=key_padding_mask)  # [B, T' , D]
        enc = self.norm(enc)

        if self.pooling == "cls" and self.add_cls:
            pooled = enc[:, 0, :]                             # [B, D]
        else:
            # masked mean over time tokens (exclude CLS if present)
            if self.add_cls:
                enc_time = enc[:, 1:, :]                      # [B, T, D]
                mask_time = ~key_padding_mask[:, 1:]          # [B, T]  (True=keep)
            else:
                enc_time = enc
                mask_time = ~key_padding_mask                 # True=keep

            denom = mask_time.sum(dim=1).clamp_min(1).unsqueeze(-1)  # [B, 1]
            pooled = (enc_time * mask_time.unsqueeze(-1)).sum(dim=1) / denom

        out = {"sequence": enc, "pooled": pooled}
        if self.head is not None:
            out["logits"] = self.head(pooled)                 # [B, C]
        return out


# -----------------------------
# Simple train/eval helpers
# -----------------------------

def step_batch(model, batch, device, criterion=None):
    X = batch["X"].to(device)
    dt = batch["delta_t"].to(device)
    ta = batch["t_abs"].to(device)
    kpm = batch["key_padding_mask"].to(device)
    labels = batch["labels"].to(device)

    out = model(X, delta_t=dt, t_abs=ta, key_padding_mask=kpm)
    logits = out.get("logits", None)

    loss = None
    if logits is not None and (labels >= 0).any():
        # Only compute loss on items with valid labels (>=0)
        mask = (labels >= 0)
        if mask.any():
            loss = criterion(logits[mask], labels[mask])
    return out, loss


# -----------------------------
# CLI usage example (toy)
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients_pt", required=True, help="Path to patients.pt from the clinical encoder step")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--model_dim", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--ff_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--pooling", type=str, default="cls", choices=["cls","mean"])
    ap.add_argument("--use_abs_time", action="store_true")
    args = ap.parse_args()

    # Load dataset; if you have labels, build a {patient_id: int} mapping and pass it in
    dataset = ClinicalSeqDataset(args.patients_pt, label_map=None)
    # Peek input_dim from first item:
    d0 = dataset[0]["X"].shape[1]

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pad)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesTransformer(
        input_dim=d0,
        model_dim=args.model_dim,
        n_heads=args.heads,
        n_layers=args.layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        add_cls=True,
        use_abs_time=args.use_abs_time,
        num_classes=args.num_classes,
        pooling=args.pooling,
    ).to(device)

    # Loss/opt (only meaningful if labels are provided)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total, n_batches = 0.0, 0
        for batch in loader:
            optim.zero_grad()
            _, loss = step_batch(model, batch, device, criterion)
            if loss is not None:
                loss.backward()
                optim.step()
                total += float(loss.item())
                n_batches += 1
        if n_batches > 0:
            print(f"Epoch {epoch+1}: train loss {total / n_batches:.4f}")
        else:
            print(f"Epoch {epoch+1}: (no labels, skipped loss)")

    # Example: extract pooled embeddings for downstream tasks
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        out, _ = step_batch(model, batch, device, criterion=None)
        pooled = out["pooled"]    # [B, model_dim]
        seq = out["sequence"]     # [B, T(+1), model_dim] (includes CLS at index 0 if add_cls=True)
        print("Pooled embedding shape:", pooled.shape)
        if "logits" in out:
            print("Logits shape:", out["logits"].shape)


if __name__ == "__main__":
    main()
