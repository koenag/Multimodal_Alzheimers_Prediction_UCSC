#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build per-timestep clinical embeddings from a wide, messy longitudinal table.

Usage:
  python build_clinical_embeddings.py \
      --csv path/to/clinical.csv \
      --out_dir out/ \
      --id_col patient_id \
      --time_col days_to_visit \
      --session_col OASIS_session_label \
      --model_dim 256 \
      --min_cat_freq 5

Outputs (in --out_dir):
  - clinical_vocab.pt : learned vocabularies + scalers + feature spec
  - patients.pt       : dict[patient_id] with:
        {
          "X_timestep": [T, D],
          "feature_tokens": [T, F, d_f] (if --emit_feature_tokens),
          "delta_t": [T],
          "t": [T],                         # absolute days_to_visit
          "visit_mask": [T],                # 1 where a visit row exists
          "session_label": [T],             # encoded index per timestep
        }
"""

import argparse
import re
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Utilities
# ---------------------------

def is_int_like(series: pd.Series) -> bool:
    if pd.api.types.is_integer_dtype(series.dropna()):
        return True
    # floats that are whole numbers
    s = series.dropna()
    if s.empty:
        return False
    return np.all(np.mod(s.astype(float), 1.0) == 0.0)

def safe_log1p(x: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.clamp(x, min=0.0))

def infer_binary(series: pd.Series) -> bool:
    s = series.dropna().unique()
    if len(s) == 0:
        return False
    try:
        s = set(pd.to_numeric(pd.Series(list(s)), errors="coerce").dropna().astype(int).tolist())
    except Exception:
        return False
    return s.issubset({0, 1})

def infer_continuous(series: pd.Series, max_unique_for_cat: int = 25) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        nunique = series.nunique(dropna=True)
        return nunique > max_unique_for_cat
    return False

def infer_categorical(series: pd.Series, max_unique_for_cat: int = 25) -> bool:
    nunique = series.nunique(dropna=True)
    # treat low-card numeric as categorical too
    if nunique <= max_unique_for_cat:
        return True
    if pd.api.types.is_object_dtype(series):
        return True
    return False


# ---------------------------
# Feature spec
# ---------------------------

@dataclass
class FeatureSpec:
    name: str
    ftype: str  # "continuous" | "binary" | "categorical"
    mean: Optional[float] = None
    std: Optional[float] = None
    cat2idx: Optional[Dict[str, int]] = None  # for categorical
    dim: int = 32  # per-feature output dim before pooling
    # For multi-hot groups, we'll have separate structures.

@dataclass
class MultiHotSpec:
    name: str                     # e.g., "drug_text" or "meds_flags"
    columns: List[str]            # columns that form the bag per timestep
    bag_kind: str                 # "string_bag" | "flag_bag"
    vocab: Optional[Dict[str,int]] = None  # for string_bag only
    dim: int = 32


# ---------------------------
# Encoders
# ---------------------------

class ContinuousFeatEncoder(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        # Input: [z, miss_mask, log_dt] -> out_dim
        self.mlp = nn.Sequential(
            nn.Linear(3, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, z: torch.Tensor, miss: torch.Tensor, log_dt: torch.Tensor):
        x = torch.stack([z, miss, log_dt], dim=-1)
        return self.mlp(x)

class BinaryFeatEncoder(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        # Input: [value(0/1), miss_mask, log_dt] -> out_dim
        self.mlp = nn.Sequential(
            nn.Linear(3, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, v01: torch.Tensor, miss: torch.Tensor, log_dt: torch.Tensor):
        x = torch.stack([v01, miss, log_dt], dim=-1)
        return self.mlp(x)

class CategoricalFeatEncoder(nn.Module):
    def __init__(self, num_embeddings: int, out_dim: int):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, out_dim)

    def forward(self, idx: torch.Tensor):
        # idx: [T]
        return self.emb(idx)

class StringBagEncoder(nn.Module):
    """Bag of embeddings for text columns like drug1..drugK that contain names or tokens."""
    def __init__(self, num_embeddings: int, out_dim: int):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, out_dim)

    def forward(self, idx_list_per_row: List[List[int]]) -> torch.Tensor:
        # returns [T, out_dim] by summing embeddings in each row
        out = []
        for idxs in idx_list_per_row:
            if len(idxs) == 0:
                out.append(torch.zeros(self.emb.embedding_dim))
            else:
                ids = torch.tensor(idxs, dtype=torch.long)
                out.append(self.emb(ids).mean(dim=0))
        return torch.stack(out, dim=0)

class FlagBagEncoder(nn.Module):
    """Bag of embeddings for flag columns like meds_1..meds_K (0/1)."""
    def __init__(self, num_items: int, out_dim: int):
        super().__init__()
        self.emb = nn.Embedding(num_items, out_dim)

    def forward(self, flags_matrix: torch.Tensor) -> torch.Tensor:
        # flags_matrix: [T, K] with 0/1
        T, K = flags_matrix.shape
        out = []
        for t in range(T):
            active = (flags_matrix[t] > 0.5).nonzero(as_tuple=False).view(-1)
            if active.numel() == 0:
                out.append(torch.zeros(self.emb.embedding_dim))
            else:
                out.append(self.emb(active).mean(dim=0))
        return torch.stack(out, dim=0)


# ---------------------------
# Main model wrapper
# ---------------------------

class ClinicalEncoder(nn.Module):
    def __init__(
        self,
        feature_specs: List[FeatureSpec],
        cat_vocab_sizes: Dict[str, int],
        multihot_specs: List[MultiHotSpec],
        model_dim: int = 256,
        emit_feature_tokens: bool = False,
    ):
        super().__init__()
        self.feature_specs = feature_specs
        self.multihot_specs = multihot_specs
        self.model_dim = model_dim
        self.emit_feature_tokens = emit_feature_tokens

        # Per-feature encoders
        self.encoders = nn.ModuleDict()
        for spec in feature_specs:
            if spec.ftype == "continuous":
                self.encoders[spec.name] = ContinuousFeatEncoder(spec.dim)
            elif spec.ftype == "binary":
                self.encoders[spec.name] = BinaryFeatEncoder(spec.dim)
            elif spec.ftype == "categorical":
                n = cat_vocab_sizes[spec.name]
                self.encoders[spec.name] = CategoricalFeatEncoder(n, spec.dim)
            else:
                raise ValueError(f"Unknown ftype {spec.ftype}")

        # Multihot encoders
        self.multi_encoders = nn.ModuleDict()
        for mh in multihot_specs:
            if mh.bag_kind == "string_bag":
                assert mh.vocab is not None
                self.multi_encoders[mh.name] = StringBagEncoder(len(mh.vocab), mh.dim)
            elif mh.bag_kind == "flag_bag":
                self.multi_encoders[mh.name] = FlagBagEncoder(len(mh.columns), mh.dim)
            else:
                raise ValueError(f"Unknown bag_kind {mh.bag_kind}")

        # Project all per-feature outputs to a shared dim and pool
        total_feat_dims = sum(s.dim for s in feature_specs) + sum(mh.dim for mh in multihot_specs)
        self.project = nn.Sequential(
            nn.Linear(total_feat_dims, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, batch_row: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        batch_row contains tensors/lists for a single patient, across T timesteps:
          - For each continuous feature f: tensors "f/value" (float [T]), "f/miss" (0/1 [T]), "log_dt" (float [T])
          - For binary feature f: same as continuous but "f/value" is 0/1 float
          - For categorical feature f: "f/index" (long [T])
          - For string bag: list of list idx per row under key f"{mh.name}/bag"
          - For flag bag: float tensor [T, K] under key f"{mh.name}/flags"
          - "log_dt": float [T] shared
        Returns:
          - X_timestep: [T, model_dim]
          - feature_tokens (optional): [T, F, dim_f] in input order
        """
        T = batch_row["T"]
        log_dt = batch_row["log_dt"]  # [T]

        feat_outs = []
        token_stack = []

        for spec in self.feature_specs:
            if spec.ftype in ("continuous", "binary"):
                v = batch_row[f"{spec.name}/value"]  # [T]
                miss = batch_row[f"{spec.name}/miss"]  # [T]
                enc = self.encoders[spec.name]
                if spec.ftype == "continuous":
                    z = v  # already z-scored
                    o = enc(z, miss, log_dt)
                else:
                    o = enc(v, miss, log_dt)
                feat_outs.append(o)
                if self.emit_feature_tokens:
                    token_stack.append(o.unsqueeze(1))
            elif spec.ftype == "categorical":
                idx = batch_row[f"{spec.name}/index"]  # [T]
                enc = self.encoders[spec.name]
                # print(spec.name, spec.ftype, type(enc))
                o = enc(idx)
                feat_outs.append(o)
                if self.emit_feature_tokens:
                    token_stack.append(o.unsqueeze(1))

        for mh in self.multihot_specs:
            enc = self.multi_encoders[mh.name]
            if mh.bag_kind == "string_bag":
                o = enc(batch_row[f"{mh.name}/bag"])  # [T, dim]
            else:
                flags = batch_row[f"{mh.name}/flags"]  # [T, K]
                o = enc(flags)
            feat_outs.append(o)
            if self.emit_feature_tokens:
                token_stack.append(o.unsqueeze(1))

        concat = torch.cat(feat_outs, dim=-1)  # [T, sum dims]
        X_timestep = self.project(concat)      # [T, model_dim]

        out = {"X_timestep": X_timestep}
        if self.emit_feature_tokens:
            out["feature_tokens"] = torch.cat(token_stack, dim=1)  # [T, F, dim_f]
        return out


# ---------------------------
# Fitter: build specs, vocab, scalers from TRAIN
# ---------------------------

def build_feature_specs(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    session_col: Optional[str],
    min_cat_freq: int = 5,
    max_unique_for_cat: int = 25,
    multihot_string_regex: str = r"^drug\d+$",
    multihot_flag_regex: str = r"^meds_\d+$",
    exclude_cols: Optional[List[str]] = None,
    per_feature_dim: int = 32,
) -> Tuple[List[FeatureSpec], Dict[str,int], List[MultiHotSpec], Dict[str,Any]]:
    exclude = set([id_col, time_col])
    if session_col:
        # we will include it as categorical later—don't exclude
        pass
    if exclude_cols:
        exclude.update(exclude_cols)

    # Identify multi-hot groups
    string_cols = [c for c in df.columns if re.match(multihot_string_regex, c)]
    flag_cols   = [c for c in df.columns if re.match(multihot_flag_regex, c)]

    base_cols = [c for c in df.columns if c not in exclude and c not in string_cols and c not in flag_cols]

    feature_specs: List[FeatureSpec] = []
    cat_vocab_sizes: Dict[str, int] = {}

    # Session/test label as categorical if provided
    if session_col and session_col in df.columns:
        # Build vocab with min freq
        vc = df[session_col].astype(str).fillna("__NA__").value_counts()
        keep = set(vc[vc >= min_cat_freq].index.tolist())
        mapping = {"__UNK__": 0}
        for v in keep:
            mapping[v] = len(mapping)
        cat_vocab_sizes[session_col] = len(mapping)
        feature_specs.append(FeatureSpec(name=session_col, ftype="categorical", cat2idx=mapping, dim=per_feature_dim))

    for c in base_cols:
        s = df[c]
        # skip all-null
        if s.isna().all():
            continue

        if infer_binary(s):
            feature_specs.append(FeatureSpec(name=c, ftype="binary", dim=per_feature_dim))
        elif infer_continuous(s, max_unique_for_cat=max_unique_for_cat):
            mu = float(s.mean(skipna=True)) if s.notna().any() else 0.0
            sd = float(s.std(skipna=True)) if s.notna().sum() > 1 else 1.0
            if sd == 0.0:
                sd = 1.0
            feature_specs.append(FeatureSpec(name=c, ftype="continuous", mean=mu, std=sd, dim=per_feature_dim))
        elif infer_categorical(s, max_unique_for_cat=max_unique_for_cat):
            vc = s.astype(str).fillna("__NA__").value_counts()
            keep = set(vc[vc >= min_cat_freq].index.tolist())
            mapping = {"__UNK__": 0}
            for v in keep:
                mapping[v] = len(mapping)
            cat_vocab_sizes[c] = len(mapping)
            feature_specs.append(FeatureSpec(name=c, ftype="categorical", cat2idx=mapping, dim=per_feature_dim))

    multihot_specs: List[MultiHotSpec] = []
    if len(string_cols) > 0:
        # Build vocab over observed strings (non-empty) across all drug* cols
        tokens = set()
        for c in string_cols:
            vals = df[c].dropna().astype(str).str.strip()
            vals = vals[vals != ""]
            tokens.update(vals.tolist())
        vocab = {"__PAD__": 0}
        for t in sorted(tokens):
            vocab[t] = len(vocab)
        multihot_specs.append(MultiHotSpec(name="drug_text", columns=string_cols, bag_kind="string_bag", vocab=vocab, dim=per_feature_dim))

    if len(flag_cols) > 0:
        multihot_specs.append(MultiHotSpec(name="meds_flags", columns=flag_cols, bag_kind="flag_bag", dim=per_feature_dim))

    meta = {
        "id_col": id_col,
        "time_col": time_col,
        "session_col": session_col,
        "string_cols": string_cols,
        "flag_cols": flag_cols,
        "per_feature_dim": per_feature_dim,
    }
    return feature_specs, cat_vocab_sizes, multihot_specs, meta


# ---------------------------
# Row -> tensors builder (single patient)
# ---------------------------

def rowpack_for_patient(
    df_p: pd.DataFrame,
    feature_specs: List[FeatureSpec],
    multihot_specs: List[MultiHotSpec],
    session_col: Optional[str],
    time_col: str,
) -> Dict[str, Any]:
    # Sort by time
    df_p = df_p.sort_values(time_col)
    t = df_p[time_col].astype(float).fillna(0.0).values
    if len(t) == 0:
        return {}
    delta = np.diff(t, prepend=t[0])
    delta[0] = 0.0
    log_dt = torch.tensor(np.log1p(np.maximum(delta, 0.0)), dtype=torch.float32)
    T = len(t)

    pack: Dict[str, Any] = {"T": T, "log_dt": log_dt, "t": torch.tensor(t, dtype=torch.float32)}

    for spec in feature_specs:
        col = spec.name
        if spec.ftype == "continuous":
            v = df_p[col].astype(float)
            miss = v.isna().astype(float).values
            v = v.fillna(spec.mean)
            z = (v.values - spec.mean) / (spec.std if spec.std else 1.0)
            pack[f"{col}/value"] = torch.tensor(z, dtype=torch.float32)
            pack[f"{col}/miss"]  = torch.tensor(miss, dtype=torch.float32)
        elif spec.ftype == "binary":
            v = pd.to_numeric(df_p[col], errors="coerce")
            miss = v.isna().astype(float).values
            v = v.fillna(0.0).clip(0,1).values
            pack[f"{col}/value"] = torch.tensor(v, dtype=torch.float32)
            pack[f"{col}/miss"]  = torch.tensor(miss, dtype=torch.float32)
        elif spec.ftype == "categorical":
            mapping = spec.cat2idx or {"__UNK__": 0}
            raw = df_p[col].astype(str).fillna("__NA__").values
            idxs = [mapping.get(x, 0) for x in raw]
            pack[f"{col}/index"] = torch.tensor(idxs, dtype=torch.long)

    for mh in multihot_specs:
        if mh.bag_kind == "string_bag":
            bags: List[List[int]] = []
            for _, row in df_p.iterrows():
                idxs = []
                for c in mh.columns:
                    val = row.get(c)
                    if pd.isna(val): 
                        continue
                    s = str(val).strip()
                    if s == "":
                        continue
                    idxs.append(mh.vocab.get(s, 0))
                bags.append(idxs)
            pack[f"{mh.name}/bag"] = bags
        else:
            flags = df_p[mh.columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(0,1).values
            pack[f"{mh.name}/flags"] = torch.tensor(flags, dtype=torch.float32)

    # Visit mask (all rows are visits)
    pack["visit_mask"] = torch.ones(T, dtype=torch.float32)

    # Encode session label if present and in feature specs (already added as categorical)
    if session_col and session_col in df_p.columns:
        # Already handled as a categorical feature; we also store raw label for debugging
        pack["session_label_raw"] = df_p[session_col].astype(str).fillna("__NA__").tolist()

    return pack


# ---------------------------
# End-to-end: fit, encode, save
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--id_col", default="patient_id")
    ap.add_argument("--time_col", default="days_to_visit")
    ap.add_argument("--session_col", default="OASIS_session_label")
    ap.add_argument("--min_cat_freq", type=int, default=5)
    ap.add_argument("--max_unique_for_cat", type=int, default=25)
    ap.add_argument("--model_dim", type=int, default=256)
    ap.add_argument("--per_feature_dim", type=int, default=32)
    ap.add_argument("--emit_feature_tokens", action="store_true")
    ap.add_argument("--exclude_cols", type=str, default="")
    args = ap.parse_args()

    out_dir = args.out_dir
    df = pd.read_csv(args.csv, sep=None, engine="python")

    exclude_cols = [c.strip() for c in args.exclude_cols.split(",") if c.strip()]

    feature_specs, cat_vocab_sizes, multihot_specs, meta = build_feature_specs(
        df=df,
        id_col=args.id_col,
        time_col=args.time_col,
        session_col=args.session_col if args.session_col in df.columns else None,
        min_cat_freq=args.min_cat_freq,
        max_unique_for_cat=args.max_unique_for_cat,
        exclude_cols=exclude_cols,
        per_feature_dim=args.per_feature_dim,
    )

    # Force OASIS_session_label to be categorical even if it has only 2 unique values
    for spec in feature_specs:
        if spec.name == "OASIS_session_label":
            spec.ftype = "categorical"

    # Make sure a vocab size exists for it
    if "OASIS_session_label" not in cat_vocab_sizes:
        unique_vals = df["OASIS_session_label"].dropna().unique()
        cat_vocab_sizes["OASIS_session_label"] = len(unique_vals)

    # Persist spec + scalers/vocabs
    spec_payload = {
        "feature_specs": [
            {
                "name": s.name, "ftype": s.ftype, "mean": s.mean, "std": s.std,
                "cat2idx": s.cat2idx, "dim": s.dim
            } for s in feature_specs
        ],
        "cat_vocab_sizes": cat_vocab_sizes,
        "multihot_specs": [
            {
                "name": mh.name, "columns": mh.columns, "bag_kind": mh.bag_kind,
                "vocab": mh.vocab, "dim": mh.dim
            } for mh in multihot_specs
        ],
        "meta": meta,
        "model_dim": args.model_dim,
        "emit_feature_tokens": args.emit_feature_tokens,
    }
    torch.save(spec_payload, f"{out_dir}/clinical_vocab.pt")

    # Build model
    model = ClinicalEncoder(
        feature_specs=feature_specs,
        cat_vocab_sizes=cat_vocab_sizes,
        multihot_specs=multihot_specs,
        model_dim=args.model_dim,
        emit_feature_tokens=args.emit_feature_tokens,
    )
    model.eval()  # embeddings only

    # Encode per patient
    patients: Dict[str, Dict[str, torch.Tensor]] = {}
    for pid, df_p in df.groupby(args.id_col):
        pack = rowpack_for_patient(
            df_p=df_p,
            feature_specs=feature_specs,
            multihot_specs=multihot_specs,
            session_col=args.session_col if args.session_col in df.columns else None,
            time_col=args.time_col,
        )
        if not pack:
            continue

        with torch.no_grad():
            out = model(pack)
            item = {
                "X_timestep": out["X_timestep"],  # [T, D]
                "delta_t": pack["log_dt"].exp() - 1.0,  # recover Δt
                "t": pack["t"],
                "visit_mask": pack["visit_mask"],
            }
            if args.emit_feature_tokens:
                item["feature_tokens"] = out["feature_tokens"]
            if "session_label_raw" in pack:
                # Also store encoded indices if present
                item["session_label_raw"] = pack["session_label_raw"]

            patients[str(pid)] = item

    torch.save(patients, f"{out_dir}/patients.pt")

    print(f"Saved spec to {out_dir}/clinical_vocab.pt")
    print(f"Saved patient embeddings to {out_dir}/patients.pt")
    print(f"Patients encoded: {len(patients)}")


if __name__ == "__main__":
    main()
