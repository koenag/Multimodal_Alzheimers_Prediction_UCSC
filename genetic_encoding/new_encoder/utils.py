import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path

def build_vocabs(
    csv_path: Path,
    fields: tuple[str]     = ("CHROM","REF","ALT","QUAL","GT","GQ","PL"),
    special_tokens: list[str] = ("[PAD]","[UNK]","[CLS]","[SEP]","[MASK]"),
    bin_size_pos: int         = 100_000,
    bin_size_qual: float      = 1.0
) -> tuple[dict[str,int], dict[str,int], dict[str,int]]:
    """
    Builds:
      - token2id:   special tokens first, then FIELD_value tokens
      - field2id:   field name → unique int
      - chrom2id:   chromosome name → unique int
    """
    df = pd.read_csv(csv_path, dtype=str)

    df["POS"]       = df["POS"].astype(int)
    df["pos_bin"]   = (df["POS"] // bin_size_pos).astype(int)
    df["QUAL"]      = df["QUAL"].astype(float)
    df["QUAL_bin"]  = (df["QUAL"] // bin_size_qual).astype(int)

    # 1) field2id
    field2id = {f: i for i, f in enumerate(fields)}

    # 2) chrom2id
    chroms = sorted(
        df["CHROM"].unique(),
        key=lambda x: (x.isdigit(), int(x) if x.isdigit() else x)
    )
    chrom2id = {c: i for i, c in enumerate(chroms)}

    # 3) token2id: reserve special tokens at the front
    token2id = {}
    idx = 0
    for tok in special_tokens:
        token2id[tok] = idx
        idx += 1

    # then add all FIELD_value tokens
    for field in fields:
        if field == "QUAL":
            for b in sorted(df["QUAL_bin"].unique()):
                key = f"QUAL_{b}"
                token2id[key] = idx; idx += 1
        else:
            for val in df[field].astype(str).unique():
                key = f"{field}_{val}"
                if key not in token2id:
                    token2id[key] = idx; idx += 1

    return token2id, field2id, chrom2id


class VCFDataset(Dataset):
    """
    Produces fixed-length sequences for pretraining:
     - QUAL is discretized and emitted as QUAL_{bin} token
     - seq_length tokens total (last one is [SEP])
     - GT/GQ/PL tokens mapped to [UNK]
    Returns dict of tensors all shaped (seq_length,).
    """
    def __init__(
        self,
        csv_path: Path,
        token2id: dict[str,int],
        field2id: dict[str,int],
        chrom2id: dict[str,int],
        fields: tuple[str, ...] = ("CHROM", "REF", "ALT", "QUAL", "GT", "GQ", "PL"),
        bin_size_pos: int = 100_000,
        bin_size_qual: float = 1.0,
        seq_length: int = 512,
        sep_token: str = "[SEP]",
        unk_token: str = "[UNK]"
    ):
        super().__init__()
        df = pd.read_csv(csv_path, dtype=str, low_memory=False)
        self.fields     = fields
        self.token2id   = token2id
        self.field2id   = field2id
        self.chrom2id   = chrom2id
        self.sep_id     = token2id[sep_token]
        self.unk_id     = token2id[unk_token]
        self.seq_length = seq_length
        self.f          = len(fields)

        assert (seq_length - 1) % self.f == 0, \
            "seq_length-1 must be multiple of num fields"
        self.snps_per_seq = (seq_length - 1) // self.f

        # numeric POS
        df["POS"] = df["POS"].astype(int)
        df["pos_bin"] = (df["POS"] // bin_size_pos).astype(int)
        df["pos_off"] = (df["POS"] % bin_size_pos) / bin_size_pos
        # discretize QUAL
        df["QUAL"] = df["QUAL"].astype(float)
        df["QUAL_bin"] = (df["QUAL"] // bin_size_qual).astype(int)

        self.df = df.reset_index(drop=True)
        self.num_seqs = len(self.df) // self.snps_per_seq

    def __len__(self):
        return self.num_seqs

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * self.snps_per_seq
        toks, flds, chrs = [], [], []
        p_bins, p_offs = [], []
        for i in range(start, start + self.snps_per_seq):
            row = self.df.iloc[i]
            # build tokens per field
            for f in self.fields:
                if f in ("GT","GQ","PL"):
                    toks.append(self.unk_id)
                elif f == "QUAL":
                    toks.append(self.token2id[f"QUAL_{row['QUAL_bin']}"])
                else:
                    toks.append(self.token2id[f"{f}_{row[f]}"])
                flds.append(self.field2id[f])
                chrs.append(self.chrom2id[row["CHROM"]])
                # positional embeddings (same for all fields)
                p_bins.append(row["pos_bin"])
                p_offs.append(row["pos_off"])
        # SEP
        toks.append(self.sep_id)
        flds.append(0); chrs.append(0); p_bins.append(0); p_offs.append(0.0)
        return {
            "token_ids":   torch.tensor(toks,   dtype=torch.long),
            "field_ids":   torch.tensor(flds,   dtype=torch.long),
            "chrom_ids":   torch.tensor(chrs,   dtype=torch.long),
            "pos_bin_ids": torch.tensor(p_bins, dtype=torch.long),
            "pos_offset":  torch.tensor(p_offs, dtype=torch.float),
        }


def mask_tokens(inputs: torch.Tensor, mask_token_id: int, vocab_size: int, mlm_prob: float):
    """
    Prepare masked tokens inputs/labels for masked language modeling:
    1. mlm_prob% of tokens are selected for possible masking.
    2. Of those, 80% replaced with [MASK], 10% with random token, 10% unchanged.
    """
    labels = inputs.clone()
    # select positions to mask
    prob_matrix = torch.full(labels.shape, mlm_prob, device=inputs.device)
    masked_indices = torch.bernoulli(prob_matrix).bool()
    labels[~masked_indices] = -100  # only compute loss on masked tokens

    # 80% replace with mask_token_id
    mask_prob = 0.8
    replace_mask = torch.bernoulli(torch.full(labels.shape, mask_prob, device=inputs.device)).bool() & masked_indices
    inputs[replace_mask] = mask_token_id

    # 10% replace with random token
    rand_prob = 0.5  # half of the remainder => ~10%
    rand_mask = torch.bernoulli(torch.full(labels.shape, rand_prob, device=inputs.device)).bool() \
                & masked_indices & ~replace_mask
    random_tokens = torch.randint(vocab_size, labels.shape, dtype=torch.long, device=inputs.device)
    inputs[rand_mask] = random_tokens[rand_mask]

    # rest 10% keep original
    return inputs, labels