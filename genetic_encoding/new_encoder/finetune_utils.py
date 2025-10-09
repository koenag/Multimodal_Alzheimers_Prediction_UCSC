
import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from typing import Dict


def extend_finetune_vocabs(
    token2id: Dict[str,int],
    vcf_path: Path,
    gq_bin_size: int = 10
) -> Dict[str,int]:
    """
    Extend an existing token2id with GT, GQ, and PLMODE tokens by parsing each
    genotype cell in the VCF. GT/GQ/PL values are extracted from the FORMAT field
    and binned / indexed as needed.
    """
    # read VCF (tab‑delimited, '#' comments skipped)
    df = pd.read_csv(vcf_path, dtype=str)
    # identify genotype columns (subjects)
    fixed = ["#CHROM","POS","ID","REF","ALT","QUAL","FILTER","INFO","FORMAT"]
    subjects = [c for c in df.columns if c not in fixed]

    for _, row in df.iterrows():
        fmt = row['FORMAT'].split(':')
        for subj in subjects:
            cell = str(row[subj])
            parts = cell.split(':')
            fv = dict(zip(fmt, parts))
            # GT token
            gt = fv.get('GT')
            if gt is not None:
                tok = f"GT_{gt}"
                if tok not in token2id:
                    token2id[tok] = len(token2id)
            # GQ token (binned)
            gq = fv.get('GQ')
            if gq and gq.isdigit():
                gq_val = int(gq)
                b = (gq_val // gq_bin_size) * gq_bin_size
                tok = f"GQ_{b}-{b+gq_bin_size}"
                if tok not in token2id:
                    token2id[tok] = len(token2id)
            # PLMODE token (index of min PL)
            pl = fv.get('PL')
            if pl:
                pls = [int(x) for x in pl.split(',') if x.isdigit()]
                if pls:
                    m = pls.index(min(pls))
                    tok = f"PLMODE_{m}"
                    if tok not in token2id:
                        token2id[tok] = len(token2id)
    return token2id


class FineTuneVCFDataset(Dataset):
    """
    Loads VCF for fine-tuning, aligned with pretraining tokenization:
      - Uses REF, ALT, QUAL (binned), GT, GQ, PLMODE tokens
      - CHROM handled via chrom_ids for positional embedding
    Returns dict with:
      token_ids, field_ids, chrom_ids, pos_bin_ids, pos_offset
    """
    def __init__(
        self,
        vcf_path: Path,
        token2id: Dict[str,int],
        field2id: Dict[str,int],
        chrom2id: Dict[str,int],
        seq_length: int = 512,
        bin_size_pos: int = 100_000,
        bin_size_qual: float = 10,
        gq_bin_size: int = 10,
        sep_token: str = "[SEP]",
        unk_token: str = "[UNK]",
    ):
        super().__init__()
        df = pd.read_csv(vcf_path, dtype=str)
        self.fields       = ("#CHROM","REF","ALT","QUAL","GT","GQ","PL")
        self.token2id     = token2id
        self.field2id     = field2id
        print(field2id)
        self.chrom2id     = chrom2id
        self.seq_length   = seq_length
        self.sep_id       = token2id[sep_token]
        self.unk_id       = token2id[unk_token]
        self.bin_size_pos = bin_size_pos
        self.bin_size_qual= bin_size_qual
        self.gq_bin_size  = gq_bin_size

        # preprocess numeric POS & QUAL
        df['POS']      = df['POS'].astype(int)
        df['pos_bin']  = (df['POS'] // bin_size_pos).astype(int)
        df['pos_off']  = (df['POS'] % bin_size_pos) / bin_size_pos
        df['QUAL']     = df['QUAL'].astype(float)
        df['QUAL_bin'] = (df['QUAL'] // bin_size_qual).astype(int)
        self.df = df.reset_index(drop=True)

        f = len(self.fields)
        assert (seq_length - 1) % f == 0, "seq_length-1 must be multiple of num fields"
        self.snps_per_seq = (seq_length - 1) // f
        self.num_seqs     = len(self.df) // self.snps_per_seq

    def __len__(self):
        return self.num_seqs

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx * self.snps_per_seq
        toks, flds, chrs = [], [], []
        p_bins, p_offs = [], []
        for i in range(start, start + self.snps_per_seq):
            row = self.df.iloc[i]
            fmt = row['FORMAT'].split(':')
            # Parse the subject genotype string
            subject_id = [c for c in self.df.columns if c not in self.fields and c not in ('POS', 'ID', 'INFO') and not c.startswith('#')][0]
            cell = row[subject_id]
            fmt_values = dict(zip(fmt, cell.split(':')))

            for f in self.fields:
                if f == '#CHROM':
                    tok_id = self.unk_id  # CHROM is handled in chrom_ids, so token is unused
                elif f == 'QUAL':
                    tok_id = self.token2id.get(f"QUAL_{row['QUAL_bin']}", self.unk_id)
                elif f in ('GT', 'GQ'):
                    val = fmt_values.get(f, "")
                    if f == 'GQ' and val.isdigit():
                        b = (int(val) // self.gq_bin_size) * self.gq_bin_size
                        tok_str = f"{f}_{b}-{b+self.gq_bin_size}"
                    else:
                        tok_str = f"{f}_{val}"
                    tok_id = self.token2id.get(tok_str, self.unk_id)
                elif f == 'PL':
                    pl_str = fmt_values.get('PL', '')
                    pls = [int(x) for x in pl_str.split(',') if x.isdigit()]
                    if pls:
                        mode = pls.index(min(pls))
                        tok_str = f"PLMODE_{mode}"
                    else:
                        tok_str = f"PLMODE_0"
                    tok_id = self.token2id.get(tok_str, self.unk_id)
                else:
                    tok_id = self.token2id.get(f"{f}_{row[f]}", self.unk_id)

                toks.append(tok_id)
                flds.append(self.field2id[f])

            # Only append CHROM ID once per SNP
            chrs.append(self.chrom2id.get(row['#CHROM'], 0))
            p_bins.append(row['pos_bin'])
            p_offs.append(row['pos_off'])
        return {
            'token_ids':   torch.tensor(toks, dtype=torch.long),
            'field_ids':   torch.tensor(flds, dtype=torch.long),
            'chrom_ids':   torch.tensor(chrs, dtype=torch.long),
            'pos_bins': torch.tensor(p_bins, dtype=torch.long),
            'pos_offs':  torch.tensor(p_offs, dtype=torch.float),
        }
