
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd

# imports from your codebase
from model import Bert
from finetune_utils import extend_finetune_vocabs, FineTuneVCFDataset


class BertForDiagnosis(nn.Module):
    def __init__(self, base_model: Bert, embed_dim: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.base = base_model
        # classifier head on pooled CLS token
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_labels)
        )

    def forward(self, token_ids, field_ids, chrom_ids, pos_bins, pos_offs):
        # outputs from base: (batch, seq_len, embed_dim)
        x = self.base(
            token_ids, field_ids, chrom_ids, pos_bins, pos_offs
        )
        # take CLS token at position 0
        cls = x[:, 0, :]
        logits = self.classifier(cls)
        return x, logits


def load_diagnosis(csv_path: Path):
    """
    Reads 'diagnosis.csv', strips parentheses, coerces to numeric,
    drops NaNs, shifts to 0-based integer labels.
    Returns mapping patient_id -> label and number of unique labels.
    """
    df = pd.read_csv(csv_path)
    # strip parentheses in PTID and DIAGNOSIS
    df['PTID'] = df['PTID'].astype(str).str.strip('()')
    df['DIAGNOSIS'] = pd.to_numeric(
        df['DIAGNOSIS'].astype(str).str.strip('()'),
        errors='coerce'
    )
    df = df.dropna(subset=['DIAGNOSIS'])
    df['DIAGNOSIS'] = df['DIAGNOSIS'].astype(int) - 1  # shift to 0-based

    # build mapping and count labels
    pid2label = dict(zip(df['PTID'], df['DIAGNOSIS']))
    unique_labels = sorted(set(pid2label.values()))
    num_labels = len(unique_labels)
    return pid2label, num_labels


def freeze_layers(model: nn.Module, num_layers: int):
    """
    Freeze all transformer layers except the last `num_layers`.
    """
    layers = model.base.layers
    for i, layer in enumerate(layers):
        if i < len(layers) - num_layers:
            for p in layer.parameters():
                p.requires_grad = False
                
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


def main():
    parser = argparse.ArgumentParser("Fine‑tune SNP BERT with diagnosis head")
    parser.add_argument("--vcf_path",    type=Path, required=True)
    parser.add_argument("--diag_path",   type=Path, required=True)
    parser.add_argument("--seq_len",     type=int, default=512)
    parser.add_argument("--batch_size",  type=int, default=128)
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--epochs",      type=int, default=3)
    parser.add_argument("--freeze",      type=int, default=2,
                        help="number of top layers to unfreeze")
    parser.add_argument("--workers",     type=int, default=4)
    parser.add_argument("--output_dir",  type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load pretrained vocabs
    token2id = json.load(open(args.output_dir / 'token2id.json'))
    field2id = json.load(open(args.output_dir / 'field2id.json'))
    chrom2id = json.load(open(args.output_dir / 'chrom2id.json'))

    # extend vocab with fine-tuning tokens
    token2id = extend_finetune_vocabs(token2id, args.vcf_path)

    # load diagnoses
    pid2label, num_labels = load_diagnosis(args.diag_path)

    # create dataset
    ds = FineTuneVCFDataset(
        vcf_path    = args.vcf_path,
        token2id    = token2id,
        field2id    = field2id,
        chrom2id    = chrom2id,
        seq_length  = args.seq_len
    )
    loader = DataLoader(
        ds,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.workers,
        pin_memory  = False  # collator handles device transfer
    )

    # instantiate base model using pretrained vocab size
    pretrained_token2id = json.load(open(args.output_dir / 'token2id.json'))
    old_vocab_size = len(pretrained_token2id)
    # build model with old vocab size
    base = Bert(
        vocab_size           = old_vocab_size,
        num_fields           = len(field2id),
        num_chroms           = len(chrom2id),
        max_bins             = 2493,
        embed_dim            = 512,
        num_layers           = 12,
        num_heads            = 4,
        dim_feedforward      = 2048,
        dropout              = 0.1,
        use_flash_attention  = True
    )
    # load pretrained weights (allow for mismatched sizes)
    state_dict = torch.load(
        args.output_dir / 'bert_snp_pretrained.pt', map_location='cpu'
    )
    missing, unexpected = base.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint with {len(missing)} missing keys and {len(unexpected)} unexpected keys")

    # expand token embeddings and MLM head to new vocab size if needed
    new_vocab_size = len(token2id)
    if new_vocab_size != old_vocab_size:
        print(f"Expanding vocab from {old_vocab_size} to {new_vocab_size}")
        # token embeddings
        old_weights = base.embedding.token_emb.weight.data
        embed_dim   = old_weights.size(1)
        # random init for new tokens
        new_weights = torch.randn(
            new_vocab_size - old_vocab_size,
            embed_dim, device=old_weights.device, dtype=old_weights.dtype
        ) * 0.02
        base.embedding.token_emb.weight.data = torch.cat([old_weights, new_weights], dim=0)
        # MLM head weights and bias
        old_mlm_w  = base.mlm_head.weight.data
        old_mlm_b  = base.mlm_head.bias.data
        base.mlm_head.weight.data = torch.cat([old_mlm_w, new_weights], dim=0)
        new_bias = torch.zeros(new_vocab_size - old_vocab_size, device=old_mlm_b.device, dtype=old_mlm_b.dtype)
        base.mlm_head.bias.data   = torch.cat([old_mlm_b, new_bias], dim=0)

    # wrap in diagnosis model
    model = BertForDiagnosis(base, embed_dim=512, num_labels=num_labels).to(device).half()
    freeze_layers(model, args.freeze)

    # optimizer on trainable params
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-2
    )
    total_steps = len(loader) * args.epochs
    scheduler = LambdaLR(
        optimizer,
        lambda step: max(0.0, float(total_steps - step) / total_steps)
    )

    # training loop
    model.train()
    for epoch in range(args.epochs):
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            inputs, labels = mask_tokens(
                batch['token_ids'].clone(),
                token2id['[MASK]'],
                len(token2id),
                0.15
            )
            optimizer.zero_grad()
            mlm_out, cls_logits = model(
                inputs,
                batch['field_ids'],
                batch['chrom_ids'],
                batch['pos_bins'],
                batch['pos_offs']
            )
            # losses
            lm_loss = F.cross_entropy(
                mlm_out.view(-1, len(token2id)),
                labels.view(-1),
                ignore_index=-100
            )
            # map batch subjects to labels
            subjects = batch.get('subject_ids', [])
            ptids = [pid2label.get(str(s), 0) for s in subjects]
            ptids = torch.tensor(ptids, device=device)
            cls_loss = F.cross_entropy(cls_logits, ptids)
            loss = lm_loss + 0.5 * cls_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs} complete | Loss {loss.item():.4f}")

    # save final model
    torch.save(model.state_dict(), args.output_dir / 'bert_snp_finetuned.pt')
if __name__ == '__main__':
    main()