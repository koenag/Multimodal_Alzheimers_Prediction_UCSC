import torch
from torch import nn
from utils import mask_tokens
class VcfCollator:
    """
    Stacks a list of examples into batched tensors,
    applies MLM masking, and moves everything to GPU.
    """
    def __init__(self, mask_token_id, vocab_size, mlm_prob, device):
        self.mask_token_id = mask_token_id
        self.vocab_size    = vocab_size
        self.mlm_prob      = mlm_prob
        self.device        = device

    def __call__(self, batch):
        # 1) stack all fields
        token_ids = torch.stack([ex["token_ids"]   for ex in batch], dim=0)
        field_ids = torch.stack([ex["field_ids"]   for ex in batch], dim=0)
        chrom_ids = torch.stack([ex["chrom_ids"]   for ex in batch], dim=0)
        pos_bins  = torch.stack([ex["pos_bin_ids"] for ex in batch], dim=0)
        pos_offs  = torch.stack([ex["pos_offset"]  for ex in batch], dim=0)

        # 2) mask on CPU
        inputs, labels = mask_tokens(
            token_ids,
            self.mask_token_id,
            self.vocab_size,
            self.mlm_prob
        )

        batch = {
            "inputs":    inputs,
            "labels":    labels,
            "field_ids": field_ids,
            "chrom_ids": chrom_ids,
            "pos_bins":  pos_bins,
            "pos_offs":  pos_offs,
        }
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}


import argparse
import json
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import bitsandbytes as bnb
from utils import build_vocabs, VCFDataset, mask_tokens
from model import Bert

def main():
    parser = argparse.ArgumentParser(description="Pretrain BERT on SNP metadata")
    parser.add_argument("--csv_path",     type=Path, required=True, help="Path to metadata CSV")
    parser.add_argument("--output_dir",   type=Path, required=True, help="Directory to save checkpoints")
    parser.add_argument("--epochs",       type=int, default=1)
    parser.add_argument("--batch_size",   type=int, default=128)
    parser.add_argument("--lr",           type=float, default=1e-5)
    parser.add_argument("--mlm_prob",     type=float, default=0.25)
    parser.add_argument("--seq_len",      type=int, default=512)
    parser.add_argument("--warmup_steps", type=int, default=250)
    parser.add_argument("--max_steps",    type=int, default=10000)
    parser.add_argument("--bin_size_pos", type=int, default=100_000)
    parser.add_argument("--bin_size_qual",type=int, default=1_000)
    parser.add_argument("--num_workers",  type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build or load vocabs
    token2id, field2id, chrom2id = build_vocabs(
        args.csv_path,
        bin_size_pos=args.bin_size_pos,
        bin_size_qual=args.bin_size_qual
    )

    vocab_size = len(token2id)
    from functools import partial
    mask_token_id = token2id["[MASK]"]
    collate_fn    = VcfCollator(mask_token_id, vocab_size, args.mlm_prob, device="cpu")

    # Dataset & DataLoader
    ds = VCFDataset(
        csv_path=args.csv_path,
        token2id=token2id,
        field2id=field2id,
        chrom2id=chrom2id,
        bin_size_pos=args.bin_size_pos,
        bin_size_qual=args.bin_size_qual,
        seq_length=args.seq_len
    )
    #loader = DataLoader(
    #    ds,
    #    batch_size=args.batch_size,
    #    shuffle=False,
    #    num_workers=args.num_workers,
    #    pin_memory=True,
    #    persistent_workers=True,
    #    prefetch_factor=4,
    #    collate_fn=collate_fn
    #)
        # compute the new args for Bert()
    num_fields = len(ds.fields)                      # e.g. 7
    num_chroms = len(chrom2id)                       # number of distinct chromosomes
    max_bins   = int(ds.df["pos_bin"].max()) + 1  # total number of genomic bins
    
    model = Bert(
        vocab_size=vocab_size,
        num_fields=num_fields,
        num_chroms=num_chroms,
        max_bins=max_bins,
        embed_dim=512,
        num_layers=12,
        num_heads=4,
        dim_feedforward=2048,
        dropout=0.1,
        use_flash_attention=True
    )
    model = model.half().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    # Optimizer & Scheduler
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=1e-2)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return float(step) / float(max(1, args.warmup_steps))
        return max(0.0,
                float(args.max_steps - step) /
                float(max(1, args.max_steps - args.warmup_steps)))
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    #total_batches = len(loader)
    pbar          = tqdm(total=total_batches, desc="Training", unit="batch")

    step = 0
    model.train()
    for epoch in range(args.epochs):
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k,v in batch.items()}
            optimizer.zero_grad()
            _, logits = model(
                batch["inputs"],
                batch["field_ids"],
                batch["chrom_ids"],
                batch["pos_bins"],
                batch["pos_offs"],
            )
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                batch["labels"].view(-1),
                ignore_index=-100,
            )
            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            step+=1
            '''
            if step % 100 == 0:
                print(f"Step {step:6d} | Loss: {loss.item():.4f}")
            step += 1
            if step >= args.max_steps:
                break
            '''
        #if step >= args.max_steps:
            #break
    #pbar.close()
    # Save final model & vocabs
    torch.save(model.state_dict(), args.output_dir / "bert_snp_pretrained.pt")
    with open(args.output_dir / "token2id.json", "w") as f:
        json.dump(token2id, f, indent=2)
    with open(args.output_dir / "field2id.json", "w") as f:
        json.dump(field2id, f, indent=2)
    with open(args.output_dir / "chrom2id.json", "w") as f:
        json.dump(chrom2id, f, indent=2)

    print(f"Training complete. Model and vocab files saved to {args.output_dir}")

if __name__ == "__main__":
    main()

