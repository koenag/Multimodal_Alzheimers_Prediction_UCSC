import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from einops import rearrange

# If using flash-attn:
try:
    from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func
    HAS_FLASH = True
except ImportError:
    HAS_FLASH = False

# ---------------------------
# Rotary Positional Embedding
# ---------------------------

def rotate_half(x: Tensor) -> Tensor:
    """
    Helper to rotate half the dimensions for RoPE.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

class RotaryEmbedding(nn.Module):
    """
    Computes fixed sin/cos for rotary embeddings.
    """
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        # build [max_seq_len, dim]
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.zeros((max_seq_len, dim))
        emb[:, 0::2] = freqs
        emb[:, 1::2] = freqs
        # register cos/sin buffers
        self.register_buffer('cos_emb', emb.cos()[None, None, :, :])
        self.register_buffer('sin_emb', emb.sin()[None, None, :, :])

    def forward(self, x: Tensor, seq_len: int, offset: int = 0) -> Tensor:
        # x: [B, num_heads, S, head_dim]
        cos = self.cos_emb[:, :, offset:offset+seq_len, :].to(x.device)
        sin = self.sin_emb[:, :, offset:offset+seq_len, :].to(x.device)
        return (x * cos) + (rotate_half(x) * sin)

# ---------------------------
# Core Embedding Layer
# ---------------------------

class SNPEmbedding(nn.Module):
    """
    Combines token, field, chromosome/bin/offset embeddings.
    """
    def __init__(
        self,
        vocab_size: int,
        num_fields: int,
        num_chroms: int,
        max_bins: int,
        embed_dim: int,
        offset_hidden: int = 64
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.field_emb = nn.Embedding(num_fields, embed_dim)
        self.chrom_emb = nn.Embedding(num_chroms, embed_dim)
        self.bin_emb   = nn.Embedding(max_bins, embed_dim)
        # small MLP for normalized offset
        self.offset_mlp = nn.Sequential(
            nn.Linear(1, offset_hidden),
            nn.GELU(),
            nn.Linear(offset_hidden, embed_dim)
        )

    def forward(
        self,
        token_ids: Tensor,
        field_ids: Tensor,
        chrom_ids: Tensor,
        bin_ids: Tensor,
        offset_norm: Tensor
    ) -> Tensor:
        # each input: [B, S]
        t = self.token_emb(token_ids)
        f = self.field_emb(field_ids)
        c = self.chrom_emb(chrom_ids)
        b = self.bin_emb(bin_ids)
        offset_in = offset_norm.unsqueeze(-1).half()
        o        = self.offset_mlp(offset_in)
        return t + f + c + b + o

# ---------------------------
# Transformer Encoder Layer
# ---------------------------

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        use_flash: bool = False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.use_flash = use_flash and HAS_FLASH

        if self.use_flash:
            self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        else:
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, embed_dim)
        )
        # rotary
        self.rotary = RotaryEmbedding(self.head_dim)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        # x: [B, S, E]
        B, S, E = x.shape
        # --- Self-Attention ---
        res = x
        x = self.norm1(x)
        if self.use_flash:
            # flash-attn path
            qkv = self.qkv_proj(x)  # [B, S, 3E]
            qkv = qkv.view(B, S, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, S, D]
            # apply rotary to q and k
            q, k, v = qkv[0], qkv[1], qkv[2]
            q = self.rotary(q, seq_len=S)
            k = self.rotary(k, seq_len=S)
            qkv_packed = torch.stack([q, k, v], dim=2)  # [B,H,3,S,D]
            # flash-attn
            out = flash_attn_qkvpacked_func(qkv_packed, dropout_p=self.dropout.p if self.training else 0.0,
                                            causal=False)
            out = rearrange(out, 'b h s d -> b s (h d)')
            x = self.out_proj(out)
        else:
            # standard MHA
            # for rotary, MultiheadAttention doesn't support custom rotation -> fallback without rotary
            x, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = res + self.dropout(x)

        # --- Feed Forward ---
        res = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = res + self.dropout(x)
        return x

# ---------------------------
# Bert-style Encoder
# ---------------------------

class Bert(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_fields: int,
        num_chroms: int,
        max_bins: int,
        embed_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        use_flash_attention: bool = False
    ):
        super().__init__()
        self.embedding = SNPEmbedding(vocab_size, num_fields, num_chroms, max_bins, embed_dim)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward, dropout, use_flash_attention)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.mlm_head = nn.Linear(embed_dim, vocab_size)

    def forward(
        self,
        token_ids: Tensor,
        field_ids: Tensor,
        chrom_ids: Tensor,
        bin_ids: Tensor,
        offset_norm: Tensor,
        attention_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        x = self.embedding(token_ids, field_ids, chrom_ids, bin_ids, offset_norm)
        for layer in self.layers:
            x = layer(x, mask=attention_mask)
        x = self.norm(x)
        logits = self.mlm_head(x)
        return x, logits
