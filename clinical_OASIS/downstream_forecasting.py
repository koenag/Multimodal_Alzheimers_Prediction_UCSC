import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ===========================
# 1. Downstream forecaster
# ===========================

class DownstreamForecaster(nn.Module):
    """
    Wraps pretrained encoder and adds a small regression head
    for forecasting a single target variate (e.g., variate_i_2 at T+Δ).
    """
    def __init__(
        self,
        encoder: nn.Module,
        d_model: int,
        pool_type: str = "cls",   # "cls" or "mean"
        out_dim: int = 1,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.pool_type = pool_type

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, out_dim),
        )

    def pool(self, hidden_states, attention_mask=None):
        """
        hidden_states: (B, L, D)
        attention_mask: (B, L) with 1 for valid tokens, 0 for padding
        """
        if self.pool_type == "cls":
            # assumes [CLS] token is at position 0
            return hidden_states[:, 0, :]  # (B, D)

        # masked mean pooling
        if attention_mask is None:
            return hidden_states.mean(dim=1)

        mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        masked_hidden = hidden_states * mask
        denom = mask.sum(dim=1).clamp(min=1.0)
        return masked_hidden.sum(dim=1) / denom  # (B, D)

    def forward(self, encoder_inputs, attention_mask=None):
        """
        encoder_inputs: dict passed directly into your pretrained encoder
        attention_mask: (B, L) if used by your encoder / pooling
        """
        encoder_outputs = self.encoder(**encoder_inputs)
        # adapt this line if your encoder uses a different key:
        hidden_states = encoder_outputs["last_hidden_state"]  # (B, L, D)

        pooled = self.pool(hidden_states, attention_mask)      # (B, D)
        preds = self.head(pooled)                              # (B, out_dim)
        return preds


# ===========================
# 2. Dataset + collate
# ===========================

class ForecastDataset(Dataset):
    """
    Each item corresponds to one sequence (e.g., one patient up to time T).

    - encoder_input_dict_list[i]: dict of tensors for your encoder, e.g.
        {
            "input_ids": (L,),
            "time_ids": (L,),
            "variate_ids": (L,),
            ...
        }
    - attention_masks[i]: (L,) mask (1 = real token, 0 = pad)
    - targets[i]: scalar or 1D tensor with the future value of your target variate
    """
    def __init__(self, encoder_input_dict_list, attention_masks, targets):
        self.encoder_input_dict_list = encoder_input_dict_list
        self.attention_masks = attention_masks
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            "encoder_inputs": self.encoder_input_dict_list[idx],
            "attention_mask": self.attention_masks[idx],
            "target": self.targets[idx],
        }


def collate_forecast(batch):
    """
    Collate function to batch dict-based encoder inputs.

    Returns:
        encoder_inputs: dict of batched tensors (B, L, ...)
        attention_mask: (B, L)
        targets: (B, 1)
    """
    first_enc_inputs = batch[0]["encoder_inputs"]
    enc_keys = list(first_enc_inputs.keys())

    encoder_inputs = {}
    for k in enc_keys:
        encoder_inputs[k] = torch.stack(
            [b["encoder_inputs"][k] for b in batch],
            dim=0,
        )

    attention_mask = torch.stack(
        [b["attention_mask"] for b in batch],
        dim=0,
    )

    targets = torch.stack([b["target"] for b in batch], dim=0)
    if targets.ndim == 1:
        targets = targets.unsqueeze(-1)  # (B, 1)

    return encoder_inputs, attention_mask, targets


# ===========================
# 3. Training function
# ===========================

def train_downstream(
    encoder: nn.Module,
    d_model: int,
    train_dataset: Dataset,
    val_dataset: Dataset | None = None,
    pool_type: str = "cls",
    out_dim: int = 1,
    freeze_encoder: bool = False,
    num_epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    device: str | None = None,
):
    """
    encoder:  pretrained encoder from Task 1 (already initialized + loaded).
    d_model: hidden size of encoder outputs
    train_dataset, val_dataset: ForecastDataset instances
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = DownstreamForecaster(
        encoder=encoder,
        d_model=d_model,
        pool_type=pool_type,
        out_dim=out_dim,
        freeze_encoder=freeze_encoder,
    ).to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_forecast,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_forecast,
        )

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0

        for enc_inputs, attention_mask, target in train_loader:
            enc_inputs = {k: v.to(device) for k, v in enc_inputs.items()}
            attention_mask = attention_mask.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            preds = model(enc_inputs, attention_mask)  # (B, out_dim)
            loss = criterion(preds, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * target.size(0)

        train_loss /= len(train_dataset)

    return model
