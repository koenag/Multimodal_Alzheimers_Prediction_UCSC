import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import copy

# ==========================================
# 1. Configuration
# ==========================================
# File paths
EMBEDDING_FILE = "event_prefix_embeddings.pt"
SOURCE_CSV = "oasis3_clinical.csv"
LABELS_CSV = "patient_alzheimers_targets.csv"

# Hyperparameters (UPDATED based on analysis)
BATCH_SIZE = 32
LR = 1e-4           # Lowered from 1e-3 to reduce instability/spikes
EPOCHS = 50         # Increased max epochs since we have early stopping now
HIDDEN_DIM = 128
DROPOUT = 0.5       # Increased from 0.3 to 0.5 to reduce overfitting
PATIENCE = 5        # Stop if val loss doesn't improve for 5 epochs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 2. Data Alignment & Loading
# ==========================================
def load_aligned_data():
    print("--- Loading Data ---")
    
    # A. Load Embeddings (X)
    if not os.path.exists(EMBEDDING_FILE):
        raise FileNotFoundError(f"Missing {EMBEDDING_FILE}")
    
    # Shape: [Num_Total_Patients, Embedding_Dim]
    all_embeddings = torch.load(EMBEDDING_FILE, map_location="cpu", weights_only=False)
    print(f"Loaded {len(all_embeddings)} embeddings.")

    # B. Reconstruct Patient IDs for those embeddings
    if not os.path.exists(SOURCE_CSV):
        raise FileNotFoundError(f"Missing {SOURCE_CSV}")
        
    print(f"Reading {SOURCE_CSV} to reconstruct patient order...")
    df_source = pd.read_csv(SOURCE_CSV, low_memory=False)
    
    if "patient_id" not in df_source.columns:
        raise ValueError("Source CSV missing 'patient_id' column")
        
    embedding_pids = sorted(df_source["patient_id"].dropna().unique())
    
    if len(embedding_pids) != len(all_embeddings):
        print(f"⚠️  WARNING: Embeddings count ({len(all_embeddings)}) != Source Patient count ({len(embedding_pids)})")
        min_len = min(len(embedding_pids), len(all_embeddings))
        embedding_pids = embedding_pids[:min_len]
        all_embeddings = all_embeddings[:min_len]

    # C. Load Labels (y)
    if not os.path.exists(LABELS_CSV):
        raise FileNotFoundError(f"Missing {LABELS_CSV}")

    print(f"Reading labels from {LABELS_CSV}...")
    df_labels = pd.read_csv(LABELS_CSV)
    
    label_map = dict(zip(df_labels.patient_id, df_labels.alzheimers_label))
    
    # D. Align X and y
    valid_indices = []
    valid_labels = []
    
    print("Aligning embeddings with labels...")
    for idx, pid in enumerate(embedding_pids):
        if pid in label_map:
            label_val = label_map[pid]
            try:
                val = float(label_val)
                valid_indices.append(idx)
                valid_labels.append(val)
            except:
                continue
                
    if len(valid_indices) == 0:
        raise ValueError("No overlapping patients found between embeddings and labels!")

    X = all_embeddings[valid_indices]
    y = torch.tensor(valid_labels).float().unsqueeze(1) # [N, 1]
    
    print(f"Final Dataset: {len(X)} patients (filtered from {len(all_embeddings)})")
    return X.to(device), y.to(device)

# ==========================================
# 3. Model Definition
# ==========================================
class AlzheimerPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        return self.net(x)

# ==========================================
# 4. Training Loop
# ==========================================
def train():
    try:
        X, y = load_aligned_data()
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        return

    # Split
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = AlzheimerPredictor(X.shape[1], HIDDEN_DIM).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3) # Added stronger weight decay
    
    train_hist = []
    val_hist = []
    
    # Early Stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    print(f"\nStarting training (Max Epochs: {EPOCHS}, Patience: {PATIENCE})...")
    
    for epoch in range(EPOCHS):
        model.train()
        batch_losses = []
        for bx, by in train_loader:
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            
        train_loss = np.mean(batch_losses)
        train_hist.append(train_loss)
        
        # Validation
        model.eval()
        val_losses = []
        correct = 0
        total = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vpreds = model(vx)
                vloss = criterion(vpreds, vy)
                val_losses.append(vloss.item())
                
                probs = torch.sigmoid(vpreds)
                predicted = (probs > 0.5).float()
                correct += (predicted == vy).sum().item()
                total += vy.size(0)
                
        val_loss = np.mean(val_losses)
        val_hist.append(val_loss)
        acc = correct / total if total > 0 else 0
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.2%}")
        
        # --- Early Stopping Logic ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0  # Reset counter
            # Save the best model immediately
            torch.save(model.state_dict(), "best_alzheimers_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping triggered! No improvement for {PATIENCE} epochs.")
                break

    # Restore best model weights
    model.load_state_dict(best_model_wts)
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print("Best model saved to 'best_alzheimers_model.pth'")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_hist, label='Train Loss')
    plt.plot(val_hist, label='Val Loss')
    plt.axvline(x=len(train_hist)-PATIENCE, color='r', linestyle='--', label='Best Model')
    plt.title("Alzheimer's Prediction (with Early Stopping)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("alzheimers_training_curve.png")
    print("Plot saved to alzheimers_training_curve.png")

if __name__ == "__main__":
    train()