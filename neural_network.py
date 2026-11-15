"""PyTorch tabular neural network for binary diabetes prediction.

An inner holdout from TRAINING data is used only for early stopping. Neither the
external test set nor the unseen validation set influences network training.
"""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class DiabetesMLP(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)  # logits, not probabilities


@dataclass
class NeuralPredictor:
    model: DiabetesMLP
    device: torch.device

    def positive_probability(self, values):
        array = np.asarray(values, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(array).to(self.device)
            return torch.sigmoid(self.model(x)).cpu().numpy().ravel()


def train_neural_network(
    X_train,
    y_train,
    *,
    dropout=0.0,
    weighted=False,
    max_epochs=100,
    patience=12,
    seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X = np.asarray(X_train, dtype=np.float32)
    y = np.asarray(y_train, dtype=np.float32)
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError('Expected a binary 0/1 target')

    # Inner holdout comes ONLY from the 70% training split.
    train_idx, tune_idx = train_test_split(
        np.arange(len(y)), test_size=0.15, stratify=y.astype(int), random_state=seed
    )
    X_fit, X_tune = X[train_idx], X[tune_idx]
    y_fit, y_tune = y[train_idx], y[tune_idx]

    model = DiabetesMLP(input_dim=X.shape[1], dropout=dropout).to(device)
    positives = int(y_fit.sum())
    negatives = len(y_fit) - positives
    if positives == 0 or negatives == 0:
        raise ValueError('Training fold needs examples from both classes')

    pos_weight = torch.tensor([negatives / positives if weighted else 1.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    ds = TensorDataset(torch.from_numpy(X_fit), torch.from_numpy(y_fit))
    generator = torch.Generator().manual_seed(seed)
    batches = DataLoader(ds, batch_size=128, shuffle=True, generator=generator)
    tune_x = torch.from_numpy(X_tune).to(device)
    tune_y = torch.from_numpy(y_tune).to(device)

    best_loss = float('inf')
    best_weights = None
    best_epoch = 0
    since_improvement = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in batches:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            tune_loss = float(criterion(model(tune_x), tune_y).item())
        history.append({'epoch': epoch, 'internal_tune_loss': tune_loss})

        if tune_loss < best_loss - 1e-5:
            best_loss = tune_loss
            best_weights = deepcopy(model.state_dict())
            best_epoch = epoch
            since_improvement = 0
        else:
            since_improvement += 1
            if since_improvement >= patience:
                break

    model.load_state_dict(best_weights)
    model.eval()
    info = {
        'best_epoch': best_epoch,
        'epochs_run': len(history),
        'best_internal_tune_loss': best_loss,
        'inner_train_rows': len(train_idx),
        'inner_tune_rows': len(tune_idx),
        'class_weighting': weighted,
        'positive_class_weight': float(pos_weight.item()),
        'dropout': dropout,
    }
    return NeuralPredictor(model, device), info, history
