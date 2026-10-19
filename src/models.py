"""Model zoo: three tabular learners and one recurrent sequence model."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from config import SEED


def tabular_models(n_classes: int) -> dict:
    return {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced",
                                       random_state=SEED)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=4, random_state=SEED),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=500, max_depth=6, learning_rate=0.06, l2_regularization=1.5,
            max_leaf_nodes=31, early_stopping=True, validation_fraction=0.12,
            n_iter_no_change=25, class_weight="balanced", random_state=SEED),
    }


class BiLSTM(nn.Module):
    def __init__(self, n_features: int = 3, hidden: int = 64, layers: int = 1,
                 n_classes: int = 2):
        super().__init__()
        self.rnn = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                           bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 64), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(64, n_classes))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out.max(dim=1).values)


def train_bilstm(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
                 y_val: np.ndarray, n_classes: int = 2, epochs: int = 60,
                 seed: int = SEED) -> tuple[BiLSTM, list[dict]]:
    torch.manual_seed(seed)
    model = BiLSTM(n_features=x_train.shape[2], n_classes=n_classes)
    mean = x_train.reshape(-1, x_train.shape[2]).mean(axis=0)
    std = x_train.reshape(-1, x_train.shape[2]).std(axis=0) + 1e-6
    xt = torch.tensor((x_train - mean) / std, dtype=torch.float32)
    xv = torch.tensor((x_val - mean) / std, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    yv = torch.tensor(y_val, dtype=torch.long)
    counts = np.bincount(y_train, minlength=n_classes).astype(float)
    weights = torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1)),
                           dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    history = []
    best_state, best_loss = None, float("inf")
    batch = 64
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(xt.shape[0])
        running = 0.0
        for start in range(0, xt.shape[0], batch):
            idx = order[start:start + batch]
            optimiser.zero_grad()
            loss = criterion(model(xt[idx]), yt[idx])
            loss.backward()
            optimiser.step()
            running += loss.detach().item() * idx.numel()
        model.eval()
        with torch.no_grad():
            logits = model(xv)
            val_loss = float(criterion(logits, yv))
            val_acc = float((logits.argmax(1) == yv).float().mean())
        history.append({"epoch": epoch + 1, "train_loss": running / xt.shape[0],
                        "val_loss": val_loss, "val_accuracy": val_acc})
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    model.norm = (mean, std)
    return model, history


def bilstm_scores(model: BiLSTM, x: np.ndarray) -> np.ndarray:
    mean, std = model.norm
    with torch.no_grad():
        logits = model(torch.tensor((x - mean) / std, dtype=torch.float32))
        return torch.softmax(logits, dim=1).numpy()
