"""Train the AstroNet CNN on a built dataset and report metrics.

    python -m exoscout.ml.train --data data/trainset.npz --epochs 40

Saves the trained weights to models/astronet.pt and validation metrics
(accuracy, precision, recall, ROC-AUC) to models/metrics.json.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, f1_score)
from sklearn.model_selection import train_test_split

from exoscout.paths import MODEL_DIR, MODEL_PATH, METRICS_PATH, TRAINSET_PATH

from .astronet import AstroNet


def _load(data_path: str):
    d = np.load(data_path)
    return d["global_view"], d["local_view"], d["label"]


def train(data_path: str, epochs: int = 60, batch_size: int = 32, lr: float = 1e-3,
          seed: int = 42) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    g, l, y = _load(data_path)

    g_tr, g_va, l_tr, l_va, y_tr, y_va = train_test_split(
        g, l, y, test_size=0.2, random_state=seed, stratify=y)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AstroNet().to(dev)
    # Materialise LazyLinear before the optimizer sees the params.
    with torch.no_grad():
        model(torch.zeros(2, model.global_bins, device=dev),
              torch.zeros(2, model.local_bins, device=dev))

    # Class weighting in case the usable split is imbalanced.
    pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
                              dtype=torch.float32, device=dev)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def to_t(a):
        return torch.tensor(a, dtype=torch.float32, device=dev)

    gt, lt, yt = to_t(g_tr), to_t(l_tr), to_t(y_tr)
    n = gt.shape[0]

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            if idx.numel() < 2:      # BatchNorm needs >1 sample in train mode
                continue
            opt.zero_grad()
            logits = model(gt[idx], lt[idx])
            loss = loss_fn(logits, yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * idx.numel()
        if ep % 5 == 0 or ep == 1:
            print(f"  epoch {ep:3d}  train_loss={total / n:.4f}", flush=True)

    # Validation.
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(to_t(g_va), to_t(l_va))).cpu().numpy()
    pred = (probs >= 0.5).astype(int)
    metrics = {
        "n_train": int(n), "n_val": int(len(y_va)),
        "accuracy": round(float(accuracy_score(y_va, pred)), 3),
        "precision": round(float(precision_score(y_va, pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y_va, pred, zero_division=0)), 3),
        "f1": round(float(f1_score(y_va, pred, zero_division=0)), 3),
        "roc_auc": round(float(roc_auc_score(y_va, probs)), 3) if len(set(y_va)) > 1 else None,
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    with open(METRICS_PATH, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print("Validation metrics:", json.dumps(metrics))
    print(f"Saved model -> {MODEL_PATH}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=TRAINSET_PATH)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    train(args.data, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    main()
