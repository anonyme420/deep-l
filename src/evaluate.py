"""
Evaluation utilities:
  - Per-class precision, recall, F1
  - ICBHI score: (Se + Sp) / 2
    Se = macro recall over abnormal classes (crackle, wheeze, both)
    Sp = recall of the normal class
  - Confusion matrix plot
  - Training history plot
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import CLASS_NAMES, RUNS_DIR
from src.losses import FocalLoss, compute_class_weights


# ── Core evaluation ────────────────────────────────────────────────────────────

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    train_cycles: list = None,
    verbose: bool = True,
) -> dict:
    """
    Run inference on the full loader and compute all metrics.

    Returns a dict with keys:
        loss, accuracy, macro_recall, icbhi,
        per_class_recall, per_class_precision, per_class_f1,
        all_preds, all_labels
    """
    model.eval()

    # Build loss function (no alpha if train_cycles not supplied)
    if train_cycles is not None:
        alpha = compute_class_weights(train_cycles).to(device)
    else:
        alpha = None
    criterion = FocalLoss(alpha=alpha, gamma=2.0)

    all_preds, all_labels = [], []
    total_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for mels, labels in loader:
            mels   = mels.to(device)
            labels = labels.to(device)
            logits = model(mels)

            loss        = criterion(logits, labels)
            total_loss += loss.item()
            n_batches  += 1

            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    avg_loss   = total_loss / max(n_batches, 1)

    # ── Per-class metrics ──────────────────────────────────────────────────────
    report = classification_report(
        all_labels, all_preds,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    per_class_recall    = [report[n]["recall"]    for n in CLASS_NAMES]
    per_class_precision = [report[n]["precision"] for n in CLASS_NAMES]
    per_class_f1        = [report[n]["f1-score"]  for n in CLASS_NAMES]

    accuracy     = report["accuracy"]
    macro_recall = report["macro avg"]["recall"]

    metrics = {
        "loss":               avg_loss,
        "accuracy":           accuracy,
        "macro_recall":       macro_recall,
        "per_class_recall":   per_class_recall,
        "per_class_precision":per_class_precision,
        "per_class_f1":       per_class_f1,
        "all_preds":          all_preds,
        "all_labels":         all_labels,
    }
    metrics["icbhi"] = icbhi_score(metrics)

    if verbose:
        _print_metrics(metrics, avg_loss)

    return metrics


def icbhi_score(metrics: dict) -> float:
    """
    Official ICBHI metric: (Se + Sp) / 2
      Se = average recall of abnormal classes (crackle=1, wheeze=2, both=3)
      Sp = recall of normal class (0)
    """
    recalls = metrics["per_class_recall"]
    sp = recalls[0]                              # normal recall = specificity
    se = np.mean([recalls[1], recalls[2], recalls[3]])  # abnormal recall = sensitivity
    return (se + sp) / 2.0


# ── Threshold tuning (post-training recall boost) ─────────────────────────────

def tune_thresholds(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    """
    Find per-class probability thresholds that maximise macro recall.
    Returns thresholds array of shape (num_classes,).
    Applied at inference: predict = argmax(prob / thresholds).
    """
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for mels, labels in loader:
            logits = model(mels.to(device))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_probs  = np.array(all_probs)   # (N, 4)
    all_labels = np.array(all_labels)

    best_thresholds = np.ones(len(CLASS_NAMES))
    best_macro      = 0.0

    # Grid search on threshold scale factors
    search_vals = np.arange(0.5, 2.01, 0.1)
    for t1 in search_vals:      # crackle
        for t2 in search_vals:  # wheeze
            for t3 in search_vals:  # both
                t = np.array([1.0, t1, t2, t3])
                scaled = all_probs / t
                preds  = scaled.argmax(axis=1)
                report = classification_report(
                    all_labels, preds,
                    labels=list(range(len(CLASS_NAMES))),
                    target_names=CLASS_NAMES,
                    output_dict=True,
                    zero_division=0,
                )
                macro = report["macro avg"]["recall"]
                if macro > best_macro:
                    best_macro      = macro
                    best_thresholds = t.copy()

    print(f"\nBest thresholds: {best_thresholds}  →  macro recall={best_macro:.4f}")
    return best_thresholds


def predict_with_thresholds(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Return predictions using tuned thresholds."""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for mels, _ in loader:
            logits = model(mels.to(device))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            scaled = probs / thresholds
            preds  = scaled.argmax(axis=1)
            all_preds.extend(preds)
    return np.array(all_preds)


# ── Visualisation ──────────────────────────────────────────────────────────────

def plot_confusion_matrix(metrics: dict, save_path: str = None):
    cm = confusion_matrix(metrics["all_labels"], metrics["all_preds"])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Confusion Matrix (counts)", "Confusion Matrix (normalised)"],
        ["d", ".2f"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)

    plt.tight_layout()
    path = save_path or os.path.join(RUNS_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {path}")


def plot_history(history: dict, save_path: str = None):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["test_loss"],  label="Test")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["icbhi"])
    axes[1].set_title("ICBHI Score  (Se+Sp)/2")
    axes[1].set_xlabel("Epoch")

    axes[2].plot(epochs, history["macro_recall"])
    axes[2].set_title("Macro Recall")
    axes[2].set_xlabel("Epoch")

    plt.tight_layout()
    path = save_path or os.path.join(RUNS_DIR, "training_history.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Training history saved to {path}")


# ── Console printer ────────────────────────────────────────────────────────────

def _print_metrics(metrics: dict, loss: float):
    print("\n" + "=" * 60)
    print(f"  Loss     : {loss:.4f}")
    print(f"  Accuracy : {metrics['accuracy']*100:.2f}%")
    print(f"  ICBHI    : {metrics['icbhi']:.4f}  (Se={np.mean(metrics['per_class_recall'][1:]):.4f}  Sp={metrics['per_class_recall'][0]:.4f})")
    print(f"  Macro Recall: {metrics['macro_recall']:.4f}")
    print("-" * 60)
    print(f"  {'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    for i, name in enumerate(CLASS_NAMES):
        print(
            f"  {name:<10} "
            f"{metrics['per_class_precision'][i]:>10.4f} "
            f"{metrics['per_class_recall'][i]:>10.4f} "
            f"{metrics['per_class_f1'][i]:>10.4f}"
        )
    print("=" * 60 + "\n")
