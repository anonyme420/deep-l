"""
Loss functions:

1. FocalLoss              — down-weights easy examples; supports per-sample
                            reduction for soft-label mixing.
2. PatchMixContrastiveLoss — Patch-Mix Contrastive Loss (INTERSPEECH 2023).
                            Uses the full batch as negatives (InfoNCE denominator)
                            instead of the original 2-way softmax, giving a
                            stronger contrastive signal with small batch sizes.
3. CombinedLoss            — Focal + PatchMix weighted sum.
                            In patch-mix mode uses lam-weighted per-sample focal
                            loss across both source labels instead of picking the
                            majority label, which was noisy near lam≈0.5.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import NUM_CLASSES, FOCAL_GAMMA


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)"""

    def __init__(
        self,
        alpha: torch.Tensor = None,
        gamma: float = FOCAL_GAMMA,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma           = gamma
        self.reduction       = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce   = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none",
                               label_smoothing=self.label_smoothing)
        pt   = torch.exp(-ce)
        loss = (1.0 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss   # reduction == "none" → (B,)


# ── Patch-Mix Contrastive Loss ────────────────────────────────────────────────

class PatchMixContrastiveLoss(nn.Module):
    """
    From: "Patch-Mix Contrastive Learning with Audio Spectrogram Transformer
           on Respiratory Sound Classification" (INTERSPEECH 2023).

    Extended to use the full batch as InfoNCE negatives instead of the
    original 2-way softmax. This gives B-1 negatives per anchor instead of 1,
    producing a stronger learning signal when batch size is small (16).

    Args:
        temperature : softmax temperature (default 0.07)
        reduction   : 'mean' | 'sum'
    """

    def __init__(self, temperature: float = 0.07, reduction: str = "mean"):
        super().__init__()
        self.temperature = temperature
        self.reduction   = reduction

    def forward(
        self,
        z_mix: torch.Tensor,            # (B, D)
        z_a:   torch.Tensor,            # (B, D) — embeddings of source A
        z_b:   torch.Tensor,            # (B, D) — embeddings of source B
        lam:   torch.Tensor,            # (B,)   — mixing ratio per sample
        z_all: torch.Tensor = None,     # (B, D) — full batch embeddings for InfoNCE denom
    ) -> torch.Tensor:
        z_mix = F.normalize(z_mix, dim=1)
        z_a   = F.normalize(z_a,   dim=1)
        z_b   = F.normalize(z_b,   dim=1)
        lam   = lam.to(z_mix.device).clamp(0.0, 1.0)

        sim_a = (z_mix * z_a).sum(dim=1) / self.temperature  # (B,)
        sim_b = (z_mix * z_b).sum(dim=1) / self.temperature  # (B,)

        if z_all is not None:
            # InfoNCE with full batch as negatives
            # denominator = logsumexp over all B original embeddings
            z_all_n  = F.normalize(z_all, dim=1)                        # (B, D)
            sim_full = torch.mm(z_mix, z_all_n.T) / self.temperature    # (B, B)
            log_Z    = torch.logsumexp(sim_full, dim=1)                  # (B,)
            loss = -(lam * (sim_a - log_Z) + (1.0 - lam) * (sim_b - log_Z))
        else:
            # Original 2-way softmax (fallback)
            log_softmax = torch.log_softmax(torch.stack([sim_a, sim_b], dim=1), dim=1)
            loss = -(lam * log_softmax[:, 0] + (1.0 - lam) * log_softmax[:, 1])

        return loss.mean() if self.reduction == "mean" else loss.sum()


# ── Combined Loss ─────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    Standard batches : w_focal * FocalLoss(logits, targets)

    Patch-Mix batches: w_focal * lam-weighted_focal + w_cl * PatchMixContrastiveLoss

    The lam-weighted focal avoids the hard majority-label assignment that was
    noisy when lam was near 0.5:
        focal = mean( lam * FL(logits, labels_a) + (1-lam) * FL(logits, labels_b) )
    """

    def __init__(
        self,
        alpha:           torch.Tensor = None,
        gamma:           float = FOCAL_GAMMA,
        temperature:     float = 0.07,
        w_focal:         float = 1.0,
        w_cl:            float = 0.5,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        # Keep two focal instances: one mean-reduced (standard), one per-sample (mix)
        self.focal_mean   = FocalLoss(alpha=alpha, gamma=gamma, reduction="mean",
                                      label_smoothing=label_smoothing)
        self.focal_sample = FocalLoss(alpha=alpha, gamma=gamma, reduction="none",
                                      label_smoothing=label_smoothing)
        self.cl_loss      = PatchMixContrastiveLoss(temperature=temperature)
        self.w_focal      = w_focal
        self.w_cl         = w_cl

    def forward(
        self,
        logits:   torch.Tensor,
        targets:  torch.Tensor,
        # Patch-Mix extras (all None during standard batches)
        z_mix:    torch.Tensor = None,
        z_a:      torch.Tensor = None,
        z_b:      torch.Tensor = None,
        lam:      torch.Tensor = None,
        labels_a: torch.Tensor = None,   # source-A labels (= targets for std batch)
        labels_b: torch.Tensor = None,   # source-B labels (donor sample labels)
        z_all:    torch.Tensor = None,   # full batch embeddings for InfoNCE
    ) -> torch.Tensor:

        if z_mix is not None and z_a is not None and z_b is not None and lam is not None:
            # Patch-Mix batch
            lam_dev = lam.to(logits.device)

            if labels_a is not None and labels_b is not None:
                # Soft focal: lam-weighted mix over both source labels (per sample)
                loss_a = self.focal_sample(logits, labels_a)   # (B,)
                loss_b = self.focal_sample(logits, labels_b)   # (B,)
                focal  = (lam_dev * loss_a + (1.0 - lam_dev) * loss_b).mean()
            else:
                focal = self.focal_mean(logits, targets)

            cl = self.cl_loss(z_mix, z_a, z_b, lam, z_all=z_all)
            return self.w_focal * focal + self.w_cl * cl

        # Standard batch
        return self.focal_mean(logits, targets)


# ── Class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(cycles: list, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Inverse-frequency weights, smoothed to avoid extreme values."""
    from collections import Counter
    counts = Counter(label for _, label in cycles)
    total  = sum(counts.values())

    weights = []
    for i in range(num_classes):
        freq = counts.get(i, 1) / total
        w    = (1.0 / (freq * num_classes))
        w    = max(0.5, min(4.0, w))
        weights.append(w)

    weights_tensor = torch.tensor(weights, dtype=torch.float32)

    print("\nClass weights (inverse-frequency):")
    from src.config import CLASS_NAMES
    for name, w in zip(CLASS_NAMES, weights):
        print(f"  {name:8s}: {w:.4f}")
    return weights_tensor
