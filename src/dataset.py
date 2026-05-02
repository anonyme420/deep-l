"""
PyTorch Dataset with two preprocessing paths:

  AST  path  → (3, IMG_SIZE, IMG_SIZE) [0,1] + ImageNet normalisation
  PaSST path → (1, 128, 998) AudioSet mean/std normalisation, padded to pretrained T

Patch-Mix augmentation is applied at batch level in train.py for both paths
(patch_mix_batch handles any C×H×W shape).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.preprocess import to_melspec
from src.augment import spec_augment
from src.config import IMG_SIZE

# ── PaSST normalisation constants (computed from AudioSet) ───────────────────
_PASST_MEAN =  -4.2677393
_PASST_STD  =   4.5689974
_PASST_T    =   500        # our 5 s clip @ 32 kHz/320 hop; pos embed resized in model.py


class ICBHIDataset(Dataset):
    def __init__(self, cycles: list, training: bool = False, use_passt: bool = False):
        self.cycles    = cycles
        self.training  = training
        self.use_passt = use_passt

    def __len__(self):
        return len(self.cycles)

    def __getitem__(self, idx):
        audio, label = self.cycles[idx]
        mel = to_melspec(audio)          # raw dB, shape (N_MELS, T)

        if self.training:
            mel = spec_augment(mel)

        if self.use_passt:
            x = _to_tensor_passt(mel)    # (1, 128, 998)
        else:
            x = _to_tensor_ast(mel)      # (3, IMG_SIZE, IMG_SIZE)

        return x, torch.tensor(label, dtype=torch.long)


# ── AST tensor ────────────────────────────────────────────────────────────────

def _to_tensor_ast(mel_db: np.ndarray) -> torch.Tensor:
    """(H, W) dB mel → [0,1] normalised → (3, IMG_SIZE, IMG_SIZE) ImageNet stats."""
    vmin, vmax = mel_db.min(), mel_db.max()
    mel_01 = (mel_db - vmin) / (vmax - vmin + 1e-8)

    mel_t = torch.from_numpy(mel_01).unsqueeze(0).repeat(3, 1, 1)
    mel_t = F.interpolate(
        mel_t.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear", align_corners=False,
    ).squeeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (mel_t - mean) / std


# ── PaSST tensor ──────────────────────────────────────────────────────────────

def _to_tensor_passt(mel_db: np.ndarray) -> torch.Tensor:
    """(H, W) dB mel → AudioSet-normalised (1, 128, 998) matching PaSST pretrain dims."""
    mel_norm = (mel_db - _PASST_MEAN) / _PASST_STD   # AudioSet normalisation

    H, W = mel_norm.shape
    if W < _PASST_T:
        # Pad time with 0 (= AudioSet mean after normalisation → neutral)
        mel_norm = np.pad(mel_norm, ((0, 0), (0, _PASST_T - W)),
                          mode="constant", constant_values=0.0)
    else:
        mel_norm = mel_norm[:, :_PASST_T]

    return torch.from_numpy(mel_norm.copy()).unsqueeze(0).float()  # (1, 128, 998)


# ── Patch-Mix (batch-level) ───────────────────────────────────────────────────

def patch_mix_batch(
    mels:       torch.Tensor,   # (B, C, H, W) — works for C=1 (PaSST) or C=3 (AST)
    labels:     torch.Tensor,
    patch_size: int   = 16,
    mix_ratio:  float = None,
) -> tuple:
    """
    Apply Patch-Mix to a training batch.  Works for any C×H×W shape.

    Returns:
        mels_mix : (B, C, H, W)
        lam      : (B,)
        idx_b    : (B,)
    """
    B, C, H, W = mels.shape
    n_h = H // patch_size
    n_w = W // patch_size
    n_patches = n_h * n_w

    idx_b  = torch.randperm(B, device=mels.device)
    mels_b = mels[idx_b]

    mels_mix = mels.clone()
    lam_list = []

    for i in range(B):
        lam_i = float(np.random.uniform(0.1, 0.9)) if mix_ratio is None else mix_ratio
        n_replace = max(1, min(n_patches - 1, int(round((1.0 - lam_i) * n_patches))))

        patch_ids = torch.randperm(n_patches, device=mels.device)[:n_replace]
        for pid in patch_ids:
            row = (pid // n_w).item()
            col = (pid %  n_w).item()
            r0, r1 = row * patch_size, (row + 1) * patch_size
            c0, c1 = col * patch_size, (col + 1) * patch_size
            mels_mix[i, :, r0:r1, c0:c1] = mels_b[i, :, r0:r1, c0:c1]

        lam_list.append(1.0 - n_replace / n_patches)

    lam = torch.tensor(lam_list, dtype=torch.float32, device=mels.device)
    return mels_mix, lam, idx_b


# ── DataLoaders ───────────────────────────────────────────────────────────────

def get_loaders(train_cycles, test_cycles, batch_size: int, use_passt: bool = False):
    train_ds = ICBHIDataset(train_cycles, training=True,  use_passt=use_passt)
    test_ds  = ICBHIDataset(test_cycles,  training=False, use_passt=use_passt)

    # Per-sample weights → every class equally likely in each batch
    labels       = [label for _, label in train_cycles]
    class_counts = Counter(labels)
    total        = len(labels)
    sample_weights = torch.tensor(
        [total / class_counts[lbl] for lbl in labels], dtype=torch.float32
    )
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    return train_loader, test_loader
