"""
Training loop with:
  1. Two-phase optimizer:
       Phase 1 (frozen backbone): plain AdamW, head_lr=1e-3
       Phase 2 (full fine-tune):  SAM + AdamW, lr=5e-5
  2. Patch-Mix Contrastive Learning every N batches
  3. Focal Loss on standard batches
  4. Gradient clipping
  5. Verbose per-class recall each epoch
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.config import EPOCHS, LR, WEIGHT_DECAY, DEVICE, RUNS_DIR, FREEZE_EPOCHS
from src.sam     import SAM
from src.losses  import CombinedLoss
from src.dataset import patch_mix_batch
from src.evaluate import evaluate, icbhi_score


# How often to apply Patch-Mix (every N batches); rest are standard Focal batches
PATCH_MIX_EVERY = 2   # every 2nd batch uses Patch-Mix


def train(
    model,
    train_loader,
    test_loader,
    train_cycles: list,
    epochs:       int   = EPOCHS,
    lr:           float = LR,
    weight_decay: float = WEIGHT_DECAY,
    device:       str   = DEVICE,
    save_name:    str   = "best_model.pt",
    head_lr:      float = 1e-3,
    max_grad_norm: float = 1.0,
):
    model = model.to(device)
    has_proj = hasattr(model, "forward_with_proj")

    def set_backbone_grad(flag: bool):
        if hasattr(model, "backbone"):
            for p in model.backbone.parameters():
                p.requires_grad = flag

    # ── Phase 1: head-only, plain AdamW ───────────────────────────────────────
    set_backbone_grad(False)
    print(f"Phase 1 — backbone frozen ({FREEZE_EPOCHS} epochs), head LR={head_lr}")

    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer   = torch.optim.AdamW(head_params, lr=head_lr, weight_decay=weight_decay)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(FREEZE_EPOCHS, 1), eta_min=head_lr * 0.1
    )
    use_sam = False

    # ── Loss ──────────────────────────────────────────────────────────────────
    # Fixed balanced weights with gentle abnormal boost — don't use compute_class_weights
    # since we already balanced via per-class oversampling (CLASS_TARGETS)
    class_weights = torch.tensor([1.0, 1.0, 1.1, 2.0], device=device)
    criterion     = CombinedLoss(
        alpha=class_weights, gamma=3.0,
        temperature=0.07, w_focal=1.0, w_cl=1.0,
        label_smoothing=0.05,
    )

    best_score = -1.0
    best_path  = os.path.join(RUNS_DIR, save_name)
    history    = {"train_loss": [], "test_loss": [], "icbhi": [], "macro_recall": []}
    patience   = 12
    no_improve = 0

    print(f"\nTraining on {device} | {epochs} epochs | Patch-Mix CL: {'enabled' if has_proj else 'disabled'}\n")

    for epoch in range(1, epochs + 1):

        # ── Phase 2: unfreeze, warmup, then SAM ─────────────────────────────────
        if epoch == FREEZE_EPOCHS + 1:
            torch.cuda.empty_cache()
            set_backbone_grad(True)
            print(f"\nEpoch {epoch}: backbone unfrozen — plain AdamW warmup\n")
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr * 0.1, weight_decay=weight_decay
            )
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=5
            )
            use_sam = False   # ← stay False for warmup epochs

        if epoch == FREEZE_EPOCHS + 6:   # switch to SAM after 5 warmup epochs
            print(f"\nEpoch {epoch}: switching to SAM\n")
            optimizer = SAM(
                model.parameters(), torch.optim.AdamW,
                rho=0.05, lr=lr, weight_decay=weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer.base_optimizer,
                T_max=epochs - FREEZE_EPOCHS - 5,
                eta_min=lr * 0.01,
            )
            use_sam = True
        
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        n_batches  = 0
        t0 = time.time()

        for batch_idx, (mels, labels) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False)
        ):
            mels   = mels.to(device)
            labels = labels.to(device)

            # Only use Patch-Mix during SAM phase: backbone frozen / warmup phases
            # have no gradient flow to the backbone, so contrastive loss adds cost
            # with no benefit.  use_sam becomes True at epoch FREEZE_EPOCHS+6.
            # mels.dim()==4 check: BEATs uses (B,T) raw audio — PatchMix needs (B,C,H,W)
            use_patchmix = has_proj and use_sam and (batch_idx % PATCH_MIX_EVERY == 0) and mels.dim() == 4
            # Waveform Mixup for BEATs raw audio (dim==2) — same regularization benefit
            # as PatchMix but operates in the waveform domain
            use_wavmix   = use_sam and (batch_idx % PATCH_MIX_EVERY == 0) and mels.dim() == 2

            if use_patchmix:
                # ── Patch-Mix batch ───────────────────────────────────────────
                mels_mix, lam, idx_b = patch_mix_batch(mels, labels, patch_size=16)
                labels_b = labels[idx_b]

                # Embed all originals once (no grad) — z_all[i] = source-A embedding,
                # z_all[idx_b[i]] = source-B embedding. Used as InfoNCE negatives.
                with torch.no_grad():
                    _, z_all = model.forward_with_proj(mels)

                z_a = z_all
                z_b = z_all[idx_b]

                if use_sam:
                    # SAM step 1
                    logits_mix, z_mix = model.forward_with_proj(mels_mix)
                    loss = criterion(
                        logits_mix, labels, z_mix, z_a, z_b, lam,
                        labels_a=labels, labels_b=labels_b, z_all=z_all,
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.first_step(zero_grad=True)

                    # SAM step 2 — anchors are already detached (computed with no_grad)
                    logits_mix2, z_mix2 = model.forward_with_proj(mels_mix)
                    loss2 = criterion(
                        logits_mix2, labels, z_mix2, z_a, z_b, lam,
                        labels_a=labels, labels_b=labels_b, z_all=z_all,
                    )
                    loss2.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.second_step(zero_grad=True)
                else:
                    optimizer.zero_grad()
                    logits_mix, z_mix = model.forward_with_proj(mels_mix)
                    loss = criterion(
                        logits_mix, labels, z_mix, z_a, z_b, lam,
                        labels_a=labels, labels_b=labels_b, z_all=z_all,
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

            elif use_wavmix:
                # ── Waveform Mixup batch (BEATs raw audio) ────────────────────
                lam_v  = float(np.random.beta(0.4, 0.4))
                lam_v  = max(0.1, min(0.9, lam_v))
                idx_b  = torch.randperm(len(mels), device=device)
                mix    = lam_v * mels + (1.0 - lam_v) * mels[idx_b]
                lbl_b  = labels[idx_b]
                lam_t  = torch.full((len(labels),), lam_v, device=device)

                def _wm_loss(logits):
                    fa = criterion.focal_sample(logits, labels)
                    fb = criterion.focal_sample(logits, lbl_b)
                    return (lam_t * fa + (1.0 - lam_t) * fb).mean()

                # SAM step 1
                loss = _wm_loss(model(mix))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.first_step(zero_grad=True)

                # SAM step 2
                loss = _wm_loss(model(mix))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.second_step(zero_grad=True)

            else:
                # ── Standard Focal batch ──────────────────────────────────────
                if use_sam:
                    loss = criterion(model(mels), labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.first_step(zero_grad=True)

                    criterion(model(mels), labels).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.second_step(zero_grad=True)
                else:
                    optimizer.zero_grad()
                    loss = criterion(model(mels), labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # ── Evaluate ──────────────────────────────────────────────────────────
        metrics = evaluate(model, test_loader, device, verbose=False)
        score   = icbhi_score(metrics)
        pcr     = metrics["per_class_recall"]

        history["train_loss"].append(avg_loss)
        history["test_loss"].append(metrics["loss"])
        history["icbhi"].append(score)
        history["macro_recall"].append(metrics["macro_recall"])

        print(
            f"Epoch {epoch:03d} | train={avg_loss:.4f} | test={metrics['loss']:.4f} | "
            f"ICBHI={score:.4f} | Sp={pcr[0]:.3f} Se={sum(pcr[1:])/3:.3f} | "
            f"[N={pcr[0]:.3f} Cr={pcr[1]:.3f} Wh={pcr[2]:.3f} Bo={pcr[3]:.3f}] | "
            f"{time.time()-t0:.0f}s"
        )

        if score > best_score:
            best_score = score
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch, "model_state": model.state_dict(),
                    "score": best_score, "metrics": metrics,
                },
                best_path,
            )
            print(f"  ✓ New best  ICBHI={best_score:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping after {patience} non-improving epochs.")
                break

    print(f"\nDone. Best ICBHI={best_score:.4f}  →  {best_path}")
    return history
