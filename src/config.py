import os
import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "ICBHI_final_database")
RUNS_DIR  = os.path.join(BASE_DIR, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

# ── Audio ──────────────────────────────────────────────────────────────────────
SAMPLE_RATE  = 32000      # PaSST native rate (was 22050)
DURATION     = 5          # seconds — each cycle padded/truncated to this length
N_MELS       = 128
HOP_LENGTH   = 320        # 10 ms at 32 kHz — matches PaSST AudioSet pretraining
N_FFT        = 1024

# ── Image (mel → image for ViT) ────────────────────────────────────────────────
IMG_SIZE = 224

# ── Classes ────────────────────────────────────────────────────────────────────
NUM_CLASSES  = 4
CLASS_NAMES  = ["Normal", "Crackle", "Wheeze", "Both"]

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE        = 8     # reduced from 16; 6 GB GPU OOMs at backbone unfreeze with bs=16
EPOCHS            = 60
LR                = 5e-5          # lower LR → slower but more stable learning
WEIGHT_DECAY      = 2e-2          # increased from 1e-2; train/test loss gap shows overfitting
FOCAL_GAMMA       = 3.0    # increased from 2.0; harder focus on minority class errors
TARGET_PER_CLASS  = 900    # oversampling target per class; 1800 caused severe overfitting on rare classes
FREEZE_EPOCHS     = 5             # freeze backbone for first N epochs, train head only

# ── Model ──────────────────────────────────────────────────────────────────────
# Options: "ast" (ViT-based Audio Spectrogram Transformer)
#          "efficientnet" (EfficientNet-B3)
#          "ensemble" (both combined at inference)
MODEL_NAME = "passt"

# ── Split ──────────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.6   # 60% patients → train, 40% → test (official ICBHI split)

# ── Hardware ───────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
