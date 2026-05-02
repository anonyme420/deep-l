"""
Load raw ICBHI .wav/.txt pairs, extract annotated respiratory cycles,
convert to mel-spectrograms, and split by patient ID.

Split priority:
  1. ICBHI_train_test_list.txt in data_dir (official challenge split)
  2. Deterministic sorted-patient fallback (saves a generated list for reproducibility)
"""

import os
import glob
import numpy as np
import librosa
from collections import defaultdict
from tqdm import tqdm

from src.config import (
    DATA_DIR, SAMPLE_RATE, DURATION, N_MELS, HOP_LENGTH,
    N_FFT, TRAIN_RATIO, SEED
)

# label encoding: (crackle, wheeze) → int
_LABEL_MAP = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}

SPLIT_FILENAME = "ICBHI_train_test_list.txt"


def _load_single_file(wav_path: str):
    """Return list of (audio_np, label) for one recording file."""
    txt_path = wav_path.replace(".wav", ".txt")
    if not os.path.exists(txt_path):
        return []

    try:
        audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    except Exception:
        return []

    target_len = DURATION * SAMPLE_RATE
    cycles = []

    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                t_start = float(parts[0])
                t_end   = float(parts[1])
                crackle  = int(parts[2])
                wheeze   = int(parts[3])
            except ValueError:
                continue

            label   = _LABEL_MAP.get((crackle, wheeze), 0)
            s       = int(t_start * SAMPLE_RATE)
            e       = int(t_end   * SAMPLE_RATE)
            segment = audio[s:e]

            if len(segment) == 0:
                continue

            # Pad or truncate to fixed length
            if len(segment) < target_len:
                segment = np.pad(segment, (0, target_len - len(segment)))
            else:
                segment = segment[:target_len]

            cycles.append((segment.astype(np.float32), label))

    return cycles


def load_all_cycles(data_dir: str = DATA_DIR):
    """Load every cycle from the dataset.

    Returns:
        cycles        : list of (audio_np, label)
        patient_ids   : list of patient ID strings (one per cycle)
        recording_ids : list of recording stem strings (one per cycle)
    """
    wav_files = sorted(glob.glob(os.path.join(data_dir, "*.wav")))
    all_cycles       = []
    all_patient_ids  = []
    all_recording_ids = []

    print(f"Loading {len(wav_files)} recordings from {data_dir} ...")
    for wav_path in tqdm(wav_files, unit="file"):
        stem       = os.path.splitext(os.path.basename(wav_path))[0]
        patient_id = stem.split("_")[0]
        cycles     = _load_single_file(wav_path)
        for cycle in cycles:
            all_cycles.append(cycle)
            all_patient_ids.append(patient_id)
            all_recording_ids.append(stem)

    print(f"Loaded {len(all_cycles)} cycles total.")
    _print_distribution(all_cycles)
    return all_cycles, all_patient_ids, all_recording_ids


# ── Official / deterministic split ────────────────────────────────────────────

def _load_official_split(data_dir: str):
    """
    Try to load ICBHI_train_test_list.txt.

    Expected format (one line per recording):
        101_1b1_Al_sc_Meditron train
        101_1b1_Pr_sc_Meditron test
        ...

    Returns dict {stem: "train" | "test"}, or None if file not found.
    """
    path = os.path.join(data_dir, SPLIT_FILENAME)
    if not os.path.exists(path):
        return None

    split_map = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                stem, split = parts
                split_map[stem] = split.lower()
    return split_map if split_map else None


def _create_deterministic_split(wav_files: list, train_ratio: float, data_dir: str):
    """
    Build a deterministic 60/40 patient-level split from sorted patient IDs
    and persist it as ICBHI_train_test_list.txt so every run is identical.
    """
    # Collect stems per patient (sorted numerically by patient ID)
    patient_to_stems = defaultdict(list)
    for wav_path in wav_files:
        stem       = os.path.splitext(os.path.basename(wav_path))[0]
        patient_id = stem.split("_")[0]
        patient_to_stems[patient_id].append(stem)

    # Sort patients by numeric ID for determinism
    sorted_patients = sorted(patient_to_stems.keys(), key=lambda x: int(x))
    split_idx       = int(len(sorted_patients) * train_ratio)
    train_patients  = set(sorted_patients[:split_idx])

    split_map = {}
    lines     = []
    for patient in sorted_patients:
        tag = "train" if patient in train_patients else "test"
        for stem in sorted(patient_to_stems[patient]):
            split_map[stem] = tag
            lines.append(f"{stem} {tag}\n")

    # Save for reproducibility
    save_path = os.path.join(data_dir, SPLIT_FILENAME)
    with open(save_path, "w") as f:
        f.writelines(lines)
    print(f"  Saved deterministic split → {save_path}")

    return split_map


def split_by_patient(
    cycles,
    patient_ids,
    recording_ids=None,
    train_ratio: float = TRAIN_RATIO,
    data_dir: str = DATA_DIR,
    seed: int = SEED,
):
    """
    Split cycles into train/test sets.

    Uses the official ICBHI_train_test_list.txt when present; otherwise
    creates a deterministic sorted-patient split and saves it so the result
    is reproducible and comparable across runs.

    recording_ids is required for the official/deterministic file-level split.
    Falls back to a random patient split when recording_ids is not provided.
    """
    # ── Option 1: official or previously-generated file split ─────────────────
    if recording_ids is not None:
        split_map = _load_official_split(data_dir)

        if split_map is None:
            print(f"\n[INFO] {SPLIT_FILENAME} not found — generating deterministic split.")
            wav_files = sorted(glob.glob(os.path.join(data_dir, "*.wav")))
            split_map = _create_deterministic_split(wav_files, train_ratio, data_dir)
        else:
            print(f"\n[INFO] Using official split from {SPLIT_FILENAME}")

        train_cycles = [c for c, rid in zip(cycles, recording_ids)
                        if split_map.get(rid, "train") == "train"]
        test_cycles  = [c for c, rid in zip(cycles, recording_ids)
                        if split_map.get(rid, "train") == "test"]

        n_train_pat = len({pid for pid, rid in zip(patient_ids, recording_ids)
                           if split_map.get(rid, "train") == "train"})
        n_test_pat  = len({pid for pid, rid in zip(patient_ids, recording_ids)
                           if split_map.get(rid, "train") == "test"})

    # ── Option 2: random patient split (legacy fallback) ──────────────────────
    else:
        print("\n[WARN] recording_ids not provided — using random patient split.")
        np.random.seed(seed)
        unique_patients = sorted(set(patient_ids))
        np.random.shuffle(unique_patients)

        split_idx      = int(len(unique_patients) * train_ratio)
        train_patients = set(unique_patients[:split_idx])
        test_patients  = set(unique_patients[split_idx:])

        train_cycles = [c for c, pid in zip(cycles, patient_ids) if pid in train_patients]
        test_cycles  = [c for c, pid in zip(cycles, patient_ids) if pid in test_patients]
        n_train_pat  = len(train_patients)
        n_test_pat   = len(test_patients)

    print(f"\nSplit: {n_train_pat} train patients / {n_test_pat} test patients")
    print(f"Train cycles: {len(train_cycles)}")
    print(f"Test  cycles: {len(test_cycles)}")
    _print_distribution(train_cycles, label="Train")
    _print_distribution(test_cycles,  label="Test")

    return train_cycles, test_cycles


def to_melspec(audio: np.ndarray) -> np.ndarray:
    """Convert a 1-D audio array to a log mel-spectrogram in dB (float32).

    Returns raw dB values (not normalized to [0,1]) so each model can apply
    its own normalization: AST uses [0,1]+ImageNet stats, PaSST uses AudioSet stats.
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)  # shape (N_MELS, T), range approx [-80, 0] dB


def _print_distribution(cycles, label: str = "Dataset"):
    from src.config import CLASS_NAMES
    counts = defaultdict(int)
    for _, lbl in cycles:
        counts[lbl] += 1
    total = max(len(cycles), 1)
    print(f"\n{label} distribution ({total} cycles):")
    for i, name in enumerate(CLASS_NAMES):
        n = counts[i]
        print(f"  {name:8s}: {n:5d}  ({n/total*100:.1f}%)")
