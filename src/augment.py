"""
Augmentation strategies:

1. Label-Aware Concatenation (LCat) — proven ICBHI technique.
   Concatenates two cycles of the *same* class to create longer, richer samples.
   This doubles effective diversity without introducing label noise.

2. Patch-Mix — mixes spectrogram patches between different samples at the
   Dataset level. Used with Patch-Mix Contrastive Loss (see losses.py).

3. Standard audio augmentations: time stretch, pitch shift, noise, time shift.
4. SpecAugment on mel spectrograms.
"""

import random
import numpy as np
import librosa
from collections import defaultdict
from src.config import SAMPLE_RATE, DURATION, TARGET_PER_CLASS, CLASS_TARGETS, SEED


# ── Standard audio augmentations ─────────────────────────────────────────────

def time_stretch(audio: np.ndarray, rate: float = None) -> np.ndarray:
    if rate is None:
        rate = np.random.uniform(0.85, 1.15)
    return librosa.effects.time_stretch(audio, rate=rate)


def pitch_shift(audio: np.ndarray, steps: int = None) -> np.ndarray:
    if steps is None:
        steps = np.random.randint(-2, 3)
    return librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=steps)


def add_noise(audio: np.ndarray, snr_db: float = None) -> np.ndarray:
    if snr_db is None:
        snr_db = np.random.uniform(20, 40)
    signal_power = np.mean(audio ** 2) + 1e-10
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(audio)) * np.sqrt(noise_power)
    return audio + noise


def time_shift(audio: np.ndarray, max_shift: float = 0.15) -> np.ndarray:
    shift = int(np.random.uniform(-max_shift, max_shift) * len(audio))
    return np.roll(audio, shift)


def augment_audio(audio: np.ndarray) -> np.ndarray:
    """Apply 1–2 random augmentations."""
    target_len = DURATION * SAMPLE_RATE
    ops = [time_stretch, pitch_shift, add_noise, time_shift]
    chosen = random.sample(ops, k=random.randint(1, 2))
    for op in chosen:
        try:
            audio = op(audio)
        except Exception:
            pass
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]
    return audio.astype(np.float32)


# ── SpecAugment ───────────────────────────────────────────────────────────────

def spec_augment(
    mel: np.ndarray,
    freq_mask_param: int = 24,
    time_mask_param: int = 32,
    n_freq_masks: int = 2,
    n_time_masks: int = 2,
) -> np.ndarray:
    """Apply multiple frequency and time masks (SpecAugment)."""
    mel = mel.copy()
    H, W = mel.shape
    fill = float(mel.min())   # works for both raw-dB and [0,1] normalized mels

    for _ in range(n_freq_masks):
        if np.random.rand() < 0.5:
            f  = np.random.randint(0, freq_mask_param)
            f0 = np.random.randint(0, max(H - f, 1))
            mel[f0:f0 + f, :] = fill

    for _ in range(n_time_masks):
        if np.random.rand() < 0.5:
            t  = np.random.randint(0, time_mask_param)
            t0 = np.random.randint(0, max(W - t, 1))
            mel[:, t0:t0 + t] = fill

    return mel


# ── Label-Aware Concatenation ─────────────────────────────────────────────────

def label_aware_concat(audio1: np.ndarray, audio2: np.ndarray) -> np.ndarray:
    """
    Concatenate two audio clips of the *same* label then truncate/pad to DURATION.
    This is the key augmentation from the ICBHI literature — produces longer,
    more representative examples of each class without introducing label noise.
    """
    target_len = DURATION * SAMPLE_RATE
    combined   = np.concatenate([audio1, audio2])
    if len(combined) < target_len:
        combined = np.pad(combined, (0, target_len - len(combined)))
    else:
        # Random crop from the concatenation — more variety than always [:target]
        max_start = len(combined) - target_len
        start     = np.random.randint(0, max_start + 1)
        combined  = combined[start:start + target_len]
    return combined.astype(np.float32)


# ── Dataset balancing ─────────────────────────────────────────────────────────

def balance_dataset(
    cycles: list,
    target_per_class: int = TARGET_PER_CLASS,
    seed: int = SEED,
    use_lcat: bool = True,
    class_targets: dict = None,
) -> list:
    """
    Oversample minority classes using per-class targets (class_targets dict)
    or a flat target_per_class for all classes.
    Uses Label-Aware Concatenation (LCat) as the primary augmentation.
    """
    random.seed(seed)
    np.random.seed(seed)

    if class_targets is None:
        class_targets = CLASS_TARGETS

    by_class = defaultdict(list)
    for audio, label in cycles:
        by_class[label].append(audio)

    balanced = []
    for label, audios in by_class.items():
        target = class_targets.get(label, target_per_class)
        if len(audios) > target:
            sampled = random.sample(audios, target)
            balanced.extend((a, label) for a in sampled)
        else:
            balanced.extend((a, label) for a in audios)

    for label, audios in by_class.items():
        target  = class_targets.get(label, target_per_class)
        current = min(len(audios), target)
        needed  = target - current
        if needed <= 0:
            continue

        for i in range(needed):
            if use_lcat and len(audios) >= 2:
                a1, a2 = random.choices(audios, k=2)
                aug = label_aware_concat(a1.copy(), a2.copy())
                if random.random() < 0.5:
                    aug = augment_audio(aug)
            else:
                src = random.choice(audios)
                aug = augment_audio(src.copy())
            balanced.append((aug, label))

    random.shuffle(balanced)
    total_target = sum(class_targets.values())
    print(f"\nBalanced training set: {len(balanced)} cycles "
          f"(targets {class_targets}, LCat={'on' if use_lcat else 'off'})")
    _print_distribution(balanced)
    return balanced


def _print_distribution(cycles):
    from src.config import CLASS_NAMES
    counts = defaultdict(int)
    for _, lbl in cycles:
        counts[lbl] += 1
    total = max(len(cycles), 1)
    for i, name in enumerate(CLASS_NAMES):
        n = counts[i]
        print(f"  {name:8s}: {n:5d}  ({n/total*100:.1f}%)")
