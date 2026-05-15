"""SC09 mel-spectrogram dataset.

Returns normalized log-mel patches of shape (n_tokens, patch_dim) ready for the 1D DiT.

Patch layout: each clip produces ~100 mel frames (1s @ 16kHz, hop=160).
We split into non-overlapping chunks of `time_frames` columns:
  (80, 100) → floor(100/8) = 12 tokens of (80*8,) = 640-dim each.
"""
import os
import random
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from einops import rearrange

from .sc09 import DIGITS, TARGET_LENGTH, SAMPLE_RATE
from .melspec import MelSpecExtractor, MelNormalizer, MEL_CFG, compute_mel_stats


TIME_FRAMES = 8    # time frames per token (= 80ms at 16kHz, hop=160)


class SC09MelDataset(Dataset):
    """
    Returns flattened mel-spec tokens: (n_tokens, mel_bins * time_frames).
    n_tokens = floor(~100 / TIME_FRAMES) = 12.
    """

    def __init__(self, root: str, split: str = "train", val_fraction: float = 0.05,
                 seed: int = 42, stats_path: str = None):
        super().__init__()
        root = Path(root)
        self.extractor = MelSpecExtractor(**MEL_CFG)

        files = []
        for digit in DIGITS:
            d = root / digit
            if not d.exists():
                raise FileNotFoundError(f"Missing: {d}. Run scripts/download_sc09.sh first.")
            files.extend(sorted(d.glob("*.wav")))

        rng = random.Random(seed)
        rng.shuffle(files)
        n_val = max(1, int(len(files) * val_fraction))
        self.files = files[n_val:] if split == "train" else files[:n_val]

        # Load or compute normalizer
        if stats_path and Path(stats_path).exists():
            self.norm = MelNormalizer.load(stats_path)
        else:
            self.norm = None  # will normalize lazily or caller must call fit_stats()

    def fit_stats(self, n_samples: int = 3000, save_path: str = None):
        """Compute and cache global mean/std over training data."""
        vals = []
        for i in range(min(n_samples, len(self))):
            mel = self._raw_mel(i)
            vals.append((mel.mean().item(), mel.std().item()))
        mean = sum(v[0] for v in vals) / len(vals)
        std = (sum(v[1]**2 for v in vals) / len(vals)) ** 0.5
        self.norm = MelNormalizer(mean, std)
        if save_path:
            self.norm.save(save_path)
        return self.norm

    def _raw_mel(self, idx: int) -> torch.Tensor:
        path = self.files[idx]
        wav, sr = torchaudio.load(path)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        wav = wav.mean(0)
        if wav.shape[0] < TARGET_LENGTH:
            wav = torch.nn.functional.pad(wav, (0, TARGET_LENGTH - wav.shape[0]))
        else:
            wav = wav[:TARGET_LENGTH]
        peak = wav.abs().max().clamp(min=1e-6)
        wav = wav / peak
        return self.extractor(wav)  # (mel_bins, T)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        mel = self._raw_mel(idx)  # (80, ~100)
        if self.norm is not None:
            mel = self.norm.normalize(mel)

        mel_bins, T = mel.shape
        # Trim to multiple of TIME_FRAMES
        T_trim = (T // TIME_FRAMES) * TIME_FRAMES
        mel = mel[:, :T_trim]  # (80, T_trim)
        # → (n_tokens, mel_bins * time_frames)
        tokens = rearrange(mel, 'm (n t) -> n (m t)', t=TIME_FRAMES)
        return tokens  # (n_tokens, 640)


def build_mel_loaders(cfg, num_workers: int = 4):
    """Build train/val DataLoaders for SC09 mel-spec."""
    stats_path = f"data/sc09_mel_stats.pt"
    train_ds = SC09MelDataset(cfg.data.root, split="train", seed=cfg.train.seed,
                               stats_path=stats_path)
    if train_ds.norm is None:
        print("Computing mel normalizer stats (SC09)...")
        train_ds.fit_stats(n_samples=3000, save_path=stats_path)

    val_ds = SC09MelDataset(cfg.data.root, split="val", seed=cfg.train.seed,
                             stats_path=stats_path)
    val_ds.norm = train_ds.norm  # share normalizer

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True,
                               persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
