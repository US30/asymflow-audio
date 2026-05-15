"""SC09 dataset: Speech Commands digits 0-9, 1s @ 16kHz."""
import os
import math
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split


DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
TARGET_LENGTH = 16000
SAMPLE_RATE = 16000


def mu_law_encode(x: torch.Tensor, mu: int = 255) -> torch.Tensor:
    """Map waveform in [-1,1] to [-1,1] via µ-law compression."""
    return x.sign() * torch.log1p(mu * x.abs()) / math.log(1 + mu)


def mu_law_decode(x: torch.Tensor, mu: int = 255) -> torch.Tensor:
    return x.sign() * ((1 + mu) ** x.abs() - 1) / mu


class SC09Dataset(Dataset):
    """
    Loads all .wav files under root/<digit>/ for each digit in DIGITS.
    Returns mu-law compressed waveforms in [-1, 1], shape (16000,).
    """

    def __init__(self, root: str, split: str = "train", val_fraction: float = 0.05,
                 seed: int = 42, mu_law: bool = True):
        super().__init__()
        root = Path(root)
        self.mu_law = mu_law

        files = []
        for digit in DIGITS:
            d = root / digit
            if not d.exists():
                raise FileNotFoundError(f"Missing digit dir: {d}. Run scripts/download_sc09.sh first.")
            files.extend(sorted(d.glob("*.wav")))

        rng = random.Random(seed)
        rng.shuffle(files)

        n_val = max(1, int(len(files) * val_fraction))
        if split == "train":
            self.files = files[n_val:]
        elif split in ("val", "test"):
            self.files = files[:n_val]
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]
        wav, sr = torchaudio.load(path)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        wav = wav.mean(0)  # mono

        # Pad or truncate to TARGET_LENGTH
        if wav.shape[0] < TARGET_LENGTH:
            wav = torch.nn.functional.pad(wav, (0, TARGET_LENGTH - wav.shape[0]))
        else:
            wav = wav[:TARGET_LENGTH]

        # Normalize to [-1, 1]
        peak = wav.abs().max().clamp(min=1e-6)
        wav = wav / peak

        if self.mu_law:
            wav = mu_law_encode(wav)

        return wav  # (16000,)


def build_loaders(cfg, num_workers: int = 4):
    train_ds = SC09Dataset(cfg.data.root, split="train", seed=cfg.train.seed)
    val_ds = SC09Dataset(cfg.data.root, split="val", seed=cfg.train.seed)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader
