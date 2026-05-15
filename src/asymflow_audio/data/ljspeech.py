"""LJSpeech dataset — raw waveform and mel-spectrogram variants.

LJSpeech: 13,100 clips, ~24 hours, single speaker (Linda Johnson), CC0 license.
Native: 22.05kHz. We resample to 16kHz for consistency with SC09 and HiFi-GAN.

Mel variant returns 2-second crops as flattened mel tokens for the 1D DiT,
matching the SC09 mel-spec format (mel_bins * TIME_FRAMES = 640-dim per token).
"""
import random
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as F
from torch.utils.data import Dataset, DataLoader
from einops import rearrange

from .melspec import MelSpecExtractor, MelNormalizer, MEL_CFG


SAMPLE_RATE = 16000
CROP_SECONDS = 2.0
CROP_LENGTH = int(SAMPLE_RATE * CROP_SECONDS)   # 32000 samples
TIME_FRAMES = 8


class LJSpeechDataset(Dataset):
    """
    Raw waveform crops from LJSpeech (2s @ 16kHz = 32000 samples).
    root should point to LJSpeech-1.1/ directory containing wavs/.
    """

    def __init__(self, root: str, split: str = "train", val_fraction: float = 0.02,
                 seed: int = 42):
        super().__init__()
        root = Path(root)
        wav_dir = root / "wavs"
        if not wav_dir.exists():
            raise FileNotFoundError(f"Missing: {wav_dir}. Run scripts/download_ljspeech.sh first.")

        files = sorted(wav_dir.glob("*.wav"))
        rng = random.Random(seed)
        rng.shuffle(files)
        n_val = max(1, int(len(files) * val_fraction))
        self.files = files[n_val:] if split == "train" else files[:n_val]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]
        wav, sr = torchaudio.load(path)
        wav = wav.mean(0)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

        # Random crop
        if wav.shape[0] >= CROP_LENGTH:
            start = random.randint(0, wav.shape[0] - CROP_LENGTH)
            wav = wav[start: start + CROP_LENGTH]
        else:
            wav = torch.nn.functional.pad(wav, (0, CROP_LENGTH - wav.shape[0]))

        peak = wav.abs().max().clamp(min=1e-6)
        return wav / peak  # (32000,)


class LJSpeechMelDataset(Dataset):
    """
    Mel-spectrogram tokens from LJSpeech 2-second crops.
    Returns (n_tokens, 640) — same format as SC09MelDataset.
    """

    def __init__(self, root: str, split: str = "train", val_fraction: float = 0.02,
                 seed: int = 42, stats_path: str = None):
        super().__init__()
        self._raw = LJSpeechDataset(root, split, val_fraction, seed)
        self.extractor = MelSpecExtractor(**MEL_CFG)

        if stats_path and Path(stats_path).exists():
            self.norm = MelNormalizer.load(stats_path)
        else:
            self.norm = None

    def fit_stats(self, n_samples: int = 3000, save_path: str = None) -> MelNormalizer:
        vals = []
        for i in range(min(n_samples, len(self))):
            wav = self._raw[i]
            mel = self.extractor(wav)
            vals.append((mel.mean().item(), mel.std().item()))
        mean = sum(v[0] for v in vals) / len(vals)
        std = (sum(v[1]**2 for v in vals) / len(vals)) ** 0.5
        self.norm = MelNormalizer(mean, std)
        if save_path:
            self.norm.save(save_path)
        return self.norm

    def __len__(self) -> int:
        return len(self._raw)

    def __getitem__(self, idx: int) -> torch.Tensor:
        wav = self._raw[idx]
        mel = self.extractor(wav)  # (80, ~200)
        if self.norm is not None:
            mel = self.norm.normalize(mel)

        mel_bins, T = mel.shape
        T_trim = (T // TIME_FRAMES) * TIME_FRAMES
        mel = mel[:, :T_trim]
        tokens = rearrange(mel, 'm (n t) -> n (m t)', t=TIME_FRAMES)
        return tokens  # (n_tokens, 640)


def build_lj_loaders(cfg, num_workers: int = 4):
    """Build train/val DataLoaders for LJSpeech mel-spec."""
    stats_path = "data/ljspeech_mel_stats.pt"
    train_ds = LJSpeechMelDataset(cfg.data.root, split="train", seed=cfg.train.seed,
                                   stats_path=stats_path)
    if train_ds.norm is None:
        print("Computing mel normalizer stats (LJSpeech)...")
        train_ds.fit_stats(n_samples=3000, save_path=stats_path)

    val_ds = LJSpeechMelDataset(cfg.data.root, split="val", seed=cfg.train.seed,
                                 stats_path=stats_path)
    val_ds.norm = train_ds.norm

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True,
                               persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
