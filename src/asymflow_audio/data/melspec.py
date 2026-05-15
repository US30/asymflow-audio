"""Mel-spectrogram extraction, normalization, and vocoder inversion.

Pipeline:
  waveform → MelSpec → log-compress → normalize → [model] → denorm → HiFi-GAN → waveform

HiFi-GAN UNIVERSAL_V1 is used for vocoder inference (pretrained, no training required).
Both SC09 (16kHz) and LJSpeech (22.05kHz → resampled 16kHz) use 16kHz as the common rate.
"""
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T


# Mel-spec config — shared across all datasets (16kHz baseline)
MEL_CFG = dict(
    sample_rate=16000,
    n_fft=512,
    hop_length=160,     # 10ms at 16kHz
    win_length=320,     # 20ms
    n_mels=80,
    f_min=0.0,
    f_max=8000.0,
)
LOG_EPS = 1e-5


class MelSpecExtractor(nn.Module):
    """Extract log-mel spectrogram from raw waveform. Returns (mel_bins, T)."""

    def __init__(self, **cfg):
        super().__init__()
        self.transform = T.MelSpectrogram(**cfg)

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (..., L) → mel: (..., mel_bins, T)"""
        mel = self.transform(wav)
        return torch.log(mel + LOG_EPS)


def compute_mel_stats(dataset, extractor: MelSpecExtractor,
                      n_samples: int = 5000) -> Tuple[float, float]:
    """Compute global mean/std over n_samples waveforms from dataset."""
    vals = []
    for i in range(min(n_samples, len(dataset))):
        wav = dataset[i]
        mel = extractor(wav)
        vals.append(mel.mean().item())
        vals.append(mel.std().item())
    mean = sum(vals[::2]) / len(vals[::2])
    std = (sum(v**2 for v in vals[1::2]) / len(vals[1::2])) ** 0.5
    return mean, std


class MelNormalizer:
    """Normalize / denormalize log-mel spectrograms."""

    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def normalize(self, mel: torch.Tensor) -> torch.Tensor:
        return (mel - self.mean) / (self.std + 1e-8)

    def denormalize(self, mel: torch.Tensor) -> torch.Tensor:
        return mel * self.std + self.mean

    def save(self, path: str):
        torch.save({"mean": self.mean, "std": self.std}, path)

    @classmethod
    def load(cls, path: str) -> "MelNormalizer":
        d = torch.load(path, weights_only=True)
        return cls(d["mean"], d["std"])


# ── HiFi-GAN vocoder ──────────────────────────────────────────────────────────

class HiFiGANVocoder(nn.Module):
    """
    Thin wrapper around pretrained HiFi-GAN for mel → waveform.

    Expects jik876/hifi-gan pretrained weights. Download via:
        bash scripts/download_hifigan.sh
    Checkpoint path: data/hifigan/generator_universal.pth
    Config path:     data/hifigan/config_v1.json
    """

    def __init__(self, checkpoint_path: str, config_path: str, device: str = "cpu"):
        super().__init__()
        import json, sys
        # hifi-gan is not on PyPI — use local clone in third_party/hifigan/
        hifigan_dir = Path(__file__).parent.parent.parent.parent / "third_party" / "hifigan"
        if str(hifigan_dir) not in sys.path:
            sys.path.insert(0, str(hifigan_dir))

        from models import Generator  # type: ignore
        from env import AttrDict       # type: ignore

        with open(config_path) as f:
            h = AttrDict(json.load(f))
        self.h = h

        gen = Generator(h)
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        gen.load_state_dict(state["generator"])
        gen.eval()
        gen.remove_weight_norm()
        self.gen = gen.to(device)

    @torch.no_grad()
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        mel: (B, mel_bins, T) normalized log-mel
        Returns: (B, L) waveform in [-1, 1]
        """
        return self.gen(mel).squeeze(1)

    @staticmethod
    def load_default(device: str = "cpu") -> "HiFiGANVocoder":
        base = Path("data/hifigan")
        return HiFiGANVocoder(
            str(base / "generator_universal.pth"),
            str(base / "config_v1.json"),
            device=device,
        )


def griffin_lim_invert(mel_log: torch.Tensor, sr: int = 16000,
                        n_fft: int = 512, hop: int = 160, n_iters: int = 32) -> torch.Tensor:
    """
    Fallback vocoder (no model weights). Lower quality than HiFi-GAN.
    mel_log: (mel_bins, T) — single clip, NOT batched.
    Returns: (L,) waveform.
    """
    mel = torch.exp(mel_log) - LOG_EPS
    mel = mel.clamp(min=0)
    inv = T.InverseMelScale(n_stft=n_fft // 2 + 1, n_mels=mel.shape[0], sample_rate=sr)
    gl = T.GriffinLim(n_fft=n_fft, hop_length=hop, n_iter=n_iters)
    spec = inv(mel.unsqueeze(0))  # (1, freq, T)
    wav = gl(spec.squeeze(0))
    return wav
