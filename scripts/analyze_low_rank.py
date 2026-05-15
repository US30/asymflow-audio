"""Phase 0: Variance-explained analysis — prove the low-rank premise before training.

Computes PCA explained-variance curves for:
  - SC09 raw waveform patches (size 64)
  - SC09 mel-spectrogram patches (80×8=640 dim)
  - LJSpeech mel-spectrogram patches (same)

Optionally adds ImageNet patches as reference (if torchvision available).

Output: results/variance_explained.png  (the key motivating figure)
        results/variance_explained.csv

Usage:
    python scripts/analyze_low_rank.py \\
        --sc09_dir data/sc09 \\
        --lj_dir   data/LJSpeech-1.1 \\
        --n_patches 10000 \\
        --out_dir results
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
from einops import rearrange

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from asymflow_audio.data.melspec import MelSpecExtractor, MEL_CFG


DIGITS = ["zero","one","two","three","four","five","six","seven","eight","nine"]
SR = 16000
PATCH_RAW = 64
MEL_BINS = 80
TIME_FRAMES = 8
PATCH_MEL = MEL_BINS * TIME_FRAMES  # 640


def load_sc09_wavs(root: str, n: int) -> list:
    root = Path(root)
    files = []
    for d in DIGITS:
        files.extend((root / d).glob("*.wav"))
    random.shuffle(files)
    return files[:n]


def load_lj_wavs(root: str, n: int) -> list:
    wav_dir = Path(root) / "wavs"
    files = sorted(wav_dir.glob("*.wav"))
    random.shuffle(files)
    return files[:n]


def wav_to_raw_patches(path, n_patches=20) -> torch.Tensor:
    """Random patches of size PATCH_RAW from one wav file."""
    wav, sr = torchaudio.load(path)
    wav = wav.mean(0)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    wav = wav / (wav.abs().max().clamp(min=1e-6))

    patches = []
    for _ in range(n_patches):
        if wav.shape[0] < PATCH_RAW:
            continue
        start = random.randint(0, wav.shape[0] - PATCH_RAW)
        patches.append(wav[start: start + PATCH_RAW])
    return torch.stack(patches) if patches else None  # (n, 64)


def wav_to_mel_patches(path, extractor, n_patches=5) -> torch.Tensor:
    """Random patches of (MEL_BINS, TIME_FRAMES) from one wav file."""
    wav, sr = torchaudio.load(path)
    wav = wav.mean(0)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    wav = wav / (wav.abs().max().clamp(min=1e-6))

    mel = extractor(wav)  # (80, T)
    _, T = mel.shape
    if T < TIME_FRAMES:
        return None

    patches = []
    for _ in range(n_patches):
        start = random.randint(0, T - TIME_FRAMES)
        patch = mel[:, start: start + TIME_FRAMES]  # (80, 8)
        patches.append(patch.reshape(-1))           # (640,)
    return torch.stack(patches)  # (n, 640)


def explained_variance_curve(patches: torch.Tensor, max_rank: int) -> np.ndarray:
    """
    patches: (N, D)
    Returns fraction of variance explained by top-k PCA components, k=1..max_rank.
    """
    patches = patches.float()
    patches = patches - patches.mean(0, keepdim=True)

    # SVD (economical)
    _, S, _ = torch.linalg.svd(patches, full_matrices=False)
    var = S.pow(2)
    total = var.sum().item()
    cumvar = var.cumsum(0).numpy() / total
    return cumvar[:max_rank]


def collect_patches(files: list, kind: str, extractor=None, n_total: int = 10000) -> torch.Tensor:
    patches_per_file = max(1, n_total // len(files))
    all_patches = []
    for f in files:
        if kind == "raw":
            p = wav_to_raw_patches(f, n_patches=patches_per_file)
        else:
            p = wav_to_mel_patches(f, extractor, n_patches=patches_per_file)
        if p is not None:
            all_patches.append(p)
        if sum(pp.shape[0] for pp in all_patches) >= n_total:
            break
    return torch.cat(all_patches, dim=0)[:n_total]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sc09_dir", default="data/sc09")
    parser.add_argument("--lj_dir",   default="data/LJSpeech-1.1")
    parser.add_argument("--n_patches", type=int, default=10000)
    parser.add_argument("--out_dir",   default="results")
    args = parser.parse_args()

    random.seed(42)
    extractor = MelSpecExtractor(**MEL_CFG)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    curves = {}

    # SC09 raw
    sc09_files = load_sc09_wavs(args.sc09_dir, n=500)
    print(f"SC09 raw: collecting patches from {len(sc09_files)} files...")
    patches_sc09_raw = collect_patches(sc09_files, "raw", n_total=args.n_patches)
    print(f"  got {patches_sc09_raw.shape[0]} patches of dim {patches_sc09_raw.shape[1]}")
    curves["SC09 raw waveform (patch=64)"] = explained_variance_curve(patches_sc09_raw, PATCH_RAW)

    # SC09 mel
    print("SC09 mel-spec: collecting patches...")
    patches_sc09_mel = collect_patches(sc09_files, "mel", extractor, n_total=args.n_patches)
    print(f"  got {patches_sc09_mel.shape[0]} patches of dim {patches_sc09_mel.shape[1]}")
    curves["SC09 mel-spec (80×8=640)"] = explained_variance_curve(patches_sc09_mel, PATCH_RAW)

    # LJSpeech mel
    if Path(args.lj_dir).exists():
        lj_files = load_lj_wavs(args.lj_dir, n=500)
        print(f"LJSpeech mel-spec: collecting patches from {len(lj_files)} files...")
        patches_lj_mel = collect_patches(lj_files, "mel", extractor, n_total=args.n_patches)
        print(f"  got {patches_lj_mel.shape[0]} patches of dim {patches_lj_mel.shape[1]}")
        curves["LJSpeech mel-spec (80×8=640)"] = explained_variance_curve(patches_lj_mel, PATCH_RAW)
    else:
        print(f"LJSpeech not found at {args.lj_dir}, skipping.")

    # Plot
    _plot(curves, args.out_dir)

    # CSV
    _save_csv(curves, args.out_dir)

    # Print elbow ranks
    print("\n=== Elbow rank (where curve exceeds 0.80 explained variance) ===")
    for name, curve in curves.items():
        elbow = next((i+1 for i, v in enumerate(curve) if v >= 0.80), len(curve))
        print(f"  {name}: rank {elbow} explains ≥80% variance")

    print(f"\nPlot saved to {args.out_dir}/variance_explained.png")
    print("Use this to pick rank_freq and rank_time for your asym projector configs.")


def _plot(curves: dict, out_dir: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot. pip install matplotlib")
        return

    ranks = range(1, PATCH_RAW + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    styles = ["-", "--", ":", "-."]
    for i, (name, curve) in enumerate(curves.items()):
        ax.plot(list(ranks)[:len(curve)], curve, styles[i % 4], linewidth=2, label=name)

    ax.axhline(0.80, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="80% variance")
    ax.axhline(0.95, color="gray", linestyle=":",  linewidth=1, alpha=0.6, label="95% variance")
    ax.set_xlabel("Rank r (number of PCA components)")
    ax.set_ylabel("Fraction of variance explained")
    ax.set_title("Low-rank structure: raw waveform vs. mel-spectrogram patches")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, PATCH_RAW)
    ax.set_ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(str(Path(out_dir) / "variance_explained.png"), dpi=150)
    plt.close()


def _save_csv(curves: dict, out_dir: str):
    path = Path(out_dir) / "variance_explained.csv"
    all_names = list(curves.keys())
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank"] + all_names)
        max_len = max(len(c) for c in curves.values())
        for r in range(max_len):
            row = [r + 1] + [f"{curves[n][r]:.6f}" if r < len(curves[n]) else "" for n in all_names]
            writer.writerow(row)


if __name__ == "__main__":
    main()
