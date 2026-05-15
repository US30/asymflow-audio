"""FAD evaluation script.

Usage:
    python -m asymflow_audio.eval \\
        --ckpt runs/asym_dct/ckpt_0100000.pt \\
        --cfg configs/asym_dct.yaml \\
        --ref_dir data/sc09/test_wavs \\
        --out_dir results/asym_dct_fad \\
        --n_samples 2048

Writes FAD score to results/<name>/fad.json.
"""
import argparse
import json
import os
from pathlib import Path

import torch
import soundfile as sf
from omegaconf import OmegaConf
from tqdm import trange

from .data.sc09 import mu_law_decode
from .model.dit1d import build_model
from .flow.projector import build_projector, PCAProjector
from .flow.sampler import euler_sample


def generate_samples(model, projector, patch_size, cfg, device, dtype, n_samples, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 64
    generated = 0
    while generated < n_samples:
        bs = min(batch_size, n_samples - generated)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype, enabled=cfg.train.bf16):
            x = euler_sample(
                model,
                shape=(bs, cfg.data.length),
                steps=cfg.sample.steps,
                projector=projector,
                patch_size=patch_size,
                sigma_min=cfg.sample.sigma_min,
                device=device,
            )
        x = mu_law_decode(x.float().cpu())
        for i, wav in enumerate(x):
            sf.write(str(out_dir / f"{generated + i:05d}.wav"), wav.numpy(), samplerate=cfg.data.sample_rate)
        generated += bs
        print(f"Generated {generated}/{n_samples}")


def compute_fad(background_dir: str, eval_dir: str) -> float:
    """Compute FAD using frechet_audio_distance package."""
    try:
        from frechet_audio_distance import FrechetAudioDistance
    except ImportError:
        raise ImportError("pip install frechet-audio-distance")

    fad = FrechetAudioDistance(use_pca=False, use_activation=False, verbose=True)
    score = fad.score(background_dir, eval_dir)
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--ref_dir", required=True, help="Reference .wav files (test set)")
    parser.add_argument("--out_dir", required=True, help="Output dir for generated samples")
    parser.add_argument("--n_samples", type=int, default=2048)
    parser.add_argument("--use_ema", action="store_true", default=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if cfg.train.bf16 else torch.float32

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    key = "ema" if args.use_ema and "ema" in ckpt else "model"
    model.load_state_dict(ckpt[key])
    model.eval()

    projector = None
    if cfg.projector is not None and cfg.projector != "none":
        projector = build_projector(cfg.projector, cfg.model.patch_size, cfg.rank).to(device)
        if isinstance(projector, PCAProjector) and cfg.get("pca_basis_path"):
            A = torch.load(cfg.pca_basis_path, map_location=device, weights_only=True)
            projector.A.copy_(A)
            projector.P.copy_(A.T @ A)
            projector._fitted = True

    generate_samples(model, projector, cfg.model.patch_size, cfg, device, dtype, args.n_samples, args.out_dir)
    print("Computing FAD...")
    fad_score = compute_fad(args.ref_dir, args.out_dir)
    print(f"FAD: {fad_score:.4f}")

    result = {"config": cfg.name, "fad": fad_score, "n_samples": args.n_samples, "ckpt": args.ckpt}
    out_path = Path(args.out_dir) / "fad.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
