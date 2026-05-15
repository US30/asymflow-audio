"""Run rank sweep: train asym_dct with r in [2,4,8,16,32,48], 50k steps each.

Writes results/rank_ablation.csv and a plot.

Usage:
    python scripts/rank_ablation.py \\
        --ref_dir data/sc09/test_wavs \\
        --steps 50000 \\
        --ranks 2,4,8,16,32,48
"""
import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_dir", required=True)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--ranks", default="2,4,8,16,32,48")
    parser.add_argument("--base_cfg", default="configs/asym_dct.yaml")
    args = parser.parse_args()

    ranks = [int(r) for r in args.ranks.split(",")]
    results = []

    for r in ranks:
        print(f"\n{'='*50}")
        print(f"Running rank={r}")
        print(f"{'='*50}")

        # Create temp config with this rank and reduced steps
        base_cfg = OmegaConf.load(args.base_cfg)
        base_cfg.rank = r
        base_cfg.name = f"asym_dct_r{r}"
        base_cfg.train.steps = args.steps

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            OmegaConf.save(base_cfg, f.name)
            cfg_path = f.name

        # Train
        ret = subprocess.run(
            [sys.executable, "-m", "asymflow_audio.train", cfg_path],
            check=True
        )

        # Eval
        ckpt = f"runs/asym_dct_r{r}/ckpt_{args.steps:07d}.pt"
        out_dir = f"results/ablation_r{r}"
        ret = subprocess.run([
            sys.executable, "-m", "asymflow_audio.eval",
            "--ckpt", ckpt,
            "--cfg", cfg_path,
            "--ref_dir", args.ref_dir,
            "--out_dir", out_dir,
            "--n_samples", "2048",
        ], capture_output=True, text=True, check=True)

        # Parse FAD from output
        fad = None
        for line in ret.stdout.splitlines():
            if line.startswith("FAD:"):
                fad = float(line.split(":")[1].strip())
        results.append({"rank": r, "fad": fad})
        print(f"Rank {r} → FAD {fad:.4f}")

    # Write CSV
    Path("results").mkdir(exist_ok=True)
    csv_path = "results/rank_ablation.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "fad"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved ablation results to {csv_path}")
    _plot(results)


def _plot(results):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return
    ranks = [r["rank"] for r in results]
    fads = [r["fad"] for r in results]
    plt.figure(figsize=(7, 4))
    plt.plot(ranks, fads, "o-", linewidth=2, markersize=8)
    plt.xlabel("Rank r")
    plt.ylabel("FAD ↓")
    plt.title("Rank ablation: AsymFM (DCT) on SC09")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = "results/rank_ablation.png"
    plt.savefig(path, dpi=150)
    print(f"Plot saved to {path}")


if __name__ == "__main__":
    main()
