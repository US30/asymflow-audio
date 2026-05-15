"""Pre-compute PCA basis from SC09 training patches and save to disk.

Usage:
    python scripts/compute_pca_basis.py --data_root data/sc09 --rank 8 --out data/sc09_pca_basis.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from einops import rearrange
from omegaconf import OmegaConf

from asymflow_audio.data.sc09 import SC09Dataset
from asymflow_audio.flow.projector import PCAProjector
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/sc09")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--n_patches", type=int, default=100000)
    parser.add_argument("--out", default="data/sc09_pca_basis.pt")
    args = parser.parse_args()

    ds = SC09Dataset(args.data_root, split="train")
    loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=4)

    patches = []
    collected = 0
    for x0 in loader:
        p = rearrange(x0, 'b (n ps) -> (b n) ps', ps=args.patch_size)
        patches.append(p)
        collected += p.shape[0]
        if collected >= args.n_patches:
            break

    patches = torch.cat(patches, dim=0)[:args.n_patches]
    print(f"Fitting PCA on {patches.shape[0]} patches of dim {args.patch_size}...")

    proj = PCAProjector(args.patch_size, args.rank)
    proj.fit(patches)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(proj.A, args.out)
    print(f"Saved A ({proj.A.shape}) to {args.out}")


if __name__ == "__main__":
    main()
