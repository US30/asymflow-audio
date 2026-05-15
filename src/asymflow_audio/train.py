"""Training script for AsymFlow audio models.

Usage:
    python -m asymflow_audio.train configs/base_fm.yaml
    python -m asymflow_audio.train configs/asym_dct.yaml
"""
import copy
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler
import wandb

from .data.sc09 import build_loaders, mu_law_decode
from .model.dit1d import build_model
from .flow.projector import build_projector, PCAProjector
from .flow.loss import fm_loss, asym_fm_loss
from .flow.sampler import euler_sample


def build_ema(model: nn.Module) -> nn.Module:
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float):
    for ep, mp in zip(ema.parameters(), model.parameters()):
        ep.data.mul_(decay).add_(mp.data, alpha=1 - decay)


def train(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.train.seed)

    run = wandb.init(project="asymflow-audio", name=cfg.name, config=OmegaConf.to_container(cfg))

    # Data
    train_loader, val_loader = build_loaders(cfg)
    train_iter = iter(train_loader)

    # Model
    model = build_model(cfg).to(device)
    ema = build_ema(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=1e-4)

    # LR warmup scheduler
    def lr_lambda(step):
        if step < cfg.train.warmup_steps:
            return step / max(1, cfg.train.warmup_steps)
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    scaler = GradScaler(enabled=cfg.train.bf16)
    dtype = torch.bfloat16 if cfg.train.bf16 else torch.float32

    # Projector
    projector = None
    if cfg.projector is not None and cfg.projector != "none":
        projector = build_projector(cfg.projector, cfg.model.patch_size, cfg.rank).to(device)
        if isinstance(projector, PCAProjector):
            _fit_pca(projector, train_loader, device, cfg)

    patch_size = cfg.model.patch_size
    out_dir = Path("runs") / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    accum_steps = cfg.train.grad_accum
    loss_accum = 0.0

    for step in range(1, cfg.train.steps + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        for _ in range(accum_steps):
            try:
                x0 = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x0 = next(train_iter)
            x0 = x0.to(device)

            with torch.autocast(device_type="cuda", dtype=dtype, enabled=cfg.train.bf16):
                if cfg.loss == "fm":
                    loss = fm_loss(model, x0) / accum_steps
                elif cfg.loss == "asym_fm":
                    loss = asym_fm_loss(model, x0, projector, patch_size) / accum_steps
                else:
                    raise ValueError(f"Unknown loss: {cfg.loss}")

            scaler.scale(loss).backward()
            loss_accum += loss.item()

        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        scheduler.step()
        update_ema(ema, model, cfg.train.ema_decay)

        if step % cfg.train.log_every == 0:
            wandb.log({"train/loss": loss_accum, "train/lr": scheduler.get_last_lr()[0]}, step=step)
            print(f"step {step:6d}  loss {loss_accum:.4f}")
            loss_accum = 0.0

        if step % cfg.train.sample_every == 0:
            _save_samples(ema, projector, patch_size, cfg, device, dtype, out_dir, step, run)

        if step % cfg.train.save_every == 0:
            ckpt = {"step": step, "model": model.state_dict(), "ema": ema.state_dict(),
                    "opt": opt.state_dict()}
            torch.save(ckpt, out_dir / f"ckpt_{step:07d}.pt")

    run.finish()


@torch.no_grad()
def _save_samples(ema, projector, patch_size, cfg, device, dtype, out_dir, step, run):
    ema.eval()
    import soundfile as sf
    samples_dir = out_dir / "samples" / f"step_{step:07d}"
    samples_dir.mkdir(parents=True, exist_ok=True)

    with torch.autocast(device_type="cuda", dtype=dtype, enabled=cfg.train.bf16):
        x = euler_sample(
            ema,
            shape=(cfg.train.num_samples, cfg.data.length),
            steps=cfg.sample.steps,
            projector=projector,
            patch_size=patch_size,
            sigma_min=cfg.sample.sigma_min,
            device=device,
        )

    # Decode mu-law → waveform
    x = mu_law_decode(x.float().cpu())

    audio_list = []
    for i, wav in enumerate(x):
        path = str(samples_dir / f"{i:03d}.wav")
        sf.write(path, wav.numpy(), samplerate=cfg.data.sample_rate)
        audio_list.append(wandb.Audio(path, sample_rate=cfg.data.sample_rate, caption=f"{i}"))

    run.log({"samples": audio_list}, step=step)


def _fit_pca(projector, train_loader, device, cfg):
    """Collect ~50k patches from training data and fit PCA projector."""
    print("Fitting PCA basis from training patches...")
    patches = []
    target = 50000
    for x0 in train_loader:
        # x0: (B, L) → (B * n_patches, patch_size)
        from einops import rearrange
        p = rearrange(x0, 'b (n ps) -> (b n) ps', ps=cfg.model.patch_size)
        patches.append(p)
        if sum(pp.shape[0] for pp in patches) >= target:
            break
    patches = torch.cat(patches, dim=0)[:target]
    projector.fit(patches.to(device))
    if cfg.get("pca_basis_path"):
        torch.save(projector.A.cpu(), cfg.pca_basis_path)
    print(f"PCA fitted. A shape: {projector.A.shape}")


if __name__ == "__main__":
    train(sys.argv[1])
