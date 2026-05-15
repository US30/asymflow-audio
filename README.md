# AsymFlow Audio

Porting **rank-asymmetric flow matching** (AsymFlow, arXiv 2605.12964) from images to raw audio waveforms.

Original paper replaces standard FM velocity target `u = ε − x₀` with:

```
u_A = P·ε − x₀
```

where `P = AA^T` is a rank-r orthogonal projector. Noise prediction is restricted to a low-rank subspace; data prediction stays full-dimensional. Full velocity recovered analytically at sample time — no architecture changes.

This repo tests whether that trick transfers to speech, where audio waveforms live near a low-rank DCT manifold.

## Setup

```bash
make install
make data         # downloads SC09 (~2.4 GB)
make test         # sanity checks
```

## Train

```bash
make train-fm     # baseline: standard flow matching
make train-asym   # AsymFM with DCT projector, rank=8
```

Logs to W&B. Checkpoints in `runs/<name>/`.

## Eval (FAD)

First prepare reference WAVs (one-time):
```bash
# Copy SC09 test split wavs to data/sc09/test_wavs/
```

Then:
```bash
python -m asymflow_audio.eval \
    --ckpt runs/asym_dct/ckpt_0100000.pt \
    --cfg configs/asym_dct.yaml \
    --ref_dir data/sc09/test_wavs \
    --out_dir results/asym_dct \
    --n_samples 2048
```

## Rank ablation

```bash
make ablation
# Writes results/rank_ablation.csv and results/rank_ablation.png
```

## Architecture

| Component | Detail |
|---|---|
| Model | 1D DiT-S/64 |
| Params | ~33M |
| Patch size | 64 samples |
| Tokens | 250 (16000 / 64) |
| Hidden dim | 384 |
| Depth | 12 blocks |
| Heads | 6 |
| Conditioning | AdaLN-zero on timestep |
| Projector | DCT-II (rank 8, per-patch) |
| Loss | Asym FM vs standard FM |
| Sampler | 50-step Euler ODE |
| Dataset | SC09 (30k × 1s @ 16kHz) |
| Metric | FAD (VGGish) |

## Expected results

| Model | FAD ↓ |
|---|---|
| DiffWave (baseline, reference) | ~1.3 |
| Standard FM (this repo baseline) | TBD |
| AsymFM DCT r=8 (this repo) | TBD |

## Paper

Chen et al., *Asymmetric Flow Models*, arXiv 2605.12964v1, May 2026.
