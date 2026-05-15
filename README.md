# AsymFlow Audio

> Porting **rank-asymmetric flow matching** from images to raw audio waveforms.

Based on: [*Asymmetric Flow Models*](https://arxiv.org/abs/2605.12964) — Chen, Ackermann, Kim, Wetzstein, Guibas (Stanford + ETH, 2026)

---

## What This Is

The AsymFlow paper introduces a simple but powerful fix for flow matching in high-dimensional spaces: instead of predicting the full velocity `u = ε − x₀`, the model predicts an **asymmetric** target where noise is projected onto a low-rank subspace before being used in the velocity target. The full-rank velocity is recovered analytically at sample time — no architecture changes needed.

This repo tests whether that idea transfers to **raw audio waveforms**. Speech signals live near a low-rank manifold in the DCT/frequency domain — smooth, harmonic, sparse in high frequencies. That makes them a natural fit for the asymmetric parameterization.

We train a 1D Diffusion Transformer (DiT) on SC09 (spoken digit commands) and compare:
- Standard flow matching baseline
- AsymFlow with DCT-truncation projector (primary)
- AsymFlow with PCA projector (secondary)

Primary metric: **FAD** (Fréchet Audio Distance). Single ablation: FAD vs. projector rank `r`.

---

## The Math (Quick Version)

**Standard flow matching** trains a network to predict velocity along a linear interpolant between data and noise:

```
x_t = t · x₀ + (1 − t) · ε,    ε ~ N(0, I)

target:   u = ε − x₀
loss:     L = E[ ‖u − û(x_t, t)‖² ]
```

**AsymFlow** replaces the velocity target with:

```
u_A = P · ε − x₀
```

where `P = A Aᵀ` is a rank-r orthogonal projector (`A ∈ ℝʳˣᴰ`, rows orthonormal). This decomposes the prediction into two regimes per patch:

| Subspace | Prediction type | Behavior |
|---|---|---|
| In range(P) | `P · u_A = P · u` | Standard u-prediction (stable) |
| Orthogonal complement | `(I−P) · u_A = −(I−P) · x₀` | x₀-prediction (kills wasted noise capacity) |

**Full velocity recovery** at sample time (no architecture change):

```
u = P · u_A  +  (I − P) · (x_t + u_A) / σ_t

where σ_t = 1 − t
```

**Why this helps for audio:** High-dimensional waveforms are high-rank in raw form but low-rank in frequency space. Restricting noise prediction to top-r DCT components forces the model to spend capacity on data structure (x₀-pred) rather than modeling noise in imperceptible high-frequency dimensions.

---

## Architecture

### 1D DiT-S/64 (~33M parameters)

```
Input waveform  (B, 16000)
       │
  Patchify       patch_size=64 → 250 tokens per waveform
       │
  Linear proj    64 → 384 dim
       │
  + Positional embedding  (learned, 250 × 384)
       │
  ┌─── × 12 ────────────────────────────────────────────┐
  │                                                      │
  │   AdaLN-zero ← timestep embedding (sinusoidal MLP)  │
  │       │                                             │
  │   Multi-head Self-Attention  (6 heads, dim=384)     │
  │       │                                             │
  │   AdaLN-zero                                        │
  │       │                                             │
  │   MLP  (384 → 1536 → 384, GELU)                    │
  │                                                      │
  └──────────────────────────────────────────────────────┘
       │
  Final AdaLN-zero + Linear proj   384 → 64 (patch_size)
       │
  Unpatchify     (B, 250, 64) → (B, 16000)
       │
  Output velocity û(x_t, t)
```

| Hyperparameter | Value |
|---|---|
| Waveform length | 16000 samples (1s @ 16kHz) |
| Patch size | 64 samples |
| Tokens | 250 |
| Hidden dim | 384 |
| Depth | 12 blocks |
| Attention heads | 6 |
| MLP ratio | 4× |
| Conditioning | AdaLN-zero on timestep t |
| Parameters | ~33M |
| Training precision | bfloat16 |

### Projectors

**DCT projector (primary):**

Builds `A` from the first `r` rows of the orthonormal DCT-II matrix of size `patch_size × patch_size`. Physical motivation: DCT basis vectors are ordered by frequency — keeping top-r = keeping the low-frequency (smooth/harmonic) subspace.

```python
P = A^T @ A   # (patch_size × patch_size), applied per token
```

Guaranteed `P² = P` (idempotent) and `P = Pᵀ` (symmetric) by construction. No training required.

**PCA projector (secondary):**

Fits PCA over ~50k training patches, takes top-r eigenvectors as rows of A. More data-adaptive but requires a precompute step.

---

## Dataset: SC09

Speech Commands v0.02, digits only (zero–nine). Standard benchmark for unconditional audio diffusion.

| Property | Value |
|---|---|
| Classes | 10 (spoken digits 0–9) |
| Clips | ~30k train / ~4k val |
| Duration | 1 second per clip |
| Sample rate | 16kHz |
| Preprocessing | µ-law compression → [-1, 1] |

**Reference FAD scores (from literature):**

| Model | FAD ↓ |
|---|---|
| DiffWave | ~1.3 |
| PriorGrad | ~0.5 |
| WaveGrad | ~1.0 |

---

## Setup

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.3 with CUDA
- 40GB+ VRAM recommended (H100/A100)

### Install

```bash
git clone https://github.com/US30/asymflow-audio
cd asymflow-audio
pip install -e ".[dev]"
```

### W&B (optional but recommended)

```bash
wandb login
# All runs log to project: asymflow-audio
# To disable: export WANDB_MODE=disabled
```

---

## Running

### 1 — Download SC09

```bash
make data
# OR: bash scripts/download_sc09.sh data/sc09
```

Downloads ~2.4 GB. Extracts the 10 digit subdirs into `data/sc09/`.

### 2 — Run correctness tests

```bash
make test
```

Verifies before burning GPU time:
- `P² = P` (projector idempotent)
- `P = Pᵀ` (projector symmetric)
- `P·x + (I−P)·x = x` (complement is correct)
- `r = patch_size` → AsymFM loss **identical** to standard FM loss
- Velocity recovery shape and full-rank identity
- No NaN/Inf on synthetic sinusoid input

**All tests must pass before training.**

### 3 — Train baseline (standard FM)

```bash
make train-fm
# OR: python -m asymflow_audio.train configs/base_fm.yaml
```

### 4 — Train AsymFlow (DCT projector, rank=8)

```bash
make train-asym
# OR: python -m asymflow_audio.train configs/asym_dct.yaml
```

### 4a — Train AsymFlow (PCA projector, optional)

```bash
make pca-basis   # precompute PCA from training data (~5 min)
python -m asymflow_audio.train configs/asym_pca.yaml
```

### Training details

| Setting | Value |
|---|---|
| Steps | 100k |
| Batch size | 128 |
| Gradient accumulation | 1 |
| Optimizer | AdamW, lr=1e-4, wd=1e-4 |
| LR warmup | 2000 steps |
| EMA decay | 0.9999 |
| Precision | bfloat16 |
| Grad clip | 1.0 |
| Sampler (eval) | 50-step Euler ODE |

Checkpoints saved every 10k steps to `runs/<name>/ckpt_XXXXXXX.pt`.  
Audio samples saved every 5k steps to `runs/<name>/samples/step_XXXXXXX/`.

### 5 — Evaluate (FAD)

Prepare reference WAVs from the SC09 val split:

```bash
mkdir -p data/sc09/test_wavs
# copy ~2k .wav files from data/sc09/*/ into test_wavs/
```

Then evaluate any checkpoint:

```bash
# Baseline
python -m asymflow_audio.eval \
    --ckpt runs/base_fm/ckpt_0100000.pt \
    --cfg configs/base_fm.yaml \
    --ref_dir data/sc09/test_wavs \
    --out_dir results/base_fm \
    --n_samples 2048

# AsymFM
python -m asymflow_audio.eval \
    --ckpt runs/asym_dct/ckpt_0100000.pt \
    --cfg configs/asym_dct.yaml \
    --ref_dir data/sc09/test_wavs \
    --out_dir results/asym_dct \
    --n_samples 2048
```

FAD score written to `results/<name>/fad.json`.

### 6 — Rank ablation

Trains 6 models at 50k steps each (rank r ∈ {2, 4, 8, 16, 32, 48}) and plots FAD vs r:

```bash
make ablation
# OR:
python scripts/rank_ablation.py \
    --ref_dir data/sc09/test_wavs \
    --steps 50000 \
    --ranks 2,4,8,16,32,48
```

Output:
- `results/rank_ablation.csv`
- `results/rank_ablation.png` — U-shaped curve expected, minimum around r=8–16

---

## Expected Runtime (H100 40GB)

| Task | Time |
|---|---|
| 100k steps, batch 128, bf16 | ~8–12 hours |
| FAD eval (2048 samples) | ~10 minutes |
| Rank ablation (6 × 50k steps) | ~24–36 hours total |

---

## File Structure

```
asymflow-audio/
├── README.md
├── HOW_TO_RUN.txt          quick reference for server
├── Makefile                common commands
├── pyproject.toml          package + deps (uv/pip)
│
├── configs/
│   ├── base_fm.yaml        standard flow matching baseline
│   ├── asym_dct.yaml       AsymFM + DCT projector, rank=8
│   └── asym_pca.yaml       AsymFM + PCA projector, rank=8
│
├── src/asymflow_audio/
│   ├── data/
│   │   └── sc09.py         dataset loader, µ-law encode/decode, DataLoader builder
│   ├── model/
│   │   └── dit1d.py        1D DiT: patch embed, positional embed, DiTBlock,
│   │                         AdaLN-zero, FinalLayer, TimestepEmbedder
│   ├── flow/
│   │   ├── projector.py    DCTProjector, PCAProjector — both expose .project()
│   │   │                     and .complement(); P = A^T A computed once at init
│   │   ├── loss.py         fm_loss(), asym_fm_loss(), recover_velocity(),
│   │   │                     sample_xt(), linear_schedule()
│   │   └── sampler.py      euler_sample() — works for both FM and AsymFM
│   ├── train.py            training loop: EMA, grad accum, bf16, W&B logging,
│   │                         checkpoint saving, audio sample generation
│   └── eval.py             generate samples → compute FAD via VGGish embeddings
│
├── scripts/
│   ├── download_sc09.sh    pulls Speech Commands v0.02, extracts digit classes
│   ├── compute_pca_basis.py fits PCA over training patches, saves A matrix
│   └── rank_ablation.py    rank sweep: train → eval → CSV + plot
│
└── tests/
    └── test_asym_core.py   projector properties, loss equivalence, velocity
                              recovery, synthetic sinusoid sanity check
```

---

## Key Design Decisions

**Why DCT as the projector?**  
The paper uses PCA computed from training data. For audio, DCT-II is essentially PCA of natural speech signals (this is why MP3/AAC compression works). Using DCT makes the projector parameter-free, physically interpretable, and avoids a precompute dependency.

**Why patch-wise projection?**  
Matches the paper exactly. Each 64-sample patch (one token in the DiT) gets the projector applied independently. This is also consistent with how DCT is applied in audio codecs — per-frame, not globally.

**Why skip the variance-reduced loss?**  
Paper Eq. 7 + perceptual correction (Eq. 18) are specific to finetuning from a pretrained latent model. From-scratch training with the base AsymFM loss (Eq. 4) is cleaner and sufficient for this experiment.

**Why σ_t clamping?**  
The velocity recovery formula divides by σ_t = (1−t). Near t=1 this explodes. Clamping at `σ_min = 1e-3` keeps sampling stable — standard practice in flow models.

---

## Risks & Known Limitations

- **Audio may have more high-frequency content than natural images.** Transients, sibilants (s/sh sounds), stop consonants have energy across the full DCT spectrum. The x₀-prediction regime (orthogonal complement) may be less stable for audio than for images. Monitor for artifacts in generated samples.
- **No vocoder needed** — this is direct waveform generation. But µ-law compression is lossy at high frequencies; may limit quality ceiling.
- **FAD only.** No Inception Score in MVP (would require training an SC09 classifier). FAD is the standard metric for this benchmark.
- **Unconditional generation only.** No class label conditioning. The architecture supports it (just pass class embedding through AdaLN-zero alongside timestep) but that is out of scope.

---

## Citation

If you use this work, please cite the original AsymFlow paper:

```bibtex
@article{chen2026asymflow,
  title     = {Asymmetric Flow Models},
  author    = {Chen, Hansheng and Ackermann, Jan and Kim, Minseo and Wetzstein, Gordon and Guibas, Leonidas},
  journal   = {arXiv preprint arXiv:2605.12964},
  year      = {2026}
}
```
