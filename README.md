# AsymFlow Audio

> Porting **rank-asymmetric flow matching** from images to speech — mel-spectrograms and raw waveforms.

Based on: [*Asymmetric Flow Models*](https://arxiv.org/abs/2605.12964) (arXiv 2605.12964v1) — Chen, Ackermann, Kim, Wetzstein, Guibas · Stanford + ETH · May 2026

---

## What This Is

The AsymFlow paper replaces the standard flow matching velocity target `u = ε − x₀` with an **asymmetric** version that restricts noise prediction to a low-rank subspace:

```
u_A = P·ε − x₀      where  P = AᵀA  (rank-r orthogonal projector)
```

This exploits the fact that natural data concentrates near a low-rank manifold — so spending model capacity on predicting noise in high-frequency dimensions is wasteful. The full-rank velocity is recovered analytically at sample time; no architecture changes needed.

**This project tests whether that idea transfers to speech.** The key claim is that mel-spectrograms — not raw waveforms — are the right domain, because mel patches are demonstrably low-rank while raw waveform patches are not.

We prove that first (Phase 0), then validate it experimentally across SC09 and LJSpeech.

---

## Study Design

| Phase | Experiment | Purpose | Est. Compute |
|---|---|---|---|
| **0** | PCA variance-explained analysis | Prove low-rank premise; set rank | <30 min CPU |
| **1** | SC09 raw waveform · FM vs AsymFM | Expected-fail condition; ablation | ~24 H100-hrs |
| **2** | SC09 mel-spectrogram · FM vs AsymFM | **Primary result** | ~16 H100-hrs |
| **3** | LJSpeech mel-spectrogram · FM vs AsymFM | Generalization | ~40 H100-hrs |
| **4** | LJSpeech mel · DiT-B · AsymFM | Scaling behavior (stretch) | ~20 H100-hrs |

Target venues if results positive: **INTERSPEECH**, **ICASSP** (short paper), ICML/NeurIPS workshops.

---

## Results

*(Fill in after training)*

| Dataset | Domain | Model | FM FAD ↓ | AsymFM FAD ↓ | Δ |
|---|---|---|---|---|---|
| SC09 | raw waveform | DiT-S | — | — | predicted: small |
| SC09 | mel-spec | DiT-S | — | — | predicted: clear win |
| LJSpeech | mel-spec | DiT-S | — | — | predicted: clear win |
| LJSpeech | mel-spec | DiT-B | — | — | predicted: larger gap |

Reference FAD scores from literature (raw waveform, SC09): DiffWave ≈ 1.3 · PriorGrad ≈ 0.5 · WaveGrad ≈ 1.0

---

## The Math

### Standard flow matching

```
x_t = t·x₀ + (1−t)·ε,    ε ~ N(0, I)

target:   u = ε − x₀
loss:     L = E[ ‖u − û(x_t, t)‖² ]
```

### Asymmetric flow matching

```
u_A = P·ε − x₀
```

Decomposition per patch:

| Subspace | Behaviour | Effect |
|---|---|---|
| range(P) | standard u-prediction | numerically stable |
| orthogonal complement | x₀-prediction | kills wasted noise capacity |

**Velocity recovery at sample time** (no architecture change needed):

```
u = P·u_A  +  (I−P)·(x_t + u_A) / σ_t      [σ_t = 1−t, clamped at 1e-3]
```

---

## Architecture: 1D DiT

```
Input (B, L) — raw waveform OR flattened mel tokens
  │
Patchify  →  tokens of shape (patch_size,)
  │
Linear proj  →  dim
  │
+ Learned positional embedding
  │
  ┌── × depth ─────────────────────────────────────────┐
  │  AdaLN-zero  ←  timestep sinusoidal MLP embedding  │
  │  Multi-head self-attention                         │
  │  AdaLN-zero                                        │
  │  MLP  (4× expansion, GELU)                        │
  └────────────────────────────────────────────────────┘
  │
Final AdaLN-zero + Linear proj  →  patch_size
  │
Unpatchify  →  (B, L)
```

| Variant | patch_size | dim | depth | heads | params | used in |
|---|---|---|---|---|---|---|
| DiT-S | 64 (raw) or 640 (mel) | 384 | 12 | 6 | ~33M | Phases 1–3 |
| DiT-B | 640 | 768 | 12 | 12 | ~130M | Phase 4 |

---

## Projectors

### DCTProjector — 1D (raw waveform, patch=64)

`A` = first `r` rows of the orthonormal DCT-II matrix of size `patch_size × patch_size`.

Physical basis: DCT-II is the de-facto PCA of natural signals (same transform used in MP3/AAC compression). Keeping top-r rows = selecting the low-frequency subspace. No training, no precomputation.

### DCT2DProjector — 2D (mel-spectrogram, patch=80×8=640)

```python
A = kron(A_freq[:rank_freq], A_time[:rank_time])   # shape: (rank_freq*rank_time, 640)
```

Separable 2D DCT basis via Kronecker product. Applied to flattened 640-dim mel patches. `rank_freq=4, rank_time=4` → rank=16 out of 640.

Idempotency guaranteed by construction: `P² = P`, `P = Pᵀ`.

### PCAProjector — data-driven (secondary control)

Top-r PCA eigenvectors fitted from ~50k training patches. Matches the original paper's approach. Requires precompute step (`scripts/compute_pca_basis.py`).

---

## Mel-Spectrogram Pipeline

```
waveform  →  MelSpectrogram(n_mels=80, hop=160, win=320, n_fft=512)
          →  log(mel + 1e-5)
          →  normalize (per-dataset mean/std, cached)
          →  patch into (80 mel × 8 time) = 640-dim tokens
          →  [model generates tokens]
          →  denormalize
          →  HiFi-GAN UNIVERSAL_V1 vocoder
          →  waveform for FAD evaluation
```

**Fair FAD evaluation:** both generated and reference audio pass through the same `mel → vocoder` round-trip. The vocoder's quality ceiling cancels out. FAD measures model quality above that floor.

---

## Datasets

| Dataset | Clips | Duration | SR | Used in |
|---|---|---|---|---|
| SC09 (Speech Commands digits) | ~30k | ~8h | 16kHz | Phases 1, 2 |
| LJSpeech (single speaker) | 13,100 | 24h | 22→16kHz | Phase 3, 4 |

---

## Setup

### Requirements
- Python ≥ 3.10
- PyTorch ≥ 2.3 with CUDA
- 40GB+ VRAM (H100 / A100)

### Install
```bash
git clone https://github.com/US30/asymflow-audio
cd asymflow-audio
pip install -e ".[dev]"
```

---

## Quick Start

```bash
# 1. Data + vocoder
make data && make data-lj && make hifigan

# 2. Phase 0 — run this before anything else
make analyze
# → results/variance_explained.png  (read this plot)

# 3. Correctness tests
make test

# 4. Phase 1 — SC09 raw (comparison condition)
make train-fm && make train-asym

# 5. Phase 2 — SC09 mel (primary experiment)
make train-fm-mel && make train-asym-mel

# 6. Phase 3 — LJSpeech mel (generalization)
make train-fm-lj && make train-asym-lj

# 7. Phase 4 — DiT-B scale (stretch, only if Phase 3 is clean)
make train-asym-lj-b
```

### Evaluate any model

```bash
python -m asymflow_audio.eval \
    --ckpt runs/asym_dct_sc09_mel/ckpt_0100000.pt \
    --cfg  configs/asym_dct_sc09_mel.yaml \
    --ref_dir data/sc09/test_wavs \
    --out_dir results/asym_sc09_mel \
    --n_samples 2048
# FAD → results/asym_sc09_mel/fad.json
```

For the full step-by-step run guide, runtime estimates, common errors, and checkpoint details: **see `HOW_TO_RUN.txt`**.

---

## File Structure

```
asymflow-audio/
├── HOW_TO_RUN.txt                   full execution guide
├── Makefile                         all commands
├── pyproject.toml
│
├── configs/
│   ├── base_fm.yaml                 Phase 1 · SC09 raw · baseline FM
│   ├── asym_dct.yaml                Phase 1 · SC09 raw · AsymFM DCT r=8
│   ├── asym_pca.yaml                Phase 1 · SC09 raw · AsymFM PCA r=8
│   ├── base_fm_sc09_mel.yaml        Phase 2 · SC09 mel · baseline FM
│   ├── asym_dct_sc09_mel.yaml       Phase 2 · SC09 mel · AsymFM DCT2D rank=16
│   ├── base_fm_lj_mel.yaml          Phase 3 · LJSpeech mel · baseline FM
│   ├── asym_dct_lj_mel.yaml         Phase 3 · LJSpeech mel · AsymFM
│   └── asym_dct_lj_mel_b.yaml       Phase 4 · LJSpeech mel · DiT-B
│
├── src/asymflow_audio/
│   ├── data/
│   │   ├── sc09.py                  SC09 loader · µ-law encode/decode
│   │   ├── melspec.py               MelSpec · MelNormalizer · HiFi-GAN vocoder
│   │   ├── sc09_mel.py              SC09 mel dataset → (12 tokens, 640-dim)
│   │   └── ljspeech.py              LJSpeech raw + mel → (25 tokens, 640-dim)
│   ├── model/
│   │   └── dit1d.py                 DiT: TimestepEmbedder · DiTBlock · AdaLN-zero
│   ├── flow/
│   │   ├── projector.py             DCTProjector · DCT2DProjector · PCAProjector
│   │   ├── loss.py                  fm_loss · asym_fm_loss · recover_velocity
│   │   └── sampler.py               euler_sample (50-step Euler ODE)
│   ├── train.py                     training loop · domain dispatch · EMA · W&B
│   └── eval.py                      FAD eval · mel→vocoder pipeline
│
├── scripts/
│   ├── analyze_low_rank.py          Phase 0 · PCA variance curves  ← run first
│   ├── download_sc09.sh
│   ├── download_ljspeech.sh
│   ├── download_hifigan.sh
│   ├── compute_pca_basis.py
│   └── rank_ablation.py             rank sweep r ∈ {2,4,8,16,32,48}
│
└── tests/
    └── test_asym_core.py            projector math + loss correctness (17 tests)
```

---

## Key Design Decisions

**Why mel-spectrogram, not raw waveform?**
Raw 16kHz waveforms have broadband DCT energy from transients (consonants, sibilants). Mel-spectrograms are frequency-compressed and band-limited — patches are smooth and low-rank. Phase 0 quantifies this gap before any GPU time is spent.

**Why DCT2D via Kronecker product?**
Applying 2D DCT to an `(80, 8)` mel patch selects the joint (frequency, time) low-energy subspace. The Kronecker construction gives a single `(rank, 640)` matrix `A` with orthonormal rows — so `P = AᵀA` satisfies `P² = P` and `P = Pᵀ` exactly, by construction.

**Why σ_t clamping?**
Velocity recovery divides by `σ_t = 1−t`. Near `t=1`, σ_t → 0 and the x₀-prediction term explodes. Clamping at `1e-3` is standard practice in flow models.

**Why skip the variance-reduced loss (paper Eq. 7)?**
That loss is for finetuning a pretrained latent model. Training from scratch only needs the base AsymFM loss (Eq. 4). Simpler = fewer confounds in the ablation.

---

## Risks

| Risk | Mitigation |
|---|---|
| Raw waveform asym FM doesn't win | Expected and informative — Phase 0 predicts this; included as ablation |
| Vocoder mismatch (HiFi-GAN @ 22kHz vs our 16kHz) | Round-trip eval design: both sides go through same vocoder, so mismatch is symmetric |
| OOM on DiT-B with 40GB | Reduce batch to 32, grad_accum to 4 in config |
| HiFi-GAN download fails (GDrive) | Code auto-falls-back to Griffin-Lim; see HOW_TO_RUN.txt |

---

## Citation

```bibtex
@article{chen2026asymflow,
  title   = {Asymmetric Flow Models},
  author  = {Chen, Hansheng and Ackermann, Jan and Kim, Minseo and
             Wetzstein, Gordon and Guibas, Leonidas},
  journal = {arXiv preprint arXiv:2605.12964},
  year    = {2026}
}
```
