# AsymFlow Audio

> Porting **rank-asymmetric flow matching** from images to raw audio and mel-spectrograms.

Based on: [*Asymmetric Flow Models*](https://arxiv.org/abs/2605.12964) — Chen, Ackermann, Kim, Wetzstein, Guibas (Stanford + ETH, 2026)

**This project tests whether the asym-velocity trick transfers to speech** — and first proves *where* it should work (mel-spectrogram) vs. where it won't (raw waveform), then validates that empirically.

---

## The Core Idea

Standard flow matching trains a network to predict velocity along a linear path between data x₀ and noise ε:

```
x_t = t·x₀ + (1−t)·ε

target:  u = ε − x₀
loss:    L = E[ ‖u − û(x_t, t)‖² ]
```

**AsymFlow** replaces the velocity target with:

```
u_A = P·ε − x₀
```

where `P = AᵀA` is a rank-r orthogonal projector (`A ∈ ℝʳˣᴰ`, rows orthonormal). The prediction decomposes per patch into two regimes:

| Subspace | Prediction type | Why it helps |
|---|---|---|
| In range(P) | Standard u-prediction | Stable training |
| Orthogonal complement | x₀-prediction | Zeros out wasted noise capacity |

**Full velocity** recovered analytically at sample time — no architecture change:

```
u = P·u_A  +  (I−P)·(x_t + u_A) / σ_t      [σ_t = 1−t]
```

**Why this matters for audio:** The trick only wins when data has strong low-rank structure. Raw waveforms have broadband energy (transients, sibilants). Mel-spectrograms are frequency-compressed and demonstrably low-rank. This project proves that empirically before training (Phase 0) and then validates it.

---

## Study Design (4 Phases)

| Phase | What | Purpose | Compute |
|---|---|---|---|
| **0** | Variance-explained analysis | Prove premise, pick rank | ~10 min CPU |
| **1** | SC09 raw waveform (FM vs AsymFM) | Baseline + expected-fail condition | ~24 H100-hrs |
| **2** | SC09 mel-spec (FM vs AsymFM) | Primary contribution | ~16 H100-hrs |
| **3** | LJSpeech mel-spec (FM vs AsymFM) | Generalization claim | ~40 H100-hrs |
| **4** | LJSpeech mel-spec, DiT-B scale | Scaling behavior (stretch) | ~20 H100-hrs |

Target venues (if positive result): **INTERSPEECH**, **ICASSP**, workshop tracks at ICML/NeurIPS.

---

## Architecture: 1D DiT

Plain transformer operating on flattened patches — no architectural hacks, matching the paper's philosophy.

```
Input  (B, L)          — raw waveform or flattened mel tokens
  │
Patchify               — chunk into patch_size-dim tokens
  │
Linear proj → dim      — input embedding
  │
+ Positional embedding (learned)
  │
┌── × depth ───────────────────────────────────────────────┐
│  AdaLN-zero ← timestep MLP embedding                    │
│  Multi-head Self-Attention                               │
│  AdaLN-zero                                              │
│  MLP (dim → 4×dim → dim, GELU)                         │
└──────────────────────────────────────────────────────────┘
  │
Final AdaLN-zero + Linear proj → patch_size
  │
Unpatchify → (B, L)
```

| Config | patch_size | dim | depth | heads | params | used for |
|---|---|---|---|---|---|---|
| DiT-S (default) | 64 or 640 | 384 | 12 | 6 | ~33M | Phases 1–3 |
| DiT-B (stretch) | 640 | 768 | 12 | 12 | ~130M | Phase 4 |

---

## Projectors

### DCTProjector (1D, raw waveform)

`A` = first r rows of orthonormal DCT-II matrix of size `patch_size × patch_size`.
Physical motivation: DCT-II is the de-facto PCA of natural signals (same basis used in MP3/AAC compression). Top-r = low-frequency subspace.

### DCT2DProjector (mel-spec)

For mel patches of shape `(mel_bins=80, time_frames=8)`:

```
A_2D = kron(A_freq[:rank_freq], A_time[:rank_time])
```

Separable 2D DCT basis. Total rank = `rank_freq × rank_time`. Applied to the flattened 640-dim mel patch. Both dimensions sorted by frequency → projector selects the smooth/low-energy subspace of the mel patch.

### PCAProjector (data-driven, secondary)

Top-r PCA eigenvectors fitted from training patches. Matches the original paper. Precomputed via `scripts/compute_pca_basis.py`.

---

## Datasets

| Dataset | Clips | Duration | SR | Domain |
|---|---|---|---|---|
| **SC09** | ~30k | ~8h | 16kHz | 10 spoken digit classes |
| **LJSpeech** | 13,100 | 24h | 22→16kHz | Single speaker, real speech |

---

## Setup

### Requirements
- Python ≥ 3.10
- PyTorch ≥ 2.3 with CUDA
- 40GB+ VRAM (H100/A100) for training

### Install
```bash
git clone https://github.com/US30/asymflow-audio
cd asymflow-audio
make install
```

---

## Running: Step by Step

### Step 1 — Download data

```bash
make data        # SC09 (~2.4 GB)
make data-lj     # LJSpeech (~2.6 GB)
make hifigan     # HiFi-GAN pretrained vocoder (~50 MB)
```

### Step 2 — Run variance analysis (Phase 0, required)

```bash
make analyze
```

Runs `scripts/analyze_low_rank.py` on SC09 and LJSpeech patches.
Outputs `results/variance_explained.png` — the motivating figure.

**Read the output before training.** The elbow ranks tell you what to set for `rank_freq` and `rank_time` in the asym configs.

### Step 3 — Run tests

```bash
make test
```

Verifies projector math (idempotency, symmetry, full-rank identity) for both 1D and 2D projectors. **All tests must pass.**

### Step 4 — Phase 1: SC09 raw waveform

```bash
make train-fm      # baseline FM, ~12 hrs
make train-asym    # AsymFM + DCT r=8, ~12 hrs
make ablation      # rank sweep r∈{2,4,8,16,32,48}, ~24 hrs
```

### Step 5 — Phase 2: SC09 mel-spectrogram (primary experiment)

```bash
make train-fm-mel      # baseline FM on mel, ~8 hrs
make train-asym-mel    # AsymFM + DCT2D (rank=4×4=16), ~8 hrs
```

### Step 6 — Phase 3: LJSpeech mel-spectrogram (generalization)

```bash
make train-fm-lj       # baseline FM on LJSpeech mel, ~20 hrs
make train-asym-lj     # AsymFM + DCT2D on LJSpeech, ~20 hrs
```

### Step 7 — Phase 4: DiT-B scale (stretch)

```bash
make train-asym-lj-b   # only if Phase 3 shows clean improvement
```

### Evaluate (FAD)

```bash
python -m asymflow_audio.eval \
    --ckpt runs/asym_dct_sc09_mel/ckpt_0100000.pt \
    --cfg configs/asym_dct_sc09_mel.yaml \
    --ref_dir data/sc09/test_wavs \
    --out_dir results/asym_sc09_mel \
    --n_samples 2048
# FAD written to results/asym_sc09_mel/fad.json
```

**For mel-spec models:** both generated and reference audio pass through the vocoder pipeline, so the vocoder's quality floor cancels out in the FAD comparison — fair evaluation by design.

---

## Expected Results

### Cross-domain table (to be filled after training)

| Dataset | Domain | Model | FM FAD ↓ | AsymFM FAD ↓ | Δ |
|---|---|---|---|---|---|
| SC09 | raw waveform | DiT-S | — | — | predicted: small |
| SC09 | mel-spec | DiT-S | — | — | predicted: clear win |
| LJSpeech | mel-spec | DiT-S | — | — | predicted: clear win |
| LJSpeech | mel-spec | DiT-B | — | — | predicted: larger win |

### Reference FAD (raw waveform, from literature)

| Model | SC09 FAD ↓ |
|---|---|
| DiffWave | ~1.3 |
| PriorGrad | ~0.5 |
| WaveGrad | ~1.0 |

---

## File Structure

```
asymflow-audio/
├── README.md
├── HOW_TO_RUN.txt              quick-start reference
├── Makefile                    all commands
├── pyproject.toml
│
├── configs/
│   ├── base_fm.yaml            SC09 raw — baseline FM
│   ├── asym_dct.yaml           SC09 raw — AsymFM + DCT r=8
│   ├── asym_pca.yaml           SC09 raw — AsymFM + PCA r=8
│   ├── base_fm_sc09_mel.yaml   SC09 mel — baseline FM       [Phase 2]
│   ├── asym_dct_sc09_mel.yaml  SC09 mel — AsymFM + DCT2D   [Phase 2]
│   ├── base_fm_lj_mel.yaml     LJSpeech mel — baseline FM  [Phase 3]
│   ├── asym_dct_lj_mel.yaml    LJSpeech mel — AsymFM       [Phase 3]
│   └── asym_dct_lj_mel_b.yaml  LJSpeech mel — DiT-B        [Phase 4]
│
├── src/asymflow_audio/
│   ├── data/
│   │   ├── sc09.py             SC09 raw waveform loader + µ-law encode/decode
│   │   ├── melspec.py          MelSpecExtractor, MelNormalizer, HiFi-GAN vocoder
│   │   ├── sc09_mel.py         SC09 mel-spec dataset (12 tokens × 640-dim)
│   │   └── ljspeech.py         LJSpeech raw + mel datasets (25 tokens × 640-dim)
│   ├── model/
│   │   └── dit1d.py            1D DiT: TimestepEmbedder, DiTBlock, AdaLN-zero, FinalLayer
│   ├── flow/
│   │   ├── projector.py        DCTProjector, DCT2DProjector, PCAProjector
│   │   ├── loss.py             fm_loss, asym_fm_loss, recover_velocity, sample_xt
│   │   └── sampler.py          euler_sample (50-step Euler ODE)
│   ├── train.py                training loop: EMA, bf16, grad accum, W&B, multi-domain
│   └── eval.py                 FAD eval with mel→vocoder pipeline for fair comparison
│
├── scripts/
│   ├── analyze_low_rank.py     Phase 0: PCA variance curves (run first!)
│   ├── download_sc09.sh
│   ├── download_ljspeech.sh
│   ├── download_hifigan.sh
│   ├── compute_pca_basis.py
│   └── rank_ablation.py        rank sweep r∈{2,4,8,16,32,48}
│
└── tests/
    └── test_asym_core.py       correctness tests: 1D + 2D projectors, loss equivalence,
                                  velocity recovery, synthetic sanity check
```

---

## Key Design Decisions

**DCT2D projector via Kronecker product.** For mel patches `(80, 8)`, the separable 2D DCT basis is `kron(A_freq, A_time)`. This is equivalent to applying 1D DCT along each axis and thresholding. Advantage: parameter-free, physically motivated, no training or precomputation. Rows are orthonormal → `P = AᵀA` is exact.

**Patch size = 640 for mel.** One token = 80 mel bins × 8 time frames = 80ms at 16kHz. Large enough to capture formant structure; small enough that the low-rank argument holds per-patch.

**Vocoder round-trip for fair FAD.** For mel-spec models, both generated and reference audio pass through `mel → HiFi-GAN → wav → VGGish`. The vocoder's quality floor cancels. The FAD measures the model's generative quality above that floor, not the vocoder quality itself.

**σ_t clamping.** Velocity recovery divides by `σ_t = 1−t`. Clamped at `1e-3` to prevent explosion near `t=1`. Standard in flow models.

---

## Risks

- **Vocoder mismatch.** HiFi-GAN was trained at 22.05kHz; we use 16kHz. Mitigated by the round-trip eval design — both sides go through the same vocoder, so the mismatch is symmetric.
- **Raw waveform may not show improvement.** This is expected and informative — the variance analysis (Phase 0) predicts this and explains why.
- **Compute overrun.** Phase 4 is explicitly stretch — skip if Phase 3 takes longer than ~40 GPU-hours.

---

## Citation

If you use this work, please cite the original AsymFlow paper:

```bibtex
@article{chen2026asymflow,
  title     = {Asymmetric Flow Models},
  author    = {Chen, Hansheng and Ackermann, Jan and Kim, Minseo and
               Wetzstein, Gordon and Guibas, Leonidas},
  journal   = {arXiv preprint arXiv:2605.12964},
  year      = {2026}
}
```
