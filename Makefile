.PHONY: install data data-lj hifigan pca-basis test \
        analyze \
        train-fm train-asym train-fm-mel train-asym-mel train-fm-lj train-asym-lj train-asym-lj-b \
        repro repro-mel ablation

# ── Setup ─────────────────────────────────────────────────────────────────────

install:
	pip install -e ".[dev]"

data:
	bash scripts/download_sc09.sh data/sc09

data-lj:
	bash scripts/download_ljspeech.sh data

hifigan:
	bash scripts/download_hifigan.sh

pca-basis:
	python scripts/compute_pca_basis.py --data_root data/sc09 --rank 8 --out data/sc09_pca_basis.pt

# ── Phase 0: Variance analysis (run before training!) ─────────────────────────

analyze:
	python scripts/analyze_low_rank.py \
		--sc09_dir data/sc09 \
		--lj_dir   data/LJSpeech-1.1 \
		--n_patches 10000 \
		--out_dir  results
	@echo ""
	@echo "Key figure: results/variance_explained.png"
	@echo "Use elbow ranks to set rank_freq/rank_time in asym configs."

# ── Tests ─────────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v

# ── Phase 1: SC09 raw waveform ────────────────────────────────────────────────

train-fm:
	python -m asymflow_audio.train configs/base_fm.yaml

train-asym:
	python -m asymflow_audio.train configs/asym_dct.yaml

# ── Phase 2: SC09 mel-spectrogram ─────────────────────────────────────────────

train-fm-mel:
	python -m asymflow_audio.train configs/base_fm_sc09_mel.yaml

train-asym-mel:
	python -m asymflow_audio.train configs/asym_dct_sc09_mel.yaml

repro-mel: train-fm-mel train-asym-mel
	@echo "SC09 mel models trained. Evaluate with make eval-mel."

# ── Phase 3: LJSpeech mel-spectrogram ─────────────────────────────────────────

train-fm-lj:
	python -m asymflow_audio.train configs/base_fm_lj_mel.yaml

train-asym-lj:
	python -m asymflow_audio.train configs/asym_dct_lj_mel.yaml

# ── Phase 4: DiT-B stretch ────────────────────────────────────────────────────

train-asym-lj-b:
	python -m asymflow_audio.train configs/asym_dct_lj_mel_b.yaml

# ── Rank ablation (Phase 1 only, SC09 raw) ────────────────────────────────────

ablation:
	python scripts/rank_ablation.py \
		--ref_dir data/sc09/test_wavs \
		--steps 50000 \
		--ranks 2,4,8,16,32,48

# ── Convenience bundles ───────────────────────────────────────────────────────

repro: train-fm train-asym
	@echo "SC09 raw models trained."
