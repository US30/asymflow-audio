.PHONY: install data test train-fm train-asym repro ablation

install:
	pip install -e ".[dev]"

data:
	bash scripts/download_sc09.sh data/sc09

pca-basis:
	python scripts/compute_pca_basis.py --data_root data/sc09 --rank 8 --out data/sc09_pca_basis.pt

test:
	pytest tests/ -v

train-fm:
	python -m asymflow_audio.train configs/base_fm.yaml

train-asym:
	python -m asymflow_audio.train configs/asym_dct.yaml

repro: train-fm train-asym
	@echo "Both models trained. Now run eval manually."

ablation:
	python scripts/rank_ablation.py \
		--ref_dir data/sc09/test_wavs \
		--steps 50000 \
		--ranks 2,4,8,16,32,48
