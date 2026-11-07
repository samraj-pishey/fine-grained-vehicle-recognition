# Common tasks. DATA_ROOT defaults to ./data_root (override: make validate-data DATA_ROOT=/path).
PY ?= .venv/bin/python
DATA_ROOT ?= data_root
export DATA_ROOT

.PHONY: setup data validate-data cloud-package notebooks test test-fast lint secret-scan classical app docker-build docker-run clean-caches

setup:            ## create .venv from the lock file (research + dev extras)
	uv sync --extra research --extra dev

data:             ## acquire original archives (resumable) and extract read-only
	scripts/acquire_dataset.sh $(DATA_ROOT) && scripts/extract_dataset.sh $(DATA_ROOT)

validate-data:    ## parse annotations, validate integrity, build canonical tables + split
	$(PY) scripts/validate_dataset.py --data-root $(DATA_ROOT)

cloud-package:    ## train/val-only table for Colab/Kaggle (no test rows)
	$(PY) scripts/prepare_training_data.py --task package --canonical $(DATA_ROOT)/metadata/canonical_annotations.csv --out $(DATA_ROOT)/cloud_package

notebooks:        ## regenerate Colab/Kaggle notebooks
	$(PY) scripts/build_notebooks.py

test:             ## full test suite (data/model tests skip if artefacts are absent)
	$(PY) -m pytest

test-fast:
	$(PY) -m pytest -m "not data and not models"

lint:
	$(PY) -m ruff check src scripts tests app.py

secret-scan:      ## run before every public push
	$(PY) scripts/secret_scan.py

classical:        ## classical CV feature extraction + ablations (validation only)
	$(PY) scripts/run_classical.py --data-root $(DATA_ROOT)

app:              ## run the Streamlit app locally
	$(PY) -m streamlit run app.py

docker-build:
	docker build -t vehicle-recognition:latest .

docker-run:
	docker run --rm -p 7860:7860 vehicle-recognition:latest

clean-caches:     ## list regenerable caches and delete only after confirmation (raw data is never touched)
	@du -sh $(DATA_ROOT)/roi_cache $(DATA_ROOT)/processed 2>/dev/null || true
	@read -p "Delete the regenerable caches listed above? [y/N] " a; if [ "$$a" = "y" ]; then rm -rf $(DATA_ROOT)/roi_cache/* $(DATA_ROOT)/processed/*; echo deleted; else echo kept; fi
