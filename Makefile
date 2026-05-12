PYTHON ?= python

.PHONY: setup data features train predict submit eda clean

setup:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) scripts/make_dataset.py

features:
	$(PYTHON) scripts/build_features.py

train:
	$(PYTHON) scripts/train.py

predict:
	$(PYTHON) scripts/predict.py

submit:
	$(PYTHON) scripts/make_submission.py

eda:
	$(PYTHON) scripts/run_eda.py

clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__') if p.is_dir()]; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.ipynb_checkpoints') if p.is_dir()]"

