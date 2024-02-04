download:
	poetry run kaggle competitions download -c rsna-2023-abdominal-trauma-detection -p input

extract:
	7z x input/rsna-2023-abdominal-trauma-detection.zip -oinput/rsna-2023-abdominal-trauma-detection

# venv:
# 	poetry new --src rsna-2023
# 	poetry add -G dev kaggle
# 	poetry source add --priority=explicit torch-cu118 https://download.pytorch.org/whl/cu118
# 	poetry add --source torch-cu118 torch
# 	poetry add --source torch-cu118 torchvision
# 	cd .venv/lib/python3.10/site-packages/torch/lib/; ln -s libnvrtc-672ee683.so.11.2 libnvrtc.so

run:
	mkdir -p working/
	cd working/; \
	export HYDRA_FULL_ERROR=1; \
	poetry run python ../scripts/run.py hydra.job.chdir=False datamodule.fold=0

# export HYDRA_FULL_ERROR=1; poetry run python ../scripts/run.py --config-name "$(cfg_name)".yaml hydra.job.chdir=False datamodule.fold=0
run_cfg: # make run_cfg cfg="selected_config.yaml"
	mkdir -p working/
	cd working/; \
	export HYDRA_FULL_ERROR=1; \
	poetry run python ../scripts/run.py --config-name "$(cfg)" hydra.job.chdir=False datamodule.fold=0

run_folds: clean_working
	mkdir -p assets/
	rm -f assets/model_*.ckpt
	mkdir -p working/
	cd working/; \
	export HYDRA_FULL_ERROR=1; \
	for n in $$(seq 0 4); \
	do \
	poetry run python ../scripts/run.py hydra.job.chdir=False datamodule.fold=$$n; \
	cp $$(find . -name "*.ckpt" | sort -V | tail -n 1) ../assets/model_$$n.ckpt; \
	done

tensorboard:
	poetry run tensorboard --logdir working/lightning_logs/

notebook:
	mkdir -p notebooks; cd notebooks; poetry run jupyter notebook

clean_all: clean_temp clean_working

clean_temp:
	rm -rf temp/

clean_working:
	rm -rf working/

submission: # make submission msg="Add something"
	cp -r src/rsna_2023 assets/
	rm -rf assets/rsna_2023/__pycache__
	poetry run kaggle datasets version -p assets -m "$(msg)" -r zip
# cp configs/config.yaml assets/
# cp $$(find . -name "*.ckpt" | sort -V | tail -n 1) assets/model.ckpt
# kaggle datasets init -p assets
# edit config!
# kaggle datasets create -p assets -r zip

# wget https://github.com/Project-MONAI/model-zoo/releases/download/hosting_storage_v1/wholeBody_ct_segmentation_v0.1.9.zip -P assets
# 7z x assets/wholeBody_ct_segmentation_v0.1.9.zip -oassets

.PHONY: download extract run tensorboard clean_all clean_temp clean_working submission
