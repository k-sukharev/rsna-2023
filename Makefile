download:
	poetry run kaggle competitions download -c rsna-2023-abdominal-trauma-detection -p input

extract:
	7z x input/rsna-2023-abdominal-trauma-detection.zip -oinput/rsna-2023-abdominal-trauma-detection

run:
	mkdir -p working/
	cd working/; \
	export HYDRA_FULL_ERROR=1; \
	poetry run python ../scripts/run.py hydra.job.chdir=False datamodule.fold=0

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

.PHONY: download extract run tensorboard clean_all clean_temp clean_working submission
