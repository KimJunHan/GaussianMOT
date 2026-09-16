IMAGE_NAME := gaussiancar
TAG_NAME := v1.0.0
CONTAINER_NAME := $(IMAGE_NAME)_container
WANDB_API_KEY := $(shell echo $$WANDB_API_KEY)
PATH_TO_NUSCENES := /home/junhan/nas2/nuscenes_original_junhan

# Named volume: conda 환경 영속성 보장
CONDA_VOLUME := gaussiancar_conda

GPUS ?= "device=2,3,4"

define run_docker
	@docker run -it --rm \
		--net host \
		--gpus '$(GPUS)' \
		-e NVIDIA_VISIBLE_DEVICES=2,3,4 \
		--ipc host \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		--name=$(CONTAINER_NAME) \
		-v $(CONDA_VOLUME):/opt/conda/envs \
		-v $(PATH_TO_NUSCENES):/data/nuscenes \
		-e WANDB_API_KEY=$(WANDB_API_KEY) \
		-e TERM=xterm-256color \
		$(IMAGE_NAME):$(TAG_NAME) \
		/bin/bash -c "source /entrypoint.sh && bash"
endef

.PHONY: build run attach clear volume-create volume-delete

volume-create:
	docker volume create $(CONDA_VOLUME)
	@echo "✅ Volume '$(CONDA_VOLUME)' created."

volume-delete:
	docker volume rm $(CONDA_VOLUME)
	@echo "🗑️  Volume '$(CONDA_VOLUME)' deleted."

build:
	docker build . -t $(IMAGE_NAME):$(TAG_NAME)
	@echo "\n✅ Build complete!"

run:
	$(call run_docker)

attach:
	docker exec -it $(CONTAINER_NAME) /bin/bash

clear:
	@rm -rf gaussiancar.egg-info/
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@rm -rf gaussiancar/ops/diff-gaussian-rasterization/build/
	@rm -rf gaussiancar/ops/diff-gaussian-rasterization.egg-info/
	@echo "✅ Cleaned."
