DIST_DIR ?= ${PWD}/dist
BUILD_DIR ?= ${PWD}/build
TAG ?= aed-network

TASK ?= 4
CONFIG_FN ?= configs/dcase2019_${TASK}.yml
JUPYTER_PORT ?= 8891
DEBUG ?= False

.PHONY: container
container:
	docker build \
		-t ${TAG} \
		.

.PHONY: run
run:
	nvidia-docker run \
		--mount type=bind,source="$(shell pwd)",target=/opt \
		--mount type=bind,source=/media/aagnone/wd/corpora,target=/corpora \
		--mount type=bind,source=/media/aagnone/wd/corpora,target=/home/aagnone/corpora \
		-ti \
		--rm \
		${TAG} \
		python main.py -c ${CONFIG_FN}

.PHONY: jup
jup:
	nvidia-docker run \
		--mount type=bind,source="$(shell pwd)",target=/opt \
		--mount type=bind,source=/media/aagnone/wd/corpora,target=/corpora \
		--mount type=bind,source=/media/aagnone/wd/corpora,target=/home/aagnone/corpora \
		-e DISABLE_MP=1 \
		-p ${JUPYTER_PORT}:${JUPYTER_PORT} \
		-ti \
		--rm \
		${TAG} \
		jupyter-lab --allow-root --ip=0.0.0.0 --port=${JUPYTER_PORT} --no-browser 2>&1 | tee log.txt

.PHONY: extract
extract:
	nvidia-docker run \
		--mount type=bind,source="$(shell pwd)",target=/opt \
		--mount type=bind,source=/media/aagnone/wd/corpora,target=/corpora \
		--mount type=bind,source=/media/aagnone/wd/corpora,target=/home/aagnone/corpora \
		-ti \
		--rm \
		${TAG} \
		python feature_extraction/extract.log_mfb_spec.py

.PHONY: debug
debug:
	nvidia-docker run \
		--mount type=bind,source="$(shell pwd)",target=/opt \
		--mount type=bind,source=/media/aagnone/wd/corpora,target=/corpora \
		--mount type=bind,source=/media/aagnone/wd/corpora,target=/home/aagnone/corpora \
		-e DISABLE_MP=1 \
		-ti \
		--rm \
		${TAG} \
		bash
