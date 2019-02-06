DIST_DIR ?= ${PWD}/dist
BUILD_DIR ?= ${PWD}/build
TAG ?= aed-network

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
		python main.py -c configs/freesound.yml
