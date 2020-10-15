# setup (use XXX=<value> make <target> to override)

# update this if you need everything to rebuild for backwards-incompatible changes
VERSION = v0.1.0
GCR_URL ?= us.gcr.io/vcm-ml
CUDA ?= n
FORTRAN_NAME = fv3core-build-fortran
WRAPPER_NAME = fv3core-build-wrapper
FV3CORE_NAME = fv3core
FV3CORE_WRAPPER_NAME = fv3core-wrapper

FORTRAN_DOCKERFILE = dependencies.Dockerfile
FV3CORE_DOCKERFILE = Dockerfile

# fv3core or fv3core-wrapper
FV3CORE_TARGET ?= fv3core

# base images w/ or w/o CUDA
ifeq ($(CUDA),n)
	BASE_IMAGE = ubuntu:19.10
	TAG_NAME = $(VERSION)-gnu9-mpich314-nocuda
	GT4PY_OPTIONALS = ""
else
	BASE_IMAGE = nvidia/cuda:10.2-devel-ubuntu18.04
	TAG_NAME = $(VERSION)-gnu8-mpich314-cuda102
	GT4PY_OPTIONALS = "[cuda102]"
endif
BUILD_ARGS += --network host --build-arg BASE_IMAGE=$(BASE_IMAGE)

# dependency image names, update sha to update version
GT4PY_IMAGE ?= $(GCR_URL)/gt4py-src:$(TAG_NAME)
MPI_IMAGE ?= $(GCR_URL)/mpi-build:$(TAG_NAME)
FMS_IMAGE ?= $(GCR_URL)/fms-build:$(TAG_NAME)
ESMF_IMAGE ?= $(GCR_URL)/esmf-build:$(TAG_NAME)
SERIALBOX_IMAGE ?= $(GCR_URL)/serialbox-build:$(TAG_NAME)

# built image names
FORTRAN_IMAGE = $(GCR_URL)/$(FORTRAN_NAME):$(TAG_NAME)
WRAPPER_IMAGE = $(GCR_URL)/$(WRAPPER_NAME):$(TAG_NAME)
FV3CORE_IMAGE ?= $(GCR_URL)/$(FV3CORE_NAME):$(TAG_NAME)
FV3CORE_WRAPPER_IMAGE ?= $(GCR_URL)/$(FV3CORE_WRAPPER_NAME):$(TAG_NAME)

BUILD_ARGS += \
	--build-arg FMS_IMAGE=$(FMS_IMAGE) \
	--build-arg ESMF_IMAGE=$(ESMF_IMAGE) \
	--build-arg MPI_IMAGE=$(MPI_IMAGE) \
	--build-arg GT4PY_IMAGE=$(GT4PY_IMAGE) \
	--build-arg SERIALBOX_IMAGE=$(SERIALBOX_IMAGE) \
	--build-arg FORTRAN_IMAGE=$(FORTRAN_IMAGE) \
	--build-arg WRAPPER_IMAGE=$(WRAPPER_IMAGE) \
	--build-arg GT4PY_OPTIONALS=$(GT4PY_OPTIONALS)

# sarus paths
CORE_TAR=$(FV3_TAG).tar
CORE_BUCKET_LOC=gs://vcm-jenkins/$(CORE_TAR)

build: fv3core_image

image_name:
	$(FV3CORE_IMAGE)

tag_name:
	$(TAG_NAME)

dev_wrapper:
	FV3CORE_WRAPPER_IMAGE=$(FV3CORE_WRAPPER_IMAGE) docker-compose run fv3core_wrapper bash

.PHONY: dev fortran_image wrapper_image fv3core_image build build_deps push_deps pull_deps

fortran_image: ## build production container image
	docker build \
		$(BUILD_ARGS) \
		-f $(FORTRAN_DOCKERFILE) \
		-t $(FORTRAN_IMAGE) \
		../external/fv3gfs-fortran

fv3core_image: fortran_image
	docker build \
		$(BUILD_ARGS) \
		-f $(FV3CORE_DOCKERFILE) \
		-t $(FV3CORE_IMAGE) \
		--target=$(FV3CORE_TARGET) \
		..

fv3core_wrapper_image: fortran_image
	FV3CORE_TARGET=fv3core_wrapper FV3CORE_IMAGE=$(FV3CORE_WRAPPER_IMAGE) $(MAKE) fv3core_image

build_deps: ## build container images of dependencies
	docker build -f $(FORTRAN_DOCKERFILE) -t $(MPI_IMAGE) $(BUILD_ARGS) --target fv3gfs-mpi ../external/fv3gfs-fortran
	docker build -f $(FORTRAN_DOCKERFILE) -t $(FMS_IMAGE) $(BUILD_ARGS) --target fv3gfs-fms ../external/fv3gfs-fortran
	docker build -f $(FORTRAN_DOCKERFILE) -t $(ESMF_IMAGE) $(BUILD_ARGS) --target fv3gfs-esmf ../external/fv3gfs-fortran
	docker build -f $(FORTRAN_DOCKERFILE) -t $(SERIALBOX_IMAGE) $(BUILD_ARGS) --target fv3gfs-environment-serialbox .
	docker build -f $(FORTRAN_DOCKERFILE) -t $(GT4PY_IMAGE) $(BUILD_ARGS) --target gt4py-src .

push_deps: ## push container images of dependencies to GCP
	docker push $(MPI_IMAGE)
	docker push $(FMS_IMAGE)
	docker push $(ESMF_IMAGE)
	docker push $(SERIALBOX_IMAGE)
	docker push $(GT4PY_IMAGE)

pull_deps: ## pull container images of dependencies from GCP (for faster builds)
	docker pull $(MPI_IMAGE)
	docker pull $(FMS_IMAGE)
	docker pull $(ESMF_IMAGE)
	docker pull $(SERIALBOX_IMAGE)
	docker pull $(GT4PY_IMAGE)

push:
	docker push $(FV3CORE_IMAGE)

pull:
	docker pull $(FV3CORE_IMAGE)

tar_core:
	docker save $(FV3CORE_IMAGE) -o $(CORE_TAR)
	gsutil copy $(CORE_TAR) $(CORE_BUCKET_LOC)
	$(RM) $(CORE_TAR)

sarus_load_tar:
	export FOUND_IMAGE=`sarus images | grep $(TAG_NAME)`
	if [ -z $(FOUND_IMAGE) ] && [ ! -f `pwd`/$(CORE_TAR) ]; then \
		gsutil copy $(CORE_BUCKET_LOC) . && \
		sarus load ./$(CORE_TAR) $(TAG_NAME); \
        fi

cleanup_remote:
	$(MAKE) -C docker cleanup_remote
	gsutil rm $(CORE_BUCKET_LOC)
	gcloud container images delete -q --force-delete-tags $(FV3_IMAGE)
