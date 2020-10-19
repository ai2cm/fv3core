# setup (use XXX=<value> make <target> to override)
include Makefile.image_names

GT4PY_TAG ?= vf462965
GT4PY_REFERENCE = git+https://github.com/VulcanClimateModeling/gt4py.git@$(GT4PY_TAG)

BUILD_ARGS += \
	--build-arg FMS_IMAGE=$(FMS_IMAGE) \
	--build-arg ESMF_IMAGE=$(ESMF_IMAGE) \
	--build-arg MPI_IMAGE=$(MPI_IMAGE) \
	--build-arg SERIALBOX_IMAGE=$(SERIALBOX_IMAGE) \
	--build-arg FORTRAN_IMAGE=$(FORTRAN_IMAGE) \
	--build-arg WRAPPER_IMAGE=$(WRAPPER_IMAGE) \
	--build-arg ENV_GT4PY_OPTIONALS=$(ENV_GT4PY_OPTIONALS) \
	--build-arg ENV_CUDA_PATH=$(ENV_CUDA_PATH) \
	--build-arg GT4PY_REFERENCE=$(GT4PY_REFERENCE)

# sarus paths
CORE_TAR=$(FV3_TAG).tar
CORE_BUCKET_LOC=gs://vcm-jenkins/$(CORE_TAR)

build: fv3core_image

dev_wrapper:
	FV3CORE_WRAPPER_IMAGE=$(FV3CORE_WRAPPER_IMAGE) docker-compose run fv3core_wrapper bash

.PHONY: dev fortran_image wrapper_image fv3core_image build build_deps push_deps pull_deps

fortran_image:
	docker build \
		$(BUILD_ARGS) \
		-f $(FORTRAN_DOCKERFILE) \
		-t $(FORTRAN_IMAGE) \
		../external/fv3gfs-fortran

fv3core_image: ## build production container image
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

pull_deps: ## pull container images of dependencies from GCP (for faster builds)
	docker pull $(MPI_IMAGE)
	docker pull $(FMS_IMAGE)
	docker pull $(ESMF_IMAGE)
	docker pull $(SERIALBOX_IMAGE)

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
		sarus load ./$(CORE_TAR) $(SARUS_FV3CORE_IMAGE); \
        fi

cleanup_remote:
	$(MAKE) -C docker cleanup_remote
	gsutil rm $(CORE_BUCKET_LOC)
	gcloud container images delete -q --force-delete-tags $(FV3_IMAGE)
