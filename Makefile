include docker/Makefile.image_names

GCR_URL = us.gcr.io/vcm-ml
REGRESSION_DATA_STORAGE_BUCKET = gs://vcm-fv3gfs-serialized-regression-data
EXPERIMENT ?=c12_6ranks_standard
FV3CORE_VERSION=0.1.0
FORTRAN_SERIALIZED_DATA_VERSION=7.1.1

SHELL=/bin/bash
CWD=$(shell pwd)
TEST_ARGS ?=-v -s -rsx
PULL ?=True
NUM_RANKS ?=6
VOLUMES ?=
MOUNTS ?=
CONTAINER_ENGINE ?=docker
RUN_FLAGS ?=--rm
TEST_DATA_HOST ?=$(CWD)/test_data/$(EXPERIMENT)
FV3UTIL_DIR=$(CWD)/external/fv3gfs-util
USE_GT4PY_DEVELOP ?=n

ifeq ($(USE_GT4PY_DEVELOP),y)
	RUN_FLAGS += --env USE_GT4PY_DEVELOP=y
endif

FV3=fv3core
ifeq ($(CONTAINER_ENGINE),sarus)
	FV3_IMAGE = load/library/$(SARUS_FV3CORE_IMAGE)
else ifeq ($(CONTAINER_ENGINE),srun)
	FV3_IMAGE = load/library/$(SARUS_FV3CORE_IMAGE)
else
	FV3_IMAGE ?= $(FV3CORE_IMAGE)
endif
FV3_TAG ?= $(TAG_NAME)

TEST_DATA_CONTAINER=/test_data
PYTHON_FILES = $(shell git ls-files | grep -e 'py$$' | grep -v -e '__init__.py')
PYTHON_INIT_FILES = $(shell git ls-files | grep '__init__.py')
TEST_DATA_TARFILE=dat_files.tar.gz
TEST_DATA_TARPATH=$(TEST_DATA_HOST)/$(TEST_DATA_TARFILE)
CORE_TAR=$(FV3_TAG).tar
CORE_BUCKET_LOC=gs://vcm-jenkins/$(CORE_TAR)
MPIRUN_CALL ?=mpirun -np $(NUM_RANKS)
BASE_INSTALL?=$(FV3)-install-serialbox
DEV_MOUNTS = -v $(CWD)/$(FV3):/$(FV3)/$(FV3) -v $(CWD)/tests:/$(FV3)/tests -v $(FV3UTIL_DIR):/usr/src/fv3gfs-util -v $(TEST_DATA_HOST):$(TEST_DATA_CONTAINER)

clean:
	find . -name ""
	$(RM) -rf comparison/wrapped/output/*
	$(MAKE) -C external/fv3gfs-wrapper clean
	$(MAKE) -C external/fv3gfs-fortran clean

update_submodules:
	if [ ! -f $(FV3UTIL_DIR)/requirements.txt  ]; then \
		git submodule update --init --recursive; \
	fi

constraints.txt: requirements.txt requirements_lint.txt
	pip-compile $^ --output-file constraints.txt

# Image build instructions have moved to docker/Makefile but are kept here for backwards-compatibility

build_environment:
	$(MAKE) -C docker build_deps

build_wrapped_environment: build_environment

build: update_submodules
	if [ $(PULL) == True ]; then \
		$(MAKE) pull_environment_if_needed; \
	else \
		$(MAKE) build_environment; \
	fi
	$(MAKE) -C docker fv3core_image

build_wrapped: update_submodules build_wrapped_environment
	$(MAKE) -C docker fv3core_wrapper_image

pull_environment_if_needed: pull_environment  # now too complicated to check for 5 images, let docker check if needed instead

pull_environment:
	$(MAKE) -C docker pull_deps

push_environment:
	$(MAKE) -C docker push_deps

rebuild_environment: build_environment push_environment

push_core:
	$(MAKE) -C docker push

pull_core:
	$(MAKE) -C docker pull

tar_core:
	$(MAKE) -C docker tar_core

sarus_load_tar:
	$(MAKE) -C docker sarus_load_tar

cleanup_remote:
	$(MAKE) -C docker cleanup_remote

# end of image build targets which have been moved to docker/Makefile

tests: #build
	$(MAKE) get_test_data
	$(MAKE) run_tests_sequential

test: tests

tests_mpi: build
	$(MAKE) get_test_data
	$(MAKE) run_tests_parallel

test_mpi: tests_mpi

test_gt4py_develop:
	USE_GT4PY_DEVELOP=y $(MAKE) test test_mpi

dev:
	docker run --rm -it \
		--network host \
		-v $(TEST_DATA_HOST):$(TEST_DATA_CONTAINER) \
		-v $(CWD):/port_dev \
		$(FV3_IMAGE) bash

dev_wrapper:
	$(MAKE) -C docker dev_wrapper

dev_tests:
	VOLUMES=$(DEV_MOUNTS) $(MAKE) test_base

dev_tests_mpi:
	VOLUMES=$(DEV_MOUNTS) $(MAKE) test_base_parallel

dev_test_mpi: dev_tests_mpi

dev_tests_mpi_host:
	MOUNTS=$(DEV_MOUNTS) $(MAKE) run_tests_parallel_host

test_base:
	$(CONTAINER_ENGINE) run $(RUN_FLAGS) $(VOLUMES) $(MOUNTS) $(FV3_IMAGE) pytest --data_path=$(TEST_DATA_CONTAINER) $(TEST_ARGS) /$(FV3)/tests

test_base_parallel:
	$(CONTAINER_ENGINE) run $(RUN_FLAGS) $(VOLUMES) $(MOUNTS) $(FV3_IMAGE) \
	$(MPIRUN_CALL) \
	pytest --data_path=$(TEST_DATA_CONTAINER) $(TEST_ARGS) -m parallel /$(FV3)/tests

run_tests_sequential:
	VOLUMES='--mount=type=bind,source=$(TEST_DATA_HOST),destination=$(TEST_DATA_CONTAINER) --mount=type=bind,source=$(CWD)/.jenkins,destination=/.jenkins' \
	$(MAKE) test_base

run_tests_parallel:
	VOLUMES='--mount=type=bind,source=$(TEST_DATA_HOST),destination=$(TEST_DATA_CONTAINER) --mount=type=bind,source=$(CWD)/.jenkins,destination=/.jenkins' \
	$(MAKE) test_base_parallel

sync_test_data:
	mkdir -p $(TEST_DATA_HOST) && gsutil -m rsync $(REGRESSION_DATA_STORAGE_BUCKET)/$(FORTRAN_SERIALIZED_DATA_VERSION)/$(EXPERIMENT)/ $(TEST_DATA_HOST)

get_test_data:
	if [ ! -f "$(TEST_DATA_HOST)/input.nml" ]; then \
	$(MAKE) sync_test_data && \
	$(MAKE) unpack_test_data ;\
	fi

unpack_test_data:
	if [ -f $(TEST_DATA_TARPATH) ]; then \
	cd $(TEST_DATA_HOST) && tar -xf $(TEST_DATA_TARFILE) && \
	rm $(TEST_DATA_TARFILE); fi

list_test_data_options:
	gsutil ls $(REGRESSION_DATA_STORAGE_BUCKET)/$(FORTRAN_SERIALIZED_DATA_VERSION)

lint:
	pre-commit run
	# pre-commit runs black for now. Will also run flake8 eventually.
	# black --diff --check $(PYTHON_FILES) $(PYTHON_INIT_FILES)
	# disable flake8 tests for now, re-enable when dycore is "running"
	#@flake8 $(PYTHON_FILES)
	# ignore unused import error in __init__.py files
	#@flake8 --ignore=F401 $(PYTHON_INIT_FILES)
	# @echo "LINTING SUCCESSFUL"

.PHONY: update_submodules build_environment build dev dev_tests dev_tests_mpi flake8 lint get_test_data unpack_test_data \
	 list_test_data_options pull_environment pull_test_data push_environment \
	rebuild_environment reformat run_tests_sequential run_tests_parallel test_base test_base_parallel \
	tests update_submodules push_core pull_core tar_core sarus_load_tar cleanup_remote
