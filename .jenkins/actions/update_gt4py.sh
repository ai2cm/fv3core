#!/bin/bash
set -e
set -x
# Run when we change the gt4py source code
export CUDA=y
make pull_environment
make -C docker build_gt4py
make container_gt4py_tests
make tests
make tests_mpi
# TODO uncomment when verify the above works
#make -C docker push_gt4py
