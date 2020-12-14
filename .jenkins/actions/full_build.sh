#!/bin/bash
set -e
set -x
export PULL=False
make tests
make tests_mpi
make push_environment
export CUDA=y
make build_environment
make push_environment
