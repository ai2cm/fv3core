#!/bin/bash
set -e
set -x
# Run when dependency.Dockerfile changes the environment image, serialbox or mpich
export PULL=False
make tests
make tests_mpi
make push_environment
export CUDA=y
make build_environment
make push_environment
