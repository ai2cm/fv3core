#!/bin/bash
set -e
set -x
# Run when we change the gt4py source code
export CUDA=y
make pull_environment
make gt4py_tests_gpu
