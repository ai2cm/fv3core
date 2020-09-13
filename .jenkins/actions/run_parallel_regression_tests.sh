#!/bin/bash
set -e -x
BACKEND=$1
EXPNAME=$2
ARGS="-v -s -rsx --backend=${BACKEND}"
export EXPERIMENT=${EXPNAME}

# The default of this set to 1 causes a segfault
export MPICH_RDMA_ENABLED_CUDA=0
make run_tests_parallel TEST_ARGS="${ARGS}"
