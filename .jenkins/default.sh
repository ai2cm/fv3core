#!/bin/bash
set -e
set -x
BACKEND=$1
ARGS="-v -s -rsx --backend=${BACKEND}"
make tests TEST_ARGS="${ARGS}"
make tests_mpi TEST_ARGS="${ARGS}"
