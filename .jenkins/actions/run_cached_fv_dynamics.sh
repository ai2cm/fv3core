#!/bin/bash
set -e -x
BACKEND=$1
EXPNAME=$2
export TEST_ARGS="-v -s -rsx --backend=${BACKEND} --which_modules=FVDynamics"

# sync the test data
make get_test_data

make tests_venv_mpi

