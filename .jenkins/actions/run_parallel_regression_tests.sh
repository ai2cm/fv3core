#!/bin/bash
set -e -x
BACKEND=$1
EXPNAME=$2
export TEST_ARGS="-v -s -rsx --which_modules=CubedToLatLon --backend=${BACKEND} ${THRESH_ARGS}"

# sync the test data
make get_test_data

if [ ${host} == "daint" ]; then
    make test_venv_parallel
else
    make tests_mpi
fi
