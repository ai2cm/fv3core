#!/bin/bash
set -e -x
BACKEND=$1
EXPNAME=$2
ARGS="-v -s -rsx --backend=${BACKEND} ${THRESH_ARGS}"
# sync the test data
make get_test_data
make tests_mpi TEST_ARGS="${ARGS} --which_modules=FVSubgridZ"
set +e
make tests_mpi TEST_ARGS="${ARGS} --python_regression"
if [ $? -ne 0 ] ; then
    echo "PYTHON REGRESSIONS failed, looking for errors in the substeps:"
    set -e
    make tests TEST_ARGS="${ARGS}"
    make tests_mpi TEST_ARGS="${ARGS}"
    exit 1
fi
set -e
exit 0
