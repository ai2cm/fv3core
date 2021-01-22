#!/bin/bash
set -e -x
BACKEND=$1
EXPNAME=$2
ARGS="-v -s -rsx --backend=${BACKEND} ${THRESH_ARGS}"

# sync the test data
make get_test_data

# The default of this set to 1 causes a segfault
make run_tests_parallel TEST_ARGS="${ARGS}"
