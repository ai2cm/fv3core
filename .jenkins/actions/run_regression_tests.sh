#!/bin/bash
set -e -x
BACKEND=$1
EXPNAME=$2
ARGS="-v -s -rsx --backend=${BACKEND} --junitxml=/.jenkins/sequential_test_results.xml ${THRESH_ARGS} --which_modules=XPPM"

# sync the test data
make get_test_data

make run_tests_sequential TEST_ARGS="${ARGS}"
