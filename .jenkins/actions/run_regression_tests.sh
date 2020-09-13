#!/bin/bash
set -e -x
BACKEND=$1
EXPNAME=$2
ARGS="-v -s -rsx --backend=${BACKEND}"
export EXPERIMENT=${EXPNAME}

# sync the test data if it does not live in /scratch
if [ ! -d ${TEST_DATA_HOST} ] ; then
    make get_test_data
fi

make run_tests_sequential TEST_ARGS="${ARGS}"
