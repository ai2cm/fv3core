#!/bin/bash
set -e -x
BACKEND=$1
EXPNAME=$2
echo "${JOB_NAME}-${BUILD_NUMBER}"
echo `pip list`
echo `which python`
echo "FV3_PATH"
echo ${FV3_PATH}
ARGS="-v -s -rsx --backend=${BACKEND} --which_modules=CubedToLatLon"
TEST_ARGS="${ARGS}" make test_venv_parallel
