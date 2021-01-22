#!/bin/bash
set -e -x
echo "${JOB_NAME}-${BUILD_NUMBER}"
echo `pip list`
echo `which python`
echo "FV3_PATH"
echo ${FV3_PATH}
make test_venv_parallel
