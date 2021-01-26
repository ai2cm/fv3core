#!/bin/bash
set -e -x
BACKEND=$1
EXPNAME=$2
XML_REPORT="sequential_test_results.xml"
export TEST_ARGS="-v -s -rsx --backend=${BACKEND} ${THRESH_ARGS}"

# sync the test data
make get_test_data
if [ ${host} == "daint" ]; then
     export TEST_ARGS="${TEST_ARGS} --junitxml=${root}/${XML_REPORT}"
     BASH_PREFIX="srun" make test_venv
else
    export TEST_ARGS="${TEST_ARGS} --junitxml=/.jenkins/${XML_REPORT}"
    make tests
fi
