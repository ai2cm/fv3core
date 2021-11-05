#!/bin/bash
set -e -x
BACKEND=$1
EXPNAME=$2
XML_REPORT="sequential_test_results.xml"
export TEST_ARGS="-v -s -rsx --backend=${BACKEND} "

# sync the test data
make get_test_data

if [ ${python_env} == "virtualenv" ]; then
    export TEST_ARGS="${TEST_ARGS} --junitxml=${jenkins_dir}/${XML_REPORT}"
    export CONTAINER_CMD="srun"
    COVERAGE=y make tests
    make savepoint_tests
else
    export TEST_ARGS="${TEST_ARGS} --junitxml=/.jenkins/${XML_REPORT}"
    export VOLUMES="-v ${pwd}/.jenkins:/.jenkins"
    COVERAGE=y make tests
    make savepoint_tests
fi
