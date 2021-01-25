#!/bin/bash
set -e -x

if [ ${host} == "daint" ]; then
    if [ -d ${VIRTUALENV} ]; then
        echo "Using existing virtualenv ${VIRTUALENV}"
    else
        ${root}/install_virtualenv.sh ${VIRTUALENV}
    fi
else
    export DOCKER_BUILDKIT=1
    make pull_environment
    make build
    make push_core
fi
