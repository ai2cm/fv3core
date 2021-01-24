#!/bin/bash
set -e -x
if [ -z ${VIRTUALENV} ]; then
    echo "setting VIRTUALENV"
    export ${VIRTUALENV}="${SCRATCH}/vcm_env_${JENKINS_TAG}"
fi
if [ ${host} == "daint" ]; then
    if [ -d ${VIRTUALENV} ]; then
        echo "Using existing virtualenv ${VIRTUALENV}"
    else
        ${root}/install_virtualenv.sh ${VIRTUALENV}
    fi
fi
