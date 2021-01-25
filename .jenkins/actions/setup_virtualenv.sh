#!/bin/bash
set -e -x

if [ ${host} == "daint" ]; then
    if [ -d ${VIRTUALENV} ]; then
        echo "Using existing virtualenv ${VIRTUALENV}"
    else
        ${root}/install_virtualenv.sh ${VIRTUALENV}
    fi
fi
