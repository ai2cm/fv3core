#!/bin/bash
set -e -x
if [ -z ${daintenv} ]; then
    echo "daintenv is not set"
    exit 1
fi
if [ ${host} == "daint" ]; then
    if [ -d ${daintev} ]; then
        echo "Using existing virtualenv ${daintenv}"
    else
        ${root}/install_virtualenv.sh ${daintenv}
    fi
fi
