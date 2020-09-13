#!/bin/bash
set -e -x
BACKEND=$1
EXPNAME=$2
ARGS="-v -s -rsx --backend=${BACKEND}"
export EXPERIMENT=${EXPNAME}

# get the test data version from the Makefile
FORTRAN_VERSION=`grep "FORTRAN_SERIALIZED_DATA_VERSION=" Makefile  | cut -d '=' -f 2`
if [ -z ${SCRATCH} ] ; then
    SCRATCH=`pwd`
fi

export TEST_DATA_HOST="${SCRATCH}/fv3core_fortran_data/${FORTRAN_VERSION}/${EXPNAME}/"
# sync the test data if it does not live in /scratch
if [ ! -d ${TEST_DATA_HOST} ] ; then
    make get_test_data
fi
if [ ${host} == "daint" ] ; then
    export CONTAINER_ENGINE="sarus"
    export RM_FLAG=""
    export FV3_IMAGE="load/library/fv3core"
fi

make run_tests_sequential TEST_ARGS="${ARGS}"
