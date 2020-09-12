#!/bin/bash
set -e -x
BACKEND=$1
EXPNAME=$2
ARGS="-v -s -rsx --backend=${BACKEND}"
export EXPERIMENT=${EXPNAME}
export NUM_RANKS=`echo ${EXPNAME} | grep -o -E '[0-9]+ranks' | grep -o -E '[0-9]+'`
shopt -s expand_aliases
envloc=`pwd`
. ${envloc}/.jenkins/env/env.${host}.sh
module add /project/d107/install/modulefiles/
module load gcloud
module load daint-gpu
module load sarus

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
if [ ${host} == "daint" ] ; then #container_engine == sarus
    make sarus_load_tar
    export CONTAINER_ENGINE="sarus"
    export RM_FLAG="--mpi"
    export FV3_IMAGE="load/library/fv3core"
    export MPIRUN_CALL=""
fi

make run_tests_parallel TEST_ARGS="${ARGS}"
