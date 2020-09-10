#!/bin/bash
set -e -x
BACKEND=$1
EXPNAME=$2
ARGS="-v -s -rsx --backend=${BACKEND}"
export EXPERIMENT=${EXPNAME}
shopt -s expand_aliases
envloc=`pwd`
. ${envloc}/.jenkins/env/env.${host}.sh
module add /project/d107/install/modulefiles/
module load gcloud
module load daint-gpu
module load sarus
#if [ "`hostname | grep daint`" != "" ] ; then

# get the test data version from the Makefile
FORTRAN_VERSION=`grep "FORTRAN_SERIALIZED_DATA_VERSION=" Makefile  | cut -d '=' -f 2`
if [ -z ${SCRATCH} ] ; then
    SCRATCH=`pwd`
fi
#cd $SCRATCH
#mkdir -p $EXPNAME
#cd $EXPNAME
export TEST_DATA_HOST="${SCRATCH}/fv3core_fortran_data/${FORTRAN_VERSION}/${EXPNAME}/"
# sync the test data if it does not live in /scratch
if [ ! -d ${TEST_DATA_HOST} ] ; then
    make get_test_data
fi
if [ ${host} == "daint" ] ; then
    make sarus_load_tar
    export CONTAINER_ENGINE="sarus"
    export RM_FLAG="-m"
    export FV3_IMAGE="load/library/fv3core"
fi
#get_container
# define command
#cmd="${mpilaunch} sarus run --mount=type=bind,source=${DATA_DIR},destination=/test_data load/library/${FV3_CONTAINER}:latest pytest --data_path=/test_data ${ARGS} /fv3core/tests"

#else
   
#fi
make run_tests_sequential TEST_ARGS="${ARGS}"
make run_tests_parallel TEST_ARGS="${ARGS}"

#cd ../
#rm -r $EXPNAME
