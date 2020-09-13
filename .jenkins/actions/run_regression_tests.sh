#!/bin/bash
set -e -x
BACKEND=$1
EXPNAME=$2
ARGS="-v -s -rsx --backend=${BACKEND}"
export EXPERIMENT=${EXPNAME}
export NUM_RANKS=`echo ${EXPNAME} | grep -o -E '[0-9]+ranks' | grep -o -E '[0-9]+'`

#shopt -s expand_aliases
#cd=`pwd`
#. ${cd}/.jenkins/env/env.${host}.sh
#. ${cd}/.jenkins/env/schedulerTools.sh
module add /project/d107/install/modulefiles/
module load gcloud
module load daint-gpu
module load sarus

scheduler_script="`dirname $0`/env/submit.${host}.${scheduler}"
if [ -f ${scheduler_script} ] ; then
    cp  ${scheduler_script} job_sequential.sh
    scheduler_script_sequential=job_sequential.sh
fi


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
    make sarus_load_tar
    export CONTAINER_ENGINE="sarus"
    export RM_FLAG=""
    export FV3_IMAGE="load/library/fv3core"
fi

run_command "make run_tests_sequential TEST_ARGS=\"${ARGS}\"" FV3CoreSequentialTests ${scheduler_script_sequential}


if [ -f ${scheduler_script} ] ; then
    cp  ${scheduler_script} job_parallel.sh
    scheduler_script_parallel=job_parallel.sh
    sed -i 's|<NTASKS>|"'${NUM_RANKS}'"|g' ${scheduler_script_parallel}
fi
if [ ${host} == "daint" ] ; then #container_engine == sarus
    export CONTAINER_ENGINE="srun sarus"
    export RM_FLAG="--mpi"
    export MPIRUN_CALL=""
fi
# The default of this set to 1 causes a segfault
export MPICH_RDMA_ENABLED_CUDA=0
run_command "make run_tests_parallel TEST_ARGS=\"${ARGS}\"" FV3CoreParallelTests ${scheduler_script_parallel}
