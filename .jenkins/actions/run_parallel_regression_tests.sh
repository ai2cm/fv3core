#!/bin/bash
set -e -x
BACKEND=$1
EXPNAME=$2
ARGS="-v -s -rsx --backend=${BACKEND}"
export EXPERIMENT=${EXPNAME}
export NUM_RANKS=`echo ${EXPNAME} | grep -o -E '[0-9]+ranks' | grep -o -E '[0-9]+'`

# get the test data version from the Makefile
FORTRAN_VERSION=`grep "FORTRAN_SERIALIZED_DATA_VERSION=" Makefile  | cut -d '=' -f 2`
if [ -z ${SCRATCH} ] ; then
    SCRATCH=`pwd`

fi

export TEST_DATA_HOST="${SCRATCH}/fv3core_fortran_data/${FORTRAN_VERSION}/${EXPNAME}/"
# assume test data is already present from sequential testing
if [ ${host} == "daint" ] ; then #container_engine == sarus
    export CONTAINER_ENGINE="srun sarus"
    export RM_FLAG="--mpi"
    export FV3_IMAGE="load/library/fv3core"
    export MPIRUN_CALL=""
fi
# The default of this set to 1 causes a segfault
export MPICH_RDMA_ENABLED_CUDA=0
make run_tests_parallel TEST_ARGS="${ARGS}"
