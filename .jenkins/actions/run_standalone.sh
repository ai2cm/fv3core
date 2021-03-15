#!/bin/bash

# Jenkins action to run a benchmark of dynamics.py on Piz Daint
# 3/11/2021, Tobias Wicky, Vulcan Inc

# stop on all errors and echo commands
set -e -x

# utility function for error handling
exitError()
{
    echo "ERROR $1: $3" 1>&2
    echo "ERROR     LOCATION=$0" 1>&2
    echo "ERROR     LINE=$2" 1>&2
    exit $1
}

# check arguments
if [ "$1" == "profile" ] ; then
    DO_PROFILE="true"
fi
if [ "$1" == "build_cache" ] ; then
    BUILD_CACHE="true"
fi
# only save timings if this is neither a cache build nor a profiling run
if [ "${BUILD_CACHE}" != "true" -a "${DO_PROFILE}" != "true" ] ; then
    SAVE_TIMINGS="true"
fi

# configuration
SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
ROOT_DIR="$(dirname "$(dirname "$SCRIPTPATH")")"
FORTRAN_SERIALIZED_DATA_VERSION=7.2.5
TIMESTEPS=2
RANKS=6
BENCHMARK_DIR=${ROOT_DIR}/examples/standalone/benchmarks
DATA_DIR="/project/s1053/fv3core_serialized_test_data/${FORTRAN_SERIALIZED_DATA_VERSION}/${experiment}"
ARTIFACT_DIR="/project/s1053/performance/fv3core_monitor/${backend}"
CACHE_DIR="/scratch/snx3000/olifu/jenkins/scratch/store_gt_caches/${experiment}/${backend}"

# check sanity of environment
test -n "${experiment}" || exitError 1001 ${LINENO} "experiment is not defined"
test -n "${backend}" || exitError 1002 ${LINENO} "backend is not defined"
if [ ! -d "${DATA_DIR}" ] ; then
    exitError 1003 ${LINENO} "test data directory ${DATA_DIR} does not exist"
fi
if [ ! -d "${ARTIFACT_DIR}" ] ; then
    exitError 1004 ${LINENO} "Artifact directory ${ARTIFACT_DIR} does not exist"
fi
if [ ! -d "${BENCHMARK_DIR}" ] ; then
    exitError 1005 ${LINENO} "Benchmark directory ${BENCHMARK_DIR} does not exist"
fi

# run standalone
if [ "${DO_PROFILE}" != "true" ] ; then
    profile="--profile"
fi
cmd="${run_script} ${TIMESTEPS} ${RANKS} ${backend} ${DATA_DIR} '' '${profile}'"
echo "Run command: ${cmd}"
${cmd}

# store cache artifacts (and remove caches afterwards)
if [ "${BUILD_CACHE}" == "true" ] ; then
    mkdir -p ${CACHE_DIR}
    rm -rf ${CACHE_DIR}/.gt_cache*
    mv .gt_cache* ${CACHE_DIR}/
fi
rm -rf .gt_cache*

# store timing artifacts
if [ "${SAVE_TIMINGS}" ] ; then
    cp $ROOT_DIR/*.json ${ARTIFACT_DIR}/
fi

# run analysis and store profiling artifacts
if [ "${DO_PROFILE}" != "true" ] ; then
    ${BENCHMARK_DIR}/process_profiling.sh
    cp $ROOT_DIR/*.prof ${ARTIFACT_DIR}/
fi

# remove venv (too many files!)
rm -rf $ROOT_DIR/venv

