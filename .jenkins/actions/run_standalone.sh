#!/bin/bash

# Jenkins action to run a benchmark of dynamics.py on Piz Daint
# 3/11/2021, Tobias Wicky, Vulcan Inc

# stop on all errors and echo commands
set -e

# utility function for error handling
exitError()
{
    echo "ERROR $1: $3" 1>&2
    echo "ERROR     LOCATION=$0" 1>&2
    echo "ERROR     LINE=$2" 1>&2
    exit $1
}

# check arguments
DO_PROFILE="false"
SAVE_CACHE="false"
SAVE_TIMINGS="false"
if [ "$1" == "profile" ] ; then
    DO_PROFILE="true"
fi
if [ "$1" == "build_cache" ] ; then
    SAVE_CACHE="true"
fi
# only save timings if this is neither a cache build nor a profiling run
if [ "${SAVE_CACHE}" != "true" -a "${DO_PROFILE}" != "true" ] ; then
    SAVE_TIMINGS="true"
fi

# configuration
SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
ROOT_DIR="$(dirname "$(dirname "$SCRIPTPATH")")"
DATA_VERSION=`grep 'FORTRAN_SERIALIZED_DATA_VERSION *=' ${ROOT_DIR}/Makefile | cut -d '=' -f 2`
TIMESTEPS=2
RANKS=6
BENCHMARK_DIR=${ROOT_DIR}/examples/standalone/benchmarks
DATA_DIR="/project/s1053/fv3core_serialized_test_data/${DATA_VERSION}/${experiment}"
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

# echo config
echo "=== $0 configuration ==========================="
echo "Script:               ${SCRIPT}"
echo "Do profiling:         ${DO_PROFILE}"
echo "Save GT4Py caches:    ${SAVE_CACHE}"
echo "Save timings:         ${SAVE_TIMINGS}"
echo "Root directory:       ${ROOT_DIR}"
echo "Experiment:           ${experiment}"
echo "Backend:              ${backend}"
echo "Data version:         ${DATA_VERSION}"
echo "Timesteps:            ${TIMESTEPS}"
echo "Ranks:                ${RANKS}"
echo "Benchmark directory:  ${BENCHMARK_DIR}"
echo "Data directory:       ${DATA_DIR}"
echo "Artifact directory:   ${ARTIFACT_DIR}"
echo "Cache directory:      ${CACHE_DIR}"

# run standalone
echo "=== Running standalone ========================="
if [ "${DO_PROFILE}" == "true" ] ; then
    profile="--profile"
fi
cmd="${BENCHMARK_DIR}/run_on_daint.sh ${TIMESTEPS} ${RANKS} ${backend} ${DATA_DIR}"
echo "Run command: ${cmd} \"\" \"${profile}\""
${cmd} "" "${profile}"

echo "=== Post-processing ============================"

# store timing artifacts
if [ "${SAVE_TIMINGS}" == "true" ] ; then
    echo "Copying timing information to ${ARTIFACT_DIR}"
    cp $ROOT_DIR/*.json ${ARTIFACT_DIR}/
fi

# store cache artifacts (and remove caches afterwards)
if [ "${SAVE_CACHE}" == "true" ] ; then
    echo "Copying GT4Py cache directories to ${CACHE_DIR}"
    mkdir -p ${CACHE_DIR}
    rm -rf ${CACHE_DIR}/.gt_cache*
    cp -rp .gt_cache* ${CACHE_DIR}/
fi
rm -rf .gt_cache*

# run analysis and store profiling artifacts
if [ "${DO_PROFILE}" == "true" ] ; then
    echo "Analyzing profiling results"
    ${BENCHMARK_DIR}/process_profiling.sh
    echo "Copying profiling information to ${ARTIFACT_DIR}"
    cp $ROOT_DIR/*.prof ${ARTIFACT_DIR}/
fi

# remove venv (too many files!)
rm -rf $ROOT_DIR/venv

echo "=== Done ======================================="
