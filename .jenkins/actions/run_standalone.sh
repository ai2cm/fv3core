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

# configuration
SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
ROOT_DIR="$(dirname "$(dirname "$SCRIPTPATH")")"
TESTDATA_PATH="/project/s1053/fv3core_serialized_test_data"
ARTIFACT_PATH="/project/s1053/performance/fv3core_monitor/${backend}"
CACHE_PATH="/scratch/snx3000/olifu/jenkins/scratch/store_gt_caches"
FORTRAN_SERIALIZED_DATA_VERSION=7.2.5
TIMESTEPS=2

# check sanity of environment
test -n "${experiment}" || exitError 1001 ${LINENO} "experiment is not defined"
test -n "${backend}" || exitError 1002 ${LINENO} "backend is not defined"
if [ ! -d "${TESTDATA_PATH}" ] ; then
    exitError 1003 ${LINENO} "test data directory ${TESTDATA_PATH} does not exist"
fi
if [ ! -d "${ARTIFACT_PATH}" ] ; then
    exitError 1004 ${LINENO} "Artifact directory ${ARTIFACT_PATH} does not exist"
fi
if [ ! -d "${CACHE_PATH}" ] ; then
    exitError 1005 ${LINENO} "Cache directory ${CACHE_PATH} does not exist"
fi
CACHE_PATH="${CACHE_PATH}/${experiment}/${backend}"

# run benchmark for with profiling
data_path=${TESTDATA_PATH}/${FORTRAN_SERIALIZED_DATA_VERSION}/${experiment}
run_script=${ROOT_DIR}/examples/standalone/benchmarks/run_on_daint.sh
if [ "${DO_PROFILE}" != "true" ] ; then

    # run performance benchmark
    ${run_script} ${TIMESTEPS} ${TIMESTEPS} ${backend} ${data_path} "" ""

    # save timings if this is not just building cache
    if [ "${BUILD_CACHE}" != "true" ] ; then
        cp $ROOT_DIR/*.json ${ARTIFACT_PATH}/
    fi

else

    # run benchmark with profiling using cProfile
    ${run_script} ${TIMESTEPS} ${TIMESTEPS} ${backend} ${data_path} "" "--profile"
    cp $ROOT_DIR/*.prof ${ARTIFACT_PATH}/

    # generate simple profile listing
    source $ROOT_DIR/venv/bin/activate
    cat > $ROOT_DIR/profile.py <<EOF
#!/usr/bin/env python3
import pstats
stats = pstats.Stats("$ROOT_DIR/fv3core_${experiment}_${backend}_0.prof")
stats.strip_dirs()
stats.sort_stats('cumulative')
stats.print_stats(200)
print('=================================================================')
stats.sort_stats('calls')
stats.print_stats(200)
EOF
    chmod 755 $ROOT_DIR/profile.py
    $ROOT_DIR/profile.py > profile.txt

    # convert to html
    mkdir -p html
    echo "<html><body><pre>" > html/index.html
    cat profile.txt >> html/index.html
    echo "</pre></body></html>" >> html/index.html

fi

# save and remove GT4Py cache directories
if [ "${BUILD_CACHE}" == "true" ] ; then
    mkdir -p ${CACHE_PATH}
    rm -rf ${CACHE_PATH}/.gt_cache*
    mv .gt_cache* ${CACHE_PATH}/
fi
rm -rf .gt_cache*

# remove venv (we can use vcm_1.0 for manual stuff)
rm -rf $ROOT_DIR/venv
