#!/bin/bash
set -e -x
BACKEND=$1
EXPNAME=$2
export TEST_ARGS="-v -s -rsx --backend=${BACKEND} --which_modules=FVDynamics"

# sync the test data
make get_test_data

if [ ! -d $(pwd)/.gt_cache_000000 ]; then
    version_file=/scratch/snx3000/olifu/jenkins/scratch/store_gt_caches/$EXPNAME/$BACKEND/GT4PY_VERSION.txt
    if [ -f ${version_file} ]; then
	version=`cat ${version_file}`
    else
	version=""
    fi
    if [ "$version" == "$GT4PY_VERSION" ]; then
        cp -r /scratch/snx3000/olifu/jenkins/scratch/store_gt_caches/$EXPNAME/$BACKEND/.gt_cache_0000* .
        find . -name m_\*.py -exec sed -i "s|\/scratch\/snx3000\/olifu\/jenkins_submit\/workspace\/fv3core-cache-setup\/backend\/$BACKEND\/experiment\/$EXPNAME\/slave\/daint_submit|$(pwd)|g" {} +
    fi
fi
CONTAINER_CMD="" make translate_tests_mpi
