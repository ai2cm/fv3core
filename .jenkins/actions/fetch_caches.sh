#!/bin/bash
set -e -x
BACKEND=$1
EXPNAME=$2
SANITIZED_BACKEND=`echo $BACKEND | sed 's/:/_/g'` #sanitize the backend from any ':'

if [ ! -d $(pwd)/.gt_cache ]; then
    # version_file=/scratch/snx3000/olifu/jenkins/scratch/store_gt_caches/$EXPNAME/$SANITIZED_BACKEND/GT4PY_VERSION.txt
    version_file=/scratch/snx3000/tobwi/sbox/test_pr/$EXPNAME/$SANITIZED_BACKEND/GT4PY_VERSION.txt
    if [ -f ${version_file} ]; then
	version=`cat ${version_file}`
    else
	version=""
    fi
    if [ "$version" == "$GT4PY_VERSION" ]; then
        # cp -r /scratch/snx3000/olifu/jenkins/scratch/store_gt_caches/$EXPNAME/$SANITIZED_BACKEND/.gt_cache* .
        cp -r /scratch/snx3000/tobwi/sbox/test_pr/$EXPNAME/$SANITIZED_BACKEND/.gt_cache* .
        # find . -name m_\*.py -exec sed -i "s|\/scratch\/snx3000\/olifu\/jenkins_submit\/workspace\/fv3core-cache-setup\/backend\/$SANITIZED_BACKEND\/experiment\/$EXPNAME\/slave\/daint_submit|$(pwd)|g" {} +
        find . -name m_\*.py -exec sed -i "s|\/scratch\/snx3000\/olifu\/jenkins_submit\/workspace\/gtc_cache_setup\/backend\/$SANITIZED_BACKEND\/experiment\/$EXPNAME\/slave\/daint_submit|$(pwd)|g" {} +
    fi
fi
