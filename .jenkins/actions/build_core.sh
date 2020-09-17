#!/bin/bash
set -e -x
export DOCKER_BUILDKIT=1
make pull_environment
cp .jenkins/artifact_vars.sh .jenkins/collect_artifact_vars.sh
sed -i 's|<BUILD_NUM>|"'${BUILD_NUMBER}'"|g' .jenkins/collect_artifact_vars.sh
sed -i 's|<TRIGGER>|"'${JOB_NAME}'"|g' .jenkins/collect_artifact_vars.sh
export FV3_TAG="${JOB_NAME}-${BUILD_NUMBER}"
make build
make push_core
make tar_core
