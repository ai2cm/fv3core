#!/bin/bash
set -e -x
VERSION="${JOB_NAME}-${BUILD_NUMBER}" make cleanup_remote
