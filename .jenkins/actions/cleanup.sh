#!/bin/bash
set -e -x

if [ ${host} == "daint" ]; then
  rm -rf ${VIRTUALENV}
else
  make cleanup_remote
fi
