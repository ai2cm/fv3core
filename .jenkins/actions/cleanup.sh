#!/bin/bash
set -e -x

if [ ${python_env} == "virtualenv" ]; then
  rm -rf ${VIRTUALENV}
else
  make cleanup_remote
fi
