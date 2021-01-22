#!/bin/bash
set -e -x

if [ ${host} == "daint" ]; then
  rm -rf ${daintenv}
else
  make cleanup_remote
fi
